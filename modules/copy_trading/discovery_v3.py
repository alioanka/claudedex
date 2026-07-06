"""
COPY v3 profitable-wallet discovery engine.

Finds the best wallets to copy from FREE data sources, scores them on
trailing REALIZED performance (wallet_profitability — walk-forward, no
forward returns), writes the ranked universe to copy_discovered_wallets,
and proposes top wallets into copy_leader_candidates (status='pending')
for OPERATOR APPROVAL.

Sources (all free / already-paid-for):
    smart_money    — smart_money_wallet_scores + smart_money_wallet_events
                     (mig 133, READ-ONLY). Best source: USD-priced swap
                     events per wallet, EVM chains.
    onchain        — our own copytrading_trades mirrored history per
                     source_wallet (realized, USD-exact).
    leader_scores  — existing copy_leader_scores rows (v2 discovery sweep
                     output) re-ranked under the v3 realized-only scorer.
    dexscreener    — v2 wallet_discovery DexScreener fetcher (candidate
                     addresses only; they rank low until another source
                     supplies trade history — honest, not a defect).
    rpc_solana     — bounded Helius enhanced-tx enrichment for top Solana
                     candidates lacking history (SOL-numeraire pricing,
                     flagged pnl_basis='sol_numeraire'). Skipped without a
                     key. Also feeds the shadow simulator via record_events.

PROMOTION SAFETY: this module NEVER adds a live leader silently.
    * Default: candidates land in copy_leader_candidates as 'pending'.
    * Auto-promote requires BOTH copy_v3_auto_promote_enabled=true AND
      copy_v3_auto_promote_max_leaders > 0 (seeded 0), AND the wallet must
      have >= copy_v3_auto_promote_min_shadow_fills simulated fills with
      positive realized shadow PnL. Promotion appends to
      copytrading_config.target_wallets and stamps reviewed_by='auto_promote'.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import aiohttp  # type: ignore
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore

from modules.copy_trading.wallet_profitability import (
    WalletScore,
    score_wallet,
    trades_to_events,
)

logger = logging.getLogger("CopyDiscoveryV3")

# Wave-F5: per-source WARNING throttle (1/source/hour) so a chronically-failing
# source is visible without spamming the log.
import time as _time
_WARN_THROTTLE_S = 3600.0
_last_warn_at: Dict[str, float] = {}


def _warn_rate_limited(source: str, msg: str) -> None:
    now = _time.monotonic()
    if now - _last_warn_at.get(source, 0.0) >= _WARN_THROTTLE_S:
        _last_warn_at[source] = now
        logger.warning(msg)
    else:
        logger.debug(msg)

SOL_MINT = "So11111111111111111111111111111111111111112"
STABLE_MINTS = {
    SOL_MINT,
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
}
DEFAULT_SOL_PRICE_USD = 150.0


@dataclass
class DiscoveryV3Config:
    sources: Sequence[str] = ("smart_money", "onchain", "leader_scores", "dexscreener")
    window_days: int = 30
    min_score: float = 60.0
    min_trades: int = 10
    min_realized_pnl_usd: float = 500.0
    max_lucky_share: float = 0.6
    max_wash_penalty: float = 0.4
    max_candidates_per_sweep: int = 25
    max_rpc_enrich_wallets: int = 5
    auto_promote_enabled: bool = False
    auto_promote_max_leaders: int = 0
    auto_promote_min_shadow_fills: int = 10
    # Wave-F5 Helius quota discipline (mig 141 seeds). The same Helius key is
    # shared with the copy monitor's per-wallet poll, so discovery must be a
    # good tenant: a hard daily call budget + exponential backoff on 429 +
    # cross-sweep fee-payer accumulation (no more "5 swaps in one 100-tx
    # snapshot" sampling flaw — see docs/agents/wave-f5/04_copy_aitrader.md).
    helius_daily_call_budget: int = 500
    helius_tx_sample: int = 100
    discovery_min_swaps: int = 3
    sm_min_score: float = 0.6
    helius_api_key: Optional[str] = None
    shadow_cfg: Optional[object] = None  # shadow_simulator.ShadowConfig


async def load_config(db_pool) -> DiscoveryV3Config:
    """Read v3 knobs from config_settings.copytrading_config (mig 135 seeds).
    Missing keys fall back to the safe defaults above."""
    cfg = DiscoveryV3Config()
    if db_pool is None:
        return cfg
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM config_settings "
                "WHERE config_type = 'copytrading_config' "
                "  AND (key LIKE 'copy_v3_%' "
                "       OR key IN ('helius_daily_call_budget', 'helius_tx_sample', "
                "                  'discovery_min_swaps', 'copy_sm_min_score'))"
            )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"v3 config read failed: {e}")
        return cfg
    kv = {r["key"]: r["value"] for r in rows}

    def _f(key: str, default: float, lo: float, hi: float) -> float:
        try:
            return max(lo, min(hi, float(kv.get(key) or default)))
        except (TypeError, ValueError):
            return default

    def _b(key: str, default: bool) -> bool:
        v = kv.get(key)
        if v is None:
            return default
        return str(v).strip().lower() in ("1", "true", "yes", "on")

    raw_sources = str(kv.get("copy_v3_sources") or "")
    if raw_sources.strip():
        cfg.sources = tuple(
            s.strip().lower() for s in raw_sources.split(",") if s.strip()
        )
    cfg.window_days = int(_f("copy_v3_window_days", 30, 7, 90))
    cfg.min_score = _f("copy_v3_min_score", 60.0, 0.0, 100.0)
    cfg.min_trades = int(_f("copy_v3_min_trades", 10, 1, 1000))
    cfg.min_realized_pnl_usd = _f("copy_v3_min_realized_pnl_usd", 500.0, 0.0, 1e7)
    cfg.max_lucky_share = _f("copy_v3_max_lucky_share", 0.6, 0.0, 1.0)
    cfg.max_wash_penalty = _f("copy_v3_max_wash_penalty", 0.4, 0.0, 1.0)
    cfg.max_candidates_per_sweep = int(_f("copy_v3_max_candidates_per_sweep", 25, 1, 200))
    cfg.max_rpc_enrich_wallets = int(_f("copy_v3_max_rpc_enrich_wallets", 5, 0, 50))
    cfg.auto_promote_enabled = _b("copy_v3_auto_promote_enabled", False)
    cfg.auto_promote_max_leaders = int(_f("copy_v3_auto_promote_max_leaders", 0, 0, 20))
    cfg.auto_promote_min_shadow_fills = int(_f("copy_v3_auto_promote_min_shadow_fills", 10, 1, 1000))
    cfg.helius_daily_call_budget = int(_f("helius_daily_call_budget", 500, 0, 1_000_000))
    cfg.helius_tx_sample = int(_f("helius_tx_sample", 100, 1, 100))
    cfg.discovery_min_swaps = int(_f("discovery_min_swaps", 3, 1, 100))
    cfg.sm_min_score = _f("copy_sm_min_score", 0.6, 0.0, 1.0)
    return cfg


# -------------------------------------------------------------------------
# Candidate + event collection (all fail-soft, read-only on source tables)
# -------------------------------------------------------------------------

async def _smart_money_candidates(db_pool, limit: int = 100) -> List[Tuple[str, str]]:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT chain, wallet FROM smart_money_wallet_scores "
            "WHERE score IS NOT NULL ORDER BY score DESC LIMIT $1", limit)
    return [(r["chain"], r["wallet"]) for r in rows]


async def _smart_money_events(db_pool, chain: str, wallet: str,
                              window_days: int) -> List[Dict]:
    """smart_money_wallet_events → scorer event shape (USD-priced)."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT token, side, amount_usd, price_usd_at_event, block_time
            FROM smart_money_wallet_events
            WHERE chain = $1 AND wallet = $2
              AND block_time > NOW() - ($3::int || ' days')::interval
              AND block_time <= NOW()
            ORDER BY block_time ASC
            """,
            chain, wallet, int(window_days) + 7,  # buys slightly before window
        )
    out: List[Dict] = []
    for r in rows:
        price = float(r["price_usd_at_event"] or 0)
        usd = float(r["amount_usd"] or 0)
        if price <= 0 or usd <= 0:
            continue
        out.append({"token": r["token"], "side": r["side"],
                    "qty": usd / price, "price_usd": price, "ts": r["block_time"]})
    return out


async def _onchain_candidates(db_pool, limit: int = 100) -> List[Tuple[str, str]]:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT chain, source_wallet
            FROM copytrading_trades
            WHERE COALESCE(exit_timestamp, entry_timestamp) > NOW() - INTERVAL '90 days'
            GROUP BY chain, source_wallet
            ORDER BY COUNT(*) DESC
            LIMIT $1
            """, limit)
    return [(r["chain"], r["source_wallet"]) for r in rows if r["source_wallet"]]


async def _onchain_events(db_pool, chain: str, wallet: str,
                          window_days: int) -> List[Dict]:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT entry_timestamp, exit_timestamp, profit_loss, entry_usd,
                   token_address
            FROM copytrading_trades
            WHERE chain = $1 AND source_wallet = $2
              AND exit_timestamp IS NOT NULL
              AND exit_timestamp <= NOW()
              AND exit_timestamp > NOW() - ($3::int || ' days')::interval
            """,
            chain, wallet, int(window_days) + 7,
        )
    return trades_to_events([dict(r) for r in rows])


async def _leader_score_candidates(db_pool, limit: int = 100) -> List[Tuple[str, str]]:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT chain, wallet_address FROM copy_leader_scores "
            "WHERE NOT COALESCE(is_dead, FALSE) AND score IS NOT NULL "
            "ORDER BY score DESC LIMIT $1", limit)
    return [(r["chain"], r["wallet_address"]) for r in rows]


async def _dexscreener_candidates(cfg: DiscoveryV3Config) -> List[Tuple[str, str]]:
    """Reuse the v2 fetcher: candidate addresses only (no history)."""
    try:
        from modules.copy_trading.wallet_discovery import (
            DiscoveryConfig, _RateLimiter, fetch_dexscreener_top_traders,
        )
    except Exception:  # noqa: BLE001
        return []
    if aiohttp is None:
        return []
    v2cfg = DiscoveryConfig(max_candidates_per_source=20)
    rl = _RateLimiter(qps=v2cfg.rate_limit_qps)
    out: List[Tuple[str, str]] = []
    try:
        timeout = aiohttp.ClientTimeout(total=v2cfg.request_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for chain in ("solana", "ethereum", "base"):
                cands = await fetch_dexscreener_top_traders(session, chain, v2cfg, rl)
                out.extend((c.chain, c.wallet_address) for c in cands)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"dexscreener candidates failed: {e}")
    return out


# --- Solana RPC enrichment (Helius enhanced-tx REST; bounded) -------------

def parse_helius_swaps(wallet: str, txs: Sequence[Dict],
                       sol_price_usd: float) -> List[Dict]:
    """Pure: Helius enhanced transactions → scorer events for `wallet`.

    SOL-numeraire: a buy is SOL out / token in; price_usd is derived from
    the SOL leg at a SINGLE current SOL price — an approximation that keeps
    relative ranking honest but absolute USD noisy (pnl_basis flag set by
    the caller). Stable-mint legs are treated as quote, never as the token.
    """
    events: List[Dict] = []
    for tx in txs:
        if not isinstance(tx, dict):
            continue
        swap = (tx.get("events") or {}).get("swap") or {}
        ts = tx.get("timestamp")
        if not swap or not isinstance(ts, (int, float)):
            continue
        when = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        sig = str(tx.get("signature") or "")

        sol_in = sol_out = 0.0
        ni, no = swap.get("nativeInput") or {}, swap.get("nativeOutput") or {}
        try:
            if ni.get("account") == wallet:
                sol_in = float(ni.get("amount") or 0) / 1e9   # wallet SPENT sol
            if no.get("account") == wallet:
                sol_out = float(no.get("amount") or 0) / 1e9  # wallet RECEIVED sol
        except (TypeError, ValueError):
            continue

        def _legs(key: str) -> List[Tuple[str, float]]:
            out = []
            for leg in swap.get(key) or []:
                if not isinstance(leg, dict) or leg.get("userAccount") != wallet:
                    continue
                mint = leg.get("mint")
                raw = leg.get("rawTokenAmount") or {}
                try:
                    qty = float(raw.get("tokenAmount") or 0) / (10 ** int(raw.get("decimals") or 0))
                except (TypeError, ValueError):
                    continue
                if mint and mint not in STABLE_MINTS and qty > 0:
                    out.append((mint, qty))
            return out

        token_paid = _legs("tokenInputs")     # wallet sent these tokens
        token_recv = _legs("tokenOutputs")    # wallet received these tokens

        if sol_in > 0 and token_recv:         # BUY: paid SOL, received token
            mint, qty = max(token_recv, key=lambda x: x[1])
            price = (sol_in * sol_price_usd) / qty
            if price > 0:
                events.append({"token": mint, "side": "buy", "qty": qty,
                               "price_usd": price, "ts": when, "source_ref": sig})
        elif sol_out > 0 and token_paid:      # SELL: paid token, received SOL
            mint, qty = max(token_paid, key=lambda x: x[1])
            price = (sol_out * sol_price_usd) / qty
            if price > 0:
                events.append({"token": mint, "side": "sell", "qty": qty,
                               "price_usd": price, "ts": when, "source_ref": sig})
    events.sort(key=lambda e: e["ts"])
    return events


# --- Helius daily call budget (persistent, shared across sweeps) ----------

async def _helius_budget_remaining(db_pool, budget: int) -> int:
    """Calls left in today's (UTC) Helius budget. Fail-soft: on DB error
    assume budget available so a transient DB blip doesn't wedge discovery."""
    if db_pool is None or budget <= 0:
        return 0 if budget <= 0 else 10 ** 9
    try:
        async with db_pool.acquire() as conn:
            used = await conn.fetchval(
                "SELECT calls FROM copy_helius_budget WHERE day = CURRENT_DATE")
        return max(0, int(budget) - int(used or 0))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"helius budget read failed: {e}")
        return int(budget)


async def _helius_budget_consume(db_pool, n: int = 1) -> None:
    if db_pool is None or n <= 0:
        return
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO copy_helius_budget (day, calls) VALUES (CURRENT_DATE, $1)
                ON CONFLICT (day) DO UPDATE SET calls = copy_helius_budget.calls + $1
                """, int(n))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"helius budget consume failed: {e}")


async def _helius_get_json(session, url: str, db_pool, cfg: "DiscoveryV3Config",
                           *, max_attempts: int = 3):
    """Budgeted Helius GET with exponential backoff on 429. Returns parsed
    JSON or None. Every attempt that actually reaches Helius consumes one unit
    of the daily budget (checked BEFORE the request)."""
    for attempt in range(max_attempts):
        if await _helius_budget_remaining(db_pool, cfg.helius_daily_call_budget) <= 0:
            logger.debug("helius daily budget exhausted; skipping request")
            return None
        await _helius_budget_consume(db_pool, 1)
        try:
            async with session.get(url, timeout=12) as resp:
                if resp.status == 429:
                    # Exponential backoff: 0.5s, 1.0s, 2.0s ... then give up.
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                if resp.status != 200:
                    return None
                return await resp.json(content_type=None)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"helius GET failed ({attempt + 1}/{max_attempts}): {e}")
            await asyncio.sleep(0.5 * (2 ** attempt))
    return None


# --- Cross-sweep fee-payer accumulation -----------------------------------

async def _accumulate_feepayers(db_pool, chain: str, source: str,
                                counts: Dict[str, int]) -> None:
    """Persist per-wallet swap counts ACROSS sweeps in copy_discovery_feepayers.
    This replaces the broken "≥N swaps inside one 100-tx snapshot" heuristic:
    a wallet that trades a handful of times per sweep accumulates over days
    until it clears discovery_min_swaps. Fail-soft."""
    if db_pool is None or not counts:
        return
    try:
        async with db_pool.acquire() as conn:
            for wallet, n in counts.items():
                if not wallet or n <= 0:
                    continue
                await conn.execute(
                    """
                    INSERT INTO copy_discovery_feepayers
                        (chain, wallet_address, source, cumulative_swaps,
                         first_seen_at, last_seen_at)
                    VALUES ($1, $2, $3, $4, NOW(), NOW())
                    ON CONFLICT (chain, wallet_address, source) DO UPDATE SET
                        cumulative_swaps =
                            copy_discovery_feepayers.cumulative_swaps + $4,
                        last_seen_at = NOW()
                    """, chain, wallet, source, int(n))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"feepayer accumulation failed: {e}")


async def _feepayer_candidates(db_pool, min_swaps: int,
                               limit: int = 200) -> List[Tuple[str, str]]:
    """Accumulated fee-payers that have cleared the cross-sweep swap floor."""
    if db_pool is None:
        return []
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT chain, wallet_address FROM copy_discovery_feepayers "
                "WHERE cumulative_swaps >= $1 "
                "ORDER BY cumulative_swaps DESC LIMIT $2",
                int(min_swaps), int(limit))
        return [(r["chain"], r["wallet_address"]) for r in rows]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"feepayer candidate read failed: {e}")
        return []


# Fee-payers that are routers / infra, never a copyable trader wallet.
_HELIUS_TOKENS_EXCLUDED = {
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",   # Jupiter v6
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Raydium AMM V4
} | STABLE_MINTS


async def _helius_tokens_candidates(db_pool, cfg: "DiscoveryV3Config",
                                    session) -> List[Tuple[str, str]]:
    """helius_tokens source (Wave-F5): the structurally-sound free Solana feed.

    1. Top-volume tokens per chain from DexScreener (free, works for tokens).
    2. Recent SWAP txs per token via Helius enhanced-tx (budget-capped).
    3. Fee-payers accumulated ACROSS sweeps in copy_discovery_feepayers.
    4. Candidate once cumulative swaps >= discovery_min_swaps — fixes the
       "5 swaps in one 100-tx snapshot" flaw. Solana only; fail-soft."""
    if session is None or not cfg.helius_api_key:
        return []
    try:
        from modules.copy_trading.wallet_discovery import (
            DiscoveryConfig, _RateLimiter, fetch_dexscreener_token_pools,
        )
    except Exception:  # noqa: BLE001
        return []
    v2cfg = DiscoveryConfig()
    rl = _RateLimiter(qps=v2cfg.rate_limit_qps)
    try:
        tokens = await fetch_dexscreener_token_pools(
            session, "solana", v2cfg, rl, max_tokens=10)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"helius_tokens token feed failed: {e}")
        tokens = []
    if not tokens:
        return []
    counts: Dict[str, int] = {}
    for tok in tokens:
        if await _helius_budget_remaining(db_pool, cfg.helius_daily_call_budget) <= 0:
            break
        addr = tok.get("token_address")
        if not addr:
            continue
        url = (f"https://api.helius.xyz/v0/addresses/{addr}/transactions"
               f"?api-key={cfg.helius_api_key}&limit={cfg.helius_tx_sample}&type=SWAP")
        data = await _helius_get_json(session, url, db_pool, cfg)
        if not isinstance(data, list):
            continue
        for tx in data:
            if not isinstance(tx, dict):
                continue
            fp = tx.get("feePayer")
            if (isinstance(fp, str) and len(fp) >= 32
                    and fp not in _HELIUS_TOKENS_EXCLUDED):
                counts[fp] = counts.get(fp, 0) + 1
    await _accumulate_feepayers(db_pool, "solana", "helius_tokens", counts)
    return await _feepayer_candidates(db_pool, cfg.discovery_min_swaps)


async def _helius_enrich(session, wallet: str, api_key: str,
                         sol_price_usd: float, db_pool=None,
                         cfg: Optional["DiscoveryV3Config"] = None) -> List[Dict]:
    limit = cfg.helius_tx_sample if cfg else 100
    url = (f"https://api.helius.xyz/v0/addresses/{wallet}/transactions"
           f"?api-key={api_key}&limit={limit}&type=SWAP")
    if cfg is not None and db_pool is not None:
        data = await _helius_get_json(session, url, db_pool, cfg)
    else:
        try:
            async with session.get(url, timeout=12) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json(content_type=None)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"helius enrich {wallet[:8]}: {e}")
            return []
    if not isinstance(data, list):
        return []
    return parse_helius_swaps(wallet, data, sol_price_usd)


# -------------------------------------------------------------------------
# Persistence + promotion gate
# -------------------------------------------------------------------------

async def _upsert_discovered(db_pool, s: WalletScore, sources: List[str],
                             pnl_basis: str, status: str) -> None:
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO copy_discovered_wallets (
                chain, wallet_address, sources, score, score_breakdown,
                realized_pnl_usd, win_rate, profit_factor, trade_count,
                max_drawdown_pct, avg_hold_seconds, consistency,
                diversification, wash_penalty, lucky_penalty, pnl_basis,
                window_days, status, last_scored_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,
                      $16,$17,$18,NOW())
            ON CONFLICT (chain, wallet_address) DO UPDATE SET
                sources = EXCLUDED.sources,
                score = EXCLUDED.score,
                score_breakdown = EXCLUDED.score_breakdown,
                realized_pnl_usd = EXCLUDED.realized_pnl_usd,
                win_rate = EXCLUDED.win_rate,
                profit_factor = EXCLUDED.profit_factor,
                trade_count = EXCLUDED.trade_count,
                max_drawdown_pct = EXCLUDED.max_drawdown_pct,
                avg_hold_seconds = EXCLUDED.avg_hold_seconds,
                consistency = EXCLUDED.consistency,
                diversification = EXCLUDED.diversification,
                wash_penalty = EXCLUDED.wash_penalty,
                lucky_penalty = EXCLUDED.lucky_penalty,
                pnl_basis = EXCLUDED.pnl_basis,
                window_days = EXCLUDED.window_days,
                status = CASE WHEN copy_discovered_wallets.status IN
                              ('promoted', 'rejected')
                         THEN copy_discovered_wallets.status
                         ELSE EXCLUDED.status END,
                last_scored_at = NOW()
            """,
            s.chain, s.wallet_address, json.dumps(sorted(set(sources))),
            s.score, json.dumps({**s.components,
                                 "wash_penalty": s.wash_penalty,
                                 "lucky_penalty": s.lucky_penalty}),
            s.realized_pnl_usd, s.win_rate, s.profit_factor, s.trade_count,
            s.max_drawdown_pct, s.avg_hold_seconds, s.consistency,
            s.diversification, s.wash_penalty, s.lucky_penalty, pnl_basis,
            s.window_days, status,
        )


async def _propose_candidate(db_pool, s: WalletScore, sources: List[str]) -> None:
    """Upsert into copy_leader_candidates (mig 092) as PENDING. Never
    overwrites an operator decision (approved/rejected rows untouched)."""
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO copy_leader_candidates (
                chain, wallet_address, source, label, score, metrics,
                status, score_breakdown, provenance
            ) VALUES ($1,$2,$3,$4,$5,$6,'pending',$7,$8)
            ON CONFLICT (chain, wallet_address) DO UPDATE SET
                score = EXCLUDED.score,
                metrics = EXCLUDED.metrics,
                score_breakdown = EXCLUDED.score_breakdown,
                provenance = EXCLUDED.provenance,
                proposed_at = NOW()
            WHERE copy_leader_candidates.status = 'pending'
            """,
            s.chain, s.wallet_address, "discovery_v3", None, s.score,
            json.dumps({
                "realized_pnl_usd": s.realized_pnl_usd, "win_rate": s.win_rate,
                "profit_factor": s.profit_factor, "trade_count": s.trade_count,
                "max_drawdown_pct": s.max_drawdown_pct,
                "window_days": s.window_days,
            }),
            json.dumps(s.components), json.dumps(sorted(set(sources))),
        )


async def _propose_smart_money_scores(db_pool, cfg: DiscoveryV3Config) -> int:
    """smart_money_scores source (Wave-F5): read smart_money_wallet_scores
    (mig 133), take EVM wallets with score >= copy_sm_min_score, and propose
    the top-N directly into copy_leader_candidates as PENDING for operator
    approval. This bypasses the realized-PnL event scorer on purpose — the
    smart_money module already scored these on forward-return; requiring our
    own event history would exclude exactly the EVM wallets we cannot yet see.

    NEVER trades: proposal only, operator-approval gate unchanged. Fail-soft
    when the smart_money table is absent (module not deployed). Returns the
    number of wallets proposed."""
    if db_pool is None:
        return 0
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT chain, wallet, score FROM smart_money_wallet_scores "
                "WHERE score >= $1 AND chain <> 'solana' AND score IS NOT NULL "
                "ORDER BY score DESC LIMIT $2",
                float(cfg.sm_min_score), int(cfg.max_candidates_per_sweep))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"smart_money_scores source unavailable: {e}")
        return 0
    proposed = 0
    for r in rows:
        chain, wallet, sm_score = r["chain"], r["wallet"], float(r["score"] or 0)
        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO copy_leader_candidates (
                        chain, wallet_address, source, label, score, metrics,
                        status, score_breakdown, provenance
                    ) VALUES ($1,$2,'smart_money_scores',NULL,$3,$4,'pending',$5,$6)
                    ON CONFLICT (chain, wallet_address) DO UPDATE SET
                        score = EXCLUDED.score,
                        metrics = EXCLUDED.metrics,
                        provenance = EXCLUDED.provenance,
                        proposed_at = NOW()
                    WHERE copy_leader_candidates.status = 'pending'
                    """,
                    chain, wallet, round(sm_score * 100.0, 2),
                    json.dumps({"smart_money_score": sm_score,
                                "source": "smart_money_wallet_scores"}),
                    json.dumps({"smart_money_score": sm_score}),
                    json.dumps(["smart_money_scores"]),
                )
            proposed += 1
        except Exception as e:  # noqa: BLE001
            logger.debug(f"smart_money_scores propose {str(wallet)[:8]} failed: {e}")
    if proposed:
        logger.info(
            f"[discovery-v3] smart_money_scores proposed {proposed} EVM "
            f"candidate(s) (score >= {cfg.sm_min_score})")
    return proposed


async def _shadow_quality(db_pool, chain: str, wallet: str) -> Tuple[int, float]:
    """(simulated fill count, realized shadow PnL) for the promotion gate."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT COUNT(*) AS n, COALESCE(SUM(realized_pnl_usd), 0) AS pnl "
            "FROM copy_shadow_fills "
            "WHERE chain = $1 AND wallet_address = $2 AND is_simulated",
            chain, wallet,
        )
    return (int(row["n"] or 0), float(row["pnl"] or 0)) if row else (0, 0.0)


async def _maybe_auto_promote(db_pool, cfg: DiscoveryV3Config,
                              ranked: List[Tuple[WalletScore, List[str]]]) -> int:
    """DOUBLE-gated auto-promotion. Appends to target_wallets and stamps the
    candidate row reviewed_by='auto_promote'. Returns promotions made."""
    if not cfg.auto_promote_enabled or cfg.auto_promote_max_leaders <= 0:
        return 0
    promoted = 0
    async with db_pool.acquire() as conn:
        already = await conn.fetchval(
            "SELECT COUNT(*) FROM copy_leader_candidates "
            "WHERE reviewed_by = 'auto_promote' AND status = 'approved'")
    budget = max(0, cfg.auto_promote_max_leaders - int(already or 0))
    for s, _src in ranked:
        if budget <= 0:
            break
        fills, shadow_pnl = await _shadow_quality(db_pool, s.chain, s.wallet_address)
        if fills < cfg.auto_promote_min_shadow_fills or shadow_pnl <= 0:
            continue
        target = (s.wallet_address if s.chain == "solana"
                  else f"{s.wallet_address}@{s.chain}")
        try:
            async with db_pool.acquire() as conn:
                raw = await conn.fetchval(
                    "SELECT value FROM config_settings "
                    "WHERE config_type='copytrading_config' AND key='target_wallets'")
                try:
                    wallets = json.loads(raw) if raw else []
                except (TypeError, ValueError):
                    wallets = []
                if not isinstance(wallets, list):
                    wallets = []
                if target in wallets or any(
                        str(w).split("@")[0].lower() == s.wallet_address.lower()
                        for w in wallets):
                    continue
                wallets.append(target)
                await conn.execute(
                    "UPDATE config_settings SET value = $1, updated_at = NOW() "
                    "WHERE config_type='copytrading_config' AND key='target_wallets'",
                    json.dumps(wallets),
                )
                await conn.execute(
                    "UPDATE copy_leader_candidates "
                    "SET status='approved', reviewed_at=NOW(), reviewed_by='auto_promote' "
                    "WHERE chain=$1 AND wallet_address=$2 AND status='pending'",
                    s.chain, s.wallet_address,
                )
                await conn.execute(
                    "UPDATE copy_discovered_wallets SET status='promoted' "
                    "WHERE chain=$1 AND wallet_address=$2",
                    s.chain, s.wallet_address,
                )
            promoted += 1
            budget -= 1
            logger.warning(
                f"[discovery-v3] AUTO-PROMOTED leader {target[:24]}... "
                f"(score={s.score}, shadow fills={fills}, shadow pnl=${shadow_pnl:.0f})"
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"auto-promote {s.wallet_address[:8]} failed: {e}")
    return promoted


# -------------------------------------------------------------------------
# Sweep orchestration
# -------------------------------------------------------------------------

async def run_discovery_sweep(db_pool, cfg: Optional[DiscoveryV3Config] = None) -> Dict:
    """End-to-end v3 sweep. Fail-soft per source/wallet; returns a summary."""
    if db_pool is None:
        return {"candidates": 0, "scored": 0, "proposed": 0, "promoted": 0}
    cfg = cfg or await load_config(db_pool)
    as_of = datetime.now(timezone.utc)

    # 1. Candidate universe with provenance.
    provenance: Dict[Tuple[str, str], List[str]] = {}
    per_source_counts: Dict[str, int] = {}

    # A Helius session is needed both by the helius_tokens candidate source
    # and by the per-wallet rpc_solana enrichment. Open it up-front so both
    # phases share it (and the daily budget).
    session = None
    if (cfg.helius_api_key and aiohttp is not None
            and ("helius_tokens" in cfg.sources
                 or ("rpc_solana" in cfg.sources and cfg.max_rpc_enrich_wallets > 0))):
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))

    async def _add(source: str, coro) -> None:
        n = 0
        try:
            for chain, wallet in await coro:
                provenance.setdefault((chain, wallet), []).append(source)
                n += 1
        except Exception as e:  # noqa: BLE001
            _warn_rate_limited(source, f"[discovery-v3] source={source} FAILED: {e}")
            per_source_counts[source] = -1
            return
        per_source_counts[source] = n

    if "smart_money" in cfg.sources:
        await _add("smart_money", _smart_money_candidates(db_pool))
    if "onchain" in cfg.sources:
        await _add("onchain", _onchain_candidates(db_pool))
    if "leader_scores" in cfg.sources:
        await _add("leader_scores", _leader_score_candidates(db_pool))
    if "dexscreener" in cfg.sources:
        await _add("dexscreener", _dexscreener_candidates(cfg))
    if "helius_tokens" in cfg.sources:
        await _add("helius_tokens", _helius_tokens_candidates(db_pool, cfg, session))

    # smart_money_scores proposes EVM wallets DIRECTLY into
    # copy_leader_candidates (bypasses the realized-PnL event scorer — see the
    # function docstring). Counted separately from provenance-based sources.
    sm_scores_proposed = 0
    if "smart_money_scores" in cfg.sources:
        try:
            sm_scores_proposed = await _propose_smart_money_scores(db_pool, cfg)
            per_source_counts["smart_money_scores"] = sm_scores_proposed
        except Exception as e:  # noqa: BLE001
            _warn_rate_limited(
                "smart_money_scores",
                f"[discovery-v3] source=smart_money_scores FAILED: {e}")
            per_source_counts["smart_money_scores"] = -1

    # 2. Build realized-event history per wallet (merge sources), score.
    scored: List[Tuple[WalletScore, List[str], str]] = []
    enriched = 0
    try:
        for (chain, wallet), sources in provenance.items():
            try:
                events: List[Dict] = []
                pnl_basis = "usd"
                if "smart_money" in cfg.sources:
                    events += await _smart_money_events(db_pool, chain, wallet, cfg.window_days)
                if "onchain" in cfg.sources:
                    events += await _onchain_events(db_pool, chain, wallet, cfg.window_days)
                if (not events and chain == "solana" and session is not None
                        and enriched < cfg.max_rpc_enrich_wallets):
                    enriched += 1
                    events = await _helius_enrich(
                        session, wallet, cfg.helius_api_key, DEFAULT_SOL_PRICE_USD,
                        db_pool=db_pool, cfg=cfg)
                    if events:
                        pnl_basis = "sol_numeraire"
                        sources = sources + ["rpc_solana"]
                        # Feed the shadow simulator the same provenance-stamped
                        # events (dedup on source_ref makes re-feeds safe).
                        try:
                            from modules.copy_trading.shadow_simulator import (
                                ShadowConfig, record_events,
                            )
                            await record_events(
                                db_pool, chain, wallet, events,
                                cfg.shadow_cfg or ShadowConfig())
                        except Exception as e:  # noqa: BLE001
                            logger.debug(f"shadow feed {wallet[:8]}: {e}")
                s = score_wallet(chain, wallet, events,
                                 as_of=as_of, window_days=cfg.window_days)
                scored.append((s, sources, pnl_basis))
            except Exception as e:  # noqa: BLE001
                logger.debug(f"scoring {wallet[:8]} failed: {e}")
    finally:
        if session is not None:
            try:
                await session.close()
            except Exception:  # noqa: BLE001
                pass

    scored.sort(key=lambda t: t[0].score, reverse=True)

    # 3. Persist universe + propose qualifying candidates.
    proposed: List[Tuple[WalletScore, List[str]]] = []
    for s, sources, pnl_basis in scored:
        qualifies = (
            s.score >= cfg.min_score
            and s.trade_count >= cfg.min_trades
            and s.realized_pnl_usd >= cfg.min_realized_pnl_usd
            and s.lucky_penalty <= cfg.max_lucky_share
            and s.wash_penalty <= cfg.max_wash_penalty
            and len(proposed) < cfg.max_candidates_per_sweep
        )
        try:
            await _upsert_discovered(
                db_pool, s, sources, pnl_basis,
                "proposed" if qualifies else "discovered")
            if qualifies:
                await _propose_candidate(db_pool, s, sources)
                proposed.append((s, sources))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"persist {s.wallet_address[:8]} failed: {e}")

    # 4. Optional, double-gated auto-promotion (default: does nothing).
    promoted = 0
    try:
        promoted = await _maybe_auto_promote(db_pool, cfg, proposed)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"auto-promote pass failed: {e}")

    # HONESTY (Wave-F5): flag when only the LOCAL/recycled sources produced
    # candidates. onchain + leader_scores both re-serve wallets we already
    # track; smart_money / helius_tokens / dexscreener / smart_money_scores are
    # the genuinely-external feeds. If the external total is 0, discovery is
    # degraded and any proposals came from already-tracked wallets.
    external_srcs = ("smart_money", "helius_tokens", "dexscreener", "smart_money_scores")
    local_srcs = ("onchain", "leader_scores")
    ext_total = sum(max(0, per_source_counts.get(s, 0)) for s in external_srcs)
    local_total = sum(max(0, per_source_counts.get(s, 0)) for s in local_srcs)
    fallback = ext_total == 0 and (local_total > 0 or len(provenance) > 0)
    if fallback:
        _warn_rate_limited(
            "v3_fallback",
            "[discovery-v3] ALL external candidate sources returned 0 "
            f"(per-source={per_source_counts}); the {len(provenance)} "
            "candidate(s) are LOCAL/recycled (already-tracked) wallets. "
            "Discovery is degraded — check Helius quota / add helius_tokens "
            "or smart_money_scores to copy_v3_sources / set ETHERSCAN_API_KEY.")

    summary = {
        "candidates": len(provenance),
        "scored": len(scored),
        "proposed": len(proposed),
        "sm_scores_proposed": sm_scores_proposed,
        "promoted": promoted,
        "rpc_enriched": enriched,
        "per_source": per_source_counts,
        "external_candidates": ext_total,
        "fallback": fallback,
    }
    logger.info(
        f"[discovery-v3] sweep: {summary['candidates']} candidates "
        f"(per-source={per_source_counts}, external={ext_total}), "
        f"{summary['proposed']} proposed + {sm_scores_proposed} sm-scores "
        f"(min_score={cfg.min_score}, min_trades={cfg.min_trades}), "
        f"{promoted} auto-promoted (gate {'ON' if cfg.auto_promote_enabled else 'OFF'})"
        f"{' [FALLBACK/degraded]' if fallback else ''}"
    )
    return summary


# -------------------------------------------------------------------------
# Self-test (pure parser; offline). Run:
#   python -m modules.copy_trading.discovery_v3
# -------------------------------------------------------------------------
def _self_test() -> None:
    w = "WaLLet1111111111111111111111111111111111111"
    txs = [
        {  # BUY: 2 SOL out, 1000 TOKEN in
            "signature": "sigBuy", "timestamp": 1780000000,
            "events": {"swap": {
                "nativeInput": {"account": w, "amount": str(int(2e9))},
                "tokenOutputs": [{"userAccount": w, "mint": "TOKENMINT",
                                  "rawTokenAmount": {"tokenAmount": "1000000000",
                                                     "decimals": 6}}],
            }},
        },
        {  # SELL: 1000 TOKEN out, 3 SOL in
            "signature": "sigSell", "timestamp": 1780003600,
            "events": {"swap": {
                "nativeOutput": {"account": w, "amount": str(int(3e9))},
                "tokenInputs": [{"userAccount": w, "mint": "TOKENMINT",
                                 "rawTokenAmount": {"tokenAmount": "1000000000",
                                                    "decimals": 6}}],
            }},
        },
        {  # someone ELSE's swap — must be ignored
            "signature": "sigOther", "timestamp": 1780003700,
            "events": {"swap": {
                "nativeInput": {"account": "OTHER", "amount": str(int(1e9))},
                "tokenOutputs": [{"userAccount": "OTHER", "mint": "X",
                                  "rawTokenAmount": {"tokenAmount": "1", "decimals": 0}}],
            }},
        },
        {  # stable-mint leg must never become the token
            "signature": "sigStable", "timestamp": 1780003800,
            "events": {"swap": {
                "nativeInput": {"account": w, "amount": str(int(1e9))},
                "tokenOutputs": [{"userAccount": w,
                                  "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                                  "rawTokenAmount": {"tokenAmount": "150000000",
                                                     "decimals": 6}}],
            }},
        },
    ]
    evs = parse_helius_swaps(w, txs, sol_price_usd=150.0)
    assert len(evs) == 2, evs
    buy, sell = evs
    assert buy["side"] == "buy" and abs(buy["qty"] - 1000.0) < 1e-9
    assert abs(buy["price_usd"] - (2 * 150.0) / 1000.0) < 1e-9   # $0.30
    assert sell["side"] == "sell" and abs(sell["price_usd"] - 0.45) < 1e-9
    assert buy["ts"] < sell["ts"]

    # FIFO over these events realizes (0.45-0.30)*1000 = $150.
    s = score_wallet("solana", w, evs,
                     as_of=datetime.fromtimestamp(1780010000, tz=timezone.utc),
                     window_days=30)
    assert abs(s.realized_pnl_usd - 150.0) < 1e-6, s.realized_pnl_usd
    print(f"discovery_v3 self-test OK (parsed {len(evs)} events, pnl={s.realized_pnl_usd})")


if __name__ == "__main__":
    _self_test()
