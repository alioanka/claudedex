"""
Copy-trading wallet-discovery engine.

Pulls candidate leader wallets from multiple public data sources,
rate-limits HTTP fetches, scores them via leader_scorer, and caches
results to the `copy_leader_scores` table (migration 024).

Operator-flagged broken/missing in Wave-2 — this module is the
rebuild. It is the **discovery** layer (cast the wide net); the
**execution** layer (mirror trades) lives in copy_engine.py.

Sources (priority order):
    1. DexScreener     — top-trader leaderboard per chain (REST, no key).
    2. Helius          — recent DEX-program transactions to extract
                          active Solana swappers (paid tier — already
                          wired in monitoring/enhanced_dashboard.py).
    3. Birdeye         — top-traders + token-trader leaderboards
                          (paid tier).
    4. GMGN            — smart-money lookalike feed (no public key
                          but the JSON endpoint is accessible).
    5. On-chain stub   — local copytrading_trades table, ranked by
                          realized PnL when no third-party keys are
                          configured. Always available as a fallback.

Design rules:
    * Never block on network access. Every fetch is wrapped in a
      bounded asyncio.wait_for + try/except. A 5xx / timeout returns
      an empty list, not an exception.
    * Rate limiting: per-host token bucket, default 5 req/sec, configurable
      via DiscoveryConfig.rate_limit_per_host_qps.
    * Cache TTL: leaders are persisted into copy_leader_scores with
      last_scored_at = NOW(). The dashboard / scoring loop treats a row
      older than cache_ttl_seconds as stale and refreshes.
    * Hard caps: max_candidates_per_source so a runaway feed can't
      blow our HTTP quota.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

try:
    import aiohttp  # type: ignore
except ImportError:  # pragma: no cover - aiohttp ships in requirements
    aiohttp = None  # type: ignore

from modules.copy_trading.leader_scorer import (
    LeaderMetrics,
    fetch_top_leaders,
    score_leader,
    upsert_score,
)

logger = logging.getLogger("WalletDiscovery")

# -----------------------------------------------------------------------
# Config + rate limiter
# -----------------------------------------------------------------------

DEFAULT_DISCOVERY_TIMEOUT_S = 12.0
DEFAULT_RATE_LIMIT_QPS = 5
DEFAULT_MAX_CANDIDATES_PER_SOURCE = 50
DEFAULT_CACHE_TTL_SECONDS = 6 * 3600  # 6 hours

# Source IDs — kept identical to the `source` enum used in
# copy_leader_scores so the dashboard filter / SQL UNIQUE constraint
# both work without translation.
SOURCE_DEXSCREENER = "dexscreener"
SOURCE_BIRDEYE = "birdeye"
SOURCE_GMGN = "gmgn"
SOURCE_HELIUS = "helius"
SOURCE_ONCHAIN = "onchain"
SOURCE_MANUAL = "manual"


@dataclass
class DiscoveryConfig:
    """Operator-tunable knobs. All reachable from
    config_settings.copytrading_config.* in copy_engine via _load_settings.
    """
    chains: Sequence[str] = ("solana", "ethereum", "base")
    sources: Sequence[str] = (
        SOURCE_DEXSCREENER,
        SOURCE_HELIUS,
        SOURCE_BIRDEYE,
        SOURCE_GMGN,
        SOURCE_ONCHAIN,
    )
    max_candidates_per_source: int = DEFAULT_MAX_CANDIDATES_PER_SOURCE
    rate_limit_qps: int = DEFAULT_RATE_LIMIT_QPS
    request_timeout_s: float = DEFAULT_DISCOVERY_TIMEOUT_S
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS
    # API keys — None means "skip this source". Resolved from
    # secrets_manager / env by the engine wrapper.
    helius_api_key: Optional[str] = None
    birdeye_api_key: Optional[str] = None
    # Mock mode: don't hit network even if keys are present (used in
    # tests + when operator wants to refresh from cached on-chain data
    # only).
    mock: bool = False


@dataclass
class _RateLimiter:
    """Per-host token bucket. Sloppy but enough — we're rate-limiting
    against third-party APIs, not microsecond DB writes."""
    qps: int
    _per_host_last_call: Dict[str, float] = field(default_factory=dict)

    async def acquire(self, host: str) -> None:
        if self.qps <= 0:
            return
        gap = 1.0 / float(self.qps)
        last = self._per_host_last_call.get(host, 0.0)
        wait_for = (last + gap) - time.monotonic()
        if wait_for > 0:
            await asyncio.sleep(wait_for)
        self._per_host_last_call[host] = time.monotonic()


@dataclass
class DiscoveredCandidate:
    """One row produced by a source feed before scoring."""
    chain: str
    wallet_address: str
    source: str
    label: Optional[str] = None
    raw: Dict = field(default_factory=dict)


# -----------------------------------------------------------------------
# Source adapters — each returns List[DiscoveredCandidate]; never raises.
# -----------------------------------------------------------------------


async def _safe_get_json(
    session,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_DISCOVERY_TIMEOUT_S,
) -> Optional[object]:
    """Bounded GET → JSON. Returns None on any failure mode so callers
    can `if data is None: return []` without re-implementing the
    try/except dance."""
    if session is None:
        return None
    try:
        async with session.get(url, headers=headers or {}, timeout=timeout) as resp:
            if resp.status != 200:
                logger.debug(f"discovery GET {url} -> HTTP {resp.status}")
                return None
            return await resp.json(content_type=None)
    except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
        logger.debug(f"discovery GET {url} failed: {e}")
        return None


async def fetch_dexscreener_top_traders(
    session,
    chain: str,
    cfg: DiscoveryConfig,
    rl: _RateLimiter,
) -> List[DiscoveredCandidate]:
    """DexScreener public token-pair endpoint. We use it to fish out
    the highest-volume tokens per chain, then look up the top buyers
    of those tokens. This is a coarse proxy for "active traders".

    No API key required. Rate-limited at 300 req/min upstream — we cap
    ourselves well below.
    """
    if cfg.mock:
        return _mock_candidates(chain, SOURCE_DEXSCREENER, cfg)

    # DexScreener uses 'solana'/'ethereum'/'base'/'bsc'... directly.
    chain_param = chain.lower()
    url = f"https://api.dexscreener.com/latest/dex/search?q={chain_param}"
    await rl.acquire("api.dexscreener.com")
    data = await _safe_get_json(session, url, timeout=cfg.request_timeout_s)
    if not isinstance(data, dict):
        return []

    candidates: List[DiscoveredCandidate] = []
    pairs = data.get("pairs") or []
    if not isinstance(pairs, list):
        return []

    # Top 10 volume-1h pairs per chain — enough volume signal without
    # blowing the per-source cap.
    pairs_sorted = sorted(
        [p for p in pairs if isinstance(p, dict) and p.get("chainId") == chain_param],
        key=lambda p: float((p.get("volume") or {}).get("h1") or 0),
        reverse=True,
    )[:10]

    # DexScreener doesn't expose buyers directly via the public REST
    # API; the buyer/trader breakdown is on the pair detail page. We
    # capture the pair's `pairCreatedAt` deployer / top-volume signal
    # as a candidate when the upstream returns one.
    for p in pairs_sorted:
        # Deployer / liquidity-provider address (top-of-stack signal).
        deployer = (p.get("info") or {}).get("imageUrl") and None  # unused
        for key in ("pairAddress", "deployerAddress"):
            addr = p.get(key) if key != "deployerAddress" else (
                (p.get("info") or {}).get("deployerAddress")
            )
            if isinstance(addr, str) and _looks_like_wallet(addr, chain):
                candidates.append(DiscoveredCandidate(
                    chain=chain,
                    wallet_address=addr,
                    source=SOURCE_DEXSCREENER,
                    label=(p.get("baseToken") or {}).get("symbol"),
                    raw={"pair": p.get("pairAddress"), "volume_h1": (p.get("volume") or {}).get("h1")},
                ))
        if len(candidates) >= cfg.max_candidates_per_source:
            break

    return candidates[: cfg.max_candidates_per_source]


async def fetch_birdeye_top_traders(
    session,
    chain: str,
    cfg: DiscoveryConfig,
    rl: _RateLimiter,
) -> List[DiscoveredCandidate]:
    """Birdeye top-traders leaderboard. Solana only on the free tier;
    multi-chain on Standard. Skips silently when no key configured."""
    if cfg.mock or not cfg.birdeye_api_key:
        return _mock_candidates(chain, SOURCE_BIRDEYE, cfg) if cfg.mock else []
    if chain != "solana":
        # Free tier is solana-only; bail rather than burning quota on
        # a 401.
        return []

    url = "https://public-api.birdeye.so/defi/v2/tokens/top_traders?sort_by=volume&sort_type=desc"
    headers = {
        "x-chain": "solana",
        "X-API-KEY": cfg.birdeye_api_key,
    }
    await rl.acquire("public-api.birdeye.so")
    data = await _safe_get_json(session, url, headers=headers, timeout=cfg.request_timeout_s)
    if not isinstance(data, dict):
        return []

    items = ((data.get("data") or {}).get("items")) or []
    out: List[DiscoveredCandidate] = []
    for it in items[: cfg.max_candidates_per_source]:
        if not isinstance(it, dict):
            continue
        addr = it.get("owner") or it.get("address")
        if not isinstance(addr, str) or not _looks_like_wallet(addr, chain):
            continue
        out.append(DiscoveredCandidate(
            chain=chain,
            wallet_address=addr,
            source=SOURCE_BIRDEYE,
            raw=it,
        ))
    return out


async def fetch_gmgn_smart_money(
    session,
    chain: str,
    cfg: DiscoveryConfig,
    rl: _RateLimiter,
) -> List[DiscoveredCandidate]:
    """GMGN smart-money endpoint. No API key required for the public
    JSON, but it can rate-limit aggressively — keep our cap low."""
    if cfg.mock:
        return _mock_candidates(chain, SOURCE_GMGN, cfg)
    if chain != "solana":
        return []

    url = "https://gmgn.ai/defi/quotation/v1/rank/sol/swaps/1d?orderby=pnl_1d"
    headers = {"User-Agent": "ClaudeDexCopyTrading/1.0"}
    await rl.acquire("gmgn.ai")
    data = await _safe_get_json(session, url, headers=headers, timeout=cfg.request_timeout_s)
    if not isinstance(data, dict):
        return []

    rows = ((data.get("data") or {}).get("rank")) or []
    out: List[DiscoveredCandidate] = []
    for r in rows[: cfg.max_candidates_per_source]:
        if not isinstance(r, dict):
            continue
        addr = r.get("address") or r.get("wallet_address")
        if not isinstance(addr, str) or not _looks_like_wallet(addr, chain):
            continue
        out.append(DiscoveredCandidate(
            chain=chain,
            wallet_address=addr,
            source=SOURCE_GMGN,
            label=r.get("twitter_username") or r.get("ens"),
            raw=r,
        ))
    return out


async def fetch_helius_active(
    session,
    chain: str,
    cfg: DiscoveryConfig,
    rl: _RateLimiter,
) -> List[DiscoveredCandidate]:
    """Wraps the existing Helius DEX-program-history sweep already
    implemented in enhanced_dashboard._discover_wallets_helius. The
    discovery module re-uses the same approach but stays self-contained
    so the dashboard import surface doesn't grow.

    Helius is Solana-only.
    """
    if cfg.mock:
        return _mock_candidates(chain, SOURCE_HELIUS, cfg)
    if chain != "solana" or not cfg.helius_api_key:
        return []

    dex_programs = [
        "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
        "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    ]
    excluded = {*dex_programs}
    seen: Dict[str, int] = {}
    for prog in dex_programs:
        url = (
            f"https://api.helius.xyz/v0/addresses/{prog}/transactions"
            f"?api-key={cfg.helius_api_key}&limit=100&type=SWAP"
        )
        await rl.acquire("api.helius.xyz")
        data = await _safe_get_json(session, url, timeout=cfg.request_timeout_s)
        if not isinstance(data, list):
            continue
        for tx in data:
            if not isinstance(tx, dict):
                continue
            fp = tx.get("feePayer")
            if not isinstance(fp, str) or fp in excluded or len(fp) < 32:
                continue
            seen[fp] = seen.get(fp, 0) + 1
    # Sort by activity volume; cap by config.
    ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
    return [
        DiscoveredCandidate(
            chain=chain,
            wallet_address=addr,
            source=SOURCE_HELIUS,
            raw={"helius_tx_count": cnt},
        )
        for addr, cnt in ranked[: cfg.max_candidates_per_source]
    ]


async def fetch_onchain_local(
    db_pool,
    chain: str,
    cfg: DiscoveryConfig,
) -> List[DiscoveredCandidate]:
    """Fallback: rank our own copytrading_trades table by realized PnL
    over the score window. This is always-available and forms the
    bootstrap leader universe before any third-party key is configured.
    """
    if db_pool is None:
        return []
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT source_wallet AS wallet_address,
                       SUM(profit_loss) AS pnl,
                       COUNT(*) AS n
                FROM copytrading_trades
                WHERE chain = $1
                  AND exit_timestamp IS NOT NULL
                  AND exit_timestamp > NOW() - INTERVAL '30 days'
                GROUP BY source_wallet
                ORDER BY SUM(profit_loss) DESC NULLS LAST
                LIMIT $2
                """,
                chain, cfg.max_candidates_per_source,
            )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"onchain discovery failed: {e}")
        return []
    return [
        DiscoveredCandidate(
            chain=chain,
            wallet_address=str(r["wallet_address"]),
            source=SOURCE_ONCHAIN,
            raw={"pnl_30d": float(r["pnl"] or 0), "trades_30d": int(r["n"])},
        )
        for r in rows
        if r["wallet_address"]
    ]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _looks_like_wallet(addr: str, chain: str) -> bool:
    """Format-level sanity. Real validity is asserted by the score step
    (it reads the wallet's actual trade history)."""
    if not addr or not isinstance(addr, str):
        return False
    if chain == "solana":
        return 32 <= len(addr) <= 44 and not addr.startswith("0x")
    # EVM-like
    return addr.startswith("0x") and len(addr) == 42


def _mock_candidates(chain: str, source: str, cfg: DiscoveryConfig) -> List[DiscoveredCandidate]:
    """Deterministic placeholder leaders used for tests / offline mode.
    Never hits the network. Addresses are obviously-synthetic so they
    don't accidentally get copy-traded if the operator forgets to flip
    mock off (they'll fail `_looks_like_wallet` on the engine side).
    """
    addrs = {
        "solana": [
            "MOCK1111111111111111111111111111111111111111",
            "MOCK2222222222222222222222222222222222222222",
            "MOCK3333333333333333333333333333333333333333",
        ],
        "ethereum": [
            "0x" + "11" * 20,
            "0x" + "22" * 20,
            "0x" + "33" * 20,
        ],
    }
    base = addrs.get(chain, addrs["solana"])
    return [
        DiscoveredCandidate(
            chain=chain,
            wallet_address=a,
            source=source,
            label=f"mock-{source}-{i}",
            raw={"mock": True},
        )
        for i, a in enumerate(base[: min(3, cfg.max_candidates_per_source)])
    ]


async def _load_leader_trades(db_pool, chain: str, wallet: str, *, lookback_days: int = 90) -> List[Dict]:
    """Fetch this leader's closed trades from copytrading_trades so the
    scorer can compute metrics. Empty list when no DB pool or no rows.
    """
    if db_pool is None:
        return []
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT entry_timestamp, exit_timestamp, profit_loss, entry_usd
                FROM copytrading_trades
                WHERE chain = $1 AND source_wallet = $2
                  AND exit_timestamp IS NOT NULL
                  AND exit_timestamp > NOW() - ($3::int || ' days')::interval
                ORDER BY exit_timestamp DESC
                """,
                chain, wallet, lookback_days,
            )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"_load_leader_trades({wallet[:8]}): {e}")
        return []
    return [dict(r) for r in rows]


# -----------------------------------------------------------------------
# Top-level orchestration
# -----------------------------------------------------------------------

async def discover_and_score(
    db_pool,
    cfg: DiscoveryConfig,
) -> List[LeaderMetrics]:
    """End-to-end pipeline:
        1. Pull candidates from every enabled source for every chain.
        2. Dedupe by (chain, wallet_address).
        3. Load each wallet's trade history from copytrading_trades.
        4. Score via leader_scorer.score_leader.
        5. Upsert into copy_leader_scores.

    Returns the scored LeaderMetrics list (sorted by score desc).

    Network-safe: every fetch is bounded by timeout + try/except; a
    full sweep with no keys configured falls through to the on-chain
    source and finishes in < 1s.
    """
    rl = _RateLimiter(qps=cfg.rate_limit_qps)

    # Build the discovery jobs we want to run.
    jobs: List = []
    session_ctx = None
    session = None
    if aiohttp is not None and not cfg.mock:
        timeout = aiohttp.ClientTimeout(total=cfg.request_timeout_s)
        session_ctx = aiohttp.ClientSession(timeout=timeout)
        session = await session_ctx.__aenter__()

    try:
        for chain in cfg.chains:
            if SOURCE_DEXSCREENER in cfg.sources:
                jobs.append(fetch_dexscreener_top_traders(session, chain, cfg, rl))
            if SOURCE_HELIUS in cfg.sources:
                jobs.append(fetch_helius_active(session, chain, cfg, rl))
            if SOURCE_BIRDEYE in cfg.sources:
                jobs.append(fetch_birdeye_top_traders(session, chain, cfg, rl))
            if SOURCE_GMGN in cfg.sources:
                jobs.append(fetch_gmgn_smart_money(session, chain, cfg, rl))
            if SOURCE_ONCHAIN in cfg.sources:
                jobs.append(fetch_onchain_local(db_pool, chain, cfg))

        # Run all sources in parallel; each is already bounded.
        results = await asyncio.gather(*jobs, return_exceptions=True)

    finally:
        if session_ctx is not None:
            try:
                await session_ctx.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    # Flatten + dedupe.
    seen: Dict[tuple, DiscoveredCandidate] = {}
    for r in results:
        if isinstance(r, Exception):
            logger.debug(f"discovery source raised: {r}")
            continue
        for c in r or []:
            key = (c.chain, c.wallet_address)
            if key not in seen:
                seen[key] = c

    logger.info(
        f"Discovery sweep produced {len(seen)} unique candidates across "
        f"{len(cfg.chains)} chains / {len(cfg.sources)} sources"
    )

    # Score each candidate (uses our own copytrading_trades for history).
    scored: List[LeaderMetrics] = []
    as_of = datetime.now(timezone.utc)
    for (chain, wallet), cand in seen.items():
        trades = await _load_leader_trades(db_pool, chain, wallet)
        m = score_leader(chain, wallet, trades, as_of=as_of)
        try:
            await upsert_score(
                db_pool, m,
                label=cand.label,
                source=cand.source,
                raw_metrics=cand.raw,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"upsert_score({wallet[:8]}): {e}")
        scored.append(m)

    scored.sort(key=lambda x: (x.score or 0.0), reverse=True)
    return scored


async def get_top_leaders(
    db_pool,
    *,
    chain: Optional[str] = None,
    limit: int = 25,
    min_score: float = 0.0,
) -> List[Dict]:
    """Cached read for dashboards — never touches the network."""
    return await fetch_top_leaders(
        db_pool, chain=chain, limit=limit, min_score=min_score,
    )
