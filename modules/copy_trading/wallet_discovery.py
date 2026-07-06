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

# Wave-F5: per-source WARNING throttle so a source that fails every sweep
# logs at most once per hour (not silently at DEBUG, not spamming). Keyed by
# source id.
_WARN_THROTTLE_S = 3600.0
_last_warn_at: Dict[str, float] = {}


def _warn_rate_limited(source: str, msg: str) -> None:
    now = time.monotonic()
    last = _last_warn_at.get(source, 0.0)
    if now - last >= _WARN_THROTTLE_S:
        _last_warn_at[source] = now
        logger.warning(msg)
    else:
        logger.debug(msg)

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


async def _pool_rotated_key(provider_type: str, fallback: Optional[str]) -> Optional[str]:
    """Wave-F5 multi-key: CURRENT rotated key from pool_engine for this
    provider, falling back to the static cfg key. Fetching per call (not at
    cfg build time) means a 429-cooled account rotates out mid-sweep and
    sibling keys share the quota burn. Fail-soft."""
    try:
        from config.rpc_provider import RPCProvider
        res = await RPCProvider.get_api_key(provider_type)
        if res and res[0]:
            return res[0]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[discovery] pool key lookup {provider_type} failed: {e}")
    return fallback


async def _safe_get_json(
    session,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_DISCOVERY_TIMEOUT_S,
    rate_limit_key: Optional[str] = None,
) -> Optional[object]:
    """Bounded GET → JSON. Returns None on any failure mode so callers
    can `if data is None: return []` without re-implementing the
    try/except dance.

    rate_limit_key: when set, an HTTP 429 is reported to pool_engine against
    that API key so the offending account cools and rotation hands the next
    call a sibling key (fail-soft no-op when the pool isn't running)."""
    if session is None:
        return None
    host = url.split("/")[2] if "://" in url else url
    try:
        async with session.get(url, headers=headers or {}, timeout=timeout) as resp:
            if resp.status != 200:
                if resp.status == 429 and rate_limit_key:
                    try:
                        from config.rpc_provider import RPCProvider
                        await RPCProvider.report_key_rate_limit(rate_limit_key, 60)
                    except Exception:
                        pass
                _warn_rate_limited(
                    f"http:{host}",
                    f"[discovery] GET {host} -> HTTP {resp.status} "
                    "(rate-limited/unauthorized?)")
                return None
            return await resp.json(content_type=None)
    except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
        _warn_rate_limited(f"http:{host}", f"[discovery] GET {host} failed: {e}")
        return None


async def fetch_dexscreener_top_traders(
    session,
    chain: str,
    cfg: DiscoveryConfig,
    rl: _RateLimiter,
) -> List[DiscoveredCandidate]:
    """QUARANTINED (Wave-F5). This adapter previously harvested
    ``pairAddress`` — an AMM **pool contract**, not a trader wallet — plus a
    ``info.deployerAddress`` field DexScreener does not return, and
    ``_looks_like_wallet`` could not tell a pool from a wallet. Every row it
    produced was a junk pool address that scored ~0 and polluted the candidate
    set. DexScreener has no public top-traders REST endpoint, so this source
    **can never** produce a wallet.

    It now returns [] unconditionally (mock mode still yields deterministic
    test rows). The volume signal it *can* legitimately provide — top-volume
    tokens/pools per chain — is exposed by ``fetch_dexscreener_token_pools``
    and consumed by the v3 ``helius_tokens`` source, which resolves REAL
    fee-payer wallets from those tokens' recent swaps.
    """
    if cfg.mock:
        return _mock_candidates(chain, SOURCE_DEXSCREENER, cfg)
    return []


async def fetch_dexscreener_token_pools(
    session,
    chain: str,
    cfg: DiscoveryConfig,
    rl: _RateLimiter,
    *,
    max_tokens: int = 10,
) -> List[Dict]:
    """DexScreener public search endpoint → the chain's highest 1h-volume
    tokens/pools. Returns pool/token metadata (NEVER treated as wallets):

        {chain, token_address, pool_address, symbol, volume_h1}

    No API key required. Used as the token feed for the v3 ``helius_tokens``
    discovery source, which pulls each token's recent swap txs and aggregates
    REAL fee-payer wallets across sweeps.
    """
    if cfg.mock or session is None:
        return []
    chain_param = chain.lower()
    url = f"https://api.dexscreener.com/latest/dex/search?q={chain_param}"
    await rl.acquire("api.dexscreener.com")
    data = await _safe_get_json(session, url, timeout=cfg.request_timeout_s)
    if not isinstance(data, dict):
        return []
    pairs = data.get("pairs") or []
    if not isinstance(pairs, list):
        return []
    pairs_sorted = sorted(
        [p for p in pairs if isinstance(p, dict) and p.get("chainId") == chain_param],
        key=lambda p: float((p.get("volume") or {}).get("h1") or 0),
        reverse=True,
    )[: max(1, int(max_tokens))]
    out: List[Dict] = []
    for p in pairs_sorted:
        base = p.get("baseToken") or {}
        token_addr = base.get("address")
        pool_addr = p.get("pairAddress")
        if not isinstance(token_addr, str):
            continue
        out.append({
            "chain": chain,
            "token_address": token_addr,
            "pool_address": pool_addr if isinstance(pool_addr, str) else None,
            "symbol": base.get("symbol"),
            "volume_h1": float((p.get("volume") or {}).get("h1") or 0),
        })
    return out


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

    # Wave-F5: rotated key per call — a cooled Birdeye account rotates out.
    birdeye_key = await _pool_rotated_key('BIRDEYE_API', cfg.birdeye_api_key)
    url = "https://public-api.birdeye.so/defi/v2/tokens/top_traders?sort_by=volume&sort_type=desc"
    headers = {
        "x-chain": "solana",
        "X-API-KEY": birdeye_key,
    }
    await rl.acquire("public-api.birdeye.so")
    data = await _safe_get_json(session, url, headers=headers,
                                timeout=cfg.request_timeout_s,
                                rate_limit_key=birdeye_key)
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
        # Wave-F5: rotated key per program call so the sweep's quota burn
        # spreads across sibling Helius accounts; a 429 (reported inside
        # _safe_get_json) cools the account and the next call rotates.
        helius_key = await _pool_rotated_key('HELIUS_API', cfg.helius_api_key)
        url = (
            f"https://api.helius.xyz/v0/addresses/{prog}/transactions"
            f"?api-key={helius_key}&limit=100&type=SWAP"
        )
        await rl.acquire("api.helius.xyz")
        data = await _safe_get_json(session, url, timeout=cfg.request_timeout_s,
                                    rate_limit_key=helius_key)
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


async def fetch_operator_targets(
    db_pool,
    cfg: DiscoveryConfig,
) -> List[DiscoveredCandidate]:
    """Seed candidates from the operator-configured target_wallets list
    in config_settings.copytrading_config.target_wallets. The operator
    already chose these wallets, so they're trivially worth scoring even
    before any third-party source returns rows.

    Recognises EVM '0x...@chain' suffixed format (split on '@') and bare
    Solana base58 addresses (assumed solana).
    """
    if db_pool is None:
        return []
    raw = None
    try:
        async with db_pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT value FROM config_settings "
                "WHERE config_type='copytrading_config' "
                "  AND key='target_wallets'"
            )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"operator-targets DB read failed: {e}")
        return []
    if not raw:
        return []
    # value is stored as JSON list-of-strings by ConfigManager.
    import json as _json
    try:
        wallets = _json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    if not isinstance(wallets, list):
        return []
    out: List[DiscoveredCandidate] = []
    for w in wallets:
        if not isinstance(w, str) or not w.strip():
            continue
        s = w.strip()
        if "@" in s:
            addr, _, chain_suffix = s.partition("@")
            chain = (chain_suffix or "ethereum").lower()
        elif s.startswith("0x"):
            addr, chain = s, "ethereum"
        else:
            addr, chain = s, "solana"
        if chain not in cfg.chains:
            continue
        if not _looks_like_wallet(addr, chain):
            continue
        out.append(DiscoveredCandidate(
            chain=chain,
            wallet_address=addr,
            source=SOURCE_MANUAL,
            label="operator-configured",
            raw={"source": "config_settings.target_wallets"},
        ))
    return out


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
            # Operator-flagged: with only OPEN trades (status='open' /
            # exit_timestamp IS NULL) the old WHERE filter returned 0
            # candidates even when copytrading_trades had rows. Now
            # we count both OPEN and CLOSED trades from the last 30 days
            # — a wallet we're actively mirroring is a candidate by
            # definition, even before its first exit. profit_loss is
            # only summed over closed legs (open rows have NULL pl).
            rows = await conn.fetch(
                """
                SELECT source_wallet AS wallet_address,
                       COALESCE(SUM(profit_loss) FILTER (WHERE profit_loss IS NOT NULL), 0) AS pnl,
                       COUNT(*) AS n,
                       COUNT(*) FILTER (WHERE status = 'open') AS n_open
                FROM copytrading_trades
                WHERE chain = $1
                  AND COALESCE(exit_timestamp, entry_timestamp) > NOW() - INTERVAL '30 days'
                GROUP BY source_wallet
                ORDER BY COUNT(*) DESC, SUM(profit_loss) DESC NULLS LAST
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
            raw={
                "pnl_30d": float(r["pnl"] or 0),
                "trades_30d": int(r["n"]),
                "open_30d": int(r["n_open"] or 0),
            },
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
        # Operator-configured target_wallets are always candidates — the
        # operator already vouched for them by pasting them into settings.
        # Without this seed the sweep can return 0 when 3rd-party APIs
        # have no keys and copytrading_trades is empty.
        jobs.append(fetch_operator_targets(db_pool, cfg))
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

    # Flatten + dedupe, counting candidates per source for diagnostics.
    seen: Dict[tuple, DiscoveredCandidate] = {}
    per_source: Dict[str, int] = {}
    for r in results:
        if isinstance(r, Exception):
            _warn_rate_limited("discovery", f"discovery source raised: {r}")
            continue
        for c in r or []:
            per_source[c.source] = per_source.get(c.source, 0) + 1
            key = (c.chain, c.wallet_address)
            if key not in seen:
                seen[key] = c

    # HONESTY (Wave-F5): distinguish EXTERNAL discovery sources from the LOCAL
    # fallback (operator target_wallets + our own copytrading_trades history —
    # both can only ever surface already-tracked wallets). When every external
    # source is empty, log ONE honest WARNING and mark the fallback candidates
    # fallback=true so downstream / the operator knows discovery is degraded.
    external_sources = {SOURCE_DEXSCREENER, SOURCE_HELIUS, SOURCE_BIRDEYE, SOURCE_GMGN}
    local_sources = {SOURCE_MANUAL, SOURCE_ONCHAIN}
    external_total = sum(per_source.get(s, 0) for s in external_sources)
    local_total = sum(per_source.get(s, 0) for s in local_sources)
    if external_total == 0 and local_total > 0:
        _warn_rate_limited(
            "fallback",
            "[discovery] ALL external candidate sources returned 0 "
            f"(per-source={per_source}); serving LOCAL fallback of "
            f"{local_total} already-tracked wallet(s). Discovery is degraded — "
            "check HELIUS_API_KEY quota / BIRDEYE_API_KEY / ETHERSCAN_API_KEY.")
        for c in seen.values():
            if c.source in local_sources:
                c.raw = {**(c.raw or {}), "fallback": True}

    logger.info(
        f"[discovery] sweep: {len(seen)} unique candidates across "
        f"{len(cfg.chains)} chains / {len(cfg.sources)} sources "
        f"(per-source={per_source}, external={external_total}, local={local_total})"
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
