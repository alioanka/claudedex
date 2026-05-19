"""
Copy-trading slippage-decay tracker (CT-W3-01).

Records per-mirrored-trade `(leader_fill_price, our_fill_price,
delta_ms)` so operators can see which leaders are too fast to mirror
profitably. Persisted to `copy_slippage_observations` (migration 026).

Design contract
---------------
* **Pure helpers** for math (median, rolling stats) so unit tests do
  not need a DB pool. The engine consumes `SlippageObservation` and
  passes it to `persist_observation(db_pool, obs)` -- which is the
  only function that touches Postgres.
* **Fail-soft writes.** Every DB call is wrapped in try/except; the
  engine never crashes a mirrored trade because the slippage
  bookkeeping side-channel hit a wire error.
* **Rolling per-leader stats** are computed by SQL (median via
  `percentile_cont(0.5)`) so we never hold weeks of observations in
  process memory.
* **Slippage sign convention.** Returned `slippage_bps` is signed:
    BUY  -> (our_px - leader_px) / leader_px * 10_000
        positive = we paid more than the leader (bad)
    SELL -> (leader_px - our_px) / leader_px * 10_000
        positive = we sold for less than the leader (bad)
  This keeps "positive is bad" intuitive across both sides.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("CopySlippageTracker")


@dataclass
class SlippageObservation:
    """One mirrored-trade slippage data point.

    All price fields are USD per token (raw decimal price, not log /
    not bps). Either side may be None if we could not derive a fill
    price -- the persistence layer handles NULLs.
    """
    chain: str
    leader_wallet: str
    token_address: str
    side: str                              # 'buy' | 'sell'
    leader_tx_hash: Optional[str] = None
    our_tx_hash: Optional[str] = None
    leader_fill_price_usd: Optional[float] = None
    our_fill_price_usd: Optional[float] = None
    leader_fill_ts: Optional[datetime] = None
    our_fill_ts: Optional[datetime] = None
    is_simulated: bool = True
    notes: Optional[Dict[str, Any]] = None

    @property
    def slippage_bps(self) -> Optional[float]:
        """Signed price-decay in basis points (positive = worse fill
        than the leader). Returns None when either fill price is missing
        or the leader price is non-positive."""
        if (self.leader_fill_price_usd is None
                or self.our_fill_price_usd is None):
            return None
        try:
            lp = float(self.leader_fill_price_usd)
            op = float(self.our_fill_price_usd)
        except (TypeError, ValueError):
            return None
        if lp <= 0:
            return None
        side = (self.side or "").lower()
        if side == "buy":
            diff = (op - lp) / lp
        elif side == "sell":
            diff = (lp - op) / lp
        else:
            return None
        return round(diff * 10_000, 4)

    @property
    def delta_ms(self) -> Optional[int]:
        """Wall-clock latency between leader fill and our fill, in ms.
        Returns None if either timestamp is missing."""
        if not self.leader_fill_ts or not self.our_fill_ts:
            return None
        try:
            return int(
                (self.our_fill_ts - self.leader_fill_ts).total_seconds() * 1000
            )
        except Exception:
            return None


# ---------------------------------------------------------------------
# Pure aggregation helpers (no DB)
# ---------------------------------------------------------------------
def median_signed(values: Sequence[float]) -> Optional[float]:
    """Median of a numeric series. Returns None on empty input."""
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return float(statistics.median(nums))


def rolling_stats(
    rows: Sequence[Dict[str, Any]],
    *,
    window_days: int = 7,
    as_of: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Compute median slippage_bps + delta_ms over a rolling window.

    `rows` is a list of dicts (matching copy_slippage_observations
    columns) -- typically the asyncpg fetch result. Returns a JSON-safe
    dict the API surface can serve unchanged.
    """
    as_of = as_of or datetime.now(timezone.utc)
    cutoff = as_of - timedelta(days=window_days)
    bps_vals: List[float] = []
    ms_vals: List[float] = []
    n_total = 0
    n_bad = 0  # rows with positive slippage_bps -- "bad" fill
    for r in rows:
        ts = r.get("recorded_at")
        if ts and isinstance(ts, datetime):
            tsu = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            if tsu < cutoff:
                continue
        bps = r.get("slippage_bps")
        if bps is not None:
            try:
                v = float(bps)
                bps_vals.append(v)
                if v > 0:
                    n_bad += 1
            except (TypeError, ValueError):
                pass
        ms = r.get("delta_ms")
        if ms is not None:
            try:
                ms_vals.append(float(ms))
            except (TypeError, ValueError):
                pass
        n_total += 1
    return {
        "window_days": window_days,
        "sample_size": n_total,
        "median_slippage_bps": median_signed(bps_vals),
        "median_delta_ms": median_signed(ms_vals),
        "bad_fill_ratio": (n_bad / len(bps_vals)) if bps_vals else None,
    }


# ---------------------------------------------------------------------
# Persistence (DB-backed)
# ---------------------------------------------------------------------
async def persist_observation(db_pool, obs: SlippageObservation) -> bool:
    """Write one observation to copy_slippage_observations.

    Fail-soft: returns False on any error and never raises. The engine
    treats slippage tracking as a side channel -- a missed write must
    not break the mirrored-trade pipeline.
    """
    if db_pool is None or obs is None:
        return False
    try:
        import json as _json
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO copy_slippage_observations (
                    chain, leader_wallet, token_address, side,
                    leader_tx_hash, our_tx_hash,
                    leader_fill_price_usd, our_fill_price_usd,
                    slippage_bps,
                    leader_fill_ts, our_fill_ts, delta_ms,
                    is_simulated, notes
                ) VALUES (
                    $1, $2, $3, $4,
                    $5, $6,
                    $7, $8,
                    $9,
                    $10, $11, $12,
                    $13, $14
                )
                """,
                obs.chain, obs.leader_wallet, obs.token_address, obs.side,
                obs.leader_tx_hash, obs.our_tx_hash,
                obs.leader_fill_price_usd, obs.our_fill_price_usd,
                obs.slippage_bps,
                obs.leader_fill_ts, obs.our_fill_ts, obs.delta_ms,
                bool(obs.is_simulated),
                _json.dumps(obs.notes) if obs.notes else None,
            )
        return True
    except Exception as e:
        logger.debug(f"persist_observation failed (fail-soft): {e}")
        return False


async def get_rolling_slippage(
    db_pool,
    *,
    leader_wallet: Optional[str] = None,
    chain: Optional[str] = None,
    window_days: int = 7,
    limit: int = 5000,
) -> Dict[str, Any]:
    """SQL-side aggregate for the /api/copytrading/slippage endpoint.

    Uses `percentile_cont(0.5)` for the medians so we never load weeks
    of observations into Python memory. Returns a JSON-safe dict.
    """
    if db_pool is None:
        return {
            "leader_wallet": leader_wallet, "chain": chain,
            "window_days": window_days,
            "sample_size": 0,
            "median_slippage_bps": None, "median_delta_ms": None,
            "bad_fill_ratio": None, "leaders": [],
        }
    try:
        async with db_pool.acquire() as conn:
            params: list = []
            where = ["recorded_at > NOW() - ($1 || ' days')::INTERVAL"]
            params.append(str(int(max(1, window_days))))
            if chain:
                where.append(f"chain = ${len(params)+1}")
                params.append(chain)
            if leader_wallet:
                where.append(f"lower(leader_wallet) = lower(${len(params)+1})")
                params.append(leader_wallet)
            where_clause = " AND ".join(where)

            # Single-leader case: scalar stats.
            if leader_wallet:
                row = await conn.fetchrow(
                    f"""
                    SELECT
                      COUNT(*) AS n,
                      PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY slippage_bps)
                        AS median_bps,
                      PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY delta_ms)
                        AS median_ms,
                      SUM(CASE WHEN slippage_bps > 0 THEN 1 ELSE 0 END)::float
                        / NULLIF(COUNT(slippage_bps), 0) AS bad_ratio
                    FROM copy_slippage_observations
                    WHERE {where_clause}
                    """,
                    *params,
                )
                return {
                    "leader_wallet": leader_wallet,
                    "chain": chain,
                    "window_days": window_days,
                    "sample_size": int(row["n"] or 0) if row else 0,
                    "median_slippage_bps": float(row["median_bps"])
                        if row and row["median_bps"] is not None else None,
                    "median_delta_ms": float(row["median_ms"])
                        if row and row["median_ms"] is not None else None,
                    "bad_fill_ratio": float(row["bad_ratio"])
                        if row and row["bad_ratio"] is not None else None,
                }

            # Multi-leader leaderboard view.
            rows = await conn.fetch(
                f"""
                SELECT
                  chain, leader_wallet,
                  COUNT(*) AS n,
                  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY slippage_bps)
                    AS median_bps,
                  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY delta_ms)
                    AS median_ms,
                  SUM(CASE WHEN slippage_bps > 0 THEN 1 ELSE 0 END)::float
                    / NULLIF(COUNT(slippage_bps), 0) AS bad_ratio,
                  MAX(recorded_at) AS last_seen
                FROM copy_slippage_observations
                WHERE {where_clause}
                GROUP BY chain, leader_wallet
                ORDER BY n DESC
                LIMIT {int(max(1, min(limit, 5000)))}
                """,
                *params,
            )
            leaders = []
            for r in rows:
                leaders.append({
                    "chain": r["chain"],
                    "leader_wallet": r["leader_wallet"],
                    "sample_size": int(r["n"] or 0),
                    "median_slippage_bps": float(r["median_bps"])
                        if r["median_bps"] is not None else None,
                    "median_delta_ms": float(r["median_ms"])
                        if r["median_ms"] is not None else None,
                    "bad_fill_ratio": float(r["bad_ratio"])
                        if r["bad_ratio"] is not None else None,
                    "last_seen": r["last_seen"].isoformat()
                        if r["last_seen"] else None,
                })
            return {
                "window_days": window_days,
                "chain": chain,
                "leaders": leaders,
            }
    except Exception as e:
        logger.debug(f"get_rolling_slippage failed: {e}")
        return {
            "leader_wallet": leader_wallet, "chain": chain,
            "window_days": window_days, "sample_size": 0,
            "median_slippage_bps": None, "median_delta_ms": None,
            "bad_fill_ratio": None, "leaders": [], "error": str(e),
        }


__all__ = [
    "SlippageObservation",
    "median_signed",
    "rolling_stats",
    "persist_observation",
    "get_rolling_slippage",
]
