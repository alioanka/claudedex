"""
COPY v3 read API — thin, read-only query helpers for the dashboard agent.

The dashboard owns the PAGES; this module owns the queries so the SQL stays
in one place. Every function takes the shared asyncpg pool, returns plain
list[dict] / dict (JSON-serializable after datetime conversion), and is
fail-soft (returns empty on any error). NOTHING here mutates state.

Suggested page mapping (dashboard agent):
    /copytrading/discovery_v3   -> get_discovery_leaderboard + get_candidates
    /copytrading/simulator      -> get_shadow_leaderboard + get_shadow_equity_curve
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("CopyV3ReadAPI")


async def get_discovery_leaderboard(
    db_pool, *, chain: Optional[str] = None, status: Optional[str] = None,
    min_score: float = 0.0, limit: int = 50,
) -> List[Dict]:
    """Ranked discovered wallets from copy_discovered_wallets (mig 135).
    Columns include score, score_breakdown, sources (provenance), pnl_basis
    ('sol_numeraire' rows are approximations — surface the flag in the UI)."""
    if db_pool is None:
        return []
    where = ["COALESCE(score, 0) >= $1"]
    args: list = [min_score]
    if chain:
        args.append(chain)
        where.append(f"chain = ${len(args)}")
    if status:
        args.append(status)
        where.append(f"status = ${len(args)}")
    args.append(max(1, min(500, int(limit))))
    sql = (
        "SELECT chain, wallet_address, sources, score, score_breakdown, "
        "       realized_pnl_usd, win_rate, profit_factor, trade_count, "
        "       max_drawdown_pct, avg_hold_seconds, consistency, "
        "       diversification, wash_penalty, lucky_penalty, pnl_basis, "
        "       window_days, status, first_seen_at, last_scored_at "
        "FROM copy_discovered_wallets "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY score DESC NULLS LAST LIMIT ${len(args)}"
    )
    try:
        async with db_pool.acquire() as conn:
            return [dict(r) for r in await conn.fetch(sql, *args)]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"get_discovery_leaderboard: {e}")
        return []


async def get_candidates(
    db_pool, *, status: str = "pending", limit: int = 50,
) -> List[Dict]:
    """copy_leader_candidates rows (mig 092 + 135 columns) for the
    approval queue. reviewed_by='auto_promote' marks machine promotions."""
    if db_pool is None:
        return []
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, chain, wallet_address, source, label, score, "
                "       metrics, score_breakdown, provenance, status, "
                "       proposed_at, reviewed_at, reviewed_by "
                "FROM copy_leader_candidates WHERE status = $1 "
                "ORDER BY score DESC NULLS LAST LIMIT $2",
                status, max(1, min(500, int(limit))),
            )
            return [dict(r) for r in rows]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"get_candidates: {e}")
        return []


async def get_shadow_leaderboard(db_pool, *, limit: int = 50) -> List[Dict]:
    """Per-wallet shadow-copy performance: latest equity snapshot joined with
    fill aggregates. ALL rows are simulated (is_simulated=true)."""
    if db_pool is None:
        return []
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH latest AS (
                    SELECT DISTINCT ON (chain, wallet_address)
                           chain, wallet_address, realized_pnl_usd,
                           unrealized_pnl_usd, equity_usd, open_positions,
                           fills_count, snapshot_at
                    FROM copy_shadow_equity
                    WHERE is_simulated
                    ORDER BY chain, wallet_address, snapshot_at DESC
                ),
                fills AS (
                    SELECT chain, wallet_address,
                           COUNT(*) FILTER (WHERE side = 'sell') AS closed_trades,
                           COUNT(*) FILTER (WHERE side = 'sell'
                                            AND realized_pnl_usd > 0) AS winning_trades,
                           MIN(event_time) AS first_fill_at,
                           MAX(event_time) AS last_fill_at
                    FROM copy_shadow_fills
                    WHERE is_simulated
                    GROUP BY chain, wallet_address
                )
                SELECT l.*, f.closed_trades, f.winning_trades,
                       f.first_fill_at, f.last_fill_at,
                       CASE WHEN COALESCE(f.closed_trades, 0) > 0
                            THEN f.winning_trades::float / f.closed_trades
                            ELSE NULL END AS shadow_win_rate
                FROM latest l
                LEFT JOIN fills f USING (chain, wallet_address)
                ORDER BY l.equity_usd DESC
                LIMIT $1
                """,
                max(1, min(500, int(limit))),
            )
            return [dict(r) for r in rows]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"get_shadow_leaderboard: {e}")
        return []


async def get_shadow_equity_curve(
    db_pool, chain: str, wallet_address: str, *, days: int = 30,
) -> List[Dict]:
    """Equity-curve points for one shadow-copied wallet (chart feed)."""
    if db_pool is None:
        return []
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT snapshot_at, realized_pnl_usd, unrealized_pnl_usd,
                       equity_usd, open_positions, fills_count
                FROM copy_shadow_equity
                WHERE chain = $1 AND wallet_address = $2 AND is_simulated
                  AND snapshot_at > NOW() - ($3::int || ' days')::interval
                ORDER BY snapshot_at ASC
                """,
                chain, wallet_address, max(1, min(365, int(days))),
            )
            return [dict(r) for r in rows]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"get_shadow_equity_curve: {e}")
        return []


async def get_shadow_fills(
    db_pool, chain: str, wallet_address: str, *, limit: int = 100,
) -> List[Dict]:
    """Recent paper fills for one wallet (simulator drill-down table)."""
    if db_pool is None:
        return []
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT token, token_symbol, side, leader_price_usd, "
                "       fill_price_usd, qty, notional_usd, fee_usd, "
                "       realized_pnl_usd, source_ref, event_source, "
                "       event_time, is_simulated "
                "FROM copy_shadow_fills "
                "WHERE chain = $1 AND wallet_address = $2 "
                "ORDER BY event_time DESC LIMIT $3",
                chain, wallet_address, max(1, min(1000, int(limit))),
            )
            return [dict(r) for r in rows]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"get_shadow_fills: {e}")
        return []


async def get_v3_overview(db_pool) -> Dict:
    """One-call summary card for the dashboard header."""
    out = {
        "discovered_total": 0, "proposed": 0, "pending_candidates": 0,
        "shadow_wallets": 0, "shadow_fills": 0, "auto_promoted": 0,
    }
    if db_pool is None:
        return out
    try:
        async with db_pool.acquire() as conn:
            out["discovered_total"] = int(await conn.fetchval(
                "SELECT COUNT(*) FROM copy_discovered_wallets") or 0)
            out["proposed"] = int(await conn.fetchval(
                "SELECT COUNT(*) FROM copy_discovered_wallets "
                "WHERE status = 'proposed'") or 0)
            out["pending_candidates"] = int(await conn.fetchval(
                "SELECT COUNT(*) FROM copy_leader_candidates "
                "WHERE status = 'pending'") or 0)
            out["shadow_wallets"] = int(await conn.fetchval(
                "SELECT COUNT(DISTINCT (chain, wallet_address)) "
                "FROM copy_shadow_fills WHERE is_simulated") or 0)
            out["shadow_fills"] = int(await conn.fetchval(
                "SELECT COUNT(*) FROM copy_shadow_fills WHERE is_simulated") or 0)
            out["auto_promoted"] = int(await conn.fetchval(
                "SELECT COUNT(*) FROM copy_leader_candidates "
                "WHERE reviewed_by = 'auto_promote' AND status = 'approved'") or 0)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"get_v3_overview: {e}")
    return out
