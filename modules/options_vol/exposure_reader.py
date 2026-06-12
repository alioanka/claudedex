"""Fleet net directional exposure (USD) read-only aggregator.

Reuses core.allocation_guard.get_all_committed (the same open-position
queries the allocation guard already runs) for the long-only spot modules,
and replaces the futures number with a SIGNED sum (long minus short notional)
from futures_positions — the only book that can be short.

Net delta proxy:
    net = sum(spot module committed USD) + (futures long - futures short)

This is a crypto-BETA proxy, not a per-asset delta: the fleet's alt/SOL spot
exposure is mapped onto the hedge currency via the fleet_beta config knob
(default 1.0 — i.e., assume the book moves 1:1 with BTC in the correlated
drawdown we are insuring against; that is the scenario that justifies this
module at all). Fail-soft: any DB error returns None, never raises.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger("OptionsVolModule.Exposure")


async def _signed_futures_usd(conn) -> float:
    """Long-minus-short open futures notional. Fail-soft to 0."""
    try:
        row = await conn.fetchrow(
            """
            SELECT COALESCE(SUM(
                CASE WHEN LOWER(side) = 'short' THEN -notional_value
                     ELSE notional_value END), 0)::float AS net
            FROM futures_positions
            """
        )
        return float(row["net"] or 0.0) if row else 0.0
    except Exception as exc:
        logger.debug("_signed_futures_usd fail-soft: %s", exc)
        return 0.0


async def get_fleet_net_delta_usd(db_pool, fleet_beta: float = 1.0
                                  ) -> Optional[Dict[str, float]]:
    """Returns {'net_delta_usd', 'spot_long_usd', 'futures_net_usd',
    'fleet_beta'} or None when the DB is unavailable."""
    if not db_pool:
        return None
    try:
        async with db_pool.acquire() as conn:
            from core.allocation_guard import get_all_committed
            committed = await get_all_committed(conn)
            futures_net = await _signed_futures_usd(conn)
        spot_long = sum(v for k, v in committed.items() if k != "futures")
        net = (spot_long + futures_net) * float(fleet_beta)
        return {
            "net_delta_usd": net,
            "spot_long_usd": spot_long,
            "futures_net_usd": futures_net,
            "fleet_beta": float(fleet_beta),
            "per_module": {k: v for k, v in committed.items() if k != "futures"},
        }
    except Exception as exc:
        logger.warning("fleet exposure read failed (fail-soft -> None): %s", exc)
        return None
