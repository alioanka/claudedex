"""
AdvisorPortfolioEngine — operator-reported holdings tracker.

Stores and queries the operator's declared portfolio (what they hold on
Midas or elsewhere). This is READ/WRITE but advice-only: no execution.
The operator updates holdings via the advisor dashboard; this engine
reads them to provide contextual advice ("you already hold 20% in AAPL,
adding more increases concentration risk").

Sim position tracking (advisor_sim_positions) also lives here:
  - open_sim_position(advice_result)    -> sim_id (int)
  - mark_to_market(sim_id, price)       -> updated SimPosition
  - close_sim_position(sim_id, price)   -> final SimPosition with pnl
  - list_open_sims()                    -> List[SimPosition]
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from modules.advisor.core.models import (
    AdviceResult,
    Direction,
    Market,
    PortfolioHolding,
    SimPosition,
)

logger = logging.getLogger("advisor.portfolio_engine")


class AdvisorPortfolioEngine:
    """
    Manages operator holdings + sim position lifecycle.

    All persistence is via asyncpg (advisor_portfolio, advisor_sim_positions
    tables defined in migration 058). Pass db_pool=None for unit-test mode
    (in-memory only).
    """

    def __init__(self, db_pool=None):
        self.db_pool = db_pool
        self._sim_cache: dict[int, SimPosition] = {}  # in-memory mirror

    # ------------------------------------------------------------------
    # Sim position management
    # ------------------------------------------------------------------

    async def open_sim_position(self, result: AdviceResult) -> Optional[int]:
        """
        Seed a new sim position from an AdviceResult.
        Returns the DB-generated sim_id, or None on failure.
        """
        if result.entry_low is None and result.entry_high is None:
            logger.warning(
                f"[portfolio] Cannot open sim for {result.symbol}: "
                "no entry price range in AdviceResult"
            )
            return None

        entry_price = (
            ((result.entry_low or 0) + (result.entry_high or 0)) / 2
            if result.entry_low and result.entry_high
            else result.entry_low or result.entry_high
        )

        sim = SimPosition(
            symbol=result.symbol,
            market=result.market,
            direction=result.direction,
            horizon=result.horizon,
            entry_price=entry_price,
            target_price=result.target_price,
            stop_price=result.stop_price,
            notional_usd=result.sim_amount_usd,
        )

        if self.db_pool is None:
            # In-memory fallback (test mode)
            sim_id = len(self._sim_cache) + 1
            self._sim_cache[sim_id] = sim
            return sim_id

        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO advisor_sim_positions
                      (symbol, market, direction, horizon,
                       entry_price, target_price, stop_price,
                       notional_usd, status, opened_at)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'open',NOW())
                    RETURNING id
                    """,
                    sim.symbol,
                    sim.market.value,
                    sim.direction.value,
                    sim.horizon.value,
                    sim.entry_price,
                    sim.target_price,
                    sim.stop_price,
                    sim.notional_usd,
                )
                sim_id = row["id"]
                self._sim_cache[sim_id] = sim
                logger.info(
                    f"[portfolio] Opened sim #{sim_id}: "
                    f"{sim.direction.value} {sim.symbol} "
                    f"@ {sim.entry_price:.4f} "
                    f"notional=${sim.notional_usd:.2f}"
                )
                return sim_id
        except Exception as exc:
            logger.error(f"[portfolio] Failed to open sim for {result.symbol}: {exc}")
            return None

    async def mark_to_market(self, sim_id: int, current_price: float) -> Optional[SimPosition]:
        """
        Update a sim position with the current market price.
        Calculates unrealised PnL and persists to DB.
        """
        sim = self._sim_cache.get(sim_id)
        if sim is None:
            return None

        if sim.direction == Direction.LONG:
            pnl_pct = (current_price - sim.entry_price) / sim.entry_price * 100
        else:  # SHORT
            pnl_pct = (sim.entry_price - current_price) / sim.entry_price * 100

        pnl_usd = sim.notional_usd * (pnl_pct / 100)

        sim.pnl_pct = pnl_pct
        sim.pnl_usd = pnl_usd
        sim.exit_price = current_price

        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE advisor_sim_positions
                        SET current_price=$1, pnl_pct=$2, pnl_usd=$3,
                            updated_at=NOW()
                        WHERE id=$4
                        """,
                        current_price, pnl_pct, pnl_usd, sim_id,
                    )
            except Exception as exc:
                logger.warning(f"[portfolio] mark_to_market DB error sim#{sim_id}: {exc}")

        return sim

    async def close_sim_position(
        self, sim_id: int, exit_price: float, reason: str = "manual"
    ) -> Optional[SimPosition]:
        """
        Close a sim position at exit_price. Calculates final PnL.
        """
        sim = self._sim_cache.get(sim_id)
        if sim is None:
            return None

        if sim.direction == Direction.LONG:
            pnl_pct = (exit_price - sim.entry_price) / sim.entry_price * 100
        else:
            pnl_pct = (sim.entry_price - exit_price) / sim.entry_price * 100

        pnl_usd = sim.notional_usd * (pnl_pct / 100)

        sim.exit_price = exit_price
        sim.pnl_pct = pnl_pct
        sim.pnl_usd = pnl_usd
        sim.status = "closed"
        sim.closed_at = datetime.utcnow()

        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE advisor_sim_positions
                        SET exit_price=$1, pnl_pct=$2, pnl_usd=$3,
                            status='closed', closed_at=NOW(),
                            close_reason=$4
                        WHERE id=$5
                        """,
                        exit_price, pnl_pct, pnl_usd, reason, sim_id,
                    )
            except Exception as exc:
                logger.warning(f"[portfolio] close_sim DB error sim#{sim_id}: {exc}")

        logger.info(
            f"[portfolio] Closed sim #{sim_id}: {sim.symbol} "
            f"pnl={pnl_pct:+.2f}% (${pnl_usd:+.2f}) reason={reason}"
        )
        return sim

    async def list_open_sims(self) -> List[SimPosition]:
        """Return all currently open sim positions."""
        if self.db_pool is None:
            return [s for s in self._sim_cache.values() if s.status == "open"]

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM advisor_sim_positions WHERE status='open'"
                )
            return [_row_to_sim(r) for r in rows]
        except Exception as exc:
            logger.error(f"[portfolio] list_open_sims error: {exc}")
            return []

    async def count_open_sims(self) -> int:
        sims = await self.list_open_sims()
        return len(sims)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _row_to_sim(row) -> SimPosition:
    from modules.advisor.core.models import Direction, Horizon, Market
    return SimPosition(
        symbol=row["symbol"],
        market=Market(row["market"]),
        direction=Direction(row["direction"]),
        horizon=Horizon(row["horizon"]),
        entry_price=float(row["entry_price"]),
        target_price=float(row["target_price"]) if row.get("target_price") else None,
        stop_price=float(row["stop_price"]) if row.get("stop_price") else None,
        notional_usd=float(row["notional_usd"]),
        advice_id=row.get("advice_id"),
        opened_at=row["opened_at"],
        closed_at=row.get("closed_at"),
        exit_price=float(row["exit_price"]) if row.get("exit_price") else None,
        pnl_pct=float(row["pnl_pct"]) if row.get("pnl_pct") else None,
        pnl_usd=float(row["pnl_usd"]) if row.get("pnl_usd") else None,
        status=row.get("status", "open"),
    )
