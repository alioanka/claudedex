"""Read per-module closed trades + orchestrator recommendations
within a date window. Used by the replay engine to compute counter-
factual P&L without re-querying the DB on each step.

Returns plain dataclasses, not asyncpg rows, so the rest of the
engine is testable with synthetic data (no DB required).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


# Same schema map as orchestrator_engine.py. Duplicated rather than
# imported because:
#  1. orchestrator_engine.py is async-only; this module is sync-safe
#     for tests.
#  2. Backtest may want a slightly different shape later (entry+exit
#     prices for slippage sim, etc.) without coupling to the
#     orchestrator's tighter contract.
_TRADE_TABLES = {
    "sniper": {
        "table": "sniper_trades",
        "time_col": "exit_timestamp",
        "fallback_time_col": "entry_timestamp",
        "pnl_col": "profit_loss",
        "has_status": True,
    },
    "arbitrage": {
        "table": "arbitrage_trades",
        "time_col": "exit_timestamp",
        "fallback_time_col": "entry_timestamp",
        "pnl_col": "profit_loss",
        "has_status": True,
    },
    "copy_trading": {
        "table": "copytrading_trades",
        "time_col": "exit_timestamp",
        "fallback_time_col": "entry_timestamp",
        "pnl_col": "profit_loss",
        "has_status": True,
    },
    "futures": {
        "table": "futures_trades",
        "time_col": "exit_time",
        "fallback_time_col": "entry_time",
        "pnl_col": "net_pnl",
        "has_status": False,
    },
    "solana": {
        "table": "solana_trades",
        "time_col": "exit_time",          # was exit_timestamp — wrong
        "fallback_time_col": "entry_time", # was entry_timestamp — wrong
        "pnl_col": "pnl_usd",             # was profit_loss — wrong
        "has_status": False,
    },
}


@dataclass
class TradeRow:
    module: str
    ts: datetime          # exit time when available, else entry time
    pnl_usd: float
    is_simulated: Optional[bool]   # None for sniper (no is_simulated col)


@dataclass
class RecRow:
    ts: datetime
    module: str
    recommended: str
    confidence: float
    approved: Optional[bool]


async def load_trades(
    pool,
    start_ts: datetime,
    end_ts: datetime,
    modules: Optional[List[str]] = None,
) -> Dict[str, List[TradeRow]]:
    """Returns {module_name: [TradeRow, ...]} for the requested window.

    Only closed trades are included (status='closed' for tables that
    have a status column; futures_trades is closed-only by schema).
    """
    selected = modules or list(_TRADE_TABLES.keys())
    out: Dict[str, List[TradeRow]] = {m: [] for m in selected}
    async with pool.acquire() as conn:
        for module in selected:
            schema = _TRADE_TABLES.get(module)
            if not schema:
                continue
            time_col = schema["time_col"]
            fallback_time = schema["fallback_time_col"]
            pnl_col = schema["pnl_col"]
            table = schema["table"]
            # COALESCE(exit_ts, entry_ts) so we count trades that
            # are still open (no exit yet) using their entry time —
            # the replay engine needs the chronological order.
            # has_status read from the schema map so we don't hard-
            # code per-module knowledge twice (futures + solana are
            # both closed-only by schema).
            has_status = schema.get("has_status", True)
            has_is_sim = module != "sniper"
            status_clause = "AND status='closed'" if has_status else ""
            is_sim_col = "is_simulated" if has_is_sim else "NULL::boolean"
            sql = (
                f"SELECT COALESCE({time_col}, {fallback_time}) AS ts, "
                f"  COALESCE({pnl_col}, 0) AS pnl, "
                f"  {is_sim_col} AS is_simulated "
                f"FROM {table} "
                f"WHERE COALESCE({time_col}, {fallback_time}) >= $1 "
                f"  AND COALESCE({time_col}, {fallback_time}) <= $2 "
                f"  {status_clause} "
                f"ORDER BY 1 ASC"
            )
            try:
                rows = await conn.fetch(sql, start_ts, end_ts)
            except Exception:
                # Table might not exist in a fresh deployment — fail soft.
                continue
            for r in rows:
                out[module].append(TradeRow(
                    module=module,
                    ts=r["ts"],
                    pnl_usd=float(r["pnl"] or 0),
                    is_simulated=(None if r["is_simulated"] is None
                                  else bool(r["is_simulated"])),
                ))
    return out


async def load_recommendations(
    pool,
    start_ts: datetime,
    end_ts: datetime,
    modules: Optional[List[str]] = None,
) -> List[RecRow]:
    """Returns orchestrator_recommendations in chronological order.
    Filters: created_at within [start_ts, end_ts], optionally per-module."""
    params = [start_ts, end_ts]
    where = "WHERE created_at >= $1 AND created_at <= $2"
    if modules:
        params.append(modules)
        where += f" AND module = ANY(${len(params)})"
    sql = (
        f"SELECT created_at, module, recommended, confidence, approved "
        f"FROM orchestrator_recommendations "
        f"{where} ORDER BY created_at ASC"
    )
    out: List[RecRow] = []
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(sql, *params)
        except Exception:
            return out
        for r in rows:
            out.append(RecRow(
                ts=r["created_at"],
                module=r["module"],
                recommended=r["recommended"],
                confidence=float(r["confidence"]) if r["confidence"] is not None else 0.0,
                approved=r["approved"],
            ))
    return out
