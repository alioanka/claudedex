"""Rebalance engine: reads per-module Sharpe from the orchestrator's
recent metrics, calls allocate(), writes proposals to
portfolio_allocations.

Runs on a tick (default hourly). Operator approves proposals via
dashboard. We never auto-apply — the operator decides when to act
on a proposal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional

from .allocator import (
    AllocationInput, AllocationProposal, AllocationReport, allocate,
)

logger = logging.getLogger("portfolio_allocator")


# Modules to consider for allocation. Same set as orchestrator.
_MODULES = ["sniper", "arbitrage", "copy_trading", "futures", "solana"]
# Module → env flag mapping. Used to determine enabled state.
_ENABLED_ENV = {
    "sniper": "SNIPER_MODULE_ENABLED",
    "arbitrage": "ARBITRAGE_MODULE_ENABLED",
    "copy_trading": "COPY_TRADING_MODULE_ENABLED",
    "futures": "FUTURES_MODULE_ENABLED",
    "solana": "SOLANA_MODULE_ENABLED",
}


def _env_flag(key: str) -> bool:
    raw = os.getenv(key, "false").strip().lower()
    return raw in ("true", "1", "yes", "on")


async def _read_module_sharpe(conn, module: str, lookback_hours: int) -> Optional[float]:
    """Most recent Sharpe from orchestrator_recommendations.metrics.
    Falls back to None when no rec has fired for this module in the
    lookback window."""
    row = await conn.fetchrow(
        "SELECT metrics FROM orchestrator_recommendations "
        "WHERE module = $1 AND created_at > NOW() - INTERVAL '%s hours' "
        "ORDER BY created_at DESC LIMIT 1" % int(lookback_hours),
        module,
    )
    if not row or not row["metrics"]:
        return None
    metrics = row["metrics"]
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except Exception:
            return None
    components = metrics.get("components", {})
    sharpe = components.get("sharpe")
    return float(sharpe) if sharpe is not None else None


async def collect_inputs(
    conn, lookback_hours: int = 168,
) -> List[AllocationInput]:
    inputs: List[AllocationInput] = []
    for module in _MODULES:
        enabled = _env_flag(_ENABLED_ENV[module])
        sharpe = await _read_module_sharpe(conn, module, lookback_hours)
        inputs.append(AllocationInput(
            module=module, enabled=enabled, sharpe=sharpe,
        ))
    return inputs


async def write_proposals(conn, report: AllocationReport) -> int:
    """Insert one row per proposal with proposed_by='allocator'.
    Returns the number of rows inserted."""
    n = 0
    for p in report.proposals:
        try:
            await conn.execute(
                "INSERT INTO portfolio_allocations "
                "(module, pct_of_book, usd_amount, proposed_by, reason, metrics) "
                "VALUES ($1, $2, $3, 'allocator', $4, $5::jsonb)",
                p.module, p.pct_of_book, p.usd_amount, p.reason,
                json.dumps(p.components),
            )
            n += 1
        except Exception as e:
            logger.warning("write_proposal(%s) failed: %s", p.module, e)
    return n


async def run_tick(
    db_pool, lookback_hours: int, total_book_usd: float,
) -> dict:
    """One tick = one proposal cycle. Returns a summary dict."""
    summary = {
        "modules": 0,
        "proposals_written": 0,
        "reserve_pct": 0.0,
        "errors": [],
    }
    async with db_pool.acquire() as conn:
        try:
            inputs = await collect_inputs(conn, lookback_hours)
            summary["modules"] = len(inputs)
            report = allocate(inputs, total_book_usd)
            summary["reserve_pct"] = report.reserve_pct
            summary["proposals_written"] = await write_proposals(conn, report)
            logger.info(
                "tick: reserve=%.2f%% proposals=%d (modules=%d)",
                report.reserve_pct, summary["proposals_written"],
                summary["modules"],
            )
        except Exception as e:
            logger.error("tick failed: %s", e, exc_info=True)
            summary["errors"].append(str(e))
    return summary


async def run_loop(
    db_pool,
    tick_interval_seconds: int = 3600,
    lookback_hours: int = 168,
    total_book_usd: float = 1000.0,
) -> None:
    """Forever loop. Caller cancels the task to stop."""
    logger.info(
        "rebalance engine starting: interval=%ds lookback=%dh book=$%.2f",
        tick_interval_seconds, lookback_hours, total_book_usd,
    )
    while True:
        try:
            summary = await run_tick(db_pool, lookback_hours, total_book_usd)
            logger.info("tick summary: %s", summary)
        except Exception as e:
            logger.error("loop iteration error: %s", e, exc_info=True)
        await asyncio.sleep(tick_interval_seconds)
