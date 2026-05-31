"""Rebalance engine: reads per-module Sharpe from the orchestrator's
recent metrics, calls allocate(), writes proposals to
portfolio_allocations.

Runs on a tick (default hourly). Operator approves proposals via
dashboard. We never auto-apply — the operator decides when to act
on a proposal.

Authority hierarchy (wave-15)
------------------------------
There are now TWO allocation authorities in the system:

  1. ORCHESTRATOR (policy brain, short-horizon):
       orchestrator_engine.run_tick writes `budget_usd` onto each
       orchestrator_recommendations row every 5 min (default). The
       AllocationGuard (core/allocation_guard.py) reads this as the
       LIVE enforced budget for that module. The orchestrator can only
       TIGHTEN or hold budgets within the operator-defined global caps;
       it cannot invent capital. Authority: ENFORCEMENT.

  2. PORTFOLIO ALLOCATOR (longer-horizon display):
       This engine runs hourly, produces fractional-Kelly proposals as
       % of total book, and writes them to portfolio_allocations for
       operator review. Proposals are ADVISORY only until the operator
       approves them via dashboard. They are NOT read by the
       AllocationGuard and do NOT block live entries.
       Authority: ADVISORY / LONGER-HORIZON.

When both produce a number for the same module:
  - Orchestrator budget_usd wins for LIVE enforcement (entry gating).
  - Allocator proposal is the operator-facing display and longer-term
    rebalancing suggestion. If the operator approves an allocator
    proposal it can update the static config_settings budget defaults,
    which the guard uses as fallback when orchestrator has no recent rec.

Why this split:
  - The orchestrator scores on 24h windows (reactive to current
    performance); the allocator uses 7-day Sharpe (smoothed).
  - They serve different latency regimes: the guard needs a sub-minute
    budget to gate entries; the allocator is for human review.
  - Keeping them separate avoids a single mis-calibration cascading
    into both systems simultaneously.

This module annotates its proposals with the current orchestrator
budget (if any) so the dashboard can show "allocator says $300 /
orchestrator has set $220" side-by-side.
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
_MODULES = ["sniper", "arbitrage", "copy_trading", "futures", "solana", "dex", "ai"]
# Module → env flag mapping. Used to determine enabled state.
_ENABLED_ENV = {
    "sniper": "SNIPER_MODULE_ENABLED",
    "arbitrage": "ARBITRAGE_MODULE_ENABLED",
    "copy_trading": "COPY_TRADING_MODULE_ENABLED",
    "futures": "FUTURES_MODULE_ENABLED",
    "solana": "SOLANA_MODULE_ENABLED",
    "dex": "DEX_MODULE_ENABLED",
    "ai": "AI_MODULE_ENABLED",
}


def _env_flag(key: str) -> bool:
    raw = os.getenv(key, "false").strip().lower()
    return raw in ("true", "1", "yes", "on")


async def _read_orchestrator_budgets(conn, ttl_minutes: int = 120) -> Dict[str, Optional[float]]:
    """Read the current orchestrator-recommended budget_usd per module.

    Returns a dict {module: budget_usd | None}. Used to annotate
    allocator proposals so the dashboard can surface both numbers.
    Fail-soft: returns {} on DB error (e.g. budget_usd column not yet
    added — the column is added by migration 046 but may be absent on
    older deployments).
    """
    result: Dict[str, Optional[float]] = {}
    try:
        rows = await conn.fetch(
            "SELECT module, budget_usd FROM orchestrator_recommendations "
            "WHERE budget_usd IS NOT NULL "
            "  AND approved IS NULL "
            "  AND superseded_at IS NULL "
            "  AND created_at > NOW() - INTERVAL '%d minutes' "
            "ORDER BY module, created_at DESC" % int(ttl_minutes)
        )
        seen = set()
        for row in rows:
            m = row["module"]
            if m not in seen:
                result[m] = float(row["budget_usd"]) if row["budget_usd"] is not None else None
                seen.add(m)
    except Exception as exc:
        logger.debug("_read_orchestrator_budgets fail-soft: %s", exc)
    return result


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


async def write_proposals(
    conn,
    report: AllocationReport,
    orchestrator_budgets: Optional[Dict[str, Optional[float]]] = None,
) -> int:
    """Insert one row per proposal with proposed_by='allocator'.
    Returns the number of rows inserted.

    Before inserting the new batch, SUPERSEDE any prior pending
    allocator proposals (one row per module). This is the missing
    invariant that the operator hit: each Recompute click was adding
    5 more pending rows on top of the previous 5, so a 3-click frenzy
    showed 15 duplicates and pollutied history. We mark old pending
    rows by setting effective_until = NOW(), which the UI/queries
    interpret as 'no longer the active proposal'. Approved rows are
    untouched.

    Wave-15: if orchestrator_budgets is provided, annotates each
    proposal's metrics with the orchestrator's current live budget for
    that module so the dashboard can surface both numbers side-by-side
    and the operator understands which authority is actively enforcing.
    """
    try:
        await conn.execute(
            "UPDATE portfolio_allocations "
            "SET effective_until = NOW() "
            "WHERE proposed_by = 'allocator' "
            "  AND approved_at IS NULL "
            "  AND effective_until IS NULL"
        )
    except Exception as e:
        logger.warning("supersede prior pending allocator proposals failed: %s", e)
    n = 0
    for p in report.proposals:
        try:
            components = dict(p.components)
            if orchestrator_budgets is not None:
                orch_budget = orchestrator_budgets.get(p.module)
                components["orchestrator_budget_usd"] = orch_budget
                components["authority_note"] = (
                    "orchestrator budget is active enforcement; "
                    "this allocator proposal is advisory / longer-horizon"
                    if orch_budget is not None
                    else "no orchestrator budget active; allocator proposal is advisory"
                )
            await conn.execute(
                "INSERT INTO portfolio_allocations "
                "(module, pct_of_book, usd_amount, proposed_by, reason, metrics) "
                "VALUES ($1, $2, $3, 'allocator', $4, $5::jsonb)",
                p.module, p.pct_of_book, p.usd_amount, p.reason,
                json.dumps(components),
            )
            n += 1
        except Exception as e:
            logger.warning("write_proposal(%s) failed: %s", p.module, e)
    return n


async def run_tick(
    db_pool, lookback_hours: int, total_book_usd: float,
) -> dict:
    """One tick = one proposal cycle. Returns a summary dict.

    Wave-15: also reads the orchestrator's current live budgets and
    annotates each allocator proposal with them so the dashboard can
    surface both numbers. The orchestrator budget is the ENFORCEMENT
    authority; this allocator is the advisory / longer-horizon authority.
    """
    summary = {
        "modules": 0,
        "proposals_written": 0,
        "reserve_pct": 0.0,
        "orchestrator_budgets_found": 0,
        "errors": [],
    }
    async with db_pool.acquire() as conn:
        try:
            inputs = await collect_inputs(conn, lookback_hours)
            summary["modules"] = len(inputs)
            report = allocate(inputs, total_book_usd)
            summary["reserve_pct"] = report.reserve_pct

            # Read orchestrator live budgets (wave-15) to annotate proposals.
            # Fail-soft: empty dict if migration 046 not yet applied.
            orch_budgets = await _read_orchestrator_budgets(conn)
            summary["orchestrator_budgets_found"] = sum(
                1 for v in orch_budgets.values() if v is not None
            )

            summary["proposals_written"] = await write_proposals(
                conn, report, orchestrator_budgets=orch_budgets
            )
            logger.info(
                "tick: reserve=%.2f%% proposals=%d (modules=%d, orch_budgets=%d)",
                report.reserve_pct, summary["proposals_written"],
                summary["modules"], summary["orchestrator_budgets_found"],
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
