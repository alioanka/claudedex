"""execution_quality engine — fetch → normalize → score → persist. READ-ONLY.

One tick:
  1. For every trading module, fetch the lookback window of closed trades from
     its own table (schema names single-sourced against orchestrator_ai's
     _MODULE_QUERIES; any drift is logged, never fatal).
  2. Normalize each row into a cost_model.TradeCostInputs — the normalization
     layer is the deliberately boring 70% of this module and is fail-soft per
     row: a bad row is skipped and counted, never raises.
  3. cost_model.compute_trade_cost() per trade (pure), persist idempotently to
     tca_trade_costs (unique module+trade_ref — re-scores are no-ops).
  4. cost_model.aggregate_scorecard() per module, insert one tca_scorecards row.

NEVER trades, NEVER writes a pause/killswitch flag, NEVER touches an order
path. The only DB writes are to the two tca_* tables. No paid LLM.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from modules.execution_quality.core.cost_model import (
    TradeCostInputs,
    TradeCostResult,
    aggregate_scorecard,
    compute_trade_cost,
)

logger = logging.getLogger("execution_quality")

_LOGS = Path("logs")
_KILLSWITCH = _LOGS / ".killswitch"
_PAUSE = _LOGS / ".pause_execution_quality"

# Single-source table/time-col names against orchestrator_ai where possible so
# the meta layers never drift. Import is fail-soft: TCA has its own map.
try:
    from modules.orchestrator_ai.core.orchestrator_engine import _MODULE_QUERIES
except Exception:  # pragma: no cover — orchestrator absent in some deploys
    _MODULE_QUERIES = {}

# Metadata keys modules may use for intent capture / costs. First hit wins.
_QUOTE_ENTRY_KEYS = ("quoted_entry_price", "expected_entry_price",
                     "expected_price", "quote_price", "quoted_price")
_QUOTE_EXIT_KEYS = ("quoted_exit_price", "expected_exit_price")
_GAS_KEYS = ("gas_cost_usd", "gas_usd", "gas_cost", "gas_fee_usd")
_FEE_KEYS = ("fees_usd", "fee_usd", "total_fees_usd")
_EXTRA_KEYS = ("flash_loan_cost", "jito_tip_usd", "priority_fee_usd", "tip_usd")


def _f(v: Any) -> Optional[float]:
    """Decimal/str/None -> float|None, never raises."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _meta(row: Any) -> Dict[str, Any]:
    raw = row["metadata"] if "metadata" in row.keys() else None
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except (TypeError, ValueError):
        return {}


def _meta_f(meta: Dict[str, Any], keys) -> Optional[float]:
    for k in keys:
        if k in meta:
            v = _f(meta.get(k))
            if v is not None:
                return v
    return None


# ───────────────────────── per-module normalizers ─────────────────────────
# Each returns TradeCostInputs or None (row unusable). Fail-soft per row.

def _common_quotes(meta: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "quoted_entry_price": _meta_f(meta, _QUOTE_ENTRY_KEYS),
        "quoted_exit_price": _meta_f(meta, _QUOTE_EXIT_KEYS),
    }


def _norm_dex(r) -> Optional[TradeCostInputs]:
    meta = _meta(r)
    notional = _f(r["usd_value"]) or 0.0
    gas = _meta_f(meta, _GAS_KEYS)
    components: Dict[str, Any] = {}
    if gas is None:
        # Legacy column; unit not guaranteed USD — say so instead of hiding it.
        gas = _f(r["gas_fee"]) or 0.0
        if gas:
            components["gas_unit_assumed"] = "usd_from_gas_fee_column"
    rec_slip = _f(r["slippage"])  # DECIMAL(5,4) fraction -> bps
    gross = _f(r["profit_loss"])
    return TradeCostInputs(
        module="dex", trade_ref=str(r["trade_id"]), venue=str(r["chain"] or ""),
        side=str(r["side"] or "buy"), is_simulated=True,  # table has no flag
        notional_usd=notional,
        entry_price=_f(r["entry_price"]), exit_price=_f(r["exit_price"]),
        recorded_entry_slippage_bps=None if rec_slip is None else rec_slip * 10_000.0,
        fee_usd=_meta_f(meta, _FEE_KEYS) or 0.0, gas_usd=gas or 0.0,
        extra_cost_usd=_meta_f(meta, _EXTRA_KEYS) or 0.0,
        gross_pnl_usd=gross, net_pnl_usd=gross,
        trade_time=r["trade_time"], components=components,
        **_common_quotes(meta),
    )


def _norm_solana(r) -> Optional[TradeCostInputs]:
    meta = _meta(r)
    sol_usd = _f(r["sol_price_usd"]) or 0.0
    notional = (_f(r["amount_sol"]) or 0.0) * sol_usd
    fee_usd = _meta_f(meta, _FEE_KEYS)
    if fee_usd is None:
        fee_usd = (_f(r["fees_sol"]) or 0.0) * sol_usd
    net = _f(r["pnl_usd"])
    return TradeCostInputs(
        module="solana", trade_ref=str(r["trade_ref"]), venue="solana",
        side=str(r["side"] or "buy"), is_simulated=bool(r["is_simulated"]),
        notional_usd=notional,
        entry_price=_f(r["entry_price"]), exit_price=_f(r["exit_price"]),
        fee_usd=fee_usd or 0.0, gas_usd=_meta_f(meta, _GAS_KEYS) or 0.0,
        extra_cost_usd=_meta_f(meta, _EXTRA_KEYS) or 0.0,
        gross_pnl_usd=None if net is None else net + (fee_usd or 0.0),
        net_pnl_usd=net, trade_time=r["trade_time"],
        components={"strategy": str(r["strategy"] or "")},
        **_common_quotes(meta),
    )


def _norm_futures(r) -> Optional[TradeCostInputs]:
    meta = _meta(r)
    return TradeCostInputs(
        module="futures", trade_ref=str(r["trade_ref"]),
        venue=str(r["exchange"] or ""), side=str(r["side"] or "long"),
        is_simulated=bool(r["is_simulated"]),
        notional_usd=_f(r["notional_value"]) or 0.0,
        entry_price=_f(r["entry_price"]), exit_price=_f(r["exit_price"]),
        fee_usd=_f(r["fees"]) or 0.0, gas_usd=0.0,
        extra_cost_usd=_meta_f(meta, _EXTRA_KEYS) or 0.0,
        gross_pnl_usd=_f(r["pnl"]), net_pnl_usd=_f(r["net_pnl"]),
        trade_time=r["trade_time"], components={},
        **_common_quotes(meta),
    )


def _norm_arbitrage(r) -> Optional[TradeCostInputs]:
    meta = _meta(r)
    net = _f(r["profit_loss"])
    gas = _meta_f(meta, _GAS_KEYS) or 0.0
    flash = _f(meta.get("flash_loan_cost")) or 0.0
    modeled_slip = _f(meta.get("slippage_cost")) or 0.0
    total_costs = _f(meta.get("total_costs"))
    gross = None
    if net is not None and total_costs is not None:
        gross = net + total_costs  # costs were deducted upstream
    components: Dict[str, Any] = {}
    if modeled_slip:
        components["modeled_slippage_usd"] = modeled_slip  # estimate, not realized
    return TradeCostInputs(
        module="arbitrage", trade_ref=str(r["trade_id"]),
        venue=str(r["chain"] or ""), side=str(r["side"] or "buy"),
        is_simulated=bool(r["is_simulated"]),
        notional_usd=_f(r["entry_usd"]) or 0.0,
        entry_price=_f(r["entry_price"]), exit_price=_f(r["exit_price"]),
        fee_usd=_meta_f(meta, _FEE_KEYS) or 0.0, gas_usd=gas,
        extra_cost_usd=flash + modeled_slip,
        gross_pnl_usd=gross, net_pnl_usd=net,
        trade_time=r["trade_time"], components=components,
        **_common_quotes(meta),
    )


def _norm_generic_usd(module: str):
    """sniper / copy_trading / ai share the entry_usd + profit_loss shape."""
    def norm(r) -> Optional[TradeCostInputs]:
        meta = _meta(r)
        net = _f(r["profit_loss"])
        fee = _meta_f(meta, _FEE_KEYS) or 0.0
        return TradeCostInputs(
            module=module, trade_ref=str(r["trade_id"]),
            venue=str(r["chain"] or ""), side=str(r["side"] or "buy"),
            is_simulated=bool(r["is_simulated"]) if r["is_simulated"] is not None else True,
            notional_usd=_f(r["entry_usd"]) or 0.0,
            entry_price=_f(r["entry_price"]), exit_price=_f(r["exit_price"]),
            fee_usd=fee, gas_usd=_meta_f(meta, _GAS_KEYS) or 0.0,
            extra_cost_usd=_meta_f(meta, _EXTRA_KEYS) or 0.0,
            gross_pnl_usd=None if net is None else net + fee, net_pnl_usd=net,
            trade_time=r["trade_time"], components={},
            **_common_quotes(meta),
        )
    return norm


# module -> (table, SELECT sql with {hours}/$1=limit, normalizer)
_TCA_SOURCES: Dict[str, Dict[str, Any]] = {
    "dex": {
        "table": "trades",
        "sql": ("SELECT trade_id, chain, side, entry_price, exit_price, usd_value, "
                "gas_fee, slippage, profit_loss, metadata, "
                "entry_timestamp AS trade_time FROM trades "
                "WHERE status='closed' AND entry_timestamp > NOW() - INTERVAL '{hours} hours' "
                "ORDER BY entry_timestamp DESC LIMIT $1"),
        "norm": _norm_dex,
    },
    "solana": {
        "table": "solana_trades",
        "sql": ("SELECT COALESCE(trade_id, id::text) AS trade_ref, side, strategy, "
                "entry_price, exit_price, amount_sol, sol_price_usd, fees_sol, "
                "pnl_usd, is_simulated, metadata, entry_time AS trade_time "
                "FROM solana_trades "
                "WHERE entry_time > NOW() - INTERVAL '{hours} hours' "
                "ORDER BY entry_time DESC LIMIT $1"),
        "norm": _norm_solana,
    },
    "futures": {
        "table": "futures_trades",
        "sql": ("SELECT id::text AS trade_ref, exchange, side, entry_price, "
                "exit_price, notional_value, fees, pnl, net_pnl, is_simulated, "
                "metadata, entry_time AS trade_time FROM futures_trades "
                "WHERE entry_time > NOW() - INTERVAL '{hours} hours' "
                "ORDER BY entry_time DESC LIMIT $1"),
        "norm": _norm_futures,
    },
    "arbitrage": {
        "table": "arbitrage_trades",
        "sql": ("SELECT trade_id, chain, side, entry_price, exit_price, entry_usd, "
                "profit_loss, is_simulated, metadata, "
                "entry_timestamp AS trade_time FROM arbitrage_trades "
                "WHERE status='closed' AND entry_timestamp > NOW() - INTERVAL '{hours} hours' "
                "ORDER BY entry_timestamp DESC LIMIT $1"),
        "norm": _norm_arbitrage,
    },
    "sniper": {
        "table": "sniper_trades",
        "sql": ("SELECT trade_id, chain, side, entry_price, exit_price, entry_usd, "
                "profit_loss, is_simulated, metadata, "
                "entry_timestamp AS trade_time FROM sniper_trades "
                "WHERE status='closed' AND entry_timestamp > NOW() - INTERVAL '{hours} hours' "
                "ORDER BY entry_timestamp DESC LIMIT $1"),
        "norm": _norm_generic_usd("sniper"),
    },
    "copy_trading": {
        "table": "copytrading_trades",
        "sql": ("SELECT trade_id, chain, side, entry_price, exit_price, entry_usd, "
                "profit_loss, is_simulated, metadata, "
                "entry_timestamp AS trade_time FROM copytrading_trades "
                "WHERE status='closed' AND entry_timestamp > NOW() - INTERVAL '{hours} hours' "
                "ORDER BY entry_timestamp DESC LIMIT $1"),
        "norm": _norm_generic_usd("copy_trading"),
    },
    "ai": {
        "table": "ai_trades",
        "sql": ("SELECT trade_id, chain, side, entry_price, exit_price, entry_usd, "
                "profit_loss, is_simulated, metadata, "
                "entry_timestamp AS trade_time FROM ai_trades "
                "WHERE status='closed' AND entry_timestamp > NOW() - INTERVAL '{hours} hours' "
                "ORDER BY entry_timestamp DESC LIMIT $1"),
        "norm": _norm_generic_usd("ai"),
    },
}

# Drift guard against orchestrator_ai's schema map (warn-only, never fatal).
for _m, _spec in _TCA_SOURCES.items():
    _ref = _MODULE_QUERIES.get(_m)
    if _ref and _ref.get("table") != _spec["table"]:
        logger.warning("TCA table drift for %s: tca=%s orchestrator=%s",
                       _m, _spec["table"], _ref.get("table"))


# ───────────────────────────── config / persist ─────────────────────────────

async def load_tca_config(pool) -> dict:
    """config_type='execution_quality' rows -> typed dict. Fail-soft to {}."""
    out: dict = {}
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM config_settings "
                "WHERE config_type='execution_quality'"
            )
        for r in rows:
            v = r["value"]
            if isinstance(v, str):
                low = v.lower()
                if low in ("true", "false"):
                    v = low == "true"
                else:
                    try:
                        v = float(v) if "." in v else int(v)
                    except ValueError:
                        pass
            out[r["key"]] = v
    except Exception as exc:
        logger.error("load_tca_config fail-soft: %s", exc)
    return out


async def _persist_trade_cost(conn, r: TradeCostResult) -> bool:
    """Insert one per-trade row; returns True iff a new row was written."""
    inserted = await conn.fetchval(
        "INSERT INTO tca_trade_costs ("
        " module, trade_ref, venue, side, is_simulated, notional_usd,"
        " fee_bps, gas_bps, extra_bps, entry_slippage_bps, exit_slippage_bps,"
        " total_cost_bps, total_cost_usd, gross_pnl_usd, net_pnl_usd,"
        " cost_to_gross_pct, quote_covered, mev_suspect, components, trade_time"
        ") VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20) "
        "ON CONFLICT (module, trade_ref) DO NOTHING RETURNING 1",
        r.module, r.trade_ref, r.venue, r.side, r.is_simulated, r.notional_usd,
        r.fee_bps, r.gas_bps, r.extra_bps, r.entry_slippage_bps,
        r.exit_slippage_bps, r.total_cost_bps, r.total_cost_usd,
        r.gross_pnl_usd, r.net_pnl_usd, r.cost_to_gross_pct, r.quote_covered,
        r.mev_suspect, json.dumps(r.components, default=str), r.trade_time,
    )
    return inserted is not None


async def _persist_scorecard(conn, module: str, window_hours: int,
                             card: Dict[str, Any], extra: Dict[str, Any]) -> None:
    await conn.execute(
        "INSERT INTO tca_scorecards ("
        " module, window_hours, trades_scored, quote_coverage_pct, avg_fee_bps,"
        " avg_gas_bps, avg_extra_bps, avg_entry_slippage_bps,"
        " median_entry_slippage_bps, avg_total_cost_bps, median_total_cost_bps,"
        " total_cost_usd, gross_pnl_usd, net_pnl_usd, cost_to_gross_pct,"
        " mev_suspect_count, breaches, components"
        ") VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)",
        module, window_hours, card["trades_scored"], card["quote_coverage_pct"],
        card["avg_fee_bps"], card["avg_gas_bps"], card["avg_extra_bps"],
        card["avg_entry_slippage_bps"], card["median_entry_slippage_bps"],
        card["avg_total_cost_bps"], card["median_total_cost_bps"],
        card["total_cost_usd"], card["gross_pnl_usd"], card["net_pnl_usd"],
        card["cost_to_gross_pct"], card["mev_suspect_count"],
        json.dumps(card["breaches"], default=str),
        json.dumps(extra, default=str),
    )


# ─────────────────────────────── tick / loop ───────────────────────────────

async def run_tick(pool, cfg: dict) -> Dict[str, Any]:
    """Score one window. Per-module errors are logged and skipped."""
    hours = max(int(cfg.get("lookback_hours", 24)), 1)
    limit = max(int(cfg.get("max_rows_per_module", 500)), 1)
    min_trades = int(cfg.get("min_trades_for_scorecard", 3))
    include_sim = bool(cfg.get("include_simulated", True))
    sandwich_bps = float(cfg.get("sandwich_suspect_bps", 150.0))
    thresholds = {
        k: float(cfg[k]) for k in
        ("fee_warn_bps", "gas_warn_bps", "slippage_warn_bps", "total_cost_warn_bps")
        if k in cfg
    }

    summary: Dict[str, Any] = {
        "modules_scored": 0, "trades_scored": 0, "rows_inserted": 0,
        "rows_skipped_bad": 0, "modules": {},
    }

    for module, spec in _TCA_SOURCES.items():
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(spec["sql"].format(hours=hours), limit)
                results: List[TradeCostResult] = []
                for row in rows:
                    try:
                        inp = spec["norm"](row)
                        if inp is None:
                            summary["rows_skipped_bad"] += 1
                            continue
                        if not include_sim and inp.is_simulated:
                            continue
                        results.append(
                            compute_trade_cost(inp, sandwich_suspect_bps=sandwich_bps)
                        )
                    except Exception as exc:
                        summary["rows_skipped_bad"] += 1
                        logger.debug("TCA %s row skipped: %s", module, exc)

                inserted = 0
                for res in results:
                    try:
                        if await _persist_trade_cost(conn, res):
                            inserted += 1
                    except Exception as exc:
                        logger.debug("TCA %s persist skipped: %s", module, exc)

                card = aggregate_scorecard(results, thresholds=thresholds,
                                           min_trades_for_breaches=min_trades)
                await _persist_scorecard(conn, module, hours, card, {
                    "rows_fetched": len(rows),
                    "include_simulated": include_sim,
                    "sandwich_suspect_bps": sandwich_bps,
                })

            summary["modules_scored"] += 1
            summary["trades_scored"] += card["trades_scored"]
            summary["rows_inserted"] += inserted
            summary["modules"][module] = {
                "trades": card["trades_scored"],
                "coverage_pct": card["quote_coverage_pct"],
                "avg_total_cost_bps": card["avg_total_cost_bps"],
                "breaches": len(card["breaches"]),
            }
            if card["breaches"]:
                logger.warning("TCA breach %s: %s", module, card["breaches"])
        except Exception as exc:
            logger.error("TCA tick failed for %s (fail-soft): %s", module, exc)

    return summary


async def run_loop(pool, *, get_config: Optional[Callable] = None) -> None:
    """Forever: load config, run a tick, sleep tick_interval_seconds."""
    while True:
        cfg = await get_config() if get_config else {}
        if _KILLSWITCH.exists():
            logger.info("killswitch present — TCA tick skipped")
        elif _PAUSE.exists():
            logger.info("execution_quality paused — tick skipped")
        else:
            try:
                s = await run_tick(pool, cfg)
                logger.info(
                    "TCA tick: modules=%d trades=%d new_rows=%d bad_rows=%d",
                    s["modules_scored"], s["trades_scored"],
                    s["rows_inserted"], s["rows_skipped_bad"],
                )
            except Exception as exc:
                logger.error("TCA tick failed (fail-soft): %s", exc)
        await asyncio.sleep(int((cfg or {}).get("tick_interval_seconds", 1800)))
