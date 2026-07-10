"""Orchestrator engine: collect per-module aggregates from DB, score
them, write recommendations.

Design:
- One tick = one scoring cycle (default 5 min). Configurable.
- Each tick:
    1. Read closed-trade aggregates from each *_trades table for the
       last `lookback_hours` window.
    2. Hand each module's row to the scorer.
    3. INSERT non-'hold' recommendations into orchestrator_recommendations.
    4. Expire pending rows older than `recommendation_ttl_minutes`.
- No live trading. No action on its own. The dashboard surfaces
  pending recommendations; operator approves manually.

Reuses asyncpg pool, no extra connection management. Logs to
logs/orchestrator_ai/.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .performance_scorer import ModuleScore, ModuleScoreInputs, score_module

logger = logging.getLogger("orchestrator_ai")


# ---- ML calibration model (optional) -----------------------------
# When data/orchestrator_ai_model.pkl exists (trained by
# modules.orchestrator_ai.core.ml_trainer), the engine multiplies
# the scorer's hard-coded confidence by P(operator_agrees | features)
# from the model. Lower confidence = more cautious; higher confidence
# = the model thinks the operator will approve.
_ML_MODEL: Optional[dict] = None
_ML_MODEL_LOADED_AT: Optional[float] = None


def _load_ml_model(path: str = "data/orchestrator_ai_model.pkl") -> Optional[dict]:
    """Best-effort load. Returns None if the file isn't there or is
    malformed — the engine falls back to the scorer's raw confidence."""
    global _ML_MODEL, _ML_MODEL_LOADED_AT
    p = Path(path)
    if not p.exists():
        _ML_MODEL = None
        return None
    try:
        mtime = p.stat().st_mtime
        # Reload only when the file changes on disk.
        if _ML_MODEL is not None and _ML_MODEL_LOADED_AT == mtime:
            return _ML_MODEL
        with open(p, "rb") as f:
            data = pickle.load(f)
        # Sanity check
        if not all(k in data for k in ("feature_columns", "weights", "bias")):
            logger.warning("ml model %s missing required keys; ignoring", path)
            return None
        _ML_MODEL = data
        _ML_MODEL_LOADED_AT = mtime
        logger.info("loaded ML model from %s (acc=%.3f, n=%d, trained=%s)",
                    path, data.get("accuracy", -1), data.get("n_examples", 0),
                    data.get("trained_at", "?"))
        return data
    except Exception as e:
        logger.warning("failed to load ml model %s: %s", path, e)
        return None


def _ml_calibrated_confidence(
    model: dict, inputs: ModuleScoreInputs, components: dict, score: float,
) -> Optional[float]:
    """Run the logistic regression on the current features. Returns
    P(approved) or None on shape mismatch."""
    try:
        feature_columns = model["feature_columns"]
        weights = model["weights"]
        bias = model["bias"]
        # Build the feature vector in the exact order the trainer used.
        # See ml_trainer.FEATURE_KEYS.
        feat_map = {
            "closed_trades": float(inputs.closed_trades),
            "total_pnl_usd": float(inputs.total_pnl_usd),
            "live_trades": float(inputs.live_trades),
            "btc_24h_change_pct": float(inputs.btc_24h_change_pct or 0.0),
            "win_rate": float(components.get("win_rate") or 0.0),
            "pnl_signal": float(components.get("pnl_signal") or 0.0),
            "volume_factor": float(components.get("volume_factor") or 0.0),
            "regime_signal": float(components.get("regime_signal") or 0.0),
            "sharpe_signal": float(components.get("sharpe_signal") or 0.0),
            "sharpe": float(components.get("sharpe") or 0.0),
            "score": float(score),
        }
        x = [feat_map.get(k, 0.0) for k in feature_columns]
        z = bias + sum(w * xi for w, xi in zip(weights, x))
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))
    except Exception as e:
        logger.debug("ml calibration failed: %s", e)
        return None


# Schema map: how to aggregate per-module. Each entry maps a module
# name to the trade table + columns we read.
#
# Module-table conventions:
#   sniper / arbitrage / copy_trading  → soft-delete tables; have a
#                                          'status' column we filter to
#                                          'closed', and entry_timestamp.
#   futures / solana                   → closed-only tables; NO 'status'
#                                          column, and use entry_time
#                                          / pnl_col differently.

# Wave-F7 (GPT-5.6 audit per-module closure): rows tagged
# metadata.excluded=true are poisoned/fabricated history — Solana
# wrong-denomination fake PnL (mig 140A) and arbitrage triangular
# phantom fills (mig 150 Part A). The dashboards already exclude them,
# but this scorer (and meta_controller, which single-sources this
# schema map) kept counting them — exactly how the audit's "suggested
# Solana to_live while PF/Sharpe were strongly negative" happened.
# Every *_trades table in _MODULE_QUERIES has a metadata JSONB column
# (migs 006/008/009/010 + the legacy dex `trades` schema), so the
# filter is applied uniformly.
EXCLUDED_ROW_FILTER = (
    "NOT COALESCE((metadata->>'excluded')::boolean, false) AND "
)
_MODULE_QUERIES = {
    "sniper": {
        "table": "sniper_trades",
        "time_col": "entry_timestamp",
        "pnl_col": "profit_loss",
        "has_status": True,
        "has_is_simulated": False,
    },
    "arbitrage": {
        "table": "arbitrage_trades",
        "time_col": "entry_timestamp",
        "pnl_col": "profit_loss",
        "has_status": True,
        "has_is_simulated": True,
    },
    "copy_trading": {
        "table": "copytrading_trades",
        "time_col": "entry_timestamp",
        "pnl_col": "profit_loss",
        "has_status": True,
        "has_is_simulated": True,
    },
    "futures": {
        "table": "futures_trades",
        "time_col": "entry_time",
        "pnl_col": "net_pnl",
        "has_status": False,   # every row is closed by schema
        "has_is_simulated": True,
    },
    "solana": {
        "table": "solana_trades",
        "time_col": "entry_time",
        "pnl_col": "pnl_usd",
        "has_status": False,   # every row is closed by schema
        "has_is_simulated": True,
    },
    "dex": {
        "table": "trades",
        "time_col": "entry_timestamp",
        "pnl_col": "profit_loss",
        "has_status": True,
        "has_is_simulated": False,
    },
    "ai": {
        "table": "ai_trades",
        "time_col": "entry_timestamp",
        "pnl_col": "profit_loss",
        "has_status": True,
        "has_is_simulated": True,
    },
}


async def _collect_module_inputs(
    conn, module: str, schema: dict, lookback_hours: int,
    market_btc_24h: Optional[float],
) -> Optional[ModuleScoreInputs]:
    """Two DB reads per module: aggregates + per-trade pnl series.
    Returns None on error so the rest of the tick can continue."""
    try:
        # has_status decides whether to filter rows by status='closed'.
        # has_is_simulated decides whether to count 'live' separately.
        # EXCLUDED_ROW_FILTER drops metadata.excluded=true rows (poisoned /
        # phantom history back-tagged by migs 140A/150A) from every score.
        closed_filter = "status='closed' AND " if schema.get("has_status") else ""
        closed_filter = EXCLUDED_ROW_FILTER + closed_filter
        if schema["has_is_simulated"]:
            row = await conn.fetchrow(
                f"SELECT "
                f"  COUNT(*) AS closed, "
                f"  COUNT(*) FILTER (WHERE {schema['pnl_col']} > 0) AS wins, "
                f"  COALESCE(SUM({schema['pnl_col']}), 0) AS pnl, "
                f"  COUNT(*) FILTER (WHERE NOT is_simulated) AS live "
                f"FROM {schema['table']} "
                f"WHERE {closed_filter}{schema['time_col']} > NOW() - INTERVAL '{lookback_hours} hours'"
            )
        else:
            # No is_simulated column — assume DRY_RUN until we can do better.
            row = await conn.fetchrow(
                f"SELECT "
                f"  COUNT(*) AS closed, "
                f"  COUNT(*) FILTER (WHERE {schema['pnl_col']} > 0) AS wins, "
                f"  COALESCE(SUM({schema['pnl_col']}), 0) AS pnl "
                f"FROM {schema['table']} "
                f"WHERE {closed_filter}{schema['time_col']} > NOW() - INTERVAL '{lookback_hours} hours'"
            )
        if row is None:
            return None
        # Per-trade pnls for Sharpe. Cap at 500 rows so a heavy-volume
        # module like sniper (10k+ closed/day) doesn't blow the
        # response budget. The 500 most recent is a tight enough
        # sample for stable Sharpe + light DB load.
        try:
            pnl_rows = await conn.fetch(
                f"SELECT {schema['pnl_col']} AS pnl FROM {schema['table']} "
                f"WHERE {closed_filter}{schema['time_col']} > NOW() - INTERVAL "
                f"'{lookback_hours} hours' "
                f"ORDER BY {schema['time_col']} DESC LIMIT 500"
            )
            trade_pnls = [float(r['pnl'] or 0) for r in pnl_rows]
        except Exception as e:
            logger.debug("trade_pnls fetch failed for %s: %s", module, e)
            trade_pnls = []
        return ModuleScoreInputs(
            module=module,
            closed_trades=int(row["closed"] or 0),
            winning_trades=int(row["wins"] or 0),
            total_pnl_usd=float(row["pnl"] or 0),
            total_volume_usd=0.0,  # not used by score_module yet
            live_trades=int(row.get("live", 0) or 0) if schema["has_is_simulated"] else 0,
            trade_pnls=trade_pnls,
            btc_24h_change_pct=market_btc_24h,
        )
    except Exception as e:
        logger.warning("collect_module_inputs(%s) failed: %s", module, e)
        return None


async def _supersede_stale_pending(conn, ttl_minutes: int) -> int:
    """Mark pending recommendations older than ttl_minutes as superseded
    so the dashboard doesn't show stale calls to action. Returns the
    number of rows expired."""
    cutoff = datetime.utcnow() - timedelta(minutes=ttl_minutes)
    result = await conn.execute(
        "UPDATE orchestrator_recommendations "
        "SET superseded_at = NOW() "
        "WHERE approved IS NULL AND superseded_at IS NULL AND created_at < $1",
        cutoff,
    )
    # asyncpg returns "UPDATE <count>"
    try:
        return int(result.split()[-1])
    except Exception:
        return 0


async def _supersede_module_pending(conn, module: str) -> None:
    """Mark this module's currently-pending (un-acted, un-superseded) rows
    as superseded so the fresh per-tick recommendation becomes the single
    current row for the module. Keeps the table bounded to ~one live row
    per module while preserving full audit history."""
    await conn.execute(
        "UPDATE orchestrator_recommendations "
        "SET superseded_at = NOW() "
        "WHERE module = $1 AND approved IS NULL AND superseded_at IS NULL",
        module,
    )


async def _read_guard_budget_defaults(conn) -> dict:
    """Load allocation_guard_config budget defaults from config_settings.
    Returns a dict {module: budget_usd, 'global_total': cap, ...}.
    Fail-soft: returns empty dict on DB error."""
    try:
        rows = await conn.fetch(
            "SELECT key, value FROM config_settings "
            "WHERE config_type = 'allocation_guard_config' "
            "  AND (key LIKE 'budget_usd_%' OR key = 'global_total_cap_usd')"
        )
        result = {}
        for row in rows:
            key, val = row["key"], row["value"]
            try:
                result[key] = float(val)
            except (TypeError, ValueError):
                pass
        return result
    except Exception as exc:
        logger.debug("_read_guard_budget_defaults fail-soft: %s", exc)
        return {}


def _compute_budget_usd(
    module: str,
    score: float,
    confidence: float,
    recommended: str,
    guard_defaults: dict,
) -> Optional[float]:
    """Derive a recommended capital budget for the module from its score.

    Design:
    - Start from the static default (budget_usd_<module> or 0 = unlimited).
    - Scale by the score (0..1) to tighten budget for poor performers and
      expand it for strong ones.
    - Apply a confidence discount so a low-confidence recommendation does
      not aggressively cut a module's budget.
    - Never exceed the static default (orchestrator can tighten or hold,
      never invent capital beyond the operator-set default).
    - Return None when no meaningful budget can be derived (e.g. zero
      default means unlimited; not_ready modules get None so the guard
      falls back to its static default).
    - A 'disable' recommendation zeros the budget; 'to_dry' halves it;
      'to_live' / 'enable' leave it at the score-scaled value.

    The global caps in config_settings are enforced by the AllocationGuard
    itself; the orchestrator only sets per-module budgets.
    """
    key = f"budget_usd_{module}"
    base_budget = guard_defaults.get(key, 0.0)

    # 0 means "unlimited" — don't emit a budget recommendation that would
    # accidentally cap an operator-set unlimited module.
    if base_budget <= 0:
        return None

    # not_ready / insufficient data: emit None so guard keeps static default.
    if score == 0.0 and confidence == 0.0:
        return None

    # Score multiplier: clamp score to [0.2, 1.0] — never cut below 20% of
    # the base budget purely on score (extreme but recoverable drawdowns
    # shouldn't fully starve a module before the operator reviews).
    score_mult = max(0.2, min(1.0, score))

    # Confidence discount: blend toward the base at low confidence.
    # At confidence=1.0 → full score_mult.  At confidence=0.0 → no change.
    effective_mult = 1.0 + (score_mult - 1.0) * confidence

    recommended_budget = round(base_budget * effective_mult, 2)

    # Verdict overrides:
    if recommended == "disable":
        recommended_budget = 0.0
    elif recommended == "to_dry":
        # Halve the budget to signal caution while allowing some DRY_RUN exposure.
        recommended_budget = round(recommended_budget * 0.5, 2)

    # Never exceed the operator-set base budget.
    recommended_budget = min(recommended_budget, base_budget)

    return recommended_budget


async def _insert_recommendation(
    conn, score_result, module: str, inputs,
    budget_usd: Optional[float] = None,
) -> None:
    metrics = {
        "closed_trades": inputs.closed_trades,
        "winning_trades": inputs.winning_trades,
        "total_pnl_usd": round(inputs.total_pnl_usd, 4),
        "live_trades": inputs.live_trades,
        "btc_24h_change_pct": inputs.btc_24h_change_pct,
        "score": round(score_result.score, 4),
        "components": score_result.components,
    }
    if budget_usd is not None:
        metrics["recommended_budget_usd"] = round(budget_usd, 2)
    await conn.execute(
        "INSERT INTO orchestrator_recommendations "
        "(module, recommended, confidence, reason, metrics, budget_usd) "
        "VALUES ($1, $2, $3, $4, $5::jsonb, $6)",
        module,
        score_result.recommended,
        round(score_result.confidence, 3),
        score_result.reason,
        json.dumps(metrics),
        budget_usd,
    )


async def run_tick(
    db_pool,
    lookback_hours: int = 24,
    recommendation_ttl_minutes: int = 60,
    market_btc_24h: Optional[float] = None,
    min_trades_for_score: int = 5,
) -> dict:
    """Run a single scoring tick. Returns a summary dict.

    Caller (the subprocess loop) decides cadence. This function is
    re-entrant and self-contained; calling twice in parallel is safe
    but doubles DB load.

    Issue 20: EVERY scored module produces exactly one current
    recommendation row per tick — including modules with too little data
    (a 'hold' tagged not_ready) and modules whose verdict is plain 'hold'.
    Previously only sniper/solana surfaced because (a) the < min_trades
    gate dropped the other 5 silently and (b) 'hold' rows were de-duped to
    once per 24h, so a module that returned 'hold' went dark after its
    first row. The operator wants to SEE a per-module recommendation, not
    silence. To keep the table bounded we supersede the module's prior
    pending row before inserting the fresh one (one live row per module,
    full audit history retained).
    """
    summary = {
        "modules_scored": 0,
        "recommendations_inserted": 0,
        "stale_expired": 0,
        "breaker_trips": 0,
        "errors": [],
    }
    # Phase 4C: run the daily-loss circuit breaker BEFORE the scorer.
    # If a module just got flipped to DRY by the breaker, the scorer's
    # 'to_live' recommendation this tick would be moot anyway, and the
    # operator gets a clearer audit trail when the trip event is the
    # first thing in the log.
    try:
        from .circuit_breaker import check_all_modules
        breaker_results = await check_all_modules(db_pool)
        summary["breaker_trips"] = sum(
            1 for r in breaker_results if r.get("tripped") is True
        )
        if summary["breaker_trips"]:
            logger.critical(
                "CIRCUIT BREAKER tick: %d module(s) tripped",
                summary["breaker_trips"],
            )
    except Exception as e:
        summary["errors"].append(f"breaker: {e}")

    async with db_pool.acquire() as conn:
        # 1. Expire stale pending rows first so the new pass starts clean.
        try:
            summary["stale_expired"] = await _supersede_stale_pending(
                conn, recommendation_ttl_minutes,
            )
        except Exception as e:
            summary["errors"].append(f"supersede: {e}")

        # Wave-15: load guard budget defaults once per tick so each module
        # can derive a recommended budget_usd alongside its verdict.
        # Fail-soft: empty dict means no budgets emitted this tick.
        guard_defaults: dict = {}
        try:
            guard_defaults = await _read_guard_budget_defaults(conn)
        except Exception as e:
            logger.debug("guard budget defaults load fail-soft: %s", e)

        # 2. Score every module and emit ONE current recommendation per
        #    module (issue 20). Insufficient-data modules get a 'hold'
        #    tagged not_ready rather than being silently dropped.
        for module, schema in _MODULE_QUERIES.items():
            inputs = await _collect_module_inputs(
                conn, module, schema, lookback_hours, market_btc_24h,
            )
            if inputs is None:
                continue
            summary["modules_scored"] += 1
            if inputs.closed_trades < min_trades_for_score:
                # Not enough data to meaningfully score, but the operator
                # still wants a per-module row. Emit an explicit
                # not_ready 'hold' with zero confidence so the dashboard
                # shows "scored, staying DRY_RUN — insufficient data".
                result = ModuleScore(
                    score=0.0,
                    confidence=0.0,
                    recommended="hold",
                    reason=(
                        f"Not ready: {inputs.closed_trades} closed trades in "
                        f"the {lookback_hours}h window (need >= "
                        f"{min_trades_for_score}). Staying DRY_RUN."
                    ),
                    components={"not_ready": True, "closed_trades": inputs.closed_trades},
                )
            else:
                result = score_module(inputs)
                # ML-calibrated confidence (optional). When a trained model
                # is on disk, multiply the hard-coded confidence by
                # P(operator_agrees | features). Caps the result so the
                # confidence shown in the rec reflects BOTH data sufficiency
                # AND the model's belief that the operator will agree.
                ml_model = _load_ml_model()
                if ml_model is not None:
                    p_agree = _ml_calibrated_confidence(
                        ml_model, inputs, result.components, result.score
                    )
                    if p_agree is not None:
                        raw_conf = result.confidence
                        # Geometric mean: any side near 0 drags the product down.
                        calibrated = math.sqrt(raw_conf * p_agree)
                        result.components["raw_confidence"] = round(raw_conf, 3)
                        result.components["p_agree_ml"] = round(p_agree, 3)
                        result.confidence = round(calibrated, 3)

            # Wave-15: derive a budget recommendation alongside the verdict.
            # The AllocationGuard reads this to set the module's live budget.
            # _compute_budget_usd returns None for not_ready modules or when
            # no static default exists (0 = unlimited), so the guard falls
            # back to its own static defaults in those cases.
            budget_usd: Optional[float] = _compute_budget_usd(
                module=module,
                score=result.score,
                confidence=result.confidence,
                recommended=result.recommended,
                guard_defaults=guard_defaults,
            )

            # Supersede this module's prior pending row, then insert the
            # fresh one. This guarantees every scored module surfaces
            # exactly one current recommendation each tick (including
            # 'hold' / not_ready) while keeping the table bounded to one
            # live row per module. The audit history is preserved via
            # superseded_at on the old rows.
            try:
                await _supersede_module_pending(conn, module)
                await _insert_recommendation(conn, result, module, inputs,
                                             budget_usd=budget_usd)
                summary["recommendations_inserted"] += 1
                budget_log = (
                    f" budget=${budget_usd:.2f}" if budget_usd is not None else ""
                )
                logger.info(
                    "tick: %s -> %s (conf=%.2f)%s %s",
                    module, result.recommended, result.confidence,
                    budget_log, result.reason,
                )
            except Exception as e:
                summary["errors"].append(f"{module}: {e}")
    return summary


async def run_loop(
    db_pool,
    tick_interval_seconds: int = 300,
    lookback_hours: int = 24,
    recommendation_ttl_minutes: int = 60,
    market_state_getter=None,
    min_trades_for_score: int = 5,
) -> None:
    """Forever loop. Caller cancels the task to stop.

    market_state_getter: optional async callable that returns the
    BTC 24h change % for the current tick. If None, the scorer's
    market signal contributes a neutral 0.5.

    min_trades_for_score: closed-trade threshold below which a module
    gets a not_ready 'hold' recommendation instead of a full score.
    """
    logger.info(
        "orchestrator engine starting: interval=%ds lookback=%dh ttl=%dm",
        tick_interval_seconds, lookback_hours, recommendation_ttl_minutes,
    )
    while True:
        try:
            btc_24h = None
            if market_state_getter is not None:
                try:
                    btc_24h = await market_state_getter()
                except Exception as e:
                    logger.debug("market_state fetch failed: %s", e)
            summary = await run_tick(
                db_pool,
                lookback_hours=lookback_hours,
                recommendation_ttl_minutes=recommendation_ttl_minutes,
                market_btc_24h=btc_24h,
                min_trades_for_score=min_trades_for_score,
            )
            logger.info("tick summary: %s", summary)
        except Exception as e:
            logger.error("tick loop error: %s", e, exc_info=True)
        await asyncio.sleep(tick_interval_seconds)
