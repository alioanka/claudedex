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

from .performance_scorer import ModuleScoreInputs, score_module

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
_MODULE_QUERIES = {
    "sniper": {
        "table": "sniper_trades",
        "time_col": "entry_timestamp",
        "pnl_col": "profit_loss",
        "has_is_simulated": False,
    },
    "arbitrage": {
        "table": "arbitrage_trades",
        "time_col": "entry_timestamp",
        "pnl_col": "profit_loss",
        "has_is_simulated": True,
    },
    "copy_trading": {
        "table": "copytrading_trades",
        "time_col": "entry_timestamp",
        "pnl_col": "profit_loss",
        "has_is_simulated": True,
    },
    "futures": {
        "table": "futures_trades",
        "time_col": "entry_time",
        "pnl_col": "net_pnl",
        "has_is_simulated": True,
    },
    "solana": {
        "table": "solana_trades",
        "time_col": "entry_timestamp",
        "pnl_col": "profit_loss",
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
        if schema["has_is_simulated"]:
            row = await conn.fetchrow(
                f"SELECT "
                f"  COUNT(*) FILTER (WHERE status='closed') AS closed, "
                f"  COUNT(*) FILTER (WHERE status='closed' AND {schema['pnl_col']} > 0) AS wins, "
                f"  COALESCE(SUM({schema['pnl_col']}) FILTER (WHERE status='closed'), 0) AS pnl, "
                f"  COUNT(*) FILTER (WHERE status='closed' AND NOT is_simulated) AS live "
                f"FROM {schema['table']} "
                f"WHERE {schema['time_col']} > NOW() - INTERVAL '{lookback_hours} hours'"
            )
        else:
            # No is_simulated column — assume DRY_RUN until we can do better.
            row = await conn.fetchrow(
                f"SELECT "
                f"  COUNT(*) FILTER (WHERE status='closed') AS closed, "
                f"  COUNT(*) FILTER (WHERE status='closed' AND {schema['pnl_col']} > 0) AS wins, "
                f"  COALESCE(SUM({schema['pnl_col']}) FILTER (WHERE status='closed'), 0) AS pnl "
                f"FROM {schema['table']} "
                f"WHERE {schema['time_col']} > NOW() - INTERVAL '{lookback_hours} hours'"
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
                f"WHERE status='closed' AND {schema['time_col']} > NOW() - INTERVAL "
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


async def _insert_recommendation(conn, score_result, module: str, inputs) -> None:
    metrics = {
        "closed_trades": inputs.closed_trades,
        "winning_trades": inputs.winning_trades,
        "total_pnl_usd": round(inputs.total_pnl_usd, 4),
        "live_trades": inputs.live_trades,
        "btc_24h_change_pct": inputs.btc_24h_change_pct,
        "score": round(score_result.score, 4),
        "components": score_result.components,
    }
    await conn.execute(
        "INSERT INTO orchestrator_recommendations "
        "(module, recommended, confidence, reason, metrics) "
        "VALUES ($1, $2, $3, $4, $5::jsonb)",
        module,
        score_result.recommended,
        round(score_result.confidence, 3),
        score_result.reason,
        json.dumps(metrics),
    )


async def run_tick(
    db_pool,
    lookback_hours: int = 24,
    recommendation_ttl_minutes: int = 60,
    market_btc_24h: Optional[float] = None,
) -> dict:
    """Run a single scoring tick. Returns a summary dict.

    Caller (the subprocess loop) decides cadence. This function is
    re-entrant and self-contained; calling twice in parallel is safe
    but doubles DB load.
    """
    summary = {
        "modules_scored": 0,
        "recommendations_inserted": 0,
        "stale_expired": 0,
        "errors": [],
    }
    async with db_pool.acquire() as conn:
        # 1. Expire stale pending rows first so the new pass starts clean.
        try:
            summary["stale_expired"] = await _supersede_stale_pending(
                conn, recommendation_ttl_minutes,
            )
        except Exception as e:
            summary["errors"].append(f"supersede: {e}")

        # 2. Score every module + write non-'hold' rows.
        for module, schema in _MODULE_QUERIES.items():
            inputs = await _collect_module_inputs(
                conn, module, schema, lookback_hours, market_btc_24h,
            )
            if inputs is None:
                continue
            summary["modules_scored"] += 1
            if inputs.closed_trades < 5:
                # Skip noise — not enough data to meaningfully score.
                continue
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
            # We deliberately DO write 'hold' rows too — gives the
            # operator visibility into "the orchestrator looked at this
            # but chose not to act". But to keep the table bounded we
            # only persist 'hold' once per module per 24h window.
            if result.recommended == "hold":
                recent_hold = await conn.fetchval(
                    "SELECT 1 FROM orchestrator_recommendations "
                    "WHERE module = $1 AND recommended = 'hold' "
                    "AND created_at > NOW() - INTERVAL '24 hours' LIMIT 1",
                    module,
                )
                if recent_hold:
                    continue
            try:
                await _insert_recommendation(conn, result, module, inputs)
                summary["recommendations_inserted"] += 1
                logger.info(
                    "tick: %s -> %s (conf=%.2f) %s",
                    module, result.recommended, result.confidence, result.reason,
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
) -> None:
    """Forever loop. Caller cancels the task to stop.

    market_state_getter: optional async callable that returns the
    BTC 24h change % for the current tick. If None, the scorer's
    market signal contributes a neutral 0.5.
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
            )
            logger.info("tick summary: %s", summary)
        except Exception as e:
            logger.error("tick loop error: %s", e, exc_info=True)
        await asyncio.sleep(tick_interval_seconds)
