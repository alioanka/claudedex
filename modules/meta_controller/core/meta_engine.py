"""META_CONTROLLER engine — collect → decide → persist → (optionally) actuate.

One tick:
  1. For every trading module, read DRY_RUN and LIVE closed-trade aggregates
     (reusing orchestrator_ai's single-sourced _MODULE_QUERIES schema map) into
     a health_scorer.ModulePerf (dry + live tracks).
  2. health_scorer.decide() -> a transparent ACTIVATE / KEEP / PAUSE decision
     with a per-module health score + readable reason.
  3. Persist one current row per module to meta_decisions (full audit history).
  4. ADVISORY BY DEFAULT. If meta_autopilot_enabled, actuate decisions by
     writing / clearing logs/.pause_<module> (the SAME short key the engines
     poll via core.dry_run.is_module_paused) — subject to a per-module dwell
     guard. NEVER touches logs/.killswitch, NEVER places a trade, NEVER edits a
     risk gate.
  5. Self-improvement: score the calibration of PAST decisions against the
     forward realized PnL that followed them, and record it to meta_calibration.

Everything is fail-soft: any per-module error is logged and skipped; the tick
continues. No paid LLM — every number is calculator-derivable by the operator.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from modules.meta_controller.core.health_scorer import (
    DECISION_ACTIVATE,
    DECISION_KEEP,
    DECISION_PAUSE,
    MetaThresholds,
    ModulePerf,
    TrackStats,
    decide,
)

# Single-source the per-module trade-table schema from orchestrator_ai so the
# two meta layers never drift on column names.
from modules.orchestrator_ai.core.orchestrator_engine import _MODULE_QUERIES

logger = logging.getLogger("meta_controller")

# Pause flag dir — the SAME files the engines poll (core.dry_run.is_module_paused
# reads logs/.pause_<module>). Autopilot only ever writes/clears these.
_PAUSE_DIR = Path("logs")
_KILLSWITCH = _PAUSE_DIR / ".killswitch"

# Modules the meta layer reasons about. Keys MUST match _MODULE_QUERIES (the
# pause-flag short key) so actuation hits the file the engine actually polls.
META_MODULES = list(_MODULE_QUERIES.keys())  # sniper arbitrage copy_trading futures solana dex ai


# ───────────────────────────── data collection ─────────────────────────────

async def _collect_track(conn, schema: dict, lookback_hours: int,
                         simulated: Optional[bool]) -> TrackStats:
    """Aggregate one track. simulated=True -> dry, False -> live, None -> all
    (used when the table has no is_simulated column; everything counts as dry).
    Two reads: aggregates + a capped per-trade pnl series for Sharpe/drawdown."""
    closed_filter = "status='closed' AND " if schema.get("has_status") else ""
    pnl = schema["pnl_col"]
    tcol = schema["time_col"]
    table = schema["table"]
    window = f"{tcol} > NOW() - INTERVAL '{int(lookback_hours)} hours'"
    sim_filter = ""
    if simulated is True:
        sim_filter = "is_simulated AND "
    elif simulated is False:
        sim_filter = "NOT is_simulated AND "
    where = f"WHERE {sim_filter}{closed_filter}{window}"
    row = await conn.fetchrow(
        f"SELECT COUNT(*) AS closed, "
        f"COUNT(*) FILTER (WHERE {pnl} > 0) AS wins, "
        f"COALESCE(SUM({pnl}), 0) AS pnl "
        f"FROM {table} {where}"
    )
    if row is None or int(row["closed"] or 0) == 0:
        return TrackStats()
    pnl_rows = await conn.fetch(
        f"SELECT {pnl} AS pnl FROM {table} {where} "
        f"ORDER BY {tcol} DESC LIMIT 500"
    )
    return TrackStats(
        closed_trades=int(row["closed"] or 0),
        winning_trades=int(row["wins"] or 0),
        total_pnl_usd=float(row["pnl"] or 0),
        trade_pnls=[float(r["pnl"] or 0) for r in pnl_rows],
    )


async def collect_module_perf(conn, module: str, lookback_hours: int) -> Optional[ModulePerf]:
    """Build ModulePerf (dry + live) for one module. Fail-soft -> None."""
    schema = _MODULE_QUERIES.get(module)
    if schema is None:
        return None
    try:
        if schema["has_is_simulated"]:
            dry = await _collect_track(conn, schema, lookback_hours, True)
            live = await _collect_track(conn, schema, lookback_hours, False)
        else:
            # No is_simulated column: treat the whole window as the dry track.
            dry = await _collect_track(conn, schema, lookback_hours, None)
            live = TrackStats()
        return ModulePerf(module=module, dry=dry, live=live)
    except Exception as exc:
        logger.warning("collect_module_perf(%s) fail-soft: %s", module, exc)
        return None


# ───────────────────────────── persistence ─────────────────────────────────

async def _persist_decision(conn, d, actuated: bool) -> None:
    try:
        await conn.execute(
            "INSERT INTO meta_decisions "
            "(module, decision, health_score, confidence, reason, components, "
            " actuated, created_at) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7, NOW())",
            d.module, d.decision, float(d.health_score), float(d.confidence),
            d.reason, json.dumps(d.components, default=str), actuated,
        )
    except Exception as exc:
        logger.error("meta_decisions insert failed for %s: %s", d.module, exc)


async def _last_actuation_at(conn, module: str) -> Optional[datetime]:
    try:
        return await conn.fetchval(
            "SELECT MAX(created_at) FROM meta_decisions "
            "WHERE module=$1 AND actuated=TRUE",
            module,
        )
    except Exception:
        return None


# ───────────────────────────── actuation ───────────────────────────────────

def _pause_path(module: str) -> Path:
    return _PAUSE_DIR / f".pause_{module}"


def _is_paused(module: str) -> bool:
    return _pause_path(module).exists()


def _actuate(module: str, decision: str) -> bool:
    """Write/clear the pause flag. Returns True if a change was made.
    NEVER touches the killswitch. Fail-soft."""
    try:
        p = _pause_path(module)
        if decision == DECISION_PAUSE and not p.exists():
            p.write_text(f"meta_controller autopilot PAUSE {datetime.utcnow().isoformat()}Z\n")
            logger.warning("AUTOPILOT: paused %s (wrote %s)", module, p)
            return True
        if decision == DECISION_ACTIVATE and p.exists():
            p.unlink()
            logger.warning("AUTOPILOT: activated %s (removed %s)", module, p)
            return True
    except Exception as exc:
        logger.error("actuate(%s,%s) fail-soft: %s", module, decision, exc)
    return False


# ───────────────────────────── self-improvement ────────────────────────────

async def _record_calibration(conn, lookback_hours: int, th: MetaThresholds) -> None:
    """Score how well PAST decisions matched the forward realized PnL.

    For each module's most recent decision that is at least one lookback window
    old, compare its verdict to the realized PnL of the window that FOLLOWED it:
      - PAUSE was 'correct' if the following window's live (else dry) PnL <= 0
      - ACTIVATE/KEEP was 'correct' if it was >= 0
    Records a simple hit-rate + per-module detail to meta_calibration. Pure
    bookkeeping — it actuates nothing; the operator reads it to judge the
    controller and tune thresholds. Fail-soft.
    """
    try:
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        rows = await conn.fetch(
            "SELECT DISTINCT ON (module) module, decision, created_at "
            "FROM meta_decisions WHERE created_at < $1 "
            "ORDER BY module, created_at DESC",
            cutoff,
        )
        hits = 0
        scored = 0
        detail: Dict[str, dict] = {}
        for r in rows:
            module = r["module"]
            schema = _MODULE_QUERIES.get(module)
            if schema is None:
                continue
            perf = await collect_module_perf(conn, module, lookback_hours)
            if perf is None:
                continue
            fwd = perf.live.total_pnl_usd if perf.live.closed_trades > 0 else perf.dry.total_pnl_usd
            n = perf.live.closed_trades + perf.dry.closed_trades
            if n == 0:
                continue
            decision = r["decision"]
            if decision == DECISION_PAUSE:
                correct = fwd <= 0
            else:  # keep / activate
                correct = fwd >= 0
            scored += 1
            hits += 1 if correct else 0
            detail[module] = {
                "prior_decision": decision,
                "forward_pnl_usd": round(fwd, 4),
                "forward_trades": n,
                "correct": correct,
            }
        if scored == 0:
            return
        hit_rate = hits / scored
        await conn.execute(
            "INSERT INTO meta_calibration "
            "(window_hours, decisions_scored, hit_rate, detail, created_at) "
            "VALUES ($1,$2,$3,$4, NOW())",
            int(lookback_hours), scored, float(hit_rate),
            json.dumps(detail, default=str),
        )
        logger.info("calibration: %d/%d prior decisions matched forward PnL (%.0f%%)",
                    hits, scored, hit_rate * 100)
    except Exception as exc:
        logger.debug("_record_calibration fail-soft: %s", exc)


# ───────────────────────────── config ──────────────────────────────────────

async def load_meta_config(conn) -> dict:
    """config_type='meta_config' rows -> typed dict. Fail-soft to {}."""
    out: dict = {}
    try:
        rows = await conn.fetch(
            "SELECT key, value FROM config_settings WHERE config_type='meta_config'"
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
        logger.error("load_meta_config fail-soft: %s", exc)
    return out


def _thresholds_from_config(cfg: dict) -> MetaThresholds:
    th = MetaThresholds()
    for f in ("pause_score", "activate_score", "min_confidence",
              "pause_loss_usd", "live_weight"):
        if f in cfg:
            try:
                setattr(th, f, float(cfg[f]))
            except (TypeError, ValueError):
                pass
    for f in ("min_trades", "n_target"):
        if f in cfg:
            try:
                setattr(th, f, int(cfg[f]))
            except (TypeError, ValueError):
                pass
    return th


# ───────────────────────────── tick + loop ─────────────────────────────────

async def run_tick(pool, cfg: dict) -> dict:
    """One full meta cycle. Returns a small summary dict (for /status)."""
    th = _thresholds_from_config(cfg)
    lookback_hours = int(cfg.get("lookback_hours", 24))
    autopilot = bool(cfg.get("meta_autopilot_enabled", False))
    dwell_minutes = float(cfg.get("autopilot_dwell_minutes", 720))  # 12h default
    killswitch_blocks = _KILLSWITCH.exists()

    summary = {"scored": 0, "actuated": 0, "decisions": {}, "autopilot": autopilot}
    async with pool.acquire() as conn:
        for module in META_MODULES:
            perf = await collect_module_perf(conn, module, lookback_hours)
            if perf is None:
                continue
            d = decide(perf, currently_paused=_is_paused(module), th=th)
            summary["scored"] += 1
            summary["decisions"][module] = {
                "decision": d.decision,
                "health": round(d.health_score, 3),
                "confidence": round(d.confidence, 3),
            }

            actuated = False
            # Actuate ONLY when autopilot is on, the decision is a real state
            # change, the killswitch is NOT engaged, and the dwell guard allows.
            if (autopilot and not killswitch_blocks
                    and d.decision in (DECISION_PAUSE, DECISION_ACTIVATE)):
                last = await _last_actuation_at(conn, module)
                dwell_ok = (last is None or
                            (datetime.utcnow() - last.replace(tzinfo=None)
                             >= timedelta(minutes=dwell_minutes)))
                if dwell_ok:
                    actuated = _actuate(module, d.decision)
                    if actuated:
                        summary["actuated"] += 1
            await _persist_decision(conn, d, actuated)

        # Self-improvement bookkeeping (advisory; actuates nothing).
        await _record_calibration(conn, lookback_hours, th)
    return summary


async def run_loop(pool, *, get_config=None) -> None:
    """Forever: load config, run a tick, sleep tick_interval_seconds."""
    while True:
        cfg = await get_config() if get_config else {}
        if _KILLSWITCH.exists():
            logger.info("killswitch present — meta tick skipped")
        elif _pause_path("meta_controller").exists():
            logger.info("meta_controller paused — tick skipped")
        else:
            try:
                s = await run_tick(pool, cfg)
                logger.info("meta tick: scored=%d actuated=%d autopilot=%s",
                            s["scored"], s["actuated"], s["autopilot"])
            except Exception as exc:
                logger.error("meta tick failed (fail-soft): %s", exc)
        await asyncio.sleep(int((cfg or {}).get("tick_interval_seconds", 900)))
