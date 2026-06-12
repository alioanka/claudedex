"""SENTINEL engine — collect → detect → persist → (optionally) freeze.

One tick (minutes-scale, default 60s):
  1. Collect: a two-source price board (Coinbase + Kraken, free public REST),
     per-module heartbeat ages from *_runtime_stats, candidate-flow counters
     from the runtime stats JSONB, and rolling window PnL per module (reusing
     orchestrator_ai's single-sourced _MODULE_QUERIES schema map).
  2. Run the pure detectors (modules.sentinel.core.detectors) — stable depeg,
     price divergence, silent module, full rejection, loss velocity,
     correlated drawdown — each individually DB-toggleable.
  3. Persist graded anomalies to sentinel_anomalies with refire dedup (a
     persisting condition updates last_seen_at/fire_count instead of spamming
     a row per tick).
  4. ADVISORY BY DEFAULT. If sentinel_autopilot_enabled (DB, default false),
     CRITICAL anomalies from FREEZE_ELIGIBLE_DETECTORS may freeze the
     offending module(s) by writing logs/.pause_<module> — the SAME flag the
     engines poll via core.dry_run.is_module_paused. Dwell-guarded per module.
     Sentinel later clears ONLY pause files it wrote itself (marker-checked)
     once the anomaly has stayed clear for unfreeze_clear_minutes. It NEVER
     touches logs/.killswitch, NEVER places a trade, NEVER edits a risk gate.

Everything is fail-soft: any per-detector or per-module error is logged and
skipped; the tick continues. No paid LLM — pure thresholds the operator can
re-derive by hand.

Boundary vs meta_controller: meta_controller scores PERFORMANCE over days;
sentinel detects ANOMALIES over minutes. Both actuate (when enabled) through
the identical pause-file mechanism and never anything stronger.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from modules.sentinel.core.detectors import (
    FREEZE_ELIGIBLE_DETECTORS,
    SEVERITY_CRITICAL,
    Anomaly,
    detect_correlated_drawdown,
    detect_full_rejection,
    detect_loss_velocity,
    detect_price_divergence,
    detect_silent_module,
    detect_stable_depeg,
    extract_candidate_flow,
)
from modules.sentinel.core.price_board import (
    REFERENCE_ASSETS,
    STABLE_ASSETS,
    fetch_price_board,
)

# Single-source the per-module trade-table schema from orchestrator_ai so the
# meta layers never drift on table/column names (same pattern as
# modules/meta_controller/core/meta_engine.py).
from modules.orchestrator_ai.core.orchestrator_engine import _MODULE_QUERIES

logger = logging.getLogger("sentinel")

_PAUSE_DIR = Path("logs")
_KILLSWITCH = _PAUSE_DIR / ".killswitch"
_PAUSE_MARKER = "sentinel autopilot FREEZE"

# Heartbeat sources: single-row (id=1) runtime_stats tables and their expected
# refresh interval in seconds (mirrors the dashboard's freshness budget).
# arbitrage is per-chain (no id=1) and handled separately. Modules without a
# runtime_stats heartbeat (futures, solana, copy_trading) are out of scope for
# the silent-module detector in v1 — documented in CLAUDE.md.
_HEARTBEAT_TABLES: Dict[str, tuple] = {
    "dex": ("dex_runtime_stats", 150),
    "sniper": ("sniper_runtime_stats", 120),
    "ai": ("ai_runtime_stats", 1800),
}
_ARB_HEARTBEAT = ("arbitrage_runtime_stats", 300)

SENTINEL_MODULES = list(_MODULE_QUERIES.keys())

DEFAULT_CONFIG: Dict[str, object] = {
    "sentinel_autopilot_enabled": False,
    "tick_interval_seconds": 60,
    "autopilot_dwell_minutes": 360,
    "unfreeze_clear_minutes": 60,
    "anomaly_refire_minutes": 30,
    "alive_window_hours": 24,
    "price_fetch_timeout_seconds": 8,
    "depeg_enabled": True,
    "depeg_warn_bps": 50.0,
    "depeg_critical_bps": 150.0,
    "divergence_enabled": True,
    "divergence_warn_bps": 100.0,
    "divergence_critical_bps": 300.0,
    "silent_module_enabled": True,
    "heartbeat_warn_factor": 3.0,
    "heartbeat_critical_factor": 10.0,
    "full_rejection_enabled": True,
    "rejection_min_seen": 25,
    "loss_velocity_enabled": True,
    "loss_velocity_window_minutes": 60,
    "loss_velocity_warn_usd_per_hr": 15.0,
    "loss_velocity_critical_usd_per_hr": 50.0,
    "corr_drawdown_enabled": True,
    "corr_drawdown_min_modules": 3,
    "corr_drawdown_module_loss_usd": 5.0,
    "corr_drawdown_critical_total_usd": 100.0,
}


# ───────────────────────────── config ──────────────────────────────────────

async def load_sentinel_config(pool) -> dict:
    """config_type='sentinel' rows -> typed dict over DEFAULT_CONFIG.
    Fail-soft to pure defaults (which are all advisory-safe)."""
    out = dict(DEFAULT_CONFIG)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM config_settings WHERE config_type='sentinel'"
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
        logger.error("load_sentinel_config fail-soft (defaults in force): %s", exc)
    return out


# ───────────────────────────── collection ──────────────────────────────────

async def _heartbeat_ages(conn) -> Dict[str, tuple]:
    """{module: (age_seconds, expected_max_age_s)} for modules with a fresh-ish
    heartbeat table. Missing table/row -> module absent (fail-soft)."""
    out: Dict[str, tuple] = {}
    for module, (table, max_age) in _HEARTBEAT_TABLES.items():
        try:
            age = await conn.fetchval(
                f"SELECT EXTRACT(EPOCH FROM (NOW() - updated_at)) "
                f"FROM {table} WHERE id = 1"
            )
            if age is not None:
                out[module] = (float(age), float(max_age))
        except Exception as exc:
            logger.debug("heartbeat read %s fail-soft: %s", table, exc)
    try:
        table, max_age = _ARB_HEARTBEAT
        age = await conn.fetchval(
            f"SELECT MIN(EXTRACT(EPOCH FROM (NOW() - updated_at))) FROM {table}"
        )
        if age is not None:
            out["arbitrage"] = (float(age), float(max_age))
    except Exception as exc:
        logger.debug("heartbeat read arbitrage fail-soft: %s", exc)
    return out


async def _candidate_flows(conn) -> Dict[str, tuple]:
    """{module: (seen, accepted)} extracted from runtime_stats JSONB where the
    module exposes flow counters (sniper-shaped today; grows with coverage)."""
    out: Dict[str, tuple] = {}
    for module, (table, _max_age) in _HEARTBEAT_TABLES.items():
        try:
            raw = await conn.fetchval(f"SELECT stats FROM {table} WHERE id = 1")
            if raw is None:
                continue
            stats = json.loads(raw) if isinstance(raw, str) else raw
            flow = extract_candidate_flow(stats)
            if flow is not None:
                out[module] = flow
        except Exception as exc:
            logger.debug("candidate flow %s fail-soft: %s", module, exc)
    return out


async def _window_pnl(conn, module: str, window_minutes: float) -> Optional[tuple]:
    """(pnl_usd, closed_trades) over the trailing window for one module's
    trade table (DRY + LIVE combined — a paper bleed is the same strategy
    defect, and freezing on it only ever blocks entries). Fail-soft -> None."""
    schema = _MODULE_QUERIES.get(module)
    if schema is None:
        return None
    try:
        closed_filter = "status='closed' AND " if schema.get("has_status") else ""
        row = await conn.fetchrow(
            f"SELECT COALESCE(SUM({schema['pnl_col']}),0) AS pnl, COUNT(*) AS n "
            f"FROM {schema['table']} WHERE {closed_filter}"
            f"{schema['time_col']} > NOW() - INTERVAL '{int(window_minutes)} minutes'"
        )
        if row is None or int(row["n"] or 0) == 0:
            return None
        return (float(row["pnl"] or 0), int(row["n"] or 0))
    except Exception as exc:
        logger.debug("window pnl %s fail-soft: %s", module, exc)
        return None


async def _was_recently_alive(conn, module: str, alive_window_hours: float) -> bool:
    """True if the module traded within the alive window — used so the
    silent-module detector never alarms on a module that is simply disabled.
    (A fresh-enough heartbeat already proves liveness by itself.)"""
    schema = _MODULE_QUERIES.get(module)
    if schema is None:
        return False
    try:
        n = await conn.fetchval(
            f"SELECT COUNT(*) FROM {schema['table']} WHERE {schema['time_col']} "
            f"> NOW() - INTERVAL '{int(alive_window_hours)} hours'"
        )
        return int(n or 0) > 0
    except Exception:
        return False


# ───────────────────────────── detection ───────────────────────────────────

async def collect_anomalies(pool, cfg: dict) -> List[Anomaly]:
    """Run every enabled detector over freshly collected inputs."""
    anomalies: List[Anomaly] = []

    # 1+2. Market detectors off the two-source price board.
    if cfg.get("depeg_enabled") or cfg.get("divergence_enabled"):
        board = await fetch_price_board(
            timeout_s=float(cfg.get("price_fetch_timeout_seconds", 8)))
        cb, kr = board.get("coinbase", {}), board.get("kraken", {})
        if cfg.get("depeg_enabled"):
            for sym in STABLE_ASSETS:
                prices = [p for p in (cb.get(sym), kr.get(sym)) if p]
                a = detect_stable_depeg(
                    sym, prices,
                    float(cfg["depeg_warn_bps"]), float(cfg["depeg_critical_bps"]))
                if a:
                    anomalies.append(a)
        if cfg.get("divergence_enabled"):
            for sym in REFERENCE_ASSETS:
                a = detect_price_divergence(
                    sym, cb.get(sym), kr.get(sym),
                    float(cfg["divergence_warn_bps"]),
                    float(cfg["divergence_critical_bps"]),
                    "coinbase", "kraken")
                if a:
                    anomalies.append(a)

    async with pool.acquire() as conn:
        # 3. Silent module (heartbeat staleness, alive-gated).
        if cfg.get("silent_module_enabled"):
            ages = await _heartbeat_ages(conn)
            for module, (age, max_age) in ages.items():
                # Fresh heartbeat proves liveness; only check the trade-table
                # alive window when the heartbeat is already stale, so a
                # disabled module (stale forever, no recent trades) is silent.
                if age > max_age and not await _was_recently_alive(
                        conn, module, float(cfg["alive_window_hours"])):
                    continue
                a = detect_silent_module(
                    module, age, max_age,
                    float(cfg["heartbeat_warn_factor"]),
                    float(cfg["heartbeat_critical_factor"]))
                if a:
                    anomalies.append(a)

        # 4. Full rejection (100% gate-block since process start).
        if cfg.get("full_rejection_enabled"):
            flows = await _candidate_flows(conn)
            for module, (seen, accepted) in flows.items():
                a = detect_full_rejection(
                    module, seen, accepted, int(cfg["rejection_min_seen"]))
                if a:
                    anomalies.append(a)

        # 5+6. Loss velocity per module + correlated drawdown across modules.
        if cfg.get("loss_velocity_enabled") or cfg.get("corr_drawdown_enabled"):
            window_m = float(cfg["loss_velocity_window_minutes"])
            module_pnls: Dict[str, float] = {}
            for module in SENTINEL_MODULES:
                res = await _window_pnl(conn, module, window_m)
                if res is None:
                    continue
                pnl, _n = res
                module_pnls[module] = pnl
                if cfg.get("loss_velocity_enabled"):
                    a = detect_loss_velocity(
                        module, pnl, window_m,
                        float(cfg["loss_velocity_warn_usd_per_hr"]),
                        float(cfg["loss_velocity_critical_usd_per_hr"]))
                    if a:
                        anomalies.append(a)
            if cfg.get("corr_drawdown_enabled"):
                a = detect_correlated_drawdown(
                    module_pnls,
                    int(cfg["corr_drawdown_min_modules"]),
                    float(cfg["corr_drawdown_module_loss_usd"]),
                    float(cfg["corr_drawdown_critical_total_usd"]))
                if a:
                    anomalies.append(a)

    return anomalies


# ───────────────────────────── persistence ─────────────────────────────────

async def _persist_anomaly(conn, a: Anomaly, refire_minutes: float) -> None:
    """Insert, or refresh the open row for the same (detector, subject,
    severity) if it last fired within the refire window (dedup)."""
    try:
        updated = await conn.fetchval(
            "UPDATE sentinel_anomalies SET last_seen_at = NOW(), "
            "fire_count = fire_count + 1, value = $4, message = $5, "
            "details = $6 "
            "WHERE id = (SELECT id FROM sentinel_anomalies "
            "            WHERE detector = $1 AND subject = $2 AND severity = $3 "
            "              AND last_seen_at > NOW() - ($7 || ' minutes')::interval "
            "            ORDER BY last_seen_at DESC LIMIT 1) "
            "RETURNING id",
            a.detector, a.subject, a.severity, float(a.value), a.message,
            json.dumps(a.details, default=str), str(float(refire_minutes)),
        )
        if updated is not None:
            return
        await conn.execute(
            "INSERT INTO sentinel_anomalies "
            "(detector, subject, severity, value, threshold, message, details, "
            " module_scoped, actuated, first_seen_at, last_seen_at, fire_count) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8, FALSE, NOW(), NOW(), 1)",
            a.detector, a.subject, a.severity, float(a.value),
            float(a.threshold), a.message, json.dumps(a.details, default=str),
            a.module_scoped,
        )
    except Exception as exc:
        logger.error("sentinel_anomalies persist failed (%s/%s): %s",
                     a.detector, a.subject, exc)


async def _record_action(conn, module: str, action: str, detector: str,
                         reason: str) -> None:
    try:
        await conn.execute(
            "INSERT INTO sentinel_actions (module, action, detector, reason, "
            "created_at) VALUES ($1,$2,$3,$4, NOW())",
            module, action, detector, reason,
        )
    except Exception as exc:
        logger.error("sentinel_actions insert failed (%s/%s): %s",
                     module, action, exc)


async def _last_action_at(conn, module: str) -> Optional[datetime]:
    try:
        return await conn.fetchval(
            "SELECT MAX(created_at) FROM sentinel_actions WHERE module=$1",
            module,
        )
    except Exception:
        return None


async def _mark_actuated(conn, detector: str, subject: str) -> None:
    try:
        await conn.execute(
            "UPDATE sentinel_anomalies SET actuated = TRUE "
            "WHERE id = (SELECT id FROM sentinel_anomalies "
            "            WHERE detector=$1 AND subject=$2 "
            "            ORDER BY last_seen_at DESC LIMIT 1)",
            detector, subject,
        )
    except Exception:
        pass


# ───────────────────────────── actuation ───────────────────────────────────

def _pause_path(module: str) -> Path:
    return _PAUSE_DIR / f".pause_{module}"


def _wrote_pause(module: str) -> bool:
    """True iff the existing pause file was written by sentinel itself.
    Sentinel must never clear an operator's or meta_controller's pause."""
    p = _pause_path(module)
    try:
        return p.exists() and p.read_text().startswith(_PAUSE_MARKER)
    except Exception:
        return False


def _freeze(module: str, detector: str) -> bool:
    """Write the pause flag (entries freeze; per-module exit loops keep
    running by design). NEVER the killswitch. Returns True on a change."""
    try:
        p = _pause_path(module)
        if p.exists():
            return False
        p.write_text(
            f"{_PAUSE_MARKER} detector={detector} "
            f"{datetime.now(timezone.utc).isoformat()}\n")
        logger.warning("AUTOPILOT: froze %s (wrote %s, detector=%s)",
                       module, p, detector)
        return True
    except Exception as exc:
        logger.error("freeze(%s) fail-soft: %s", module, exc)
        return False


def _unfreeze(module: str) -> bool:
    """Clear a pause flag ONLY if sentinel wrote it. Fail-soft."""
    try:
        if not _wrote_pause(module):
            return False
        _pause_path(module).unlink()
        logger.warning("AUTOPILOT: unfroze %s (anomaly clear)", module)
        return True
    except Exception as exc:
        logger.error("unfreeze(%s) fail-soft: %s", module, exc)
        return False


def _freeze_targets(a: Anomaly) -> List[str]:
    """Module short keys a CRITICAL freeze-eligible anomaly points at."""
    if a.module_scoped:
        return [a.subject]
    cands = a.details.get("freeze_candidates", [])
    return [m for m in cands if isinstance(m, str) and m in SENTINEL_MODULES]


async def _actuate(conn, anomalies: List[Anomaly], cfg: dict) -> int:
    """Freeze offenders + clear stale sentinel freezes. Gated by the caller on
    autopilot + killswitch. Dwell-guarded per module. Returns action count."""
    dwell = timedelta(minutes=float(cfg["autopilot_dwell_minutes"]))
    actions = 0

    async def _dwell_ok(module: str) -> bool:
        last = await _last_action_at(conn, module)
        if last is None:
            return True
        now = datetime.now(timezone.utc)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return now - last >= dwell

    # Freeze: CRITICAL anomalies from the freeze-eligible class only.
    for a in anomalies:
        if a.severity != SEVERITY_CRITICAL or a.detector not in FREEZE_ELIGIBLE_DETECTORS:
            continue
        for module in _freeze_targets(a):
            if _pause_path(module).exists():
                continue
            if not await _dwell_ok(module):
                logger.info("autopilot dwell guard blocked freeze of %s", module)
                continue
            if _freeze(module, a.detector):
                actions += 1
                await _record_action(conn, module, "freeze", a.detector, a.message)
                await _mark_actuated(conn, a.detector, a.subject)

    # Unfreeze: sentinel-written pauses whose anomalies stayed clear.
    clear_m = float(cfg["unfreeze_clear_minutes"])
    for module in SENTINEL_MODULES:
        if not _wrote_pause(module):
            continue
        try:
            still_hot = await conn.fetchval(
                "SELECT COUNT(*) FROM sentinel_anomalies "
                "WHERE severity = 'critical' AND detector = ANY($1::text[]) "
                "  AND (subject = $2 OR details->'freeze_candidates' ? $2) "
                "  AND last_seen_at > NOW() - ($3 || ' minutes')::interval",
                list(FREEZE_ELIGIBLE_DETECTORS), module, str(clear_m),
            )
        except Exception as exc:
            logger.debug("unfreeze check %s fail-soft (stay frozen): %s", module, exc)
            continue
        if int(still_hot or 0) > 0:
            continue
        if not await _dwell_ok(module):
            continue
        if _unfreeze(module):
            actions += 1
            await _record_action(conn, module, "unfreeze", "",
                                 f"no critical anomaly for {clear_m:.0f}m")
    return actions


# ───────────────────────────── tick + loop ─────────────────────────────────

async def run_tick(pool, cfg: dict) -> dict:
    """One full sentinel cycle. Returns a small summary dict (for /status)."""
    autopilot = bool(cfg.get("sentinel_autopilot_enabled", False))
    killswitch_blocks = _KILLSWITCH.exists()
    anomalies = await collect_anomalies(pool, cfg)

    summary = {
        "anomalies": len(anomalies),
        "critical": sum(1 for a in anomalies if a.severity == SEVERITY_CRITICAL),
        "actions": 0,
        "autopilot": autopilot,
        "detected": [
            {"detector": a.detector, "subject": a.subject,
             "severity": a.severity, "value": a.value}
            for a in anomalies
        ],
    }

    async with pool.acquire() as conn:
        for a in anomalies:
            log = logger.error if a.severity == SEVERITY_CRITICAL else logger.warning
            log("ANOMALY [%s] %s/%s: %s", a.severity.upper(), a.detector,
                a.subject, a.message)
            await _persist_anomaly(conn, a, float(cfg["anomaly_refire_minutes"]))

        # Actuate ONLY when autopilot is on AND the killswitch is absent.
        if autopilot and not killswitch_blocks:
            summary["actions"] = await _actuate(conn, anomalies, cfg)
    return summary


async def run_loop(pool, *, get_config=None) -> None:
    """Forever: load config, run a tick, sleep tick_interval_seconds."""
    while True:
        cfg = dict(DEFAULT_CONFIG)
        if get_config:
            try:
                cfg = await get_config()
            except Exception as exc:
                logger.error("get_config fail-soft (defaults in force): %s", exc)
        if _KILLSWITCH.exists():
            logger.info("killswitch present — sentinel tick skipped")
        elif _pause_path("sentinel").exists():
            logger.info("sentinel paused — tick skipped")
        else:
            try:
                s = await run_tick(pool, cfg)
                logger.info("sentinel tick: anomalies=%d critical=%d actions=%d autopilot=%s",
                            s["anomalies"], s["critical"], s["actions"], s["autopilot"])
            except Exception as exc:
                logger.error("sentinel tick failed (fail-soft): %s", exc)
        await asyncio.sleep(int(cfg.get("tick_interval_seconds", 60)))
