"""PARAM_TUNER engine — load registry → reward → propose → (optionally) apply.

One tick, per registered tunable:
  1. Read the tunable's CURRENT value from config_settings and its bandit
     state from param_bandit_state (creating/reconciling the discretized
     grid against the operator's [min, max] bounds in the registry).
  2. REWARD: score the owning module's rolling closed-trade performance
     (same *_trades aggregates meta_controller/orchestrator_ai read; the
     score_track math is imported from meta_controller so the two layers
     never drift) and attribute it to the arm matching the value that was
     ACTUALLY configured. At most one reward per reward_window_hours per
     key, and only when the window has >= reward_min_trades closed trades.
  3. PROPOSE: bandit.propose() -> at most one pending row in
     param_proposals (deduped against existing pending rows). SHADOW-ONLY
     BY DEFAULT: a proposal row is all that ever happens.
  4. AUTO-APPLY (default OFF): only when EVERY link of the chain holds —
     auto_apply_enabled (DB, default false) AND killswitch absent AND
     pause flag absent AND the key is still in the registry AND the
     proposal is kind='exploit' (never 'explore') AND the value re-clamps
     inside the operator's [min, max] AND the per-key cooldown elapsed.
     The ONLY write is the config_settings value (old value preserved in
     the proposal row + config_history for one-step reversal). It NEVER
     touches code, env flags, pause files, or logs/.killswitch.

Everything fail-soft: a broken registry entry is logged and skipped; the
tick continues. No paid LLM anywhere.

HONESTY: rewards come from the live/dry tape under whatever value was
running — paper-tuned parameters overfit, and a window's performance is
confounded by regime. Proposals are hypotheses for DRY_RUN validation,
not verified improvements. That is why shadow-only is the default.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from modules.param_tuner.core import bandit
from modules.param_tuner.core import challenge
from modules.param_tuner.core import challenge_engine
from modules.param_tuner.core.bandit import KIND_EXPLOIT

# Single-source the per-module trade-table schema and the track-scoring
# math from the existing meta layers so column names and score semantics
# never drift between controllers.
from modules.meta_controller.core.health_scorer import score_track
from modules.meta_controller.core.meta_engine import collect_module_perf

logger = logging.getLogger("param_tuner")

_PAUSE_DIR = Path("logs")
_KILLSWITCH = _PAUSE_DIR / ".killswitch"
_PAUSE_FLAG = _PAUSE_DIR / ".pause_param_tuner"

# Hard whitelist of config namespaces the tuner may EVER write to when
# auto-apply is on. Risk namespaces are permanently out of scope: a tuner
# that can widen stops is a martingale generator with extra steps.
_FORBIDDEN_CONFIG_TYPES = {"risk_management", "security", "wallets"}
_FORBIDDEN_KEY_TOKENS = ("stop_loss", "leverage", "max_loss", "killswitch",
                         "live_execution", "dry_run", "private_key", "secret")


def registry_entry_allowed(entry: dict) -> bool:
    """Reject registry entries that target risk/secret knobs, whatever
    the operator typed. Defense in depth on top of the seeded registry."""
    ctype = str(entry.get("config_type", "")).lower()
    key = str(entry.get("key", "")).lower()
    if ctype in _FORBIDDEN_CONFIG_TYPES:
        return False
    return not any(tok in key for tok in _FORBIDDEN_KEY_TOKENS)


# ───────────────────────────── config ──────────────────────────────────────

async def load_tuner_config(conn) -> dict:
    """config_type='param_tuner' rows -> typed dict. Fail-soft to {}."""
    out: dict = {}
    try:
        rows = await conn.fetch(
            "SELECT key, value FROM config_settings WHERE config_type='param_tuner'"
        )
        for r in rows:
            v = r["value"]
            if isinstance(v, str):
                low = v.lower()
                if low in ("true", "false"):
                    v = low == "true"
                elif low.startswith("[") or low.startswith("{"):
                    try:
                        v = json.loads(v)
                    except (ValueError, TypeError):
                        pass
                else:
                    try:
                        v = float(v) if "." in v else int(v)
                    except ValueError:
                        pass
            out[r["key"]] = v
    except Exception as exc:
        logger.error("load_tuner_config fail-soft: %s", exc)
    return out


def parse_registry(cfg: dict) -> List[dict]:
    """Validate tunable_registry entries. Bad/forbidden entries logged + skipped."""
    raw = cfg.get("tunable_registry", [])
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            logger.error("tunable_registry is not valid JSON — registry empty")
            return []
    entries: List[dict] = []
    for e in raw if isinstance(raw, list) else []:
        try:
            assert e["config_type"] and e["key"] and e["module"]
            lo, hi = float(e["min"]), float(e["max"])
            assert hi > lo
            e = {**e, "min": lo, "max": hi,
                 "steps": int(e.get("steps", 5)),
                 "value_type": str(e.get("value_type", "float"))}
            if not registry_entry_allowed(e):
                logger.warning("registry entry %s/%s targets a forbidden "
                               "namespace — skipped", e["config_type"], e["key"])
                continue
            entries.append(e)
        except (KeyError, TypeError, ValueError, AssertionError) as exc:
            logger.warning("bad registry entry %r skipped: %s", e, exc)
    return entries


# ───────────────────────────── DB helpers ──────────────────────────────────

async def _read_current_value(conn, ctype: str, key: str) -> Optional[float]:
    try:
        raw = await conn.fetchval(
            "SELECT value FROM config_settings WHERE config_type=$1 AND key=$2",
            ctype, key,
        )
        return None if raw is None else float(raw)
    except (TypeError, ValueError) as exc:
        logger.warning("current value of %s/%s is non-numeric: %s", ctype, key, exc)
        return None


async def _load_state(conn, entry: dict) -> bandit.BanditState:
    """Load bandit state, creating/reconciling against current bounds."""
    ctype, key = entry["config_type"], entry["key"]
    key_id = f"{ctype}/{key}"
    row = await conn.fetchrow(
        "SELECT state FROM param_bandit_state WHERE config_type=$1 AND key=$2",
        ctype, key,
    )
    if row is None:
        return bandit.new_state(key_id, entry["min"], entry["max"],
                                entry["steps"], entry["value_type"])
    st = bandit.from_dict(json.loads(row["state"]))
    expected = bandit.make_grid(entry["min"], entry["max"],
                                entry["steps"], entry["value_type"])
    if [a.value for a in st.arms] != expected or st.value_type != entry["value_type"]:
        logger.info("registry bounds changed for %s — reconciling grid", key_id)
        st = bandit.reconcile_grid(st, entry["min"], entry["max"],
                                   entry["steps"], entry["value_type"])
    return st


async def _save_state(conn, entry: dict, st: bandit.BanditState,
                      rewarded: bool) -> None:
    await conn.execute(
        "INSERT INTO param_bandit_state (config_type, key, module, state, "
        " last_reward_at, updated_at) "
        "VALUES ($1,$2,$3,$4, CASE WHEN $5 THEN NOW() ELSE NULL END, NOW()) "
        "ON CONFLICT (config_type, key) DO UPDATE SET "
        " module=EXCLUDED.module, state=EXCLUDED.state, updated_at=NOW(), "
        " last_reward_at=CASE WHEN $5 THEN NOW() "
        "                ELSE param_bandit_state.last_reward_at END",
        entry["config_type"], entry["key"], entry["module"],
        json.dumps(bandit.to_dict(st)), rewarded,
    )


async def _reward_due(conn, entry: dict, reward_window_hours: float) -> bool:
    last = await conn.fetchval(
        "SELECT last_reward_at FROM param_bandit_state "
        "WHERE config_type=$1 AND key=$2",
        entry["config_type"], entry["key"],
    )
    if last is None:
        return True
    now = datetime.now(timezone.utc)
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return now - last >= timedelta(hours=reward_window_hours)


# ───────────────────────────── reward ──────────────────────────────────────

async def compute_reward(conn, module: str, lookback_hours: int,
                         n_target: int, reward_min_trades: int) -> Optional[float]:
    """Module rolling performance -> reward in [0,1]. LIVE track preferred
    when it has enough trades, else DRY. None = not enough evidence."""
    perf = await collect_module_perf(conn, module, lookback_hours)
    if perf is None:
        return None
    track = perf.live if perf.live.closed_trades >= reward_min_trades else perf.dry
    if track.closed_trades < reward_min_trades:
        return None
    score, _conf, _comp = score_track(track, n_target)
    return None if score is None else max(0.0, min(1.0, score))


# ───────────────────────────── proposals ───────────────────────────────────

async def _open_pending(conn, ctype: str, key: str) -> int:
    """In-flight proposals for a key: pending (operator queue) AND
    challenge (out-of-sample gate). Both count against the per-key cap so
    a running challenge does not let the tuner also queue a duplicate."""
    return int(await conn.fetchval(
        "SELECT COUNT(*) FROM param_proposals "
        "WHERE config_type=$1 AND key=$2 AND status IN ('pending','challenge')",
        ctype, key,
    ) or 0)


async def _persist_proposal(conn, entry: dict, p: bandit.Proposal,
                            status: str = "pending") -> int:
    return await conn.fetchval(
        "INSERT INTO param_proposals "
        "(config_type, key, module, current_value, proposed_value, "
        " bound_min, bound_max, kind, reason, components, status, created_at) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11, NOW()) RETURNING id",
        entry["config_type"], entry["key"], entry["module"],
        f"{p.current_value:g}", f"{p.proposed_value:g}",
        entry["min"], entry["max"], p.kind, p.reason,
        json.dumps({
            "current_mean": p.current_mean, "proposed_mean": p.proposed_mean,
            "proposed_pulls": p.proposed_pulls, "ucb_score":
                None if p.ucb_score == float("inf") else round(p.ucb_score, 6),
        }), status,
    )


# ───────────────────────────── auto-apply ──────────────────────────────────

async def _cooldown_ok(conn, ctype: str, key: str, cooldown_hours: float) -> bool:
    last = await conn.fetchval(
        "SELECT MAX(applied_at) FROM param_proposals "
        "WHERE config_type=$1 AND key=$2 AND status='auto_applied'",
        ctype, key,
    )
    if last is None:
        return True
    now = datetime.now(timezone.utc)
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return now - last >= timedelta(hours=cooldown_hours)


async def try_auto_apply(conn, entry: dict, p: bandit.Proposal,
                         proposal_id: int, cfg: dict) -> bool:
    """The full safety chain. Returns True only if a config row was
    actually updated. Every reject path logs why and leaves the proposal
    pending for the operator."""
    if not bool(cfg.get("auto_apply_enabled", False)):
        return False
    if _KILLSWITCH.exists() or _PAUSE_FLAG.exists():
        logger.info("auto-apply blocked: killswitch/pause present")
        return False
    if p.kind != KIND_EXPLOIT:
        return False  # explore proposals are NEVER auto-applied
    lo, hi = entry["min"], entry["max"]
    value = bandit.clamp(p.proposed_value, lo, hi)
    if entry["value_type"] == "int":
        value = float(round(value))
    if not (lo <= value <= hi):   # cannot happen post-clamp; belt+braces
        return False
    if not await _cooldown_ok(conn, entry["config_type"], entry["key"],
                              float(cfg.get("auto_apply_cooldown_hours", 24))):
        logger.info("auto-apply blocked: cooldown active for %s/%s",
                    entry["config_type"], entry["key"])
        return False

    old = await conn.fetchval(
        "SELECT value FROM config_settings WHERE config_type=$1 AND key=$2",
        entry["config_type"], entry["key"],
    )
    if old is None:
        logger.warning("auto-apply blocked: %s/%s row vanished",
                       entry["config_type"], entry["key"])
        return False
    new_value = f"{value:g}"
    await conn.execute(
        "UPDATE config_settings SET value=$3, updated_at=NOW(), "
        "updated_by='param_tuner' WHERE config_type=$1 AND key=$2",
        entry["config_type"], entry["key"], new_value,
    )
    await conn.execute(
        "UPDATE param_proposals SET status='auto_applied', applied_at=NOW(), "
        "components = components || $2::jsonb WHERE id=$1",
        proposal_id, json.dumps({"old_value": old, "new_value": new_value}),
    )
    try:  # audit trail for the dashboard's existing config-history view
        await conn.execute(
            "INSERT INTO config_history (config_type, key, old_value, "
            " new_value, change_source, changed_by, reason, timestamp) "
            "VALUES ($1,$2,$3,$4,'param_tuner','param_tuner',$5, NOW())",
            entry["config_type"], entry["key"], old, new_value,
            f"bandit auto-apply (proposal {proposal_id}); revert by "
            f"restoring old_value",
        )
    except Exception as exc:
        logger.warning("config_history insert fail-soft: %s", exc)
    logger.warning("AUTO-APPLIED %s/%s: %s -> %s (proposal %d, bounds [%g, %g])",
                   entry["config_type"], entry["key"], old, new_value,
                   proposal_id, lo, hi)
    return True


# ───────────────────────────── tick + loop ─────────────────────────────────

async def run_tick(pool, cfg: dict) -> dict:
    lookback_hours = int(cfg.get("lookback_hours", 24))
    n_target = int(cfg.get("n_target", 50))
    reward_min_trades = int(cfg.get("reward_min_trades", 5))
    reward_window_hours = float(cfg.get("reward_window_hours", 6))
    c = float(cfg.get("exploration_c", 0.5))
    min_pulls = int(cfg.get("min_pulls_exploit", 3))
    margin = float(cfg.get("improvement_margin", 0.05))
    max_open = int(cfg.get("max_open_proposals_per_key", 1))

    summary = {"tunables": 0, "rewarded": 0, "proposed": 0, "applied": 0,
               "auto_apply": bool(cfg.get("auto_apply_enabled", False))}
    registry = parse_registry(cfg)
    async with pool.acquire() as conn:
        for entry in registry:
            try:
                summary["tunables"] += 1
                current = await _read_current_value(
                    conn, entry["config_type"], entry["key"])
                if current is None:
                    logger.info("%s/%s has no numeric value — skipped",
                                entry["config_type"], entry["key"])
                    continue
                st = await _load_state(conn, entry)

                rewarded = False
                if await _reward_due(conn, entry, reward_window_hours):
                    reward = await compute_reward(
                        conn, entry["module"], lookback_hours,
                        n_target, reward_min_trades)
                    if reward is not None:
                        bandit.update(st, current, reward)
                        rewarded = True
                        summary["rewarded"] += 1
                await _save_state(conn, entry, st, rewarded)

                p = bandit.propose(st, current, c=c,
                                   min_pulls_exploit=min_pulls,
                                   improvement_margin=margin,
                                   lo=entry["min"], hi=entry["max"])
                if p is None:
                    continue
                if await _open_pending(conn, entry["config_type"],
                                       entry["key"]) >= max_open:
                    continue
                await _emit_proposal(conn, entry, p, cfg, summary)
            except Exception as exc:
                logger.error("tunable %s/%s fail-soft: %s",
                             entry.get("config_type"), entry.get("key"), exc)

        # Advance every open variant challenge (score + resolve). Fail-soft.
        try:
            cs = await challenge_engine.process_challenges(conn, cfg)
            summary["challenges_open"] = cs["open"]
            summary["challenges_passed"] = cs["passed"]
            summary["challenges_failed"] = cs["failed"]
            if cs["open"] or cs["passed"] or cs["failed"]:
                logger.info(
                    "challenges: open=%d scored_trades=%d passed=%d failed=%d "
                    "abandoned=%d", cs["open"], cs["scored_trades"],
                    cs["passed"], cs["failed"], cs["abandoned"])
        except Exception as exc:
            logger.error("process_challenges fail-soft: %s", exc)
    return summary


async def _emit_proposal(conn, entry: dict, p: bandit.Proposal, cfg: dict,
                         summary: dict) -> None:
    """Route a bandit proposal. When challenges are enabled and the knob
    has an honest out-of-sample proxy (challenge.classify_route ->
    ROUTE_CHALLENGE), open a baseline-vs-variant challenge instead of
    queueing directly; the proposal only reaches the operator queue if it
    PASSES, and is NEVER auto-applied. Exempt/censored knobs (no honest
    proxy) keep the old direct-to-pending + auto-apply-eligible behavior,
    logged. Challenges disabled -> old behavior for every knob."""
    if bool(cfg.get("challenge_enabled", True)):
        route, detail = challenge.classify_route(
            entry["config_type"], entry["key"],
            p.current_value, p.proposed_value)
        if route == challenge.ROUTE_CHALLENGE:
            cooldown = float(cfg.get("challenge_retry_cooldown_hours", 72))
            if await challenge_engine.failed_recently(
                    conn, entry["config_type"], entry["key"],
                    f"{p.proposed_value:g}", cooldown):
                logger.info("%s/%s -> variant %g failed a challenge within "
                            "%gh cooldown; not re-opening",
                            entry["config_type"], entry["key"],
                            p.proposed_value, cooldown)
                return
            pid = await _persist_proposal(conn, entry, p, status="challenge")
            summary["proposed"] += 1
            await challenge_engine.open_challenge(conn, pid, entry, p, cfg)
            return
        logger.info("%s/%s challenge bypass (%s): %s — direct to pending",
                    entry["config_type"], entry["key"], route, detail)

    # Old behavior: exempt/censored knob, or challenges globally disabled.
    pid = await _persist_proposal(conn, entry, p)
    summary["proposed"] += 1
    if await try_auto_apply(conn, entry, p, pid, cfg):
        summary["applied"] += 1


async def run_loop(pool, *, get_config=None) -> None:
    """Forever: load config, run a tick, sleep tick_interval_seconds."""
    while True:
        cfg = await get_config() if get_config else {}
        if _KILLSWITCH.exists():
            logger.info("killswitch present — param_tuner tick skipped")
        elif _PAUSE_FLAG.exists():
            logger.info("param_tuner paused — tick skipped")
        else:
            try:
                s = await run_tick(pool, cfg)
                logger.info(
                    "tick: tunables=%d rewarded=%d proposed=%d applied=%d "
                    "auto_apply=%s", s["tunables"], s["rewarded"],
                    s["proposed"], s["applied"], s["auto_apply"])
            except Exception as exc:
                logger.error("param_tuner tick failed (fail-soft): %s", exc)
        await asyncio.sleep(int((cfg or {}).get("tick_interval_seconds", 3600)))
