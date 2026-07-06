"""Variant-challenge persistence/loop — the async DB side of challenge.py.

Lifecycle (all inside the normal tuner tick, fail-soft per challenge):
  open_challenge()      proposal persisted with status='challenge' gets one
                        param_variant_challenges row (baseline = current
                        value, variant = proposed value, window frozen at
                        open time).
  process_challenges()  each tick: pull the owning module's closed trades
                        ENTERED after started_at whose exits landed past
                        the incremental cursor, score baseline and variant
                        on the identical rows with the identical harness
                        (challenge.accumulate), then resolve once the
                        window elapses: PASSED -> proposal moves to the
                        existing 'pending' operator queue (NEVER
                        auto-applied); anything else -> 'challenge_failed'.

Writes only param_variant_challenges + param_proposals.status. Never
touches config_settings, never trades, no LLM.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from modules.param_tuner.core import challenge

logger = logging.getLogger("param_tuner")

_SLICE_LIMIT = 500   # max trades scored per challenge per tick


def _naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Trade tables use naive TIMESTAMP columns (server-UTC convention,
    same as the meta layers' NOW() windows); strip tz for comparisons."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


async def failed_recently(conn, ctype: str, key: str, variant_value: str,
                          cooldown_hours: float) -> bool:
    """True if the same (knob, proposed value) failed a challenge inside
    the cooldown. The bandit is deterministic — without this it would
    re-open the identical challenge every tick after a fail."""
    n = await conn.fetchval(
        "SELECT COUNT(*) FROM param_variant_challenges "
        "WHERE config_type=$1 AND key=$2 AND variant_value=$3 "
        " AND verdict=$4 AND resolved_at > NOW() - make_interval(hours => $5)",
        ctype, key, variant_value, challenge.VERDICT_FAILED,
        float(cooldown_hours),
    )
    return int(n or 0) > 0


async def open_challenge(conn, proposal_id: int, entry: dict,
                         p, cfg: dict) -> int:
    window = float(cfg.get("challenge_window_hours", 72))
    cid = await conn.fetchval(
        "INSERT INTO param_variant_challenges "
        "(proposal_id, config_type, key, module, baseline_value, "
        " variant_value, window_hours, components) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING id",
        proposal_id, entry["config_type"], entry["key"], entry["module"],
        f"{p.current_value:g}", f"{p.proposed_value:g}", window,
        json.dumps({"scorer": "gate", "kind": p.kind}),
    )
    logger.info("CHALLENGE %d opened for %s/%s: baseline %g vs variant %g "
                "over %gh (proposal %d)", cid, entry["config_type"],
                entry["key"], p.current_value, p.proposed_value, window,
                proposal_id)
    return int(cid)


async def _fetch_slice(conn, scorer: challenge.GateScorer,
                       started_at: datetime, cursor: datetime) -> list:
    """Closed trades ENTERED after the challenge opened (out-of-sample
    cut) whose exits landed past the cursor. Table/column names come from
    the static code registry in challenge.py, never from operator input,
    so the f-string interpolation is not an injection surface."""
    status_filter = "status='closed' AND " if scorer.has_status else ""
    return await conn.fetch(
        f"SELECT {scorer.feature_col} AS feature, "
        f"       {scorer.pnl_pct_col} AS pnl_pct, "
        f"       {scorer.exit_col}    AS exited_at "
        f"FROM {scorer.table} "
        f"WHERE {status_filter}{scorer.feature_col} IS NOT NULL "
        f"  AND {scorer.pnl_pct_col} IS NOT NULL "
        f"  AND {scorer.exit_col} IS NOT NULL "
        f"  AND {scorer.entry_col} > $1 AND {scorer.exit_col} > $2 "
        f"ORDER BY {scorer.exit_col} ASC LIMIT {_SLICE_LIMIT}",
        _naive_utc(started_at), _naive_utc(cursor),
    )


async def _resolve(conn, row, verdict: str, reason: str,
                   n: int, bsum: float, vsum: float) -> None:
    comp = {
        "reason": reason, "n_samples": n,
        "baseline_mean": round(bsum / n, 6) if n else None,
        "variant_mean": round(vsum / n, 6) if n else None,
    }
    await conn.execute(
        "UPDATE param_variant_challenges SET resolved_at=NOW(), verdict=$2, "
        "components = components || $3::jsonb WHERE id=$1",
        row["id"], verdict, json.dumps(comp),
    )
    if verdict == challenge.VERDICT_PASSED:
        # ONLY transition a passed challenge may make: into the normal
        # pending-operator-approval queue. Never auto-applied from here.
        new_status = "pending"
    elif verdict == challenge.VERDICT_FAILED:
        new_status = "challenge_failed"
    else:
        new_status = None   # abandoned: operator already moved the proposal
    if new_status:
        await conn.execute(
            "UPDATE param_proposals SET status=$2, "
            "components = components || $3::jsonb "
            "WHERE id=$1 AND status='challenge'",
            row["proposal_id"], new_status,
            json.dumps({"challenge_id": row["id"], "challenge_verdict": verdict,
                        "challenge_reason": reason}),
        )
    logger.info("CHALLENGE %d resolved %s for %s/%s (n=%d): %s",
                row["id"], verdict, row["config_type"], row["key"], n, reason)


async def process_challenges(conn, cfg: dict) -> dict:
    """Score + resolve every open challenge. Fail-soft per challenge."""
    min_samples = int(cfg.get("challenge_min_samples", 20))
    min_edge_pct = float(cfg.get("challenge_min_edge_pct", 5))
    cap_pct = float(cfg.get("challenge_pnl_cap_pct", 20))
    out = {"open": 0, "scored_trades": 0, "passed": 0, "failed": 0,
           "abandoned": 0}
    rows = await conn.fetch(
        "SELECT id, proposal_id, config_type, key, module, baseline_value, "
        "       variant_value, n_samples, baseline_reward_sum, "
        "       variant_reward_sum, last_scored_at, started_at, window_hours "
        "FROM param_variant_challenges WHERE resolved_at IS NULL "
        "ORDER BY id",
    )
    now = datetime.now(timezone.utc)
    for row in rows:
        try:
            out["open"] += 1
            n0 = int(row["n_samples"])
            b0, v0 = float(row["baseline_reward_sum"]), float(row["variant_reward_sum"])

            # Proposal dismissed/superseded by the operator mid-challenge?
            status = await conn.fetchval(
                "SELECT status FROM param_proposals WHERE id=$1",
                row["proposal_id"])
            if status != "challenge":
                await _resolve(conn, row, challenge.VERDICT_ABANDONED,
                               f"proposal left challenge state (now "
                               f"{status!r})", n0, b0, v0)
                out["abandoned"] += 1
                continue

            scorer = challenge.KNOB_SCORERS.get(
                (row["config_type"], row["key"]))
            if scorer is None:   # code registry changed under the challenge
                await _resolve(conn, row, challenge.VERDICT_ABANDONED,
                               "knob no longer has a registered scoring "
                               "function", n0, b0, v0)
                out["abandoned"] += 1
                continue

            started = row["started_at"]
            cursor = row["last_scored_at"] or started
            trades = await _fetch_slice(conn, scorer, started, cursor)
            if trades:
                dn, db, dv = challenge.accumulate(
                    [(t["feature"], t["pnl_pct"]) for t in trades],
                    float(row["baseline_value"]), float(row["variant_value"]),
                    cap_pct, scorer.direction)
                new_cursor = max(t["exited_at"] for t in trades)
                await conn.execute(
                    "UPDATE param_variant_challenges SET "
                    " n_samples = n_samples + $2, "
                    " baseline_reward_sum = baseline_reward_sum + $3, "
                    " variant_reward_sum = variant_reward_sum + $4, "
                    " last_scored_at = $5 WHERE id=$1",
                    row["id"], dn, db, dv,
                    new_cursor.replace(tzinfo=timezone.utc)
                    if new_cursor.tzinfo is None else new_cursor,
                )
                n0, b0, v0 = n0 + dn, b0 + db, v0 + dv
                out["scored_trades"] += dn

            started_aware = started if started.tzinfo is not None \
                else started.replace(tzinfo=timezone.utc)
            elapsed = now >= started_aware + timedelta(
                hours=float(row["window_hours"]))
            verdict = challenge.resolve(
                n0, b0, v0, min_samples=min_samples,
                min_edge_pct=min_edge_pct, window_elapsed=elapsed)
            if verdict is not None:
                await _resolve(conn, row, verdict[0], verdict[1], n0, b0, v0)
                out["passed" if verdict[0] == challenge.VERDICT_PASSED
                    else "failed"] += 1
        except Exception as exc:
            logger.error("challenge %s fail-soft: %s", row["id"], exc)
    return out


# ────────────────────────────── self-test ───────────────────────────────────
#
# Offline end-to-end proof with an in-memory fake asyncpg connection (no DB,
# no network): a challenge OPENS on a proposal, ACCUMULATES over ticks as
# out-of-sample trades close, and RESOLVES pass/fail — plus the tuner router
# BYPASSING an exempt knob. Dispatches queries by distinctive substring; it
# is a test double, not a SQL engine.

class _FakeConn:
    def __init__(self):
        self.proposals: list = []
        self.challenges: list = []
        self.ai_trades: list = []   # dicts: feature, pnl_pct, entered, exited
        self._pid = 0
        self._cid = 0

    async def fetchval(self, sql, *a):
        s = " ".join(sql.split())
        if "INSERT INTO param_proposals" in s:
            self._pid += 1
            self.proposals.append({"id": self._pid, "status": a[10]})
            return self._pid
        if "INSERT INTO param_variant_challenges" in s:
            self._cid += 1
            self.challenges.append({
                "id": self._cid, "proposal_id": a[0], "config_type": a[1],
                "key": a[2], "module": a[3], "baseline_value": a[4],
                "variant_value": a[5], "window_hours": float(a[6]),
                "n_samples": 0, "baseline_reward_sum": 0.0,
                "variant_reward_sum": 0.0, "last_scored_at": None,
                "started_at": datetime.now(timezone.utc).replace(tzinfo=None),
                "resolved_at": None, "verdict": None})
            return self._cid
        if "COUNT(*) FROM param_variant_challenges" in s:  # failed_recently
            ctype, key, variant, verdict = a[0], a[1], a[2], a[3]
            return sum(1 for c in self.challenges
                       if c["config_type"] == ctype and c["key"] == key
                       and c["variant_value"] == variant
                       and c["verdict"] == verdict)
        if "SELECT status FROM param_proposals" in s:
            return next((p["status"] for p in self.proposals
                         if p["id"] == a[0]), None)
        raise AssertionError("unhandled fetchval: " + s[:60])

    async def fetch(self, sql, *a):
        s = " ".join(sql.split())
        if "FROM param_variant_challenges WHERE resolved_at IS NULL" in s:
            return [c for c in self.challenges if c["resolved_at"] is None]
        if "AS feature" in s:   # _fetch_slice
            started, cursor = a[0], a[1]
            rows = [{"feature": t["feature"], "pnl_pct": t["pnl_pct"],
                     "exited_at": t["exited"]}
                    for t in self.ai_trades
                    if t["entered"] > started and t["exited"] > cursor]
            return sorted(rows, key=lambda r: r["exited_at"])
        raise AssertionError("unhandled fetch: " + s[:60])

    async def execute(self, sql, *a):
        s = " ".join(sql.split())
        if "UPDATE param_variant_challenges SET resolved_at=NOW()" in s:
            c = next(c for c in self.challenges if c["id"] == a[0])
            c["resolved_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
            c["verdict"] = a[1]
            return
        if "UPDATE param_variant_challenges SET n_samples" in s:
            c = next(c for c in self.challenges if c["id"] == a[0])
            c["n_samples"] += a[1]
            c["baseline_reward_sum"] += a[2]
            c["variant_reward_sum"] += a[3]
            c["last_scored_at"] = a[4]
            return
        if "UPDATE param_proposals SET status" in s:
            p = next((p for p in self.proposals if p["id"] == a[0]), None)
            if p and p["status"] == "challenge":
                p["status"] = a[1]
            return
        raise AssertionError("unhandled execute: " + s[:60])


def _selftest() -> None:
    import asyncio

    entry = {"config_type": "ai_config", "key": "confidence_threshold",
             "module": "ai"}
    cfg = {"challenge_min_samples": 20, "challenge_min_edge_pct": 5,
           "challenge_pnl_cap_pct": 20}

    class _P:  # stand-in for bandit.Proposal (only the fields used)
        current_value, proposed_value, kind = 0.35, 0.50, "explore"

    class _P2:
        current_value, proposed_value, kind = 0.35, 0.40, "explore"

    async def run():
        base = datetime.now(timezone.utc).replace(tzinfo=None)

        # --- 1. Challenge OPENS on a proposal --------------------------------
        conn = _FakeConn()
        conn.proposals.append({"id": 7, "status": "challenge"})
        cid = await open_challenge(conn, 7, entry, _P(),
                                   {"challenge_window_hours": 72})
        assert cid == 1 and conn.challenges[0]["n_samples"] == 0
        conn.challenges[0]["started_at"] = base - timedelta(hours=100)  # elapsed

        # --- 2. ACCUMULATES + 3a. RESOLVES PASS: low-confidence trades lose;
        #        variant 0.50 skips them -> higher out-of-sample mean. Both
        #        values scored on identical trades, identical harness. -----
        for feature, pnl in [(0.40, -10.0), (0.42, -6.0), (0.55, 8.0),
                             (0.60, 12.0), (0.38, -12.0), (0.70, 4.0)] * 4:
            conn.ai_trades.append({
                "feature": feature, "pnl_pct": pnl,
                "entered": base - timedelta(hours=50),
                "exited": base - timedelta(hours=40)})
        out = await process_challenges(conn, cfg)
        assert conn.challenges[0]["n_samples"] == 24, conn.challenges[0]
        assert out["passed"] == 1 and out["failed"] == 0, out
        assert conn.challenges[0]["verdict"] == challenge.VERDICT_PASSED
        assert conn.proposals[0]["status"] == "pending"   # never auto-applied
        assert not await failed_recently(conn, "ai_config",
                                         "confidence_threshold", "0.5", 72)

        # --- 3b. RESOLVES FAIL: variant with no real edge + retry cooldown ---
        conn2 = _FakeConn()
        conn2.proposals.append({"id": 3, "status": "challenge"})
        await open_challenge(conn2, 3, entry, _P2(),
                             {"challenge_window_hours": 72})
        conn2.challenges[0]["started_at"] = base - timedelta(hours=100)
        for feature, pnl in [(0.50, 5.0), (0.55, 6.0)] * 12:  # pass both values
            conn2.ai_trades.append({
                "feature": feature, "pnl_pct": pnl,
                "entered": base - timedelta(hours=50),
                "exited": base - timedelta(hours=40)})
        out2 = await process_challenges(conn2, cfg)
        assert out2["failed"] == 1, out2
        assert conn2.challenges[0]["verdict"] == challenge.VERDICT_FAILED
        assert conn2.proposals[0]["status"] == "challenge_failed"
        assert await failed_recently(conn2, "ai_config",
                                     "confidence_threshold", "0.4", 72)

        # --- 4. Thin-tape at window end FAILS (evidence != pass) -------------
        conn3 = _FakeConn()
        conn3.proposals.append({"id": 9, "status": "challenge"})
        await open_challenge(conn3, 9, entry, _P(),
                             {"challenge_window_hours": 72})
        conn3.challenges[0]["started_at"] = base - timedelta(hours=100)
        conn3.ai_trades.append({
            "feature": 0.60, "pnl_pct": 9.0,
            "entered": base - timedelta(hours=50),
            "exited": base - timedelta(hours=40)})   # 1 obs < min_samples 20
        out3 = await process_challenges(conn3, cfg)
        assert out3["failed"] == 1 and \
            conn3.challenges[0]["verdict"] == challenge.VERDICT_FAILED

        # --- 5. Window NOT elapsed -> stays open, no verdict -----------------
        conn4 = _FakeConn()
        conn4.proposals.append({"id": 11, "status": "challenge"})
        await open_challenge(conn4, 11, entry, _P(),
                             {"challenge_window_hours": 72})
        await process_challenges(conn4, cfg)   # started_at = now (default)
        assert conn4.challenges[0]["resolved_at"] is None

        # --- 6. Exempt-knob routing: the tuner BYPASSES the challenge --------
        from modules.param_tuner.core import tuner_engine

        class _PExempt:
            current_value, proposed_value = 1.0, 1.5
            current_mean = proposed_mean = 0.6
            proposed_pulls, ucb_score, kind = 5, 1.2, "exploit"
            reason = "test"
        conn5 = _FakeConn()
        exempt_entry = {"config_type": "futures_risk", "key": "atr_tp_rr_ratio",
                        "module": "futures", "min": 0.8, "max": 2.0,
                        "value_type": "float"}
        summary = {"proposed": 0, "applied": 0}
        # auto_apply defaults false -> exempt knob queues pending, no apply.
        await tuner_engine._emit_proposal(conn5, exempt_entry, _PExempt(),
                                          {"challenge_enabled": True}, summary)
        assert summary["proposed"] == 1 and summary["applied"] == 0
        assert conn5.proposals[0]["status"] == "pending"  # bypassed challenge
        assert conn5.challenges == []                      # no challenge opened

        print("challenge_engine self-test OK")

    asyncio.run(run())


if __name__ == "__main__":
    _selftest()
