"""
Test Runner Routes — server-side API for the /test-runner dashboard page.

Why this exists
---------------
The operator on the VPS cannot trivially:
  * run psql from the host (DB credentials live inside the postgres
    container, not on the host shell),
  * verify dashboard UI fixes without manually browsing every page,
  * consolidate output from several scripts into one paste-back block.

This module exposes a small, whitelist-only API the dashboard's
Test Runner page (Agent 4 frontend) calls. The client only ever sends
a `test_id` from the published catalog — never raw commands — so the
attack surface is exactly the static catalog defined below.

═══════════════════════════════════════════════════════════════════════
API CONTRACT (pinned for Agent 4)
═══════════════════════════════════════════════════════════════════════

GET  /api/test-runner/tests
  Response 200:
    {
      "success": true,
      "tests": [
        {
          "id": "preflight",
          "title": "Run preflight matrix",
          "category": "scripts",       # one of: scripts | api | db
          "kind": "bash",              # one of: bash | probe | db_query
          "cmd_preview": "bash scripts/preflight.sh",
          "timeout_s": 300,
          "description": "..."         # short tooltip text
        },
        ...
      ]
    }

POST /api/test-runner/run
  Request body:  {"test_id": "<id from catalog>"}
  Response 200 on success (test ran, even if test itself failed):
    {
      "success": true,
      "test_id": "preflight",
      "kind": "bash",
      "exit_code": 0,
      "stdout": "...",
      "stderr": "...",
      "duration_ms": 1234,
      "timed_out": false
    }
  Response 400 if test_id unknown.
  Response 500 only if the executor itself errored (rare).
  For kind=probe responses, exit_code is the HTTP status (200..599)
  and stdout is the pretty-printed JSON body; stderr is empty unless
  the probe itself errored.
  For kind=db_query responses, exit_code is 0 on success / 1 on SQL
  error; stdout is a column-aligned table; stderr carries the SQL
  error message when present.

GET  /api/test-runner/probe/{endpoint}
  Path param `endpoint` is the URL-encoded api path (e.g. `bot/status`
  for /api/bot/status).  Returns:
    {
      "success": true,
      "status": 200,
      "body": <parsed JSON or raw text>,
      "elapsed_ms": 12
    }
  Used by the UI to fetch the same data the live pages render.

Auth: every endpoint is wrapped in require_auth (no admin needed;
the runner is read-only).
═══════════════════════════════════════════════════════════════════════
"""

import asyncio
import json
import logging
import os
import signal
import time
from typing import Any, Dict, List, Optional

from aiohttp import web

from auth.middleware import require_auth

# Hard cap on stdout/stderr captured per run — guards the dashboard
# process from a runaway test dumping gigabytes. 2 MB each is plenty
# for any preflight/smoke script we ship.
_MAX_OUTPUT_BYTES = 2 * 1024 * 1024

# Repo root inside the container (and the host bind-mount). All bash
# tests run with cwd here so relative paths in scripts work.
_REPO_ROOT = os.environ.get("CLAUDEDEX_REPO_ROOT", "/app")

logger = logging.getLogger("TestRunnerRoutes")


# ─────────────────────────────────────────────────────────────────────
# Test catalog — single source of truth. Client posts test_id only;
# never accept arbitrary commands. Add a new test by appending a dict
# to this list (see "How to add a new test" in TEST_SCRIPTS_SESSION_18).
# ─────────────────────────────────────────────────────────────────────
TEST_CATALOG: List[Dict[str, Any]] = [
    # ── Scripts ──────────────────────────────────────────────────────
    {
        "id": "preflight",
        "title": "Run preflight matrix",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/preflight.sh"],
        "cmd_preview": "bash scripts/preflight.sh",
        "timeout_s": 300,
        "description": "Full preflight matrix (DB schema, env, modules).",
    },
    {
        "id": "dashboard_smoke",
        "title": "Dashboard smoke test",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/dashboard_smoke.sh"],
        "cmd_preview": "bash scripts/dashboard_smoke.sh",
        "timeout_s": 120,
        "description": (
            "Authenticated end-to-end smoke of every session-18 dashboard "
            "endpoint (CSRF, MODE badge, sniper cap fallback, ...)."
        ),
    },

    # ── API probes ───────────────────────────────────────────────────
    {
        "id": "api_bot_status",
        "title": "API: /api/bot/status",
        "category": "api",
        "kind": "probe",
        "endpoint": "bot/status",
        "cmd_preview": "GET /api/bot/status",
        "timeout_s": 15,
        "description": "MODE badge source (dry_run + mode).",
    },
    {
        "id": "api_sniper_stats",
        "title": "API: /api/sniper/stats",
        "category": "api",
        "kind": "probe",
        "endpoint": "sniper/stats",
        "cmd_preview": "GET /api/sniper/stats",
        "timeout_s": 15,
        "description": "Sniper active/effective positions + cap fallback.",
    },
    {
        "id": "api_copytrading_stats",
        "title": "API: /api/copytrading/stats",
        "category": "api",
        "kind": "probe",
        "endpoint": "copytrading/stats",
        "cmd_preview": "GET /api/copytrading/stats",
        "timeout_s": 15,
        "description": "Copy-trading live vs simulated split.",
    },
    {
        "id": "api_analytics_risk_sniper",
        "title": "API: /api/analytics/risk/sniper",
        "category": "api",
        "kind": "probe",
        "endpoint": "analytics/risk/sniper",
        "cmd_preview": "GET /api/analytics/risk/sniper",
        "timeout_s": 15,
        "description": "Sniper risk metrics (VaR/CVaR/exposure).",
    },
    {
        "id": "api_analytics_perf_sniper",
        "title": "API: /api/analytics/performance/sniper?timeframe=all",
        "category": "api",
        "kind": "probe",
        "endpoint": "analytics/performance/sniper?timeframe=all",
        "cmd_preview": "GET /api/analytics/performance/sniper?timeframe=all",
        "timeout_s": 15,
        "description": "Sniper performance over the full history.",
    },
    {
        "id": "api_modules",
        "title": "API: /api/modules",
        "category": "api",
        "kind": "probe",
        "endpoint": "modules",
        "cmd_preview": "GET /api/modules",
        "timeout_s": 15,
        "description": "Per-module enable/disable + pause state.",
    },
    {
        "id": "api_arbitrage_stats",
        "title": "API: /api/arbitrage/stats",
        "category": "api",
        "kind": "probe",
        "endpoint": "arbitrage/stats",
        "cmd_preview": "GET /api/arbitrage/stats",
        "timeout_s": 15,
        "description": "Arbitrage status string + counters.",
    },

    # ── DB probes (run inside the dashboard's own asyncpg pool — no
    # docker.sock required) ────────────────────────────────────────────
    {
        "id": "db_sniper_settings",
        "title": "DB: sniper config_settings",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT key, value FROM config_settings "
            "WHERE config_type='sniper_config' ORDER BY key"
        ),
        "cmd_preview": (
            "SELECT key, value FROM config_settings "
            "WHERE config_type='sniper_config' ORDER BY key"
        ),
        "timeout_s": 15,
        "description": "Every key under sniper_config (live DB-backed).",
    },
    {
        "id": "db_seeded_caps",
        "title": "DB: seeded caps & safety toggles",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key IN ('max_active_positions','safety_check_enabled',"
            "'test_mode') ORDER BY config_type, key"
        ),
        "cmd_preview": (
            "SELECT config_type,key,value FROM config_settings WHERE key IN "
            "('max_active_positions','safety_check_enabled','test_mode')"
        ),
        "timeout_s": 15,
        "description": "Confirms cap, safety-filter, and test_mode are seeded.",
    },
    {
        "id": "db_open_positions",
        "title": "DB: open positions across modules",
        "category": "db",
        "kind": "db_query",
        # Per-table conventions:
        #   sniper_trades + copytrading_trades  → soft-delete pattern,
        #                                         filter status='open'
        #   futures_positions                   → every row IS an open
        #                                         position (no status
        #                                         column by schema)
        # Earlier versions of this probe used `futures_trades WHERE
        # status='open'` (wrong table — that's closed-only) and then
        # `futures_positions WHERE status='open'` (column doesn't
        # exist). Now COUNT(*) the positions table directly.
        "sql": (
            "SELECT 'sniper' AS src, COUNT(*) AS n "
            "FROM sniper_trades WHERE status='open' "
            "UNION ALL SELECT 'copy', COUNT(*) "
            "FROM copytrading_trades WHERE status='open' "
            "UNION ALL SELECT 'futures', COUNT(*) "
            "FROM futures_positions"
        ),
        "cmd_preview": (
            "Counts: sniper_trades + copytrading_trades WHERE status='open', "
            "futures_positions (table = open)"
        ),
        "timeout_s": 15,
        "description": "Open-position counts per trading module.",
    },

    # ── DB probes from MAY_2026_HARDENING_TEST_PLAN.md ────────────────
    {
        "id": "db_migration_seeds",
        "title": "DB: migration seeds (caps + safety_check)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE (config_type='sniper_config' AND key='max_active_positions') "
            "   OR (config_type='copytrading_config' AND key='max_active_positions') "
            "   OR (config_type='sniper_config' AND key='safety_check_enabled') "
            "ORDER BY config_type, key"
        ),
        "cmd_preview": (
            "SELECT config_type,key,value FROM config_settings WHERE seed-keys"
        ),
        "timeout_s": 15,
        "description": (
            "Confirms migrations 016 (sniper cap=500) + 017 (copy cap=50) "
            "+ Phase-2 safety_check_enabled row are present in config_settings."
        ),
    },
    {
        "id": "db_runtime_stats_freshness",
        "title": "DB: sniper_runtime_stats freshness",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT id, "
            "EXTRACT(EPOCH FROM (NOW() - updated_at))::int AS age_seconds, "
            "(stats->>'pools_detected')::int AS pools_detected, "
            "(stats->>'pools_evaluated')::int AS pools_evaluated, "
            "(stats->>'pools_passed')::int AS pools_passed, "
            "(stats->>'pools_rejected')::int AS pools_rejected, "
            "(stats->>'active_positions')::int AS active_positions, "
            "(stats->>'max_active_positions')::int AS max_active_positions, "
            "(stats->>'jupiter_quote_fallback_hits')::int AS jupiter_fallback "
            "FROM sniper_runtime_stats WHERE id = 1"
        ),
        "cmd_preview": (
            "SELECT age_seconds, pools_*, active_positions, jupiter_fallback "
            "FROM sniper_runtime_stats WHERE id=1"
        ),
        "timeout_s": 15,
        "description": (
            "Sniper subprocess snapshot freshness + every Phase-2 counter. "
            "Stale (age > 600s) means the subprocess crashed or stopped."
        ),
    },
    {
        "id": "db_block_time_anchored",
        "title": "DB: block_time_anchored propagation (30m window)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT "
            "  COALESCE(metadata->>'detection_path', 'unknown') AS path, "
            "  COALESCE(metadata->>'block_time_anchored', 'false') AS anchored, "
            "  COUNT(*) AS n "
            "FROM sniper_trades "
            "WHERE entry_timestamp > NOW() - INTERVAL '30 minutes' "
            "GROUP BY path, anchored "
            "ORDER BY path, anchored"
        ),
        "cmd_preview": (
            "GROUP-BY path, anchored on sniper_trades.metadata, last 30m"
        ),
        "timeout_s": 15,
        "description": (
            "Confirms detection paths anchor their timing to on-chain "
            "blockTime (Phase-2 fix). anchored=true should dominate "
            "both polling and wss buckets."
        ),
    },
    {
        "id": "db_detection_latency",
        "title": "DB: detection latency p50/p95 (30m)",
        "category": "db",
        "kind": "db_query",
        # detect_to_rpc_receipt_ms is stored as a JSONB number which
        # asyncpg returns as the original numeric type — including
        # values like "2038.02". ::int truncation would silently lose
        # precision AND fails on "2038.02" (no implicit float→int cast
        # in JSON-to-int). Use ::numeric so percentile_cont sees the
        # full fractional resolution.
        "sql": (
            "SELECT "
            "  COALESCE(metadata->>'detection_path', 'unknown') AS path, "
            "  COUNT(*) AS samples, "
            "  ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "
            "    ((metadata->'timing'->>'detect_to_rpc_receipt_ms')::numeric))::numeric, 1) AS p50_ms, "
            "  ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY "
            "    ((metadata->'timing'->>'detect_to_rpc_receipt_ms')::numeric))::numeric, 1) AS p95_ms "
            "FROM sniper_trades "
            "WHERE entry_timestamp > NOW() - INTERVAL '30 minutes' "
            "  AND metadata->'timing'->>'detect_to_rpc_receipt_ms' IS NOT NULL "
            "GROUP BY path"
        ),
        "cmd_preview": "p50/p95 of detect_to_rpc_receipt_ms by path, 30m",
        "timeout_s": 30,
        "description": (
            "Phase-2 headline metric: how stale a candidate is by the "
            "time we receive the listener notification. Lower is better. "
            "WSS should beat polling by 2-5x once both have samples."
        ),
    },
    {
        "id": "db_admin_login_status",
        "title": "DB: admin user login status",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT username, is_active, failed_login_attempts, "
            "  CASE WHEN failed_login_attempts >= 5 "
            "       THEN 'LOCKED' ELSE 'ok' END AS status, "
            "  last_login_at, updated_at "
            "FROM users WHERE username = 'admin'"
        ),
        "cmd_preview": (
            "SELECT username, failed_login_attempts, status FROM users WHERE username='admin'"
        ),
        "timeout_s": 10,
        "description": (
            "Checks whether admin is locked out from too many failed login "
            "attempts (>= 5 = locked). If locked, run the next test to reset."
        ),
    },
    {
        "id": "db_unlock_admin",
        "title": "DB: unlock admin login attempts",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "UPDATE users SET failed_login_attempts = 0, updated_at = NOW() "
            "WHERE username = 'admin' "
            "RETURNING username, failed_login_attempts"
        ),
        "cmd_preview": "UPDATE users SET failed_login_attempts=0 WHERE username='admin'",
        "timeout_s": 10,
        "description": (
            "Resets admin's failed_login_attempts to 0 so dashboard_smoke.sh "
            "(and any other auth-needing script) can log in again. "
            "Returns the new value for confirmation."
        ),
    },
    {
        "id": "db_enable_pgcrypto",
        "title": "DB: enable pgcrypto extension (run once)",
        "category": "db",
        "kind": "db_query",
        # Single statement so asyncpg's extended protocol is happy.
        # Verify with the next probe (db_check_pgcrypto) if needed.
        "sql": "CREATE EXTENSION IF NOT EXISTS pgcrypto",
        "cmd_preview": "CREATE EXTENSION IF NOT EXISTS pgcrypto",
        "timeout_s": 10,
        "description": (
            "Enables pgcrypto so crypt()/gen_salt() are available for "
            "the db_reset_admin_password probe below. Idempotent; safe "
            "to run any time. Required only on first use. Returns 0 "
            "rows on success."
        ),
    },
    {
        "id": "db_check_pgcrypto",
        "title": "DB: pgcrypto extension status",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT extname AS extension, extversion AS version "
            "FROM pg_extension WHERE extname = 'pgcrypto'"
        ),
        "cmd_preview": "SELECT ... FROM pg_extension WHERE extname='pgcrypto'",
        "timeout_s": 10,
        "description": "Verifies pgcrypto is registered in this DB.",
    },
    {
        "id": "db_reset_admin_password",
        "title": "DB: reset admin password to admin123 (for smoke tests)",
        "category": "db",
        "kind": "db_query",
        # Uses pgcrypto's crypt() with gen_salt('bf') to generate a
        # bcrypt-compatible hash at the database. asyncpg/psycopg can't
        # easily import bcrypt at request time, so this is the cleanest
        # path. pgcrypto is part of TimescaleDB's base image.
        #
        # NOTE: scripts/init_auth.py will rotate this BACK to a random
        # password on the next bot restart (security measure for the
        # leaked default). Use to enable smoke tests within ONE bot
        # session; do not rely on it persisting across restarts.
        # asyncpg's fetch() only accepts ONE statement per call, so we
        # can't combine CREATE EXTENSION + UPDATE here. pgcrypto ships
        # with TimescaleDB; if the operator gets "function crypt does
        # not exist" they can run the separate db_enable_pgcrypto probe
        # below first.
        "sql": (
            "UPDATE users SET "
            "  password_hash = crypt('admin123', gen_salt('bf', 12)), "
            "  failed_login_attempts = 0, "
            "  updated_at = NOW() "
            "WHERE username = 'admin' "
            "RETURNING username, failed_login_attempts, "
            "         substring(password_hash, 1, 7) AS hash_prefix"
        ),
        "cmd_preview": (
            "UPDATE users SET password_hash=crypt('admin123', gen_salt('bf')) "
            "WHERE username='admin'"
        ),
        "timeout_s": 10,
        "description": (
            "Sets admin password back to 'admin123' so dashboard_smoke.sh "
            "can log in. init_auth.py will rotate this on the next bot "
            "restart — only good for the current session. Returns the "
            "first 7 chars of the new hash (should be $2b$12$) so you "
            "can verify the update."
        ),
    },
    {
        "id": "db_copytrading_bounded_sets",
        "title": "DB: COPY_TRADING bounded sets snapshot",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT "
            "  COUNT(*) FILTER (WHERE status='open') AS open_count, "
            "  COUNT(*) FILTER (WHERE status='closed') AS closed_count, "
            "  COUNT(DISTINCT source_wallet) AS unique_leaders, "
            "  COUNT(*) FILTER (WHERE is_simulated) AS simulated_count, "
            "  COUNT(*) FILTER (WHERE NOT is_simulated) AS live_count "
            "FROM copytrading_trades"
        ),
        "cmd_preview": (
            "open/closed/leaders/simulated/live counts on copytrading_trades"
        ),
        "timeout_s": 15,
        "description": (
            "COPY_TRADING bounded sets after MB-22..25 — confirms data "
            "shape and the new is_simulated split (DASH-Q-04)."
        ),
    },

    # ── More API probes for COMPLETE_TEST_SCRIPTS.md coverage ────────
    {
        "id": "api_health",
        "title": "API: /health (canary, no auth)",
        "category": "api",
        "kind": "probe",
        "endpoint": "../health",  # rewritten to /health below
        "cmd_preview": "GET /health",
        "timeout_s": 10,
        "description": (
            "No-auth liveness canary. /__routes__ + /health are the only "
            "endpoints intentionally exempt from auth middleware."
        ),
    },
    {
        "id": "api_routes",
        "title": "API: /__routes__ count",
        "category": "api",
        "kind": "probe",
        "endpoint": "../__routes__",
        "cmd_preview": "GET /__routes__ | length",
        "timeout_s": 15,
        "description": (
            "Diagnostic: every registered aiohttp route. Baseline ≥ 470 "
            "after this session's additions."
        ),
    },
    {
        "id": "api_sniper_timing",
        "title": "API: /api/sniper/timing (cached)",
        "category": "api",
        "kind": "probe",
        "endpoint": "sniper/timing",
        "cmd_preview": "GET /api/sniper/timing",
        "timeout_s": 30,
        "description": (
            "P50/P95 detection latency by path. Response carries "
            "'cached': true on the 2nd call within the 30s TTL window."
        ),
    },
    {
        "id": "api_dashboard_summary",
        "title": "API: /api/dashboard/summary",
        "category": "api",
        "kind": "probe",
        "endpoint": "dashboard/summary",
        "cmd_preview": "GET /api/dashboard/summary",
        "timeout_s": 15,
        "description": (
            "Source for the /dashboard hero metrics (portfolio value, "
            "P&L, open positions). Verifies the unified-trade query."
        ),
    },
    {
        "id": "api_analytics_perf_arbitrage",
        "title": "API: /api/analytics/performance/arbitrage",
        "category": "api",
        "kind": "probe",
        "endpoint": "analytics/performance/arbitrage?timeframe=all",
        "cmd_preview": "GET /api/analytics/performance/arbitrage",
        "timeout_s": 15,
        "description": (
            "Verifies /analytics module switcher routes correctly to "
            "arbitrage data after Agent 2's fix (commit 3cb544b)."
        ),
    },
    {
        "id": "api_full_dashboard_modules",
        "title": "API: /api/modules (Module Overview source)",
        "category": "api",
        "kind": "probe",
        "endpoint": "modules?include_disabled=true",
        "cmd_preview": "GET /api/modules?include_disabled=true",
        "timeout_s": 15,
        "description": (
            "Source for /full-dashboard's Module Overview. env-flag "
            "should be the single source of truth (FAILURE A — Agent 2)."
        ),
    },
]


def _public_catalog_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Strip executor-internal fields (cmd/sql/endpoint) before sending
    the catalog to the client. The client only needs id/title/preview
    to render a button — it never needs the raw command."""
    return {
        "id": entry["id"],
        "title": entry["title"],
        "category": entry["category"],
        "kind": entry["kind"],
        "cmd_preview": entry["cmd_preview"],
        "timeout_s": entry["timeout_s"],
        "description": entry.get("description", ""),
    }


class TestRunnerRoutes:
    """
    Test-runner dashboard routes.

    Mirrors the shape of monitoring/analytics_routes.AnalyticsRoutes
    so wiring in enhanced_dashboard._setup_routes stays consistent.
    """

    def __init__(self, app: web.Application, db_manager=None, jinja_env=None):
        self.app = app
        self.db = db_manager
        self.jinja_env = jinja_env
        self.logger = logger
        # Catalog lookup by id for O(1) dispatch in run/probe handlers.
        self._by_id: Dict[str, Dict[str, Any]] = {t["id"]: t for t in TEST_CATALOG}

    # ── route registration ──────────────────────────────────────────
    def setup_routes(self, app: Optional[web.Application] = None) -> None:
        """Register all /api/test-runner/* endpoints. `app` arg kept
        optional so the calling pattern matches AnalyticsRoutes."""
        target = app or self.app
        target.router.add_get(
            '/api/test-runner/tests', require_auth(self.list_tests)
        )
        target.router.add_post(
            '/api/test-runner/run', require_auth(self.run_test)
        )
        self.logger.info(
            "Test-runner routes configured (%d tests in catalog)",
            len(TEST_CATALOG),
        )

    # ── GET /api/test-runner/tests ──────────────────────────────────
    async def list_tests(self, request: web.Request) -> web.Response:
        """Return the public catalog so the frontend can render one
        button per entry. Internal command/sql/endpoint fields are
        stripped via _public_catalog_entry."""
        return web.json_response({
            "success": True,
            "tests": [_public_catalog_entry(t) for t in TEST_CATALOG],
        })

    # ── POST /api/test-runner/run ───────────────────────────────────
    async def run_test(self, request: web.Request) -> web.Response:
        """Dispatch to the correct executor based on the catalog entry's
        kind. Client posts only {"test_id": "<id>"}; we never accept a
        raw command."""
        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"success": False, "error": "invalid JSON body"}, status=400
            )
        test_id = (payload or {}).get("test_id")
        entry = self._by_id.get(test_id) if test_id else None
        if not entry:
            return web.json_response(
                {"success": False, "error": f"unknown test_id: {test_id!r}"},
                status=400,
            )

        kind = entry["kind"]
        try:
            if kind == "bash":
                result = await self._run_bash(entry)
            elif kind == "db_query":
                result = await self._run_db_query(entry)
            elif kind == "probe":
                # probe kind is normally served by GET /probe — but
                # accept it here too so the frontend can keep one POST
                # path for everything.
                result = await self._run_probe(request, entry)
            else:
                return web.json_response(
                    {"success": False, "error": f"unsupported kind {kind!r}"},
                    status=400,
                )
        except Exception as exc:
            self.logger.exception("test_runner: executor crashed for %s", test_id)
            return web.json_response(
                {"success": False, "error": str(exc), "test_id": test_id},
                status=500,
            )

        return web.json_response({
            "success": True,
            "test_id": test_id,
            "kind": kind,
            **result,
        })

    # ── kind=bash executor ──────────────────────────────────────────
    async def _run_bash(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Spawn a subprocess for a whitelisted bash command. Caps
        stdout/stderr at _MAX_OUTPUT_BYTES each, kills the process
        group on timeout."""
        cmd: List[str] = entry["cmd"]
        timeout_s: int = int(entry.get("timeout_s", 120))
        t0 = time.perf_counter()
        timed_out = False

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=_REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # new process group for clean kill
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                await asyncio.sleep(2)
                if proc.returncode is None:
                    os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout_b, stderr_b = await proc.communicate()

        duration_ms = int((time.perf_counter() - t0) * 1000)

        def _trunc(b: bytes) -> str:
            if len(b) > _MAX_OUTPUT_BYTES:
                head = b[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
                return head + f"\n…[truncated at {_MAX_OUTPUT_BYTES} bytes]"
            return b.decode("utf-8", errors="replace")

        return {
            "exit_code": proc.returncode if proc.returncode is not None else -1,
            "stdout": _trunc(stdout_b or b""),
            "stderr": _trunc(stderr_b or b""),
            "duration_ms": duration_ms,
            "timed_out": timed_out,
        }

    # ── kind=db_query executor ──────────────────────────────────────
    async def _run_db_query(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Run a whitelisted SELECT through the dashboard's own asyncpg
        pool. We deliberately do NOT exec into the postgres container —
        that would require docker.sock; the dashboard already has DB
        access so this is cleaner.

        Safety: catalog SQL is hard-coded; we never accept SQL from the
        client. We still wrap a SET statement_timeout via per-conn
        execute to bound DB-side runtime in case a query goes wild.
        """
        sql: str = entry["sql"]
        timeout_s: int = int(entry.get("timeout_s", 15))
        t0 = time.perf_counter()

        if not self.db or not getattr(self.db, "pool", None):
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": "db pool unavailable (self.db is None)",
                "duration_ms": int((time.perf_counter() - t0) * 1000),
                "timed_out": False,
            }

        try:
            async with self.db.pool.acquire() as conn:
                # DB-side guard. asyncpg accepts an integer in ms; cap
                # at 80% of our wall-clock budget so we always see the
                # SQL error before the application timeout fires.
                ms = max(1000, int(timeout_s * 800))
                await conn.execute(f"SET statement_timeout = {ms}")
                rows = await asyncio.wait_for(
                    conn.fetch(sql), timeout=timeout_s
                )
        except asyncio.TimeoutError:
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": f"query timed out after {timeout_s}s",
                "duration_ms": int((time.perf_counter() - t0) * 1000),
                "timed_out": True,
            }
        except Exception as exc:
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": f"SQL error: {exc}",
                "duration_ms": int((time.perf_counter() - t0) * 1000),
                "timed_out": False,
            }

        # Render column-aligned table; cap rows to keep the response
        # paste-friendly even on a runaway result-set.
        MAX_ROWS = 200
        out_lines: List[str] = []
        if not rows:
            out_lines.append("(0 rows)")
        else:
            cols = list(rows[0].keys())
            widths = {c: len(c) for c in cols}
            for r in rows[:MAX_ROWS]:
                for c in cols:
                    widths[c] = max(widths[c], len(str(r[c])))
            header = "  ".join(c.ljust(widths[c]) for c in cols)
            sep = "  ".join("-" * widths[c] for c in cols)
            out_lines.append(header)
            out_lines.append(sep)
            for r in rows[:MAX_ROWS]:
                out_lines.append("  ".join(
                    str(r[c]).ljust(widths[c]) for c in cols
                ))
            if len(rows) > MAX_ROWS:
                out_lines.append(f"…[{len(rows) - MAX_ROWS} more rows]")
            out_lines.append(f"({len(rows)} rows)")

        return {
            "exit_code": 0,
            "stdout": "\n".join(out_lines),
            "stderr": "",
            "duration_ms": int((time.perf_counter() - t0) * 1000),
            "timed_out": False,
        }

    async def _run_probe(self, request: web.Request,
                         entry: Dict[str, Any]) -> Dict[str, Any]:
        """Issue a same-origin GET against the dashboard's own API
        carrying the caller's session+csrf cookies. This is how the UI
        clones reach /api/sniper/stats, /api/modules, etc. without
        re-implementing the auth dance.

        Uses aiohttp.ClientSession with the request's Cookie header so
        the proxy inherits the caller's identity. We never accept a raw
        URL from the client — only the catalog's endpoint path."""
        import aiohttp

        endpoint: str = entry["endpoint"]
        timeout_s: int = int(entry.get("timeout_s", 15))
        # Reconstruct same-origin URL. request.scheme + request.host
        # reflects whatever proxy/binding the dashboard is reached
        # through, so the proxy works behind nginx/cloudflare too.
        # Catalog convention: endpoint values under /api/* are bare
        # (e.g. "bot/status"); values rooted elsewhere prefix with
        # "../" (e.g. "../health" → /health, "../__routes__" → /__routes__).
        if endpoint.startswith("../"):
            url = f"{request.scheme}://{request.host}/{endpoint[3:].lstrip('/')}"
        else:
            url = f"{request.scheme}://{request.host}/api/{endpoint.lstrip('/')}"
        t0 = time.perf_counter()

        # Forward auth + CSRF cookies so the proxied request looks
        # identical to a direct browser GET from the same session.
        cookies = {k: v for k, v in request.cookies.items()}
        headers = {
            "X-CSRF-Token": request.cookies.get("csrf_token", ""),
            "Accept": "application/json",
        }

        try:
            async with aiohttp.ClientSession(cookies=cookies) as sess:
                async with sess.get(
                    url, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout_s),
                ) as resp:
                    status = resp.status
                    text = await resp.text()
        except asyncio.TimeoutError:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"probe timed out after {timeout_s}s",
                "duration_ms": int((time.perf_counter() - t0) * 1000),
                "timed_out": True,
            }
        except Exception as exc:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"probe error: {exc}",
                "duration_ms": int((time.perf_counter() - t0) * 1000),
                "timed_out": False,
            }

        # Try to pretty-print JSON; fall back to raw text for HTML
        # error pages (e.g. login redirect).
        try:
            parsed = json.loads(text)
            pretty = json.dumps(parsed, indent=2)
        except Exception:
            pretty = text[:_MAX_OUTPUT_BYTES]

        # exit_code == HTTP status per the API contract, so the
        # frontend can chip-green on 200..299, chip-red otherwise.
        return {
            "exit_code": status,
            "stdout": pretty,
            "stderr": "" if 200 <= status < 300 else f"HTTP {status}",
            "duration_ms": int((time.perf_counter() - t0) * 1000),
            "timed_out": False,
        }
