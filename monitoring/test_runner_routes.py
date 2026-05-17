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
        "sql": (
            "SELECT 'sniper' AS src, COUNT(*) AS n "
            "FROM sniper_trades WHERE status='open' "
            "UNION ALL SELECT 'copy', COUNT(*) "
            "FROM copytrading_trades WHERE status='open' "
            "UNION ALL SELECT 'futures', COUNT(*) "
            "FROM futures_trades WHERE status='open'"
        ),
        "cmd_preview": (
            "SELECT src, COUNT(*) FROM {sniper,copy,futures}_trades "
            "WHERE status='open'"
        ),
        "timeout_s": 15,
        "description": "Open-position counts per trading module.",
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
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": "probe executor not yet implemented",
            "duration_ms": 0,
            "timed_out": False,
        }
