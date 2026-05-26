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
          "description": "...",        # short tooltip text
          "tags": ["must"]             # optional; allowed values:
                                        #   must, new, p0, flaky, expected-empty
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
    {
        "id": "settings_save_smoke",
        "title": "Per-module settings save CSRF smoke",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/settings_save_smoke.sh"],
        "cmd_preview": "bash scripts/settings_save_smoke.sh",
        "timeout_s": 120,
        "description": (
            "Logs in as admin, POSTs a benign payload to each "
            "/api/<module>/settings endpoint with X-CSRF-Token, then "
            "GETs each /api/modules/<m>/dry-run. 403 anywhere = the "
            "'CSRF token missing or invalid' regression is back."
        ),
    },
    {
        "id": "orchestrator_train_report",
        "title": "Orchestrator: train ML model (report-only) [expected-empty]",
        "category": "scripts",
        "kind": "bash",
        "cmd": [
            "python", "-m", "modules.orchestrator_ai.core.ml_trainer",
            "--report-only",
        ],
        "cmd_preview": "python -m modules.orchestrator_ai.core.ml_trainer --report-only",
        "timeout_s": 60,
        "description": (
            "Trains the orchestrator's confidence-calibration model from "
            "orchestrator_training_data WITHOUT saving the pkl. Shows the "
            "operator how many labeled examples are accumulated and the "
            "current logistic-regression weights + accuracy. Exits 1 if "
            "fewer than 30 examples exist."
        ),
    },
    {
        "id": "orchestrator_train_save",
        "title": "Orchestrator: train + save ML model [expected-empty]",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["python", "-m", "modules.orchestrator_ai.core.ml_trainer"],
        "cmd_preview": "python -m modules.orchestrator_ai.core.ml_trainer",
        "timeout_s": 60,
        "description": (
            "Same as the report-only run plus writes data/orchestrator_ai_"
            "model.pkl + JSON sidecar. Run this after operator-approval "
            "history accumulates. The orchestrator can optionally load "
            "the pkl on next restart to override its hard-coded weights."
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
            "UNION ALL SELECT 'arbitrage', COUNT(*) "
            "FROM arbitrage_trades WHERE status='open' "
            "UNION ALL SELECT 'copy', COUNT(*) "
            "FROM copytrading_trades WHERE status='open' "
            "UNION ALL SELECT 'futures', COUNT(*) "
            "FROM futures_positions "
            "UNION ALL SELECT 'solana', COUNT(*) "
            "FROM solana_positions "
            "UNION ALL SELECT 'dex', COUNT(*) "
            "FROM trades WHERE status='open' "
            "UNION ALL SELECT 'ai', COUNT(*) "
            "FROM ai_trades WHERE status='open'"
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
            "  last_login, updated_at "
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
    # api_full_dashboard_modules removed — identical response to
    # api_modules above (just adds include_disabled which is a no-op for
    # the operator since every module is currently enabled). Was emitting
    # ~250 lines per Run-All click for no extra signal.

    # ── Phase 3: per-module DRY_RUN coverage ──────────────────────────
    # One DB probe + one API probe per module so the operator can see
    # at a glance whether each module is paper-trading, what its DB-row
    # value is, and whether the engine has recorded trades recently.
    {
        "id": "db_per_module_dry_run_flags",
        "title": "DB: per-module dry_run rows",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, value FROM config_settings "
            "WHERE key = 'dry_run' "
            "ORDER BY config_type"
        ),
        "cmd_preview": "SELECT config_type, value FROM config_settings WHERE key='dry_run'",
        "timeout_s": 10,
        "description": (
            "Shows every per-module dry_run override saved in DB. "
            "Missing rows fall back to the env / global / default chain."
        ),
    },
    {
        "id": "db_trades_per_module_24h",
        "title": "DB: trades per module (last 24h)",
        "category": "db",
        "kind": "db_query",
        # UNION across all five trade tables. The shape varies per
        # table (futures_trades has no status column), so we COUNT
        # rows by a column each table actually has (entry_timestamp
        # for sniper/copy/arb; opened_at for futures).
        "sql": (
            # Per-table time columns are NOT uniform across modules:
            #   sniper / arbitrage / copy_trading / ai / dex → entry_timestamp
            #   futures / solana                             → entry_time
            "SELECT 'sniper' AS module, COUNT(*) AS trades_24h "
            "FROM sniper_trades WHERE entry_timestamp > NOW() - INTERVAL '24 hours' "
            "UNION ALL SELECT 'arbitrage', COUNT(*) "
            "FROM arbitrage_trades WHERE entry_timestamp > NOW() - INTERVAL '24 hours' "
            "UNION ALL SELECT 'copy_trading', COUNT(*) "
            "FROM copytrading_trades WHERE entry_timestamp > NOW() - INTERVAL '24 hours' "
            "UNION ALL SELECT 'futures', COUNT(*) "
            "FROM futures_trades WHERE entry_time > NOW() - INTERVAL '24 hours' "
            "UNION ALL SELECT 'solana', COUNT(*) "
            "FROM solana_trades WHERE entry_time > NOW() - INTERVAL '24 hours' "
            "UNION ALL SELECT 'dex', COUNT(*) "
            "FROM trades WHERE entry_timestamp > NOW() - INTERVAL '24 hours' "
            "UNION ALL SELECT 'ai', COUNT(*) "
            "FROM ai_trades WHERE entry_timestamp > NOW() - INTERVAL '24 hours' "
            "ORDER BY 1"
        ),
        "cmd_preview": "COUNT(*) per *_trades table, last 24h",
        "timeout_s": 30,
        "description": (
            "Quick sanity that each enabled module is actually writing "
            "trade rows. If a module is enabled but its row is 0, the "
            "engine is alive but not capturing data — investigate. "
            "Modules that are intentionally disabled return 0 (fine)."
        ),
    },
    {
        "id": "db_open_positions_per_module",
        "title": "DB: open positions per module",
        "category": "db",
        "kind": "db_query",
        "sql": (
            # Per-table conventions:
            #   sniper/arbitrage/copy/dex/ai → status='open' filter
            #   futures                      → futures_positions table (every row open)
            #   solana                       → solana_positions table (no status col)
            "SELECT 'sniper' AS module, COUNT(*) AS open_positions "
            "FROM sniper_trades WHERE status='open' "
            "UNION ALL SELECT 'arbitrage', COUNT(*) "
            "FROM arbitrage_trades WHERE status='open' "
            "UNION ALL SELECT 'copy_trading', COUNT(*) "
            "FROM copytrading_trades WHERE status='open' "
            "UNION ALL SELECT 'futures', COUNT(*) "
            "FROM futures_positions "
            "UNION ALL SELECT 'solana', COUNT(*) "
            "FROM solana_positions "
            "UNION ALL SELECT 'dex', COUNT(*) "
            "FROM trades WHERE status='open' "
            "UNION ALL SELECT 'ai', COUNT(*) "
            "FROM ai_trades WHERE status='open' "
            "ORDER BY 1"
        ),
        "cmd_preview": "COUNT(*) open positions per module (status / position tables)",
        "timeout_s": 15,
        "description": "How many positions each module currently holds open.",
    },
    {
        "id": "db_pnl_simulated_vs_live_per_module",
        "title": "DB: P&L breakdown — simulated vs live, per module",
        "category": "db",
        "kind": "db_query",
        # sniper_trades has no is_simulated column; assume sniper is
        # always simulated until proven otherwise (the operator can
        # query sniper directly to distinguish if needed).
        "sql": (
            # Per-table pnl-column conventions (the user has hit this
            # twice now): arb + copy = profit_loss, futures = net_pnl,
            # solana = pnl_usd. Sniper has no is_simulated col so it
            # cannot meaningfully split — handled elsewhere.
            "SELECT 'arbitrage' AS module, "
            "  COALESCE(SUM(profit_loss) FILTER (WHERE is_simulated), 0)::numeric(20,4) AS simulated_pnl, "
            "  COALESCE(SUM(profit_loss) FILTER (WHERE NOT is_simulated), 0)::numeric(20,4) AS live_pnl, "
            "  COUNT(*) FILTER (WHERE is_simulated) AS sim_trades, "
            "  COUNT(*) FILTER (WHERE NOT is_simulated) AS live_trades "
            "FROM arbitrage_trades "
            "UNION ALL "
            "SELECT 'copy_trading', "
            "  COALESCE(SUM(profit_loss) FILTER (WHERE is_simulated), 0)::numeric(20,4), "
            "  COALESCE(SUM(profit_loss) FILTER (WHERE NOT is_simulated), 0)::numeric(20,4), "
            "  COUNT(*) FILTER (WHERE is_simulated), "
            "  COUNT(*) FILTER (WHERE NOT is_simulated) "
            "FROM copytrading_trades "
            "UNION ALL "
            "SELECT 'futures', "
            "  COALESCE(SUM(net_pnl) FILTER (WHERE is_simulated), 0)::numeric(20,4), "
            "  COALESCE(SUM(net_pnl) FILTER (WHERE NOT is_simulated), 0)::numeric(20,4), "
            "  COUNT(*) FILTER (WHERE is_simulated), "
            "  COUNT(*) FILTER (WHERE NOT is_simulated) "
            "FROM futures_trades "
            "UNION ALL "
            "SELECT 'solana', "
            "  COALESCE(SUM(pnl_usd) FILTER (WHERE is_simulated), 0)::numeric(20,4), "
            "  COALESCE(SUM(pnl_usd) FILTER (WHERE NOT is_simulated), 0)::numeric(20,4), "
            "  COUNT(*) FILTER (WHERE is_simulated), "
            "  COUNT(*) FILTER (WHERE NOT is_simulated) "
            "FROM solana_trades "
            "ORDER BY 1"
        ),
        "cmd_preview": "SUM(pnl) split by is_simulated, per module",
        "timeout_s": 30,
        "description": (
            "Critical for live-readiness: how much real money has each "
            "module made/lost vs paper. Pre-live the live_pnl column "
            "MUST be 0 for every module. After flipping one module live, "
            "operator watches this row to see real fills land."
        ),
    },
    # api_module_dry_run_overview removed — third copy of /api/modules.
    # The effective_dry_run field is already visible in the api_modules
    # probe response (every module row carries it). Was 3× duplication.

    # ── Phase 3 D: orchestrator_ai readiness probes ─────────────────
    {
        "id": "db_orch_table_exists",
        "title": "DB: orchestrator_recommendations table present",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT table_name, "
            "  (SELECT COUNT(*) FROM information_schema.columns "
            "   WHERE table_name='orchestrator_recommendations') AS column_count "
            "FROM information_schema.tables "
            "WHERE table_name = 'orchestrator_recommendations'"
        ),
        "cmd_preview": (
            "SELECT FROM information_schema.tables WHERE table_name='orchestrator_recommendations'"
        ),
        "timeout_s": 10,
        "description": (
            "Confirms migration 018 has run and the orchestrator can "
            "write recommendations. 0 rows = migration pending."
        ),
    },
    {
        "id": "db_orch_recs_summary",
        "title": "DB: orchestrator recommendations summary",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT module, "
            "  COUNT(*) FILTER (WHERE approved IS NULL AND superseded_at IS NULL) AS pending, "
            "  COUNT(*) FILTER (WHERE approved IS TRUE) AS approved, "
            "  COUNT(*) FILTER (WHERE approved IS FALSE) AS rejected, "
            "  COUNT(*) FILTER (WHERE superseded_at IS NOT NULL) AS superseded, "
            "  MAX(created_at) AS most_recent "
            "FROM orchestrator_recommendations "
            "GROUP BY module ORDER BY most_recent DESC NULLS LAST"
        ),
        "cmd_preview": "per-module rec counts grouped by approval state",
        "timeout_s": 10,
        "description": (
            "How many recs each module has, by state. Pre-orchestrator-start "
            "this returns 0 rows; after first tick you'll see rows here."
        ),
    },
    {
        "id": "api_orch_pending_recs",
        "title": "API: /api/orchestrator/recommendations?status=pending",
        "category": "api",
        "kind": "probe",
        "endpoint": "orchestrator/recommendations?status=pending&limit=20",
        "cmd_preview": "GET /api/orchestrator/recommendations?status=pending",
        "timeout_s": 15,
        "description": (
            "Lists currently pending operator approvals. Empty = nothing "
            "to action (either no module crossed a threshold, or the "
            "orchestrator subprocess isn't running yet)."
        ),
    },
    {
        "id": "api_orch_history",
        "title": "API: /api/orchestrator/history?hours=3",
        "category": "api",
        "kind": "probe",
        # Default window dropped 72h → 3h. With 5-min tick cadence and
        # 7 modules, 72h emits ~6000 rows / ~10K log lines per probe run.
        # 3h is enough to confirm the orchestrator is alive and emitting,
        # and keeps the test_runner log size sane.
        "endpoint": "orchestrator/history?hours=3",
        "cmd_preview": "GET /api/orchestrator/history?hours=3",
        "timeout_s": 15,
        "description": (
            "Per-module score timeseries for the last 72h. Grouped by "
            "module. Each point: ts, recommended, score, components, "
            "total_pnl_usd, closed_trades. Powers the /orchestrator "
            "trend chart."
        ),
    },
    # ── Phase 4C: circuit breaker ────────────────────────────────────
    {
        "id": "db_breaker_table_exists",
        "title": "DB: circuit_breaker_events table present",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT table_name, "
            "  (SELECT COUNT(*) FROM information_schema.columns "
            "   WHERE table_name='circuit_breaker_events') AS column_count "
            "FROM information_schema.tables "
            "WHERE table_name = 'circuit_breaker_events'"
        ),
        "cmd_preview": "SELECT FROM information_schema.tables WHERE table_name='circuit_breaker_events'",
        "timeout_s": 10,
        "description": "Confirms migration 021 has run.",
    },
    {
        "id": "db_breaker_thresholds",
        "title": "DB: per-module daily-loss thresholds",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, value AS threshold_pct "
            "FROM config_settings "
            "WHERE key = 'daily_loss_breaker_pct' "
            "ORDER BY config_type"
        ),
        "cmd_preview": "SELECT … WHERE key='daily_loss_breaker_pct'",
        "timeout_s": 10,
        "description": "Each module's threshold; defaults to 5.0%.",
    },
    {
        "id": "db_breaker_active_events",
        "title": "DB: active circuit-breaker trips",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT module, tripped_at, pct_loss, threshold_pct, "
            "  action_taken "
            "FROM circuit_breaker_events "
            "WHERE cleared_at IS NULL "
            "  AND tripped_at > NOW() - INTERVAL '24 hours' "
            "ORDER BY tripped_at DESC"
        ),
        "cmd_preview": "Active (uncleared) trips in last 24h",
        "timeout_s": 10,
        "description": (
            "Source for the dashboard's circuit-breaker banner. Empty = "
            "no current trips, which is the normal state."
        ),
    },
    {
        "id": "api_breaker_active",
        "title": "API: /api/circuit-breaker/active",
        "category": "api",
        "kind": "probe",
        "endpoint": "circuit-breaker/active",
        "cmd_preview": "GET /api/circuit-breaker/active",
        "timeout_s": 10,
        "description": "Active trips JSON — what the banner polls.",
    },

    # ── Phase 4B: portfolio allocator ────────────────────────────────
    {
        "id": "db_alloc_table_exists",
        "title": "DB: portfolio_allocations table present",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT table_name, "
            "  (SELECT COUNT(*) FROM information_schema.columns "
            "   WHERE table_name='portfolio_allocations') AS column_count "
            "FROM information_schema.tables "
            "WHERE table_name = 'portfolio_allocations'"
        ),
        "cmd_preview": "SELECT FROM information_schema.tables WHERE table_name='portfolio_allocations'",
        "timeout_s": 10,
        "description": "Confirms migration 020 has run.",
    },
    {
        "id": "db_alloc_current",
        "title": "DB: current approved allocation per module",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT DISTINCT ON (module) module, pct_of_book, usd_amount, "
            "  approved_at, approved_by "
            "FROM portfolio_allocations WHERE approved_at IS NOT NULL "
            "ORDER BY module, approved_at DESC"
        ),
        "cmd_preview": "Most recent approved allocation per module",
        "timeout_s": 10,
        "description": (
            "Shows the operative allocation per module. Sum + reserve "
            "should equal 100%. Empty = no operator approvals yet."
        ),
    },
    {
        "id": "api_alloc_current",
        "title": "API: /api/portfolio/allocations/current",
        "category": "api",
        "kind": "probe",
        "endpoint": "portfolio/allocations/current",
        "cmd_preview": "GET /api/portfolio/allocations/current",
        "timeout_s": 15,
        "description": "Current approved per-module allocation, JSON shape.",
    },
    {
        "id": "api_alloc_pending",
        "title": "API: /api/portfolio/allocations?status=pending",
        "category": "api",
        "kind": "probe",
        "endpoint": "portfolio/allocations?status=pending&limit=20",
        "cmd_preview": "GET /api/portfolio/allocations?status=pending",
        "timeout_s": 15,
        "description": "Pending allocator proposals awaiting operator approval.",
    },

    # ── Phase 4A: backtest replay ────────────────────────────────────
    {
        "id": "api_backtest_strategies",
        "title": "API: /api/backtest/strategies",
        "category": "api",
        "kind": "probe",
        "endpoint": "backtest/strategies",
        "cmd_preview": "GET /api/backtest/strategies",
        "timeout_s": 10,
        "description": (
            "Lists the replay strategies the engine knows about. "
            "Must include approve_all, approve_on_confidence, "
            "never_approve, operator_replay."
        ),
    },
    # ── Per-module settings page round-trip probes ────────────────────
    # Verifies that each /<module>/settings page can read its config
    # via GET. POST/save verification stays in the bash smoke layer
    # because the catalog has no POST kind.
    {
        "id": "api_settings_arbitrage_get",
        "title": "API: GET /api/arbitrage/settings",
        "category": "api",
        "kind": "probe",
        "endpoint": "arbitrage/settings",
        "cmd_preview": "GET /api/arbitrage/settings",
        "timeout_s": 15,
        "description": (
            "Source for the /arbitrage/settings page. 200 = settings page "
            "will populate. 401/403 = session expired. 500 = config_settings "
            "table broken."
        ),
    },
    {
        "id": "api_settings_sniper_get",
        "title": "API: GET /api/sniper/settings",
        "category": "api",
        "kind": "probe",
        "endpoint": "sniper/settings",
        "cmd_preview": "GET /api/sniper/settings",
        "timeout_s": 15,
        "description": "Source for the /sniper/settings page.",
    },
    {
        "id": "api_settings_copytrading_get",
        "title": "API: GET /api/copytrading/settings",
        "category": "api",
        "kind": "probe",
        "endpoint": "copytrading/settings",
        "cmd_preview": "GET /api/copytrading/settings",
        "timeout_s": 15,
        "description": "Source for the /copytrading/settings page.",
    },
    {
        "id": "api_settings_ai_get",
        "title": "API: GET /api/ai/settings",
        "category": "api",
        "kind": "probe",
        "endpoint": "ai/settings",
        "cmd_preview": "GET /api/ai/settings",
        "timeout_s": 15,
        "description": "Source for the /ai/settings page.",
    },
    {
        "id": "api_settings_futures_get",
        "title": "API: GET /api/settings/futures",
        "category": "api",
        "kind": "probe",
        "endpoint": "settings/futures",
        "cmd_preview": "GET /api/settings/futures",
        "timeout_s": 15,
        "description": "Source for the /futures/settings page.",
    },
    {
        "id": "api_settings_solana_get",
        "title": "API: GET /api/settings/solana",
        "category": "api",
        "kind": "probe",
        "endpoint": "settings/solana",
        "cmd_preview": "GET /api/settings/solana",
        "timeout_s": 15,
        "description": "Source for the /solana/settings page.",
    },
    {
        "id": "api_credentials_list",
        "title": "API: GET /api/credentials (admin)",
        "category": "api",
        "kind": "probe",
        "endpoint": "credentials",
        "cmd_preview": "GET /api/credentials",
        "timeout_s": 15,
        "description": (
            "Source for the credentials settings page. Returns 403 if "
            "the current session is not an admin."
        ),
    },
    {
        "id": "api_rpc_pool_endpoints_list",
        "title": "API: GET /api/rpc-pool/endpoints",
        "category": "api",
        "kind": "probe",
        "endpoint": "rpc-pool/endpoints",
        "cmd_preview": "GET /api/rpc-pool/endpoints",
        "timeout_s": 15,
        "description": "Source for the RPC/API endpoint settings page.",
    },

    # ── Per-module DRY_RUN GET probes (canonical toggle endpoint) ─────
    # Each /api/modules/<m>/dry-run GET returns
    #   {db_value: "true"|"false"|null, effective_dry_run: bool}
    # so the operator can confirm both the persisted override and the
    # value the engine subprocess will actually resolve on startup.
    {
        "id": "api_dry_run_arbitrage",
        "title": "API: GET /api/modules/arbitrage/dry-run",
        "category": "api",
        "kind": "probe",
        "endpoint": "modules/arbitrage/dry-run",
        "cmd_preview": "GET /api/modules/arbitrage/dry-run",
        "timeout_s": 10,
        "description": (
            "Per-module DRY_RUN read for ARBITRAGE. Save settings on "
            "/arbitrage/settings with the DRY RUN checkbox to flip; "
            "subprocess restart required for effect."
        ),
    },
    {
        "id": "api_dry_run_sniper",
        "title": "API: GET /api/modules/sniper/dry-run",
        "category": "api",
        "kind": "probe",
        "endpoint": "modules/sniper/dry-run",
        "cmd_preview": "GET /api/modules/sniper/dry-run",
        "timeout_s": 10,
        "description": "Per-module DRY_RUN read for SNIPER.",
    },
    {
        "id": "api_dry_run_copytrading",
        "title": "API: GET /api/modules/copy_trading/dry-run",
        "category": "api",
        "kind": "probe",
        "endpoint": "modules/copy_trading/dry-run",
        "cmd_preview": "GET /api/modules/copy_trading/dry-run",
        "timeout_s": 10,
        "description": "Per-module DRY_RUN read for COPY_TRADING.",
    },
    {
        "id": "api_dry_run_ai",
        "title": "API: GET /api/modules/ai/dry-run",
        "category": "api",
        "kind": "probe",
        "endpoint": "modules/ai/dry-run",
        "cmd_preview": "GET /api/modules/ai/dry-run",
        "timeout_s": 10,
        "description": "Per-module DRY_RUN read for AI_ANALYSIS.",
    },
    {
        "id": "api_dry_run_futures",
        "title": "API: GET /api/modules/futures/dry-run",
        "category": "api",
        "kind": "probe",
        "endpoint": "modules/futures/dry-run",
        "cmd_preview": "GET /api/modules/futures/dry-run",
        "timeout_s": 10,
        "description": "Per-module DRY_RUN read for FUTURES_TRADING.",
    },
    {
        "id": "api_dry_run_solana",
        "title": "API: GET /api/modules/solana/dry-run",
        "category": "api",
        "kind": "probe",
        "endpoint": "modules/solana/dry-run",
        "cmd_preview": "GET /api/modules/solana/dry-run",
        "timeout_s": 10,
        "description": "Per-module DRY_RUN read for SOLANA.",
    },
    {
        "id": "api_dry_run_dex",
        "title": "API: GET /api/modules/dex/dry-run",
        "category": "api",
        "kind": "probe",
        "endpoint": "modules/dex/dry-run",
        "cmd_preview": "GET /api/modules/dex/dry-run",
        "timeout_s": 10,
        "description": "Per-module DRY_RUN read for DEX_TRADING.",
    },
    {
        "id": "api_performance_metrics",
        "title": "API: GET /api/performance/metrics",
        "category": "api",
        "kind": "probe",
        "endpoint": "performance/metrics",
        "cmd_preview": "GET /api/performance/metrics",
        "timeout_s": 30,
        "description": (
            "Aggregate performance from unified trades table. Used by "
            "/performance and dashboard hero cards. Used to 500 on "
            "Decimal/NaT inputs — hardened in 6970edc."
        ),
    },

    {
        "id": "db_orch_training_data",
        "title": "DB: orchestrator ML training-data view",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT recommended, operator_agreed, COUNT(*) AS n "
            "FROM orchestrator_training_data "
            "GROUP BY recommended, operator_agreed "
            "ORDER BY recommended, operator_agreed"
        ),
        "cmd_preview": (
            "SELECT recommended, operator_agreed, COUNT(*) FROM orchestrator_training_data"
        ),
        "timeout_s": 15,
        "description": (
            "Confirms the labeled-data view exists and shows the per-"
            "action accept/reject distribution. After 30+ days of "
            "operator interactions this becomes the training set for "
            "a confidence-calibration model."
        ),
    },

    # ════════════════════════════════════════════════════════════════════
    # Wave-2 T1 catalog additions: DEX / ARBITRAGE / SOLANA / SNIPER
    # coverage for the commits enumerated in PM_PLAN "T1 / T2 brief".
    # ════════════════════════════════════════════════════════════════════

    # ── DEX (A1 wave-2: f7d7941, e872121, 23d860d, 48d5f20,
    #        162f711, a40f69a, 869eed3) ─────────────────────────────────
    {
        "id": "script_dex_decimals_unit_tests",
        "title": "Script: DEX decimals + route-quality regression tests (869eed3)",
        "category": "scripts",
        "kind": "bash",
        "cmd": [
            "python", "-m", "pytest",
            "tests/unit/test_dex_decimals.py", "-v", "--tb=short", "-x",
        ],
        "cmd_preview": "pytest tests/unit/test_dex_decimals.py -v",
        "timeout_s": 120,
        "description": (
            "Pins MB-01 (decimals on input + output legs for USDC/WBTC/"
            "WETH) and the route-quality scoring regressions (prefers "
            "lower gas + lower price-impact). Failure = direct_dex "
            "decimals fix or _score_quote ranker regressed."
        ),
    },
    {
        "id": "script_dex_web3_v6_imports",
        "title": "Script: DEX web3 v6 API drift import smoke (48d5f20)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/dex_web3_v6_smoke.sh"],
        "cmd_preview": "bash scripts/dex_web3_v6_smoke.sh",
        "timeout_s": 30,
        "description": (
            "Imports trading.executors.direct_dex + mev_protection and "
            "asserts the v6 snake_case Web3 helpers + ExtraDataToPOAMiddleware "
            "import path resolve. Failure = web3>=6 install drift or a "
            "regression of the toChecksumAddress/PoA fallback shim."
        ),
    },
    {
        "id": "script_dex_mev_unbound_check",
        "title": "Script: DEX mev_protection bundle_id default (e872121)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/dex_mev_unbound_check.sh"],
        "cmd_preview": "bash scripts/dex_mev_unbound_check.sh",
        "timeout_s": 15,
        "description": (
            "Source-grep: verifies trading/executors/mev_protection.py "
            "declares bundle_id=None before the if/else branches so the "
            "low-risk ADVANCED path can't UnboundLocalError. Also "
            "asserts Flashbots-on-ETH gating string is present."
        ),
    },
    {
        "id": "db_dex_recent_trades_24h",
        "title": "DB: DEX trades last 24h (decimals + scoring sanity)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            # trades schema has no 'dex_used' column (data/storage/database.py
            # line 235); the executor stashes the routed DEX inside metadata.
            "SELECT chain, "
            "  COALESCE(metadata->>'dex', strategy) AS dex_used, "
            "  status, COUNT(*) AS n, "
            "  ROUND(AVG(slippage)::numeric, 5) AS avg_slippage, "
            "  ROUND(AVG(gas_fee)::numeric, 8) AS avg_gas_fee "
            "FROM trades "
            "WHERE entry_timestamp > NOW() - INTERVAL '24 hours' "
            "GROUP BY chain, dex_used, status "
            "ORDER BY chain, dex_used, status"
        ),
        "cmd_preview": (
            "trades last 24h: chain, metadata.dex, status, COUNT, "
            "AVG(slippage), AVG(gas_fee)"
        ),
        "timeout_s": 15,
        "description": (
            "Per-chain DEX trade flow + average slippage / gas. After "
            "f7d7941 the slippage column should always be populated "
            "(self.max_slippage init fix). avg_gas wildly off chain "
            "ceiling indicates the per-chain gwei cap (162f711) is "
            "misconfigured."
        ),
    },
    {
        "id": "db_dex_settings_keys",
        "title": "DB: DEX settings (max_slippage + gas + mev)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE config_type LIKE 'dex%' "
            "  AND key IN ("
            "    'max_slippage','max_slippage_bps',"
            "    'max_gas_price','max_gas_price_gwei',"
            "    'mev_protection','flashbots_enabled'"
            "  ) "
            "ORDER BY config_type, key"
        ),
        "cmd_preview": (
            "SELECT … WHERE config_type LIKE 'dex%' AND key IN slip/gas/mev"
        ),
        "timeout_s": 10,
        "description": (
            "Wave-2 DEX wiring: confirms max_slippage (f7d7941), gas "
            "ceiling (162f711) and MEV toggle (e872121) are seeded. "
            "Missing rows = DirectDEXExecutor falls back to defaults "
            "(0.5% slippage, per-chain gwei map, Flashbots-ETH-only)."
        ),
    },
    {
        "id": "db_dex_open_positions",
        "title": "DB: DEX open positions (status='open')",
        "category": "db",
        "kind": "db_query",
        "sql": (
            # trades has no 'dex_used' or 'amount_in' — use metadata.dex
            # and the canonical 'amount' column.
            "SELECT chain, "
            "  COALESCE(metadata->>'dex', strategy) AS dex_used, "
            "  COUNT(*) AS open_n, "
            "  ROUND(SUM(amount)::numeric, 4) AS total_amount, "
            "  MIN(entry_timestamp) AS oldest, "
            "  MAX(entry_timestamp) AS newest "
            "FROM trades WHERE status = 'open' "
            "GROUP BY chain, dex_used ORDER BY chain, dex_used"
        ),
        "cmd_preview": (
            "Open trades: chain, metadata.dex, COUNT, SUM(amount), oldest/newest"
        ),
        "timeout_s": 10,
        "description": (
            "Snapshot of currently-held DEX positions. amount_in being "
            "honest (post-23d860d decimals fix) is the key invariant — "
            "a USDC position must report units of USDC, not 10^12× more."
        ),
    },

    # ── ARBITRAGE (A2 wave-2: 9e6a7d1, 744ee48, 4adcd29, 8cf0143,
    #              89175d4, 014384d) ──────────────────────────────────
    {
        "id": "db_arb_cost_profile_keys",
        "title": "DB: ARBITRAGE cost-profile + min_profit_spread (744ee48)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE config_type = 'arbitrage_config' "
            "  AND key IN ("
            "    'min_profit_spread','min_profit_bps',"
            "    'gas_budget_usd_per_hour','adaptive_min_profit_enabled',"
            "    'chain_cost_profile_enabled'"
            "  ) "
            "ORDER BY key"
        ),
        "cmd_preview": (
            "SELECT … WHERE config_type='arbitrage_config' AND key IN profit/cost knobs"
        ),
        "timeout_s": 10,
        "description": (
            "Wave-2 ARB knobs landed by 744ee48 (UI knob honored), "
            "4adcd29 (cost helpers), 89175d4 (hourly gas budget). "
            "Empty result = settings_arbitrage.html saves are not "
            "reaching the engine; engine falls back to chain defaults."
        ),
    },
    {
        "id": "db_arb_flash_loan_receiver_secrets",
        "title": "DB: ARBITRAGE flash-loan receiver addresses (89175d4)",
        "category": "db",
        "kind": "db_query",
        # The fix moved FLASH_LOAN_RECEIVER_CONTRACT_* from os.getenv to
        # the secrets_manager DB-backed table. We don't surface the
        # plaintext value — just whether the encrypted row exists per
        # chain so the operator can confirm the migration ran.
        "sql": (
            # config_sensitive uses column 'key' (not key_name) per
            # migration 002. secure_credentials (migration 012) is the
            # newer table with key_name. ARB receivers can live in either
            # — UNION so this probe lights up regardless of which path
            # the operator chose.
            "SELECT key AS key_name, "
            "  CASE WHEN encrypted_value IS NOT NULL "
            "       AND length(encrypted_value) > 0 "
            "       THEN 'present' ELSE 'missing' END AS status, "
            "  updated_at "
            "FROM config_sensitive "
            "WHERE key LIKE 'FLASH_LOAN_RECEIVER%' "
            "UNION ALL "
            "SELECT key_name, "
            "  CASE WHEN encrypted_value IS NOT NULL "
            "       AND length(encrypted_value) > 0 "
            "       THEN 'present' ELSE 'missing' END AS status, "
            "  updated_at "
            "FROM secure_credentials "
            "WHERE key_name LIKE 'FLASH_LOAN_RECEIVER%' "
            "ORDER BY key_name"
        ),
        "cmd_preview": (
            "config_sensitive.key + secure_credentials.key_name LIKE 'FLASH_LOAN_RECEIVER%'"
        ),
        "timeout_s": 10,
        "description": (
            "Confirms per-chain Aave V3 receiver-contract addresses "
            "exist in the encrypted DB store (89175d4). Engines fall "
            "back to os.getenv only if DB row absent; visible 'missing' "
            "rows = `_get_decrypted_key` returns None, flash-loan path "
            "errors at execute time."
        ),
    },
    {
        "id": "db_arb_recent_pnl_costs",
        "title": "DB: ARBITRAGE recent PnL with real costs (8cf0143)",
        "category": "db",
        "kind": "db_query",
        # The 8cf0143 fix replaces the $15 / 30%-of-spread magic numbers
        # with chain-aware live costs. After the fix, gas_cost should
        # vary by chain (ETH > ARB > BASE) instead of every row being
        # exactly 15.0.
        "sql": (
            # arbitrage_trades schema has no gas_cost_usd / slippage_cost_usd
            # columns (migration 009) — the 8cf0143 fix writes the cost
            # breakdown into the metadata JSONB. Read it back.
            "SELECT chain, status, COUNT(*) AS n, "
            "  ROUND(AVG(NULLIF((metadata->>'gas_cost_usd')::numeric, 0))::numeric, 4) AS avg_gas_usd, "
            "  ROUND(AVG(NULLIF((metadata->>'slippage_cost_usd')::numeric, 0))::numeric, 4) AS avg_slip_usd, "
            "  ROUND(SUM(profit_loss)::numeric, 4) AS sum_pnl "
            "FROM arbitrage_trades "
            "WHERE entry_timestamp > NOW() - INTERVAL '24 hours' "
            "GROUP BY chain, status ORDER BY chain, status"
        ),
        "cmd_preview": (
            "Per-chain ARB roll-up — gas / slippage / PnL last 24h "
            "(costs read from metadata JSONB)"
        ),
        "timeout_s": 15,
        "description": (
            "Per-chain ARB cost roll-up. After 8cf0143 the avg_gas_usd "
            "should differ between chains; the legacy $15 constant "
            "would show identical 15.0 across ethereum/arbitrum/base. "
            "Slippage cost should track entry_usd × default_slippage_pct."
        ),
    },
    # api_arb_settings_get removed — duplicate of api_settings_arbitrage_get
    # earlier in this file. The min_profit_spread key is already visible
    # in that response (line 36 of the body). Was emitting ~120 lines of
    # identical settings JSON.
    {
        "id": "script_arb_nameerror_regression",
        "title": "Script: ARBITRAGE spatial-arb NameError grep (9e6a7d1)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/arb_nameerror_check.sh"],
        "cmd_preview": "bash scripts/arb_nameerror_check.sh",
        "timeout_s": 10,
        "description": (
            "Source-grep: asserts arbitrage_engine.py no longer "
            "references the renamed `forward_output` / `final_output` "
            "identifiers from inside `_check_arb_opportunity`. Their "
            "reappearance = wave-1 NameError regression that silently "
            "dropped every spatial-arb opportunity."
        ),
    },

    # ── SOLANA (A3 wave-2: 09a5c85, 661cee6, 83df4ad, b1b358f) ──────
    {
        "id": "db_solana_adaptive_priority_fee",
        "title": "DB: SOLANA adaptive priority-fee + quote TTL (83df4ad)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key IN ("
            "  'adaptive_priority_fee_enabled',"
            "  'adaptive_priority_fee_percentile',"
            "  'adaptive_priority_fee_min_lamports',"
            "  'adaptive_priority_fee_max_lamports',"
            "  'adaptive_priority_fee_ttl_s',"
            "  'jupiter_quote_max_age_s'"
            ") "
            "ORDER BY config_type, key"
        ),
        "cmd_preview": (
            "SELECT … WHERE key IN adaptive_priority_fee_*/jupiter_quote_max_age_s"
        ),
        "timeout_s": 10,
        "description": (
            "Wave-2 Solana profitability levers: adaptive priority-fee "
            "controller (off by default) + Jupiter quote freshness TTL "
            "(default 10s). Empty result = JupiterHelper falls back to "
            "static priority_fee and 10s TTL."
        ),
    },
    {
        "id": "db_solana_drift_guards",
        "title": "DB: SOLANA Drift MB-15 pre-trade guards (661cee6)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE config_type = 'solana_drift' "
            "  AND key IN ("
            "    'drift_enabled','drift_max_leverage',"
            "    'drift_max_funding_pct_annual',"
            "    'drift_oracle_deviation_max_pct',"
            "    'drift_min_oracle_conf_bps'"
            "  ) "
            "ORDER BY key"
        ),
        "cmd_preview": (
            "SELECT … WHERE config_type='solana_drift' AND key IN MB-15 caps"
        ),
        "timeout_s": 10,
        "description": (
            "MB-15 fail-closed guards: leverage cap, funding sanity "
            "cap, oracle-deviation cap, Pyth confidence cap. Drift "
            "stays drift_enabled=false until operator flips; on flip, "
            "missing rows fall back to conservative defaults (3x / "
            "50%/yr / 1% / 500 bps)."
        ),
    },
    {
        "id": "db_solana_ml_rug_gate",
        "title": "DB: SOLANA ML rug-gate config (b1b358f)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE config_type = 'solana_ml' "
            "  AND key IN ("
            "    'solana_ml_enabled','solana_ml_max_rug_prob',"
            "    'solana_ml_min_pump_prob'"
            "  ) "
            "ORDER BY key"
        ),
        "cmd_preview": (
            "SELECT … WHERE config_type='solana_ml' AND key IN ml gate knobs"
        ),
        "timeout_s": 10,
        "description": (
            "P1-07 ML rug-gate wiring (b1b358f). Default off "
            "(solana_ml_enabled=false). When enabled, RugClassifier "
            "lazy-loads at first _open_position; refuses entry when "
            "rug_prob > solana_ml_max_rug_prob."
        ),
    },
    {
        "id": "db_solana_recent_trades_decimals",
        "title": "DB: SOLANA recent trades sanity (decimals + execution path)",
        "category": "db",
        "kind": "db_query",
        # After 09a5c85 the close path resolves on-chain decimals
        # instead of hardcoding 6. Real-world: BONK is 5 decimals,
        # most modern memecoins are 6 or 9. A sniped position whose
        # close-side qty looks orders-of-magnitude off would have
        # caught fire pre-fix.
        "sql": (
            # solana_trades is closed-only by schema (migration 008) — no
            # 'status' column. Group by strategy + is_simulated instead.
            "SELECT strategy, is_simulated, "
            "  COUNT(*) AS n, "
            "  ROUND(AVG(amount_sol)::numeric, 4) AS avg_amount_sol, "
            "  ROUND(AVG(pnl_usd)::numeric, 4) AS avg_pnl_usd, "
            "  MIN(entry_time) AS oldest, "
            "  MAX(entry_time) AS newest "
            "FROM solana_trades "
            "WHERE entry_time > NOW() - INTERVAL '24 hours' "
            "GROUP BY strategy, is_simulated ORDER BY strategy, is_simulated"
        ),
        "cmd_preview": (
            "GROUP-BY strategy,is_simulated on solana_trades, 24h"
        ),
        "timeout_s": 15,
        "description": (
            "24h solana_trades roll-up. Post-09a5c85 the close path "
            "uses on-chain decimals (no more 10x oversell / 1000x "
            "undersell on BONK-like tokens). avg_amount_sol grossly "
            "different from configured position_size = misconfig."
        ),
    },
    {
        "id": "db_solana_position_size_caps",
        "title": "DB: SOLANA position-size + capital caps",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE config_type LIKE 'solana_%' "
            "  AND key IN ("
            "    'capital','position_size','max_positions','min_position',"
            "    'daily_loss_limit','stop_loss','take_profit'"
            "  ) "
            "ORDER BY config_type, key"
        ),
        "cmd_preview": (
            "SELECT … WHERE config_type LIKE 'solana_%' AND key IN caps"
        ),
        "timeout_s": 10,
        "description": (
            "Snapshot of Solana engine capital + risk caps. These are "
            "read by SolanaConfigManager → SolanaEngine init. Missing "
            "rows fall back to DEFAULTS (10 SOL capital, 1 SOL/pos, "
            "3 positions, 5% daily loss)."
        ),
    },

    # ── SNIPER (A4 wave-2: 6612be2, 77b22e7, 87c5523, 4adcd29,
    #           adee9c2, 5a0a3e9) ─────────────────────────────────────
    {
        "id": "db_sniper_processed_hit_ratio",
        "title": "DB: SNIPER processed→confirmed readback hit ratio (6612be2)",
        "category": "db",
        "kind": "db_query",
        # processed_hit + processed_miss_fallback are surfaced in
        # sniper_runtime_stats.stats by _persist_runtime_stats. The
        # two-stage readback (R1 fix) should land processed_hit on
        # the supermajority of getTransaction calls for fresh pools.
        "sql": (
            "SELECT id, "
            "  EXTRACT(EPOCH FROM (NOW() - updated_at))::int AS age_seconds, "
            "  (stats->>'processed_hit')::int AS processed_hit, "
            "  (stats->>'processed_miss_fallback')::int AS processed_miss, "
            "  CASE WHEN COALESCE((stats->>'processed_hit')::int, 0) "
            "          + COALESCE((stats->>'processed_miss_fallback')::int, 0) > 0 "
            "       THEN ROUND(100.0 * "
            "            COALESCE((stats->>'processed_hit')::numeric, 0) / "
            "            NULLIF(COALESCE((stats->>'processed_hit')::numeric, 0) "
            "                 + COALESCE((stats->>'processed_miss_fallback')::numeric, 0), 0), "
            "            1) "
            "       ELSE NULL END AS pct_processed "
            "FROM sniper_runtime_stats WHERE id = 1"
        ),
        "cmd_preview": (
            "processed_hit / (hit+miss) ratio from sniper_runtime_stats"
        ),
        "timeout_s": 15,
        "description": (
            "R1 fix: two-stage `processed`→`confirmed` readback. "
            "pct_processed > 80% means the fast path is paying off "
            "(~300 ms vs 3-13 s commitment wait). Low ratio implies "
            "Raydium logMessages aren't populating at `processed` for "
            "this RPC — investigate provider commitment lag."
        ),
    },
    {
        "id": "db_sniper_safety_check_errors",
        "title": "DB: SNIPER safety-check exception counter (77b22e7)",
        "category": "db",
        "kind": "db_query",
        # safety_check_errors + jupiter_quote_fallback_hits +
        # birdeye_fallback_hits are all preserved across the 1-min
        # _log_stats_if_needed reset so the dashboard sees cumulative
        # totals (87c5523 + adee9c2 carry-over rule).
        "sql": (
            "SELECT id, "
            "  EXTRACT(EPOCH FROM (NOW() - updated_at))::int AS age_seconds, "
            "  (stats->>'safety_check_errors')::int AS safety_errors, "
            "  (stats->>'jupiter_quote_fallback_hits')::int AS jup_fallback, "
            "  (stats->>'birdeye_fallback_hits')::int AS birdeye_fallback, "
            "  (stats->>'tokens_analyzed')::int AS tokens_analyzed, "
            "  (stats->>'tokens_rejected')::int AS tokens_rejected "
            "FROM sniper_runtime_stats WHERE id = 1"
        ),
        "cmd_preview": (
            "safety_check_errors + jup/birdeye fallback counters from runtime stats"
        ),
        "timeout_s": 15,
        "description": (
            "R2 fix: GoPlus/Honeypot.is exception path now records "
            "cooldown + counter so the same failing token can't "
            "busy-loop the safety API. Sustained non-zero growth = "
            "external safety provider outage, not a bug."
        ),
    },
    {
        "id": "db_sniper_wss_carry_over",
        "title": "DB: SNIPER WSS dispatched/inflight peak surface (87c5523)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT id, "
            "  EXTRACT(EPOCH FROM (NOW() - updated_at))::int AS age_seconds, "
            "  (stats->>'wss_dispatched')::int AS wss_dispatched, "
            "  (stats->>'wss_inflight_peak')::int AS wss_inflight_peak, "
            "  (stats->>'pools_detected')::int AS pools_detected "
            "FROM sniper_runtime_stats WHERE id = 1"
        ),
        "cmd_preview": (
            "wss_dispatched + wss_inflight_peak from sniper_runtime_stats"
        ),
        "timeout_s": 15,
        "description": (
            "R3 fix: WSS counters now carry across the 1-min stats "
            "reset window. inflight_peak approaching the semaphore "
            "ceiling (SNIPER_WSS_CONCURRENCY) means bad-RPC backup "
            "is saturating the dispatch queue."
        ),
    },
    {
        "id": "db_sniper_quorum_outcomes_30m",
        "title": "DB: SNIPER quorum-decision outcomes 30m (R4 quorum)",
        "category": "db",
        "kind": "db_query",
        # R4 (token_safety._quorum_honeypot_decision) split honeypot
        # rejection into agreement-based + asymmetric fail-safe paths.
        # rejected_safety_error is the 77b22e7 outcome label for the
        # exception cooldown branch; rejected_safety is generic R4 +
        # other safety rejections.
        "sql": (
            "SELECT "
            "  COALESCE(metadata->>'outcome', 'unknown') AS outcome, "
            "  COUNT(*) AS n "
            "FROM sniper_trades "
            "WHERE entry_timestamp > NOW() - INTERVAL '30 minutes' "
            "GROUP BY outcome ORDER BY n DESC"
        ),
        "cmd_preview": (
            "GROUP-BY metadata->>'outcome' on sniper_trades, 30m"
        ),
        "timeout_s": 15,
        "description": (
            "Distribution of sniper_trades.metadata.outcome over the "
            "last 30 min. rejected_safety_error column comes from "
            "77b22e7 (R2); other rejected_safety rows are the R4 "
            "quorum gate. Healthy mix = both sources up."
        ),
    },
    # api_sniper_timing_per_chain removed — same endpoint as
    # api_sniper_timing earlier; the per-chain GROUP BY is delivered
    # client-side from the same data, so two probes hit identical bytes.
    {
        "id": "script_sniper_listener_widget_present",
        "title": "Script: SNIPER per-chain widget HTML presence (5a0a3e9)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/sniper_listener_widget_check.sh"],
        "cmd_preview": "bash scripts/sniper_listener_widget_check.sh",
        "timeout_s": 10,
        "description": (
            "Grep test for the per-chain listener-health widget id in "
            "dashboard/templates/performance_sniper.html (5a0a3e9). "
            "Confirms the operator-visible WSS-saturation / "
            "processed-hit panel survived template refactors."
        ),
    },

    # ════════════════════════════════════════════════════════════════════
    # Wave-2 T2 catalog additions: FUTURES / AI / COPY_TRADING coverage
    # for the commits enumerated in PM_PLAN "T1 / T2 brief" section.
    # ════════════════════════════════════════════════════════════════════

    # ── FUTURES (A5 wave-2: FUT-RM-01..07) ───────────────────────────────
    {
        "id": "db_futures_leverage_caps",
        "title": "DB: FUTURES leverage + position caps (FUT-RM-01)",
        "category": "db",
        "kind": "db_query",
        # The b1b8df9 fix patched only the dashboard wrapper; FUT-RM-01
        # propagates futures_max_leverage / max_positions through to the
        # main_futures.py subprocess. This probe confirms the DB rows
        # exist — otherwise FuturesRiskManager silently falls back to
        # max_leverage=3 on next subprocess restart.
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key IN ('futures_max_leverage','max_positions',"
            "'capital_allocation','default_leverage') "
            "  AND config_type LIKE 'futures%' "
            "ORDER BY config_type, key"
        ),
        "cmd_preview": (
            "SELECT … WHERE key IN ('futures_max_leverage','max_positions',…)"
        ),
        "timeout_s": 10,
        "description": (
            "FUT-RM-01 wiring sanity: every key main_futures.py merges "
            "into risk_cfg before constructing FuturesRiskManager. "
            "Missing rows = engine falls back to hard-coded defaults "
            "(leverage=3x, positions=3) regardless of dashboard setting."
        ),
    },
    {
        "id": "db_futures_funding_gate",
        "title": "DB: FUTURES funding-rate gate config (FUT-RM-05)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key IN ('skip_long_funding_bps','skip_short_funding_bps') "
            "ORDER BY config_type, key"
        ),
        "cmd_preview": (
            "SELECT … WHERE key IN ('skip_long_funding_bps','skip_short_funding_bps')"
        ),
        "timeout_s": 10,
        "description": (
            "FUT-RM-05 directional funding gate. Default 5 bps ≈ 55% "
            "APR ceiling for longs. Missing rows mean the engine falls "
            "back to the dataclass default."
        ),
    },
    {
        "id": "db_futures_atr_sizing",
        "title": "DB: FUTURES ATR sizing toggles (FUT-RM-06)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key IN ('atr_sizing_enabled','atr_risk_pct',"
            "'atr_stop_multiplier') "
            "ORDER BY config_type, key"
        ),
        "cmd_preview": (
            "SELECT … WHERE key IN ('atr_sizing_enabled','atr_risk_pct',…)"
        ),
        "timeout_s": 10,
        "description": (
            "FUT-RM-06 ATR-based per-symbol sizing. Opt-in (default "
            "off). When enabled, _calculate_position_size routes to "
            "the ATR branch so a 5% ATR symbol gets ~1/5 the notional "
            "of a 1% ATR symbol."
        ),
    },
    {
        "id": "db_futures_isolated_enforce",
        "title": "DB: FUTURES isolated-margin enforcement (FUT-RM-07)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key = 'enforce_isolated_margin' "
            "ORDER BY config_type"
        ),
        "cmd_preview": "SELECT … WHERE key='enforce_isolated_margin'",
        "timeout_s": 10,
        "description": (
            "FUT-RM-07 defence-in-depth: after fill, _verify_isolated_"
            "or_close() reads back the position and emergency-closes "
            "on margin_type != ISOLATED. Default True."
        ),
    },
    # api_settings_futures_post_wave2 removed — duplicate of
    # api_settings_futures_get earlier. Wave-2 keys (skip_long_funding_bps,
    # atr_sizing_enabled, atr_risk_pct, enforce_isolated_margin) are
    # already visible in that probe's response. Was 60 lines of identical
    # JSON.

    # ════════════════════════════════════════════════════════════════════
    # Wave-4 T2 catalog additions: FUTURES coverage for
    # FUT-RM-07b (Telegram alert) + FUT-RM-09b (funding-forecast widget).
    # Commits: 34e6c95 / 6f66608 / 142250b / b805626.
    # ════════════════════════════════════════════════════════════════════
    {
        "id": "api_futures_funding_forecast",
        "title": "API: GET /api/futures/funding-forecast (FUT-RM-09b)",
        "category": "api",
        "kind": "probe",
        # Default 24h window; widget polls this exact URL.
        "endpoint": "futures/funding-forecast?window_hours=24",
        "cmd_preview": "GET /api/futures/funding-forecast?window_hours=24",
        "timeout_s": 15,
        "description": (
            "FUT-RM-09b per-symbol 24h forward funding-cost forecast. "
            "Reads the latest snapshot per (symbol, side) from "
            "futures_funding_payments and projects predicted_usd × "
            "intervals_per_window. Empty rows[] = no recent snapshot "
            "(engine has not yet observed a funding interval for any "
            "symbol on this network)."
        ),
    },
    {
        "id": "db_futures_funding_payments_recent",
        "title": "DB: futures_funding_payments recent rows (mig 029)",
        "category": "db",
        "kind": "db_query",
        # The funding-forecast widget reads from this table. Empty = the
        # engine has not yet seen a funding interval close OR migration
        # 029 has not been applied. The per-symbol projection in the
        # /api/futures/funding-forecast handler short-circuits when no
        # row is present.
        "sql": (
            "SELECT symbol, side, exchange, source, "
            "  ROUND(predicted_usd::numeric, 4) AS predicted_usd, "
            "  ROUND(realized_usd::numeric, 4) AS realized_usd, "
            "  ROUND(notional_usd::numeric, 2) AS notional_usd, "
            "  hour_bucket "
            "FROM futures_funding_payments "
            "WHERE hour_bucket > NOW() - INTERVAL '24 hours' "
            "ORDER BY hour_bucket DESC, symbol "
            "LIMIT 20"
        ),
        "cmd_preview": (
            "SELECT symbol,side,predicted/realized_usd FROM "
            "futures_funding_payments WHERE hour_bucket > now()-24h"
        ),
        "timeout_s": 15,
        "description": (
            "Last 24h of per-(symbol, side, source) funding payments. "
            "Source 'engine' = predicted at gate time; 'income' = "
            "exchange-confirmed funding income row; 'exit' = realized "
            "at position close. Powers the FUT-RM-09b forecast widget."
        ),
    },
    {
        "id": "db_futures_telegram_alert_flag",
        "title": "DB: FUTURES Telegram emergency-close flag (FUT-RM-07b)",
        "category": "db",
        "kind": "db_query",
        # Gating flag for the FUT-RM-07b Telegram payload. Default TRUE
        # (alert fires whenever the verify path detects a CROSS-margin
        # fill). Operator can disable per-config_type if Telegram is
        # noisy in a particular environment.
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key = 'telegram_emergency_close_enabled' "
            "  AND config_type LIKE 'futures%' "
            "ORDER BY config_type"
        ),
        "cmd_preview": (
            "SELECT … WHERE key='telegram_emergency_close_enabled' "
            "AND config_type LIKE 'futures%'"
        ),
        "timeout_s": 10,
        "description": (
            "FUT-RM-07b alert gate. Default True; row absent = engine "
            "falls back to the FuturesLeverageConfig dataclass default "
            "(also True). Set to 'false' to silence the critical "
            "Telegram payload (the emergency-close itself still runs)."
        ),
    },
    {
        "id": "script_futures_funding_forecast_widget_present",
        "title": "Script: FUTURES funding-forecast widget HTML presence (b805626)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/futures_funding_forecast_widget_check.sh"],
        "cmd_preview": "bash scripts/futures_funding_forecast_widget_check.sh",
        "timeout_s": 10,
        "description": (
            "Grep test for the funding-forecast widget DOM ids "
            "(funding-forecast-window/total/table) and the "
            "/api/futures/funding-forecast URL in "
            "dashboard/templates/dashboard_futures.html. Catches "
            "template refactors that drop the FUT-RM-09b panel."
        ),
    },
    {
        "id": "script_futures_fut_rm_07b_notify_helper",
        "title": "Script: FUTURES FUT-RM-07b notify helper presence (34e6c95/6f66608)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/futures_fut_rm_07b_notify_check.sh"],
        "cmd_preview": "bash scripts/futures_fut_rm_07b_notify_check.sh",
        "timeout_s": 10,
        "description": (
            "Source-grep: asserts futures_engine.py declares "
            "`_notify_fut_rm_07_emergency_close`, awaits it from the "
            "verify-close path, AND gates it on "
            "telegram_emergency_close_enabled. Refactors that delete "
            "the call or drop the flag check fail this probe."
        ),
    },

    # ── AI (A6 wave-2: E1 quorum / E2 calibration / E3 bandit) ───────────
    {
        "id": "db_ai_calibration_table",
        "title": "DB: ai_confidence_calibration table present (mig 023)",
        "category": "db",
        "kind": "db_query",
        # New table from migration 023_add_ai_confidence_calibration.sql.
        # Verifies the table exists with the columns A6 E2 expects:
        # trade_id, provider, predicted_score/confidence, realized_*.
        # Empty column_count = migration not run; calibration endpoint
        # will return success=true but empty bins.
        "sql": (
            "SELECT table_name, "
            "  (SELECT COUNT(*) FROM information_schema.columns "
            "   WHERE table_name='ai_confidence_calibration') AS column_count, "
            "  (SELECT COUNT(*) FROM information_schema.columns "
            "   WHERE table_name='ai_confidence_calibration' "
            "     AND column_name IN ('trade_id','provider','predicted_score',"
            "       'predicted_confidence','realized_pnl_pct','realized_won',"
            "       'quorum_required','closed_at')) AS expected_cols "
            "FROM information_schema.tables "
            "WHERE table_name = 'ai_confidence_calibration'"
        ),
        "cmd_preview": "information_schema check for ai_confidence_calibration",
        "timeout_s": 10,
        "description": (
            "Confirms migration 023 has been applied. expected_cols "
            "should be 8 (the union of columns A6 E2 writes). The "
            "calibration endpoint and sentiment_engine close-hook will "
            "no-op gracefully when missing but no learning happens."
        ),
    },
    {
        "id": "db_ai_calibration_sample",
        "title": "DB: ai calibration sample (last 90d)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT "
            "  COUNT(*) AS total_rows, "
            "  COUNT(*) FILTER (WHERE realized_won IS NOT NULL) AS closed_rows, "
            "  COUNT(*) FILTER (WHERE quorum_required) AS quorum_rows, "
            "  ROUND(AVG(predicted_confidence)::numeric, 4) AS avg_pred_conf, "
            "  ROUND(AVG(CASE WHEN realized_won THEN 1.0 ELSE 0.0 END)::numeric, 4) AS avg_win_rate "
            "FROM ai_confidence_calibration "
            "WHERE created_at > NOW() - INTERVAL '90 days'"
        ),
        "cmd_preview": "COUNT + AVG predicted_confidence vs realized_won, 90d",
        "timeout_s": 15,
        "description": (
            "Source of the /api/ai/calibration reliability plot. "
            "avg_pred_conf far from avg_win_rate = miscalibrated LLM. "
            "Closed_rows < 20 = not enough data for a reliable plot."
        ),
    },
    {
        "id": "db_ai_quorum_bandit_config",
        "title": "DB: ai quorum + bandit settings (A6 E1/E3)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key IN ('quorum_required','quorum_max_disagreement',"
            "'bandit_enabled','bandit_epsilon','ai_provider') "
            "  AND config_type='ai_config' "
            "ORDER BY key"
        ),
        "cmd_preview": (
            "SELECT … WHERE key IN ('quorum_required','bandit_enabled',…)"
        ),
        "timeout_s": 10,
        "description": (
            "Wave-2 AI tunables: E1 multi-provider quorum + E3 prompt-"
            "bandit ε-greedy controller. Defaults are off — set "
            "quorum_required=true once both openai+anthropic keys are "
            "loaded; flip bandit_enabled=true after the calibration "
            "table has ≥30 closed trades."
        ),
    },
    {
        "id": "api_ai_calibration",
        "title": "API: GET /api/ai/calibration (A6 E2)",
        "category": "api",
        "kind": "probe",
        "endpoint": "ai/calibration",
        "cmd_preview": "GET /api/ai/calibration | .bins[] / .brier",
        "timeout_s": 15,
        "description": (
            "Reliability-diagram bins + Brier score for the LLM "
            "sentiment predictor. Returns success=true with empty "
            "bins + brier=null when the table is missing or has no "
            "closed trades — UI shows a 'no data' panel either way."
        ),
    },
    {
        "id": "db_ai_bandit_state",
        "title": "DB: prompt-bandit per-arm state (A6 E3)",
        "category": "db",
        "kind": "db_query",
        # prompt_bandit persists its (count, sum_reward, last_used_at)
        # per-template stats to ai_feature_store.feature_vector under
        # the 'bandit_v1' key. This probe shows the arm distribution
        # so the operator can see ε-greedy exploration vs exploit ratio.
        "sql": (
            "SELECT "
            "  feature_vector->'bandit_v1'->>'template' AS template, "
            "  COUNT(*) AS selections, "
            "  ROUND(AVG((feature_vector->'bandit_v1'->>'reward')::numeric)::numeric, 4) AS avg_reward, "
            "  MAX(timestamp) AS last_used "
            "FROM ai_feature_store "
            "WHERE feature_vector->'bandit_v1'->>'template' IS NOT NULL "
            "  AND timestamp > NOW() - INTERVAL '14 days' "
            "GROUP BY template "
            "ORDER BY selections DESC"
        ),
        "cmd_preview": "GROUP BY bandit_v1.template, AVG reward, 14d window",
        "timeout_s": 15,
        "description": (
            "Per-arm pull count and mean reward for the prompt-bandit. "
            "When bandit_enabled=true the engine writes one row per "
            "LLM call. Skewed selections (one arm ≫ others) = the "
            "bandit has converged."
        ),
    },

    # ════════════════════════════════════════════════════════════════════
    # Wave-4 T2 catalog additions: AI coverage for AI-Q-05 (calibrated
    # booster inference wrap) + quorum-metrics observability.
    # Commits: 16b7dab / a6c3a89 / c7e4a27 / 50bd9c3 / 68b20fb.
    # ════════════════════════════════════════════════════════════════════
    {
        "id": "api_ai_quorum_metrics",
        "title": "API: GET /api/ai/quorum-metrics?hours=24 (wave-4 50bd9c3)",
        "category": "api",
        "kind": "probe",
        "endpoint": "ai/quorum-metrics?hours=24",
        "cmd_preview": "GET /api/ai/quorum-metrics?hours=24",
        "timeout_s": 15,
        "description": (
            "Multi-provider quorum agreement metrics for the dashboard "
            "agreement-rate chart. Reads ai_feature_store rows whose "
            "metadata.quorum_outcome is populated. Empty payload = "
            "quorum_required=false OR the engine has not run since "
            "wave-4 (no rows yet)."
        ),
    },
    {
        "id": "db_ai_quorum_outcomes",
        "title": "DB: ai_feature_store quorum_outcome rows (c7e4a27)",
        "category": "db",
        "kind": "db_query",
        # _persist_quorum_outcome writes one row per tick into
        # ai_feature_store with metadata.quorum_outcome set to one of
        # the documented labels (agree / disagree / collapsed / no_vote).
        # This probe surfaces the per-outcome distribution over the
        # last 24h — the same window the /api/ai/quorum-metrics
        # endpoint defaults to.
        "sql": (
            "SELECT "
            "  COALESCE(metadata->>'quorum_outcome', 'unknown') AS outcome, "
            "  COUNT(*) AS n, "
            "  MAX(timestamp) AS most_recent "
            "FROM ai_feature_store "
            "WHERE metadata->>'quorum_outcome' IS NOT NULL "
            "  AND timestamp > NOW() - INTERVAL '24 hours' "
            "GROUP BY outcome "
            "ORDER BY n DESC"
        ),
        "cmd_preview": (
            "GROUP BY metadata->>'quorum_outcome' FROM ai_feature_store, 24h"
        ),
        "timeout_s": 15,
        "description": (
            "Distribution of quorum-vote outcomes (agree / disagree / "
            "collapsed / no_vote) for the last 24h. Source of the "
            "dashboard agreement-rate chart. Empty = quorum_required "
            "is false OR no LLM tick has run since wave-4 deployed."
        ),
    },
    {
        "id": "db_ai_calibrated_predictions_flag",
        "title": "DB: AI ai_calibrated_predictions_enabled flag (AI-Q-05)",
        "category": "db",
        "kind": "db_query",
        # Default-OFF flag operator flips after AI-Q-06 trainer has
        # produced calibrated_*.pkl sidecars. Row absent = engine reads
        # the EnsemblePredictor config default (False).
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key = 'ai_calibrated_predictions_enabled' "
            "  AND config_type IN ('ai_config','ai_analysis','ml_config') "
            "ORDER BY config_type"
        ),
        "cmd_preview": (
            "SELECT … WHERE key='ai_calibrated_predictions_enabled'"
        ),
        "timeout_s": 10,
        "description": (
            "AI-Q-05 calibrated booster inference gate. Default FALSE — "
            "EnsemblePredictor base scores unchanged. Set to true once "
            "the operator has run the AI-Q-06 trainer and confirmed "
            "calibrated_<name>.pkl artefacts are present in the "
            "models directory (db_ai_calibrated_model_artefacts probe)."
        ),
    },
    {
        "id": "db_ai_calibrated_model_artefacts",
        "title": "DB: AI calibrated booster artefacts on disk (AI-Q-05)",
        "category": "db",
        "kind": "db_query",
        # `pg_ls_dir` works inside the trading-postgres container even
        # though we run the probe from the dashboard process — we read
        # via asyncpg, which only sees what the DB process can see. The
        # /app/models directory is the canonical container-side path
        # (matches Dockerfile.dashboard + main bind-mount).
        # Empty = AI-Q-06 trainer never ran OR the bind-mount points
        # elsewhere; either way the flag has nothing to consume.
        "sql": (
            "SELECT name FROM pg_ls_dir('/app/models') AS name "
            "WHERE name LIKE 'calibrated_%.pkl' "
            "   OR name LIKE '%_calibrated.pkl' "
            "   OR name LIKE '%_calibrated.joblib' "
            "ORDER BY name"
        ),
        "cmd_preview": (
            "SELECT FROM pg_ls_dir('/app/models') WHERE name LIKE 'calibrated_%.pkl'"
        ),
        "timeout_s": 10,
        "description": (
            "Lists calibrated booster sidecars the EnsemblePredictor "
            "loader expects. Names match either deliverable form "
            "(calibrated_<name>.pkl) or the AI-Q-06 trainer form "
            "(<name>_calibrated.{pkl,joblib}). Empty = no calibration "
            "artefacts on disk; flipping the flag has no effect."
        ),
    },
    {
        "id": "script_ai_calibrated_helper_present",
        "title": "Script: AI calibrated_predict_proba helper presence (16b7dab/a6c3a89)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/ai_calibrated_helper_check.sh"],
        "cmd_preview": "bash scripts/ai_calibrated_helper_check.sh",
        "timeout_s": 10,
        "description": (
            "Source-grep: asserts ml/models/ensemble_model.py declares "
            "`calibrated_predict_proba` + `_load_calibrated_models`, "
            "calls the wrapper from the predict path, AND references "
            "`ai_calibrated_predictions_enabled`. Refactors that drop "
            "any of these silently regress AI-Q-05 to the raw booster."
        ),
    },
    {
        "id": "script_ai_quorum_persist_present",
        "title": "Script: AI _persist_quorum_outcome helper presence (c7e4a27)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/ai_quorum_persist_check.sh"],
        "cmd_preview": "bash scripts/ai_quorum_persist_check.sh",
        "timeout_s": 10,
        "description": (
            "Source-grep: asserts sentiment_engine.py declares "
            "`_persist_quorum_outcome` + `_record_quorum_outcome` AND "
            "awaits the persist helper from the engine tick. Catches "
            "regressions that would leave ai_feature_store empty and "
            "blank out the /api/ai/quorum-metrics chart."
        ),
    },
    {
        "id": "script_ai_quorum_widget_present",
        "title": "Script: AI quorum-metrics widget URL presence (50bd9c3)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/ai_quorum_widget_check.sh"],
        "cmd_preview": "bash scripts/ai_quorum_widget_check.sh",
        "timeout_s": 10,
        "description": (
            "Grep test for the /api/ai/quorum-metrics URL on "
            "dashboard/templates/dashboard_ai.html. Confirms the "
            "agreement-rate chart still polls the wave-4 endpoint."
        ),
    },

    # ── COPY_TRADING (A7 wave-2: operator-priority quant rebuild) ────────
    # Wallet-discovery + leader-scorer + Kelly sizing is the headline
    # feature this wave. Catalog gives the operator one-button checks
    # of every layer: migration -> scoring -> ranking -> refresh -> sizing.
    {
        "id": "db_copy_leader_scores_table",
        "title": "DB: copy_leader_scores table present (mig 024)",
        "category": "db",
        "kind": "db_query",
        # Migration 024_copy_leader_scores.sql adds the persistent
        # leader-scoring table. Verify the table exists AND has the
        # composite-score / kelly_fraction columns the engine reads
        # (CT-Q-02). Missing columns = wallet_discovery refresh will
        # fail at write time.
        "sql": (
            "SELECT table_name, "
            "  (SELECT COUNT(*) FROM information_schema.columns "
            "   WHERE table_name='copy_leader_scores') AS column_count, "
            "  (SELECT COUNT(*) FROM information_schema.columns "
            "   WHERE table_name='copy_leader_scores' "
            "     AND column_name IN ('chain','wallet_address','source',"
            "       'realized_pnl_usd_30d','sharpe_30d','hit_rate',"
            "       'max_drawdown_pct','score','kelly_fraction',"
            "       'raw_metrics','last_scored_at')) AS expected_cols "
            "FROM information_schema.tables "
            "WHERE table_name = 'copy_leader_scores'"
        ),
        "cmd_preview": "information_schema check for copy_leader_scores",
        "timeout_s": 10,
        "description": (
            "Confirms migration 024 has been applied. expected_cols "
            "should be 11 (the canonical set wallet_discovery writes "
            "and copy_engine reads for Kelly sizing). 0 rows = run "
            "the migration before the next refresh."
        ),
    },
    {
        "id": "db_copy_leader_scores_top",
        "title": "DB: copy_leader_scores top-10 by score",
        "category": "db",
        "kind": "db_query",
        # Source of /copytrading/leaders dashboard page + Kelly sizing.
        # NULL score = discovered but not yet scored (leader_scorer
        # didn't have enough history). Empty result = either migration
        # not applied OR no refresh has run yet.
        "sql": (
            "SELECT chain, "
            "  substring(wallet_address, 1, 12) || '...' AS wallet_short, "
            "  source, "
            "  ROUND(score::numeric, 2) AS score, "
            "  ROUND(sharpe_30d::numeric, 3) AS sharpe_30d, "
            "  ROUND(hit_rate::numeric, 3) AS hit_rate, "
            "  ROUND(kelly_fraction::numeric, 4) AS kelly, "
            "  last_scored_at "
            "FROM copy_leader_scores "
            "WHERE score IS NOT NULL "
            "ORDER BY score DESC NULLS LAST "
            "LIMIT 10"
        ),
        "cmd_preview": (
            "SELECT top-10 by score from copy_leader_scores"
        ),
        "timeout_s": 10,
        "description": (
            "Top-10 ranked leaders fed to the /copytrading/leaders "
            "page and copy_engine's per-leader Kelly sizing. Empty = "
            "either migration 024 not applied or no discovery refresh "
            "yet - run api_copytrading_leaders_refresh to populate."
        ),
    },
    {
        "id": "db_copy_leader_scores_by_source",
        "title": "DB: copy_leader_scores discovery-source breakdown",
        "category": "db",
        "kind": "db_query",
        # wallet_discovery.py pulls from 5 sources: dexscreener,
        # birdeye, gmgn, helius, manual. Skew towards one source =
        # likely the others are rate-limited or missing API keys.
        "sql": (
            "SELECT source, "
            "  COUNT(*) AS rows, "
            "  COUNT(*) FILTER (WHERE score IS NOT NULL) AS scored, "
            "  ROUND(AVG(score)::numeric, 2) AS avg_score, "
            "  MAX(last_scored_at) AS newest "
            "FROM copy_leader_scores "
            "GROUP BY source "
            "ORDER BY rows DESC"
        ),
        "cmd_preview": "GROUP BY source on copy_leader_scores",
        "timeout_s": 10,
        "description": (
            "Per-source row + scored count from wallet_discovery's "
            "5-source sweep (dexscreener, birdeye, gmgn, helius, "
            "manual). Heavy skew = the other sources are rate-limited "
            "or missing API keys in secrets_manager."
        ),
    },
    {
        "id": "api_copytrading_leaders_list",
        "title": "API: GET /api/copytrading/leaders (ranked top-N)",
        "category": "api",
        "kind": "probe",
        "endpoint": "copytrading/leaders?limit=10",
        "cmd_preview": "GET /api/copytrading/leaders?limit=10",
        "timeout_s": 15,
        "description": (
            "Cached ranked-leader list. NEVER hits the network - "
            "discovery refresh is a separate admin POST so paid quotas "
            "aren't burned on dashboard reload. 200 + leaders[] array "
            "= page will render; empty leaders[] is normal pre-refresh."
        ),
    },
    {
        "id": "api_copytrading_leaders_refresh",
        "title": "API: POST /api/copytrading/leaders/refresh — route exists (expect 405)",
        "category": "api",
        "kind": "probe",
        # Probe kind is GET-only in this catalog, so we use the GET form
        # to verify the route exists. Expected response is HTTP 405
        # (Method Not Allowed) - that proves the route was registered
        # with add_post only. A 404 means the route was never
        # registered (regression). A 403 means require_admin rejected
        # the caller (also healthy).
        "endpoint": "copytrading/leaders/refresh",
        "cmd_preview": "GET probes POST route (expect HTTP 405 or 403)",
        "timeout_s": 15,
        "description": (
            "Probes route registration for the admin-only POST "
            "/api/copytrading/leaders/refresh. HTTP 405 = route "
            "registered, GET correctly rejected. HTTP 404 = route "
            "missing (regression). HTTP 403 = caller is not admin "
            "(also fine - proves require_admin wrap). Operator runs "
            "the actual sweep via curl with X-CSRF-Token."
        ),
    },
    {
        "id": "db_copy_leader_scores_post_refresh",
        "title": "DB: copy_leader_scores rows post-refresh (CT freshness)",
        "category": "db",
        "kind": "db_query",
        # Companion to api_copytrading_leaders_refresh - operator runs
        # the refresh POST, then this probe to confirm rows landed.
        # mock=true via the API yields synthetic rows; mock=false hits
        # paid Helius/Birdeye quotas. Either way the row-count should
        # bump and last_scored_at should be < 5 min old.
        "sql": (
            "SELECT chain, source, COUNT(*) AS rows, "
            "  COUNT(*) FILTER (WHERE last_scored_at > NOW() - INTERVAL '5 minutes') AS fresh_5m, "
            "  MAX(last_scored_at) AS most_recent "
            "FROM copy_leader_scores "
            "GROUP BY chain, source "
            "ORDER BY most_recent DESC NULLS LAST"
        ),
        "cmd_preview": (
            "SELECT chain,source,COUNT(*),fresh_5m FROM copy_leader_scores"
        ),
        "timeout_s": 10,
        "description": (
            "Run AFTER POST /api/copytrading/leaders/refresh. fresh_5m > 0 "
            "for the chains in the refresh body = sweep wrote rows. "
            "rows but fresh_5m=0 = previous refresh, no new sweep. "
            "Empty = refresh never ran or DiscoveryConfig errored out."
        ),
    },

    # ════════════════════════════════════════════════════════════════════
    # Wave-4 T1 catalog additions: DEX / SOLANA / COPY_TRADING coverage
    # for the commits enumerated in the T1-W4 brief.
    # ════════════════════════════════════════════════════════════════════

    # ── COPY_TRADING (A7 wave-4: CT-Q-09 probation + CT-Q-12 exposure) ──
    {
        "id": "db_copy_probation_thresholds",
        "title": "DB: COPY probation-gate thresholds (CT-Q-09, mig 030)",
        "category": "db",
        "kind": "db_query",
        # Migration 030_copy_probation_gate_defaults.sql seeds the four
        # copy_probation_* knobs the engine BUY path consults. The
        # engine also accepts bare ('probation_*') keys; this probe
        # surfaces both so the operator can see which form is active.
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key IN ("
            "  'copy_probation_gate_enabled','probation_gate_enabled',"
            "  'copy_probation_score_threshold','probation_score_threshold',"
            "  'copy_probation_loss_pct_threshold','probation_loss_pct_threshold',"
            "  'copy_probation_days','probation_days'"
            ") "
            "ORDER BY key, config_type"
        ),
        "cmd_preview": (
            "SELECT … WHERE key IN copy_probation_* / probation_* knobs"
        ),
        "timeout_s": 10,
        "description": (
            "CT-Q-09 probation-gate config. Defaults: enabled=true, "
            "score<30 + >=10 trades auto-benches, mirrored-trade loss "
            "worse than -25% auto-benches, bench duration 7 days. "
            "Empty result = mig 030 not applied; engine falls back to "
            "_load_settings dataclass defaults (still safe)."
        ),
    },
    {
        "id": "db_copy_probation_state",
        "title": "DB: COPY leaders currently on probation (CT-Q-09)",
        "category": "db",
        "kind": "db_query",
        # copy_engine._is_leader_on_probation reads on_probation +
        # probation_until from copy_leader_scores (cols added in mig
        # 026). BUY path refuses with `[replay] reason=probation` when
        # any leader is benched. This probe lists who, why, until-when.
        "sql": (
            "SELECT chain, "
            "  substring(wallet_address, 1, 12) || '...' AS wallet_short, "
            "  on_probation, "
            "  probation_until, "
            "  probation_reason "
            "FROM copy_leader_scores "
            "WHERE on_probation = TRUE "
            "ORDER BY probation_until DESC NULLS LAST "
            "LIMIT 50"
        ),
        "cmd_preview": (
            "SELECT chain,wallet,on_probation,probation_until,probation_reason"
        ),
        "timeout_s": 10,
        "description": (
            "Currently benched leaders. Empty = nobody on probation "
            "(all leaders eligible to mirror). Non-empty rows = engine "
            "will refuse BUY broadcasts for these wallets until "
            "probation_until passes; SELLs are NEVER gated. "
            "probation_reason values: 'score<threshold', "
            "'closed_trade_loss>threshold'."
        ),
    },
    {
        "id": "db_copy_exposure_breakdown",
        "title": "DB: COPY cross-module exposure breakdown per chain (CT-Q-12)",
        "category": "db",
        "kind": "db_query",
        # Mirrors what modules/copy_trading/exposure_aggregator.py
        # sums at runtime. Per-chain UNION of open-position USD across
        # DEX (trades.usd_value), SNIPER, AI, COPY (entry_usd) -- the
        # SOLANA positions table has no entry_usd column yet (carry-
        # over noted in COPY_TRADING/CLAUDE.md) so it appears as a
        # zero row to make the gap visible.
        "sql": (
            "SELECT 'dex' AS module, chain, "
            "  COUNT(*) AS open_positions, "
            "  ROUND(COALESCE(SUM(usd_value),0)::numeric, 2) AS open_usd "
            "FROM trades WHERE status='open' GROUP BY chain "
            "UNION ALL "
            "SELECT 'sniper', chain, COUNT(*), "
            "  ROUND(COALESCE(SUM(entry_usd),0)::numeric, 2) "
            "FROM sniper_trades WHERE status='open' GROUP BY chain "
            "UNION ALL "
            "SELECT 'ai', chain, COUNT(*), "
            "  ROUND(COALESCE(SUM(entry_usd),0)::numeric, 2) "
            "FROM ai_trades WHERE status='open' GROUP BY chain "
            "UNION ALL "
            "SELECT 'copy', chain, COUNT(*), "
            "  ROUND(COALESCE(SUM(entry_usd),0)::numeric, 2) "
            "FROM copytrading_trades WHERE status='open' GROUP BY chain "
            "ORDER BY module, chain"
        ),
        "cmd_preview": (
            "UNION ALL open-position USD by module across DEX/SNIPER/AI/COPY"
        ),
        "timeout_s": 15,
        "description": (
            "Per-module per-chain open exposure. Sums the same tables "
            "exposure_aggregator.get_exposure_usd consults at BUY "
            "time. Any chain whose total exceeds "
            "copy_cross_module_exposure_cap_usd (default $5000) will "
            "see COPY BUYs refused with [replay] reason=cross_module_"
            "cap. SOLANA noop until solana_positions.entry_usd ships."
        ),
    },
    {
        "id": "db_copy_cross_module_cap",
        "title": "DB: COPY cross-module exposure cap config (CT-Q-12)",
        "category": "db",
        "kind": "db_query",
        # Migration 030 seeds the two CT-Q-12 flags. copy_engine.
        # _check_cross_module_exposure consults both: the enable flag
        # gates the lookup, the cap_usd compares against
        # existing + intended.
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key IN ("
            "  'copy_cross_module_exposure_check_enabled',"
            "  'cross_module_exposure_check_enabled',"
            "  'copy_cross_module_exposure_cap_usd',"
            "  'cross_module_exposure_cap_usd'"
            ") "
            "ORDER BY key, config_type"
        ),
        "cmd_preview": (
            "SELECT … WHERE key IN copy_cross_module_exposure_* knobs"
        ),
        "timeout_s": 10,
        "description": (
            "CT-Q-12 cross-module exposure-cap config. Defaults: "
            "check_enabled=true, cap_usd=5000. Empty = mig 030 not "
            "applied; engine falls back to dataclass defaults. "
            "Disabling the check removes the cross-module safety net; "
            "per-module max_copy_amount + max_active_positions stay."
        ),
    },

    # ── DEX (A1 wave-4: 115cb35 V3 impact, 6f96075 bloXroute BSC) ──────
    {
        "id": "db_dex_v3_quoter_addresses",
        "title": "DB: DEX V3 QuoterV2 per-chain address overrides (115cb35)",
        "category": "db",
        "kind": "db_query",
        # The chunked-probe V3 impact path (wave-4) uses the same
        # `_quote_v3` -> QuoterV2 helper added in wave-3. Operators
        # CAN override the per-chain quoter address via the
        # `v3_quoter_addresses` config key (one row whose value is a
        # JSON dict). Empty result = no override; the hard-coded
        # UNISWAP_V3_QUOTER_V2_ADDRESSES literal in direct_dex.py
        # (asserted by script_dex_quoter_v2_addresses_present) is the
        # full source of truth. Non-empty rows = operator pointed at a
        # fork (SushiSwap V3, PancakeSwap V3, etc.) and the override
        # value should be valid JSON parseable as a chain->addr dict.
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key = 'v3_quoter_addresses' "
            "ORDER BY config_type"
        ),
        "cmd_preview": "SELECT … WHERE key='v3_quoter_addresses'",
        "timeout_s": 10,
        "description": (
            "Per-chain V3 QuoterV2 address overrides. Empty = engine "
            "uses the hard-coded UNISWAP_V3_QUOTER_V2_ADDRESSES map "
            "(eth/poly/arb/base/op/bsc). Non-empty = operator pointed "
            "at a fork quoter; value should be a JSON dict keyed by "
            "chain. Companion to script_dex_quoter_v2_addresses_present."
        ),
    },
    {
        "id": "db_dex_bloxroute_config",
        "title": "DB: DEX bloXroute BSC private-tx config (6f96075)",
        "category": "db",
        "kind": "db_query",
        # Wave-4 wired a real BSC private-tx send. Two operator-tunable
        # config rows: bloxroute_enabled (gate, default false) and
        # bloxroute_bsc_endpoint (default https://api.blxrbdn.com).
        # The auth header is a SECRET (loaded from secrets_manager or
        # BLOXROUTE_AUTH_HEADER env) and is NEVER read here.
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key IN ('bloxroute_enabled','bloxroute_bsc_endpoint') "
            "ORDER BY config_type, key"
        ),
        "cmd_preview": (
            "SELECT … WHERE key IN ('bloxroute_enabled','bloxroute_bsc_endpoint')"
        ),
        "timeout_s": 10,
        "description": (
            "bloXroute BSC private-tx config. Default off (enabled="
            "false). When flipped on, mev_protection routes BSC swaps "
            "via blxr_private_tx instead of public mempool. Endpoint "
            "default https://api.blxrbdn.com. Auth header is a secret "
            "(not surfaced here); missing header => fallback to "
            "public-mempool path. Ethereum still routes via Flashbots."
        ),
    },
    {
        "id": "script_dex_quoter_v2_addresses_present",
        "title": "Script: DEX QuoterV2 per-chain map presence (115cb35)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/dex_quoter_v2_addresses_check.sh"],
        "cmd_preview": "bash scripts/dex_quoter_v2_addresses_check.sh",
        "timeout_s": 10,
        "description": (
            "Source-grep: asserts UNISWAP_V3_QUOTER_V2_ADDRESSES in "
            "trading/executors/direct_dex.py has entries for "
            "ethereum/polygon/arbitrum/base/optimism/bsc. Missing "
            "chain rows = the chunked V3 impact probe returns None "
            "and the max_price_impact_bps refusal gate silently "
            "no-ops on that chain."
        ),
    },

    # ── SOLANA (A3 wave-4: d6a4a8c Jito wiring, 3edd27e warmup) ────────
    {
        "id": "db_solana_jito_flag",
        "title": "DB: SOLANA Jito-bundle enable + tip config (d6a4a8c)",
        "category": "db",
        "kind": "db_query",
        # Two rows under config_type='solana_jupiter' (see
        # modules/solana_trading/config/solana_config_manager.py
        # CONFIG_KEY_MAPPING). Flag default False, tip default
        # 50_000 lamports (~$0.01 @ SOL=$200 -- documented competitive
        # floor per Jito ops doc; arbitrage's 10k default lands less
        # reliably during peak hours). Empty = engine falls back to
        # dataclass defaults (still safe; flag stays off).
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key IN ("
            "  'solana_jito_bundle_enabled',"
            "  'solana_jito_tip_lamports'"
            ") "
            "ORDER BY key"
        ),
        "cmd_preview": (
            "SELECT … WHERE key IN ('solana_jito_bundle_enabled',"
            "'solana_jito_tip_lamports')"
        ),
        "timeout_s": 10,
        "description": (
            "Jito bundle path config. Defaults: enabled=false, "
            "tip=50000 lamports. When flipped on, engine submits "
            "signed Jupiter swap + tip via JitoClient.send_bundle "
            "and falls back to vanilla execute_swap on rejection / "
            "rate-limit / timeout. Watch logs/solana_trading/ for "
            "SEND / LANDED / REJECTED / SKIPPED / fell-back lines."
        ),
    },
    {
        "id": "db_solana_pump_predictor_flag",
        "title": "DB: SOLANA pump-predictor enable flag (3edd27e)",
        "category": "db",
        "kind": "db_query",
        # Wave-4 pump-predictor warmup pre-fills the per-token
        # TokenPriceBuffer at engine startup so the gate has 60 mins
        # of history from minute 0. The gate itself stays opt-in via
        # this flag (default False). When operator flips it on, the
        # gate works immediately instead of dropping every entry for
        # 30 min waiting for the buffer to fill.
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key = 'solana_pump_predictor_enabled' "
            "ORDER BY config_type"
        ),
        "cmd_preview": (
            "SELECT … WHERE key='solana_pump_predictor_enabled'"
        ),
        "timeout_s": 10,
        "description": (
            "Pump-predictor gate flag. Default false. Setting true "
            "activates the warmup-prefilled price buffer as a "
            "rug/pump filter on entries. Empty result = engine reads "
            "config_manager default (false). Companion to the "
            "warmup pre-fetch which seeds 60x 1m Birdeye bars when "
            "BIRDEYE_API_KEY is present, else 1 Jupiter spot bar."
        ),
    },
    {
        "id": "script_solana_jito_helper_present",
        "title": "Script: SOLANA _execute_swap_via_jito helper presence (d6a4a8c)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/solana_jito_helper_check.sh"],
        "cmd_preview": "bash scripts/solana_jito_helper_check.sh",
        "timeout_s": 10,
        "description": (
            "Source-grep: asserts modules/solana_trading/core/"
            "solana_engine.py defines _execute_swap_via_jito() AND "
            "_open_position calls it AND JitoClient is imported. "
            "Catches a regression that quietly turns "
            "solana_jito_bundle_enabled=True into a no-op fall-back "
            "to vanilla Jupiter swaps with no MEV protection."
        ),
    },

    # ── Wave-5 diagnostic endpoints (W5-ARB / W5-AI 'Why no trades?' panels) ─
    # Both endpoints surface skip-reason ledgers + current config; PM_FINAL_W5
    # flagged the missing probes as a Wave-6 carry-over — closed here instead.
    {
        "id": "api_arb_diagnostics",
        "title": "API: GET /api/arbitrage/diagnostics — Why no trades? (W5 012887a)",
        "category": "api",
        "kind": "probe",
        "endpoint": "arbitrage/diagnostics",
        "cmd_preview": "GET /api/arbitrage/diagnostics | .near_misses + .cost_profile",
        "timeout_s": 15,
        "tags": ["new", "must"],
        "description": (
            "Wave-5 ARB diagnostic surface. Returns per-chain cost profile, "
            "near-miss counters by reason, last 20 rejected opportunities, "
            "last 10 trades, gas-spend snapshot, and a stale flag when the "
            "engine hasn't written a runtime-stats row in >10 min. Use this "
            "before assuming the engine is dead: empty trade list + healthy "
            "near-miss counters means the engine is alive but every "
            "opportunity is below profit threshold."
        ),
    },
    {
        "id": "api_ai_diagnostics",
        "title": "API: GET /api/ai/diagnostics — Why no trades? (W5 71988d2)",
        "category": "api",
        "kind": "probe",
        "endpoint": "ai/diagnostics",
        "cmd_preview": "GET /api/ai/diagnostics | .skip_reasons + .effective_config",
        "timeout_s": 15,
        "tags": ["new", "must"],
        "description": (
            "Wave-5 AI diagnostic surface. Returns signal/trade counts, skip "
            "reasons grouped by enum (direct_trading_off / confidence_below "
            "/ quorum_disagree / position_cap / risk_gate / cooldown), the "
            "redacted effective config (confidence_threshold, max_positions, "
            "quorum_required, direct_trading), and a 512KB-bounded log tail. "
            "ROOT-CAUSE TOOL: operator saw '50 signals 0 trades'; this "
            "endpoint immediately reveals direct_trading=false as the gate."
        ),
    },
    {
        "id": "db_arb_near_miss_counters",
        "title": "DB: ARB near-miss counters (W5 d2e1019)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT chain, "
            "  jsonb_pretty(stats->'near_miss_counters') AS counters, "
            "  jsonb_array_length(COALESCE(stats->'near_misses', '[]'::jsonb)) AS recent_n, "
            "  EXTRACT(EPOCH FROM (NOW() - updated_at))::int AS age_seconds "
            "FROM arbitrage_runtime_stats "
            "ORDER BY chain"
        ),
        "cmd_preview": (
            "near_miss_counters JSONB + last-update age from arbitrage_runtime_stats"
        ),
        "timeout_s": 10,
        "tags": ["new", "must"],
        "description": (
            "Per-chain breakdown of why arb opportunities were rejected "
            "(daily_cap / cooldown / gas_budget / risk_manager / "
            "raw_spread_negative / min_profit). age_seconds > 600 = the "
            "arbitrage subprocess hasn't snapshotted recently (likely dead "
            "or paused). Companion to /api/arbitrage/diagnostics."
        ),
    },
    {
        "id": "db_copy_unrealized_pnl_open",
        "title": "DB: COPY open positions w/ tokens_received (W5 6fe0a36)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT trade_id, source_wallet, "
            "  token_address, "
            "  entry_usd, "
            "  ROUND(entry_price::numeric, 6) AS entry_price_usd_per_token, "
            "  ROUND(amount::numeric, 4) AS tokens, "
            "  CASE WHEN metadata::jsonb ? 'tokens_received' THEN 'yes' ELSE 'no — legacy' END AS has_tokens_received, "
            "  CASE WHEN COALESCE((metadata::jsonb->>'tokens_received_approx')::bool, FALSE) "
            "       THEN 'backfilled (approx)' ELSE 'native' END AS source "
            "FROM copytrading_trades "
            "WHERE status = 'open' AND chain = 'solana' "
            "ORDER BY entry_timestamp DESC LIMIT 25"
        ),
        "cmd_preview": "Open Solana COPY rows: entry_price + tokens + has_tokens_received",
        "timeout_s": 10,
        "tags": ["new", "must", "p0"],
        "description": (
            "After 6fe0a36 + backfill_copy_tokens_received.py --force, "
            "every OPEN Solana row should have has_tokens_received='yes' "
            "and entry_price_usd_per_token <<< 1.0 for memecoins (or "
            "≈ 1.0 for USDC). If entry_price still shows 80-200 (SOL "
            "range), the operator hasn't run the backfill yet."
        ),
    },
    {
        "id": "db_futures_default_leverage_post_mig031",
        "title": "DB: FUTURES default leverage (post mig 031 — should be 5x)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, key, value "
            "FROM config_settings "
            "WHERE config_type = 'futures_leverage' "
            "  AND key IN ('default_leverage', 'max_leverage') "
            "ORDER BY key"
        ),
        "cmd_preview": "futures_leverage.default_leverage + max_leverage after mig 031",
        "timeout_s": 10,
        "tags": ["new", "p0"],
        "description": (
            "Migration 031 (Wave-5 FUT-RM-18) lowered default_leverage "
            "from 10 to 5. If default_leverage still reads 10, the "
            "migration didn't apply (re-pull and check trading-bot logs)."
        ),
    },
    {
        "id": "db_futures_atr_dynamic_flag",
        "title": "DB: FUTURES ATR dynamic SL/TP flag (W5 FUT-RM-16)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT config_type, key, value FROM config_settings "
            "WHERE key IN ("
            "  'futures_atr_dynamic_sl_tp_enabled', "
            "  'futures_min_signal_confluence_count', "
            "  'futures_post_loss_cooloff_minutes' "
            ") ORDER BY key"
        ),
        "cmd_preview": "Wave-5 FUTURES risk knobs (ATR / confluence / cool-off)",
        "timeout_s": 10,
        "tags": ["new"],
        "description": (
            "Verifies Wave-5 risk-tightening knobs are present: ATR-scaled "
            "SL/TP (default true), confluence count (default 2), post-loss "
            "cool-off minutes (default 240 = 4h). Missing rows = engine "
            "falls back to module defaults (which are the same values, so "
            "missing is non-fatal — just means operator hasn't tuned)."
        ),
    },
    # ── DASHBOARD (W6 timezone-helper) ───────────────────────────────────
    {
        "id": "script_timezone_helper_present",
        "title": "Script: DASHBOARD timezone helper presence (W6 2500f5f)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/timezone_helper_check.sh"],
        "cmd_preview": "bash scripts/timezone_helper_check.sh",
        "timeout_s": 10,
        "tags": ["new"],
        "description": (
            "Asserts dashboard/static/js/timezone.js exists, exports "
            "window.parseUtcTimestamp / formatLocalDateTime / "
            "formatTimeAgo, AND that base.html loads the script. The "
            "server emits naive UTC ISO timestamps; without these "
            "helpers browser JS parses them as LOCAL time and shifts "
            "displays by the operator's UTC offset (UTC+3 -> '3h ago' "
            "on fresh rows). Operator-reported regression 2026-05-21."
        ),
    },

    # ════════════════════════════════════════════════════════════════════
    # Wave-6 T1-W6 catalog additions — AI key-resolution + COPY BUY/SELL
    # detector rewrite + ARB engine-health surface + dashboard/futures
    # operator-reported fixes. Commits cca8d94 / 3939a20 / b675ab1 /
    # 49400d6 / f81144f / 2d30f3c / 2466ec0 / f0fb2bd / 0770912 /
    # a2849b7 / e32cf63 / 5a857e0. See docs/agents/CAMPAIGN_BRIEF.md.
    # ════════════════════════════════════════════════════════════════════

    # ── AI (W6 cca8d94 / 3939a20 / b675ab1 / 49400d6) ────────────────────
    {
        "id": "api_ai_diagnostics_subprocess_health",
        "title": "API: /api/ai/diagnostics — subprocess_health surface (W6 49400d6)",
        "category": "api",
        "kind": "probe",
        "endpoint": "ai/diagnostics",
        "cmd_preview": "GET /api/ai/diagnostics | .subprocess_health",
        "timeout_s": 15,
        "tags": ["new", "must", "p0"],
        "description": (
            "Wave-6 49400d6 added a subprocess_health block to the AI "
            "diagnostics payload so the dashboard's Offline badge can "
            "distinguish 'subprocess crashed' from 'alive but not signalling'. "
            "Response MUST carry subprocess_health.last_sentiment_tick_at "
            "(set by sentiment_engine._last_tick_at on every loop "
            "iteration). null = subprocess never ticked since boot."
        ),
    },
    {
        "id": "script_ai_secrets_after_db_pool",
        "title": "Script: AI main_ai secrets-after-db-pool ordering (W6 cca8d94)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/ai_secrets_after_db_pool_check.sh"],
        "cmd_preview": "bash scripts/ai_secrets_after_db_pool_check.sh",
        "timeout_s": 10,
        "tags": ["new", "must", "p0"],
        "description": (
            "Root-cause guard for 'AI signals but 0 trades': asserts "
            "main_ai.py calls asyncpg.create_pool(...) FIRST, then "
            "_secrets.initialize(db_pool), then _secrets.get_async("
            "OPENAI_API_KEY / ANTHROPIC_API_KEY). Any reorder leaves "
            "secrets_manager in bootstrap mode and ai_provider boots "
            "with empty keys, silently skipping every tick."
        ),
    },
    {
        "id": "script_ai_per_tick_liveness_log",
        "title": "Script: AI sentiment_engine per-tick liveness log (W6 b675ab1)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/ai_per_tick_liveness_log_check.sh"],
        "cmd_preview": "bash scripts/ai_per_tick_liveness_log_check.sh",
        "timeout_s": 10,
        "tags": ["new"],
        "description": (
            "Asserts modules/ai_analysis/core/sentiment_engine.py emits "
            "the 'sentiment cycle tick' liveness log line at the top "
            "of every run() iteration AND stamps self._last_tick_at. "
            "These feed the subprocess_health.last_sentiment_tick_at "
            "surface; dropping either re-introduces the 'looks alive "
            "but no trades' diagnostic dead end."
        ),
    },
    {
        "id": "db_ai_stats_health_keys",
        "title": "DB: AI provider keys in secure_credentials / config_sensitive (W6 cca8d94)",
        "category": "db",
        "kind": "db_query",
        # Wave-6 the AI provider keys live ENCRYPTED — either in the
        # newer secure_credentials table (key_name col, migration 012)
        # or the older config_sensitive table (key col, migration 002).
        # We don't surface the plaintext value — only whether an
        # encrypted row exists per provider so the operator can confirm
        # the migration ran and the secrets_manager will resolve.
        "sql": (
            "SELECT 'secure_credentials' AS src, key_name AS k, "
            "  CASE WHEN encrypted_value IS NOT NULL "
            "       AND length(encrypted_value) > 0 "
            "       THEN 'present' ELSE 'missing' END AS status, "
            "  updated_at "
            "FROM secure_credentials "
            "WHERE key_name IN ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY') "
            "UNION ALL "
            "SELECT 'config_sensitive', key, "
            "  CASE WHEN encrypted_value IS NOT NULL "
            "       AND length(encrypted_value) > 0 "
            "       THEN 'present' ELSE 'missing' END, "
            "  updated_at "
            "FROM config_sensitive "
            "WHERE key IN ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY') "
            "ORDER BY src, k"
        ),
        "cmd_preview": (
            "secure_credentials + config_sensitive WHERE key IN "
            "('OPENAI_API_KEY','ANTHROPIC_API_KEY')"
        ),
        "timeout_s": 10,
        "tags": ["new"],
        "description": (
            "Surfaces whether OPENAI_API_KEY + ANTHROPIC_API_KEY rows "
            "exist (encrypted) in either secure_credentials or "
            "config_sensitive. Empty result = secrets_manager DB path "
            "has nothing to resolve and ai_provider falls back to env. "
            "Companion to script_ai_secrets_after_db_pool which checks "
            "the call ordering, not the data."
        ),
    },
    # ── COPY (W6 f81144f / 2d30f3c / 2466ec0 / f0fb2bd / 0770912) ────────
    {
        "id": "script_copy_stablecoin_guard_present",
        "title": "Script: COPY stablecoin guard + open-position check (W6 f81144f)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/copy_stablecoin_guard_check.sh"],
        "cmd_preview": "bash scripts/copy_stablecoin_guard_check.sh",
        "timeout_s": 10,
        "tags": ["new", "must", "p0"],
        "description": (
            "Source-grep: asserts modules/copy_trading/copy_engine.py "
            "declares STABLECOIN_MINTS + EVM_STABLECOIN_ADDRESSES sets, "
            "the _has_open_copy_position SELL-side gate, AND raises the "
            "stablecoin_not_tradeable refusal label. Missing any one "
            "re-introduces ghost SELLs against USDC/USDT/DAI mints."
        ),
    },
    {
        "id": "script_copy_buy_sell_detector_rewrite",
        "title": "Script: COPY delta-based BUY/SELL detector (W6 2d30f3c)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/copy_buy_sell_detector_check.sh"],
        "cmd_preview": "bash scripts/copy_buy_sell_detector_check.sh",
        "timeout_s": 10,
        "tags": ["new", "must", "p0"],
        "description": (
            "Source-grep: asserts copy_engine.py partitions balance "
            "deltas into base_deltas (non-stablecoin mints) vs "
            "quote_deltas (stablecoin mints) — the W6 rewrite that "
            "correctly classifies a USDC->memecoin swap as a BUY of "
            "the memecoin. Companion to script_copy_stablecoin_guard_"
            "present which checks the constant set itself."
        ),
    },
    {
        "id": "db_copy_stablecoin_refusals_recent",
        "title": "DB: COPY stuck stablecoin rows (pre-W6 era leftovers)",
        "category": "db",
        "kind": "db_query",
        # USDC + USDT Solana mints. After f81144f + 2d30f3c the engine
        # refuses any new BUY where token_address is in EVM_STABLECOIN_
        # ADDRESSES or STABLECOIN_MINTS; the 5 pre-fix open rows stay
        # in the DB until an operator manually closes them. This probe
        # surfaces those rows so the operator can see they exist AND
        # confirm no NEW rows are landing after Wave-6 deploy.
        "sql": (
            "SELECT trade_id, "
            "  substring(source_wallet, 1, 12) || '...' AS leader, "
            "  token_address, "
            "  status, "
            "  entry_timestamp "
            "FROM copytrading_trades "
            "WHERE token_address IN ("
            "  'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',"
            "  'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB'"
            ") "
            "  AND status = 'open' "
            "ORDER BY entry_timestamp DESC "
            "LIMIT 20"
        ),
        "cmd_preview": (
            "SELECT trade_id, leader, status FROM copytrading_trades "
            "WHERE token_address IN (USDC_mint, USDT_mint) AND status='open'"
        ),
        "timeout_s": 10,
        "tags": ["new"],
        "description": (
            "Wave-6 stablecoin-guard regression check. Pre-fix the "
            "engine opened 5 ghost rows against USDC/USDT mints. After "
            "f81144f + 2d30f3c those rows should NOT grow — re-run "
            "this probe after a few hours of live traffic; the row "
            "count must be stable. Pre-existing rows stay open until "
            "the operator closes them via /api/copytrading/close."
        ),
    },
    {
        "id": "script_copy_trade_ui_icons_present",
        "title": "Script: COPY trades-page UI icons (W6 2466ec0 / f0fb2bd)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/copy_trade_ui_icons_check.sh"],
        "cmd_preview": "bash scripts/copy_trade_ui_icons_check.sh",
        "timeout_s": 10,
        "tags": ["new"],
        "description": (
            "Grep test: trades_copytrading.html ships the W6 UI "
            "enrichment - copyTokenAddress() copy-to-clipboard, "
            "birdeye.so/token chart link, solscan.io/token explorer "
            "link. Template refactors that drop any silently regress "
            "the one-click jump from the trade row to the explorer."
        ),
    },
    # ── ARB (W6 a2849b7 / e32cf63) ───────────────────────────────────────
    {
        "id": "db_arb_runtime_health",
        "title": "DB: ARB engine health (last_tick_at + last_error, W6 a2849b7)",
        "category": "db",
        "kind": "db_query",
        # Wave-6 a2849b7 added last_tick_at + last_error + last_error_at
        # to arbitrage_runtime_stats.stats so the operator can
        # distinguish "subprocess dead" from "alive but rejecting".
        # age_seconds > 600 (10m) with last_tick_at null = dead;
        # age_seconds low + last_error populated = alive but erroring.
        "sql": (
            "SELECT chain, "
            "  EXTRACT(EPOCH FROM (NOW() - updated_at))::int AS age_seconds, "
            "  stats->>'last_tick_at' AS last_tick, "
            "  stats->>'last_error' AS last_error, "
            "  stats->>'last_error_at' AS last_error_at "
            "FROM arbitrage_runtime_stats "
            "ORDER BY chain"
        ),
        "cmd_preview": (
            "last_tick_at + last_error + age_seconds per chain from "
            "arbitrage_runtime_stats"
        ),
        "timeout_s": 10,
        "tags": ["new", "must"],
        "description": (
            "Wave-6 ARB health surface. Distinguishes dead-engine "
            "(age_seconds > 600 AND last_tick is null) from "
            "alive-but-rejecting (low age + populated last_error). "
            "Companion to /api/arbitrage/diagnostics; the operator "
            "can trigger a restart with `touch logs/.restart_arbitrage` "
            "which main.py picks up within 5s."
        ),
    },
    {
        "id": "script_arb_engine_health_script_present",
        "title": "Script: ARB engine-health operator diagnostic (W6 e32cf63)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/arb_engine_health_script_check.sh"],
        "cmd_preview": "bash scripts/arb_engine_health_script_check.sh",
        "timeout_s": 10,
        "tags": ["new"],
        "description": (
            "Asserts scripts/arb_engine_health.py exists, is executable, "
            "and queries arbitrage_runtime_stats. The script is the "
            "single-command operator diagnostic for the 'ARB engine "
            "silent' symptom (stale runtime stats, last_trades months "
            "old, near_miss_counters returning 0 rows). Suggests the "
            "next action — typically restart via the W6 flag-file "
            "pattern `touch logs/.restart_arbitrage`."
        ),
    },
    # ── Foreground fixes (W6 5a857e0) — close-button, CSRF, hot-wallets,
    #    futures W5 knobs surfaced on settings_futures.html.
    {
        "id": "script_close_button_endpoint_dual_table",
        "title": "Script: COPY close-button dual-table fallback (W6 5a857e0)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/copy_close_button_dual_table_check.sh"],
        "cmd_preview": "bash scripts/copy_close_button_dual_table_check.sh",
        "timeout_s": 10,
        "tags": ["new", "must"],
        "description": (
            "Source-grep: asserts copy_engine.py _process_close_flag_"
            "files declares the from_trades_table fallback so rows that "
            "live ONLY in copytrading_trades (legacy soft-delete) are "
            "still closeable via the dashboard's Close button. Operator-"
            "reported regression closed in 5a857e0."
        ),
    },
    {
        "id": "script_reconcile_csrf_header",
        "title": "Script: COPY reconcile button CSRF header (W6 5a857e0)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/copy_reconcile_csrf_check.sh"],
        "cmd_preview": "bash scripts/copy_reconcile_csrf_check.sh",
        "timeout_s": 10,
        "tags": ["new"],
        "description": (
            "Grep test: dashboard_copytrading.html wraps the "
            "/api/copytrading/reconcile POST through "
            "window.withCsrfHeaders('POST') so the token is attached "
            "automatically. Operator-reported '403 on Reconcile' "
            "regression closed in 5a857e0."
        ),
    },
    {
        "id": "script_dashboard_hot_wallets_filter_sort",
        "title": "Script: DASHBOARD hot-wallets active filter + sort (W6 5a857e0)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/dashboard_hot_wallets_filter_check.sh"],
        "cmd_preview": "bash scripts/dashboard_hot_wallets_filter_check.sh",
        "timeout_s": 10,
        "tags": ["new"],
        "description": (
            "Grep test: discovery_copytrading.html filters the hot-"
            "wallets card to ACTIVE wallets only (>0 trades OR nonzero "
            "PnL) and sorts by trades DESC, PnL DESC. Operator-"
            "reported 'card is full of 0-trade noise' regression "
            "closed in 5a857e0."
        ),
    },
    {
        "id": "script_futures_settings_w5_knobs_present",
        "title": "Script: FUTURES settings W5 risk knobs (W6 5a857e0)",
        "category": "scripts",
        "kind": "bash",
        "cmd": ["bash", "scripts/futures_settings_w5_knobs_check.sh"],
        "cmd_preview": "bash scripts/futures_settings_w5_knobs_check.sh",
        "timeout_s": 10,
        "tags": ["new"],
        "description": (
            "Grep test: settings_futures.html exposes the three W5 "
            "risk knobs (futures_min_signal_confluence_count + "
            "futures_atr_dynamic_sl_tp_enabled + "
            "futures_post_loss_cooloff_minutes) so the operator can "
            "tune them without psql. Closes the W5 carry-over flagged "
            "in PM_FINAL_W5."
        ),
    },

    # ── Wave-7 DASHBOARD fixes (issues 18/1/3/2/5+6/15) ───────────────
    {
        "id": "api_full_dashboard_charts_no_500",
        "title": "API: /api/dashboard/charts/full (issue 18 tz crash)",
        "category": "api",
        "kind": "probe",
        "endpoint": "dashboard/charts/full",
        "cmd_preview": "GET /api/dashboard/charts/full",
        "timeout_s": 20,
        "tags": ["new", "p0"],
        "description": (
            "Full-dashboard Performance Analytics chart data. Issue 18: this "
            "endpoint raised 'can't compare offset-naive and offset-aware "
            "datetimes' and returned success=false, blanking all 15 charts. "
            "PASS = HTTP 200 with success:true (no 500, no tz error). "
            "Normalized via _as_utc in _unified_closed_trades."
        ),
    },
    {
        "id": "api_funding_accounts",
        "title": "API: /api/funding/accounts (issue 15 wallet panel)",
        "category": "api",
        "kind": "probe",
        "endpoint": "funding/accounts",
        "cmd_preview": "GET /api/funding/accounts",
        "timeout_s": 20,
        "tags": ["new"],
        "description": (
            "Consolidated 'which wallet/exchange funds which module' panel "
            "source. PASS = HTTP 200 with data.accounts containing all 7 "
            "modules; each reports a public wallet/exchange or 'not "
            "initialized'. Public addresses only — never private keys."
        ),
    },
    {
        "id": "api_performance_charts_all_modules",
        "title": "API: /api/performance/charts (issue 3 all-7-modules)",
        "category": "api",
        "kind": "probe",
        "endpoint": "performance/charts?timeframe=all",
        "cmd_preview": "GET /api/performance/charts?timeframe=all",
        "timeout_s": 20,
        "tags": ["new"],
        "description": (
            "Main-dashboard equity/PnL/strategy charts. Issue 3: this read "
            "only the DEX `trades` table; now built from _unified_closed_trades "
            "so strategy_performance spans all 7 modules. PASS = HTTP 200 "
            "success:true; inspect strategy_performance for non-DEX labels."
        ),
    },
    {
        "id": "api_modules_health_strings",
        "title": "API: /api/modules (issue 1 health fallbacks)",
        "category": "api",
        "kind": "probe",
        "endpoint": "modules",
        "cmd_preview": "GET /api/modules",
        "timeout_s": 20,
        "tags": ["new"],
        "description": (
            "Module health status. Issue 1(a): dex/arb/copy/ai now derive "
            "'ENABLED + RUNNING' from DB-freshness heartbeats when no health "
            "port answers (was always 'ENABLED (no health)'). PASS = HTTP 200; "
            "enabled+active modules should show 'ENABLED + RUNNING'."
        ),
    },
    {
        "id": "db_copy_execution_wallets",
        "title": "DB: copytrading_diagnostics execution wallets (issue 15)",
        "category": "db",
        "kind": "db_query",
        "sql": (
            "SELECT key, value FROM config_settings "
            "WHERE config_type='copytrading_diagnostics' "
            "AND key IN ('evm_execution_wallet','solana_execution_wallet') "
            "ORDER BY key"
        ),
        "cmd_preview": (
            "SELECT key,value FROM config_settings WHERE "
            "config_type='copytrading_diagnostics'"
        ),
        "timeout_s": 15,
        "tags": ["new"],
        "description": (
            "Source rows for the COPY card on the Funding/Accounts panel. "
            "Populated by CopyTradingEngine._persist_execution_wallets once "
            "the copy subprocess initializes (public addresses only). Empty "
            "until the module has run at least once."
        ),
    },

    # ── Wave-8 DEX fixes (heartbeat / ML provenance / state restore) ──
    # These three validate the Wave-8 DEX work. Two preconditions are
    # NOT yet deployed on the operator's VPS at branch-cut and are
    # handled FAIL-SOFT so a clean run isn't littered with red:
    #   * migration 032 (dex_runtime_stats) may be unapplied → the
    #     freshness probe uses to_regclass() and returns an explicit
    #     "run migration 032" row instead of a hard SQL error.
    #   * the ML ensemble is in honest heuristic_fallback until the
    #     operator trains models (see ml/CLAUDE.md) → ml_source is
    #     EXPECTED to read 'heuristic_fallback', NOT a fabricated
    #     confidence. Both are tagged expected-empty so an empty/absent
    #     result is not misread as a regression.
    {
        "id": "db_dex_runtime_stats_freshness",
        "title": "DB: dex_runtime_stats heartbeat freshness (migration 032)",
        "category": "db",
        "kind": "db_query",
        # to_regclass() returns NULL (not an error) when the table is
        # absent, so a pre-migration-032 VPS gets a clean one-row
        # "MIGRATION_MISSING — run migration 032" message instead of a
        # red SQL failure. When the table exists we report the heartbeat
        # age; fresh = updated within ~150s of NOW() while DEX runs.
        "sql": (
            "SELECT CASE "
            "  WHEN to_regclass('public.dex_runtime_stats') IS NULL "
            "    THEN 'MIGRATION_MISSING — run migration 032 (dex_runtime_stats)' "
            "  WHEN NOT EXISTS (SELECT 1 FROM dex_runtime_stats WHERE id = 1) "
            "    THEN 'NO_HEARTBEAT_ROW — DEX module has not written stats yet' "
            "  WHEN (SELECT EXTRACT(EPOCH FROM (NOW() - updated_at)) "
            "        FROM dex_runtime_stats WHERE id = 1) <= 150 "
            "    THEN 'FRESH — age ' || (SELECT ROUND(EXTRACT(EPOCH FROM "
            "         (NOW() - updated_at)))::text FROM dex_runtime_stats "
            "         WHERE id = 1) || 's (DEX heartbeat alive)' "
            "  ELSE 'STALE — age ' || (SELECT ROUND(EXTRACT(EPOCH FROM "
            "       (NOW() - updated_at)))::text FROM dex_runtime_stats "
            "       WHERE id = 1) || 's (>150s: DEX stopped or crashed)' "
            "END AS heartbeat_status"
        ),
        "cmd_preview": (
            "to_regclass-guarded freshness of dex_runtime_stats.updated_at "
            "(<=150s = FRESH)"
        ),
        "timeout_s": 15,
        "tags": ["new", "expected-empty"],
        "description": (
            "Wave-8 DEX liveness heartbeat (migration 032 / dex_runtime_stats). "
            "FRESH = the DEX subprocess wrote updated_at within ~150s, proving "
            "the per-tick heartbeat fires. MIGRATION_MISSING = run migration "
            "032 first. NO_HEARTBEAT_ROW = DEX not running yet. STALE = DEX "
            "stopped/crashed. Fail-soft: never hard-errors pre-migration."
        ),
    },
    {
        "id": "db_dex_ml_source_provenance",
        "title": "DB: DEX ml_source provenance (Wave-8 DEFECT-2)",
        "category": "db",
        "kind": "db_query",
        # ml_source lives in trades.metadata for DEX rows. Pre-training
        # it MUST read 'heuristic_fallback' — the honest current state.
        # It only reads 'ensemble' after the operator trains + persists
        # ensemble artifacts (ml/CLAUDE.md). Group-by surfaces the split
        # so any fabricated optimistic label (e.g. a bogus 'ensemble'
        # with no trained model) would stand out immediately.
        "sql": (
            "SELECT "
            "  COALESCE(metadata->>'ml_source', '(none)') AS ml_source, "
            "  COUNT(*) AS n, "
            "  MAX(entry_timestamp) AS most_recent "
            "FROM trades "
            "WHERE entry_timestamp > NOW() - INTERVAL '24 hours' "
            "GROUP BY ml_source "
            "ORDER BY n DESC"
        ),
        "cmd_preview": (
            "GROUP BY trades.metadata->>'ml_source', last 24h"
        ),
        "timeout_s": 15,
        "tags": ["new", "expected-empty"],
        "description": (
            "Wave-8 DEFECT-2: every DEX opportunity carries an HONEST "
            "ml_source provenance label, never a fabricated ML confidence. "
            "EXPECTED value today is 'heuristic_fallback' (the ensemble runs "
            "in fail-soft fallback until models are trained — see ml/CLAUDE.md). "
            "It reads 'ensemble' ONLY after the operator trains + persists "
            "artifacts and restarts DEX. 0 rows = no DEX trades in 24h (fine "
            "in DRY_RUN with no candidates)."
        ),
    },
    {
        "id": "db_dex_load_state_open_trades",
        "title": "DB: DEX open trades for _load_state restore (manual-check)",
        "category": "db",
        "kind": "db_query",
        # _load_state restore is hard to auto-probe without the running
        # engine's in-memory positions dict, so this is a documented
        # manual-check: it lists the open DEX trades the engine SHOULD
        # repopulate into engine.positions on restart. The operator
        # cross-checks this count against /api/dashboard/summary open
        # positions or the DEX log's "restored N open positions" line.
        "sql": (
            "SELECT token_address, entry_price, amount, entry_timestamp "
            "FROM trades "
            "WHERE status = 'open' "
            "ORDER BY entry_timestamp DESC "
            "LIMIT 50"
        ),
        "cmd_preview": "Open DEX trades (status='open') — restore reference set",
        "timeout_s": 15,
        "tags": ["new", "expected-empty"],
        "description": (
            "Wave-8 _load_state restore — manual cross-check (low priority). "
            "Lists open DEX trades the engine should reflect in its in-memory "
            "positions after a restart. Compare this count to the DEX log's "
            "'restored N open positions' line (or /api/dashboard/summary open "
            "count). Not an auto-pass/fail probe: the engine's positions dict "
            "isn't queryable from the dashboard. 0 rows = no open DEX "
            "positions (normal in DRY_RUN with no entries)."
        ),
    },

    # ── Wave-9 DEX engine quant-audit fixes (code-presence sanity) ────
    # All three are READ-ONLY source-greps / import checks run inline
    # via `bash -c` / `python -c` (no helper-script files added, no DB,
    # no engine instantiation). They assert STATIC behavior that holds
    # whether or not any module is running, so they have no
    # not-deployed precondition and are plain sanity checks.
    {
        "id": "script_dex_contract_gate_honest",
        "title": "Script: DEX _check_smart_contract honesty grep (Wave-9)",
        "category": "scripts",
        "kind": "bash",
        # Asserts the gate no longer returns a blanket verified:True and
        # that the honest verified:False / status:'unknown' return is
        # present. Tolerant of single/double quote style.
        "cmd": [
            "bash", "-c",
            "set -euo pipefail; cd \"${CLAUDEDEX_REPO_ROOT:-/app}\"; "
            "f=core/engine.py; "
            "if [ ! -f \"$f\" ]; then echo \"FAIL — $f not found\"; exit 1; fi; "
            "blk=$(awk '/async def _check_smart_contract/{c=1} "
            "c{print} /async def _analyze_holder_distribution/{if(c)exit}' \"$f\"); "
            "if echo \"$blk\" | grep -Eq \"['\\\"]verified['\\\"][[:space:]]*:[[:space:]]*True\"; "
            "then echo 'FAIL — _check_smart_contract still returns verified:True (blanket safe)'; "
            "echo \"$blk\" | grep -nE \"['\\\"]verified['\\\"][[:space:]]*:[[:space:]]*True\"; exit 1; fi; "
            "if echo \"$blk\" | grep -Eq \"['\\\"]verified['\\\"][[:space:]]*:[[:space:]]*False\" "
            "&& echo \"$blk\" | grep -Eq \"['\\\"]status['\\\"][[:space:]]*:[[:space:]]*['\\\"]unknown['\\\"]\"; "
            "then echo 'PASS — _check_smart_contract returns honest verified:False/status:unknown'; exit 0; "
            "else echo 'FAIL — honest verified:False/status:unknown return missing from _check_smart_contract'; exit 1; fi",
        ],
        "cmd_preview": (
            "grep core/engine.py::_check_smart_contract for honest "
            "verified:False/status:'unknown' (no blanket verified:True)"
        ),
        "timeout_s": 20,
        "tags": ["new", "must"],
        "description": (
            "Wave-9 contract-gate honesty: _check_smart_contract previously "
            "returned a blanket verified:True — a FALSE positive safety signal. "
            "PASS asserts the blanket verified:True is gone AND the honest "
            "verified:False / status:'unknown' return is present, so the "
            "downstream gate logs a caution instead of asserting an "
            "unperformed verification. Static source-grep; no engine run."
        ),
    },
    {
        "id": "script_dex_random_feature_stub_removed",
        "title": "Script: DEX np.random.rand feature-stub removed (Wave-9)",
        "category": "scripts",
        "kind": "bash",
        # Asserts ZERO non-comment np.random.rand in core/engine.py. The
        # deleted _extract_features stub returned np.random.rand(10) — a
        # random feature vector on a live path. We tolerate the named
        # token inside the tombstone COMMENT (line starts with #), only
        # flag a code-level appearance.
        "cmd": [
            "bash", "-c",
            "set -euo pipefail; cd \"${CLAUDEDEX_REPO_ROOT:-/app}\"; "
            "f=core/engine.py; "
            "if [ ! -f \"$f\" ]; then echo \"FAIL — $f not found\"; exit 1; fi; "
            "hits=$(grep -nE 'np\\.random\\.rand' \"$f\" "
            "| grep -vE '^[[:space:]]*[0-9]+:[[:space:]]*#') || true; "
            "if [ -n \"$hits\" ]; then "
            "echo 'FAIL — np.random.rand present in core/engine.py (non-comment):'; "
            "echo \"$hits\" | head -5; exit 1; fi; "
            "echo 'PASS — no non-comment np.random.rand in core/engine.py (stub removed; tombstone only)'",
        ],
        "cmd_preview": (
            "grep core/engine.py for non-comment np.random.rand → expect none "
            "(tombstone comment OK)"
        ),
        "timeout_s": 20,
        "tags": ["new", "must"],
        "description": (
            "Wave-9: the dead _extract_features stub that returned "
            "np.random.rand(10) (random features on a live path) was DELETED. "
            "PASS = zero non-comment np.random.rand occurrences in "
            "core/engine.py; the Wave-9 tombstone comment naming the removed "
            "code is tolerated. Reappearance = a fabricated-feature foot-gun "
            "is back. Static source-grep."
        ),
    },
    {
        "id": "script_ensemble_feature_contract_82",
        "title": "Script: ENSEMBLE_FEATURE_NAMES == 82 + clean import (Wave-9)",
        "category": "scripts",
        "kind": "bash",
        # Imports the canonical feature-name list and asserts len == 82
        # (the fixed-order contract documented in ml/CLAUDE.md). Guards
        # against silent feature-order/length drift between train and
        # inference. Import must also succeed (catches syntax/dep drift).
        "cmd": [
            "python", "-c",
            "from ml.models.ensemble_model import ENSEMBLE_FEATURE_NAMES as F; "
            "n=len(F); "
            "import sys; "
            "print('PASS — ENSEMBLE_FEATURE_NAMES imports cleanly, len=%d (==82)' % n) "
            "if n==82 else (print('FAIL — ENSEMBLE_FEATURE_NAMES len=%d, expected 82 (feature-order drift)' % n) or sys.exit(1))",
        ],
        "cmd_preview": (
            "python -c 'assert len(ENSEMBLE_FEATURE_NAMES)==82' (clean import)"
        ),
        "timeout_s": 30,
        "tags": ["new", "must"],
        "description": (
            "Wave-9 ensemble feature contract: the inference vector is a "
            "fixed-order 82-element list (ml/CLAUDE.md). PASS = the module "
            "imports cleanly AND len(ENSEMBLE_FEATURE_NAMES)==82. Guards "
            "against future feature add/remove/reorder drift that would "
            "silently desync train-time from inference-time. Needs the "
            "dashboard image's ML deps (pandas/torch) on PATH; an ImportError "
            "here flags a dependency/packaging regression, not a feature "
            "count change."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────
# Tag system — operator-facing badges. Tags are computed centrally
# (NOT inlined into each catalog dict) so the 121-entry catalog stays
# diff-reviewable and a single edit here re-tags the whole set.
#
# Allowed values (matched by frontend pill colors):
#   must            — sanity / cap-seed / dry-run probes the operator
#                     should run on every deploy. Red pill.
#   new             — Wave-3 + Wave-4 additions. Green pill so the
#                     operator can spot recent coverage at a glance.
#   p0              — migration-present probes (023/024/029/030);
#                     orange pill — a 0-row response means a missing
#                     migration, which silently breaks an engine path.
#   flaky           — known-intermittent; yellow pill so the operator
#                     can deprioritize a single red without alarm.
#   expected-empty  — probes that ROUTINELY return 0 rows in DRY_RUN
#                     mode (e.g. live PnL splits, allocator proposals).
#                     Grey pill so an empty result is NOT misread as
#                     a regression.
#
# When adding a new test:
#   1. Append the dict to TEST_CATALOG as before.
#   2. If it deserves any tag(s), add an entry below. Untagged entries
#      simply render without pills.
# ─────────────────────────────────────────────────────────────────────
ALLOWED_TAGS = frozenset({"must", "new", "p0", "flaky", "expected-empty"})

# Sanity scripts — single-purpose source-greps the operator should run
# on every deploy. Failure = a regression in the named commit's fix.
_MUST_SANITY_SCRIPTS = {
    "script_dex_web3_v6_imports",
    "script_dex_mev_unbound_check",
    "script_arb_nameerror_regression",
    "script_sniper_listener_widget_present",
}

# Per-module DRY_RUN GET probes — every engine MUST report dry_run
# explicitly so the operator can see at a glance which modules will
# place real orders on next subprocess restart.
_MUST_DRY_RUN_PROBES = {
    "api_dry_run_arbitrage", "api_dry_run_sniper", "api_dry_run_copytrading",
    "api_dry_run_ai", "api_dry_run_futures", "api_dry_run_solana",
    "api_dry_run_dex",
}

# Cap / safety-toggle seed probes — confirm the engines have the
# config rows they need to enforce per-module caps. Missing rows mean
# the engine silently falls back to dataclass defaults (still safe but
# the dashboard knobs become no-ops).
_MUST_CAP_SEED_PROBES = {
    "db_seeded_caps",
    "db_migration_seeds",
    "db_breaker_thresholds",
    "db_futures_leverage_caps",
    "db_copy_probation_thresholds",
}

# Migration-present probes — surface whether migrations 023/024/029/030
# have been applied. A 0-row response = engine path is silently broken.
_P0_MIGRATION_PROBES = {
    "db_ai_calibration_table",          # mig 023
    "db_copy_leader_scores_table",      # mig 024
    "db_futures_funding_payments_recent",  # mig 029
    "db_copy_probation_thresholds",     # mig 030 (also tagged must)
}

# Wave-3 + Wave-4 catalog additions — auto-flagged via commit-hash
# grep below + explicit Wave-4 section IDs. Operator wanted a visual
# badge for recent additions so they don't have to memorize which
# entries are "new" since the c9adaa8 close-out.
_NEW_WAVE34_COMMIT_HASHES = frozenset({
    "115cb35", "142250b", "16b7dab", "34e6c95", "3edd27e", "50bd9c3",
    "68b20fb", "6f66608", "6f96075", "a6c3a89", "b805626", "c7e4a27",
    "d6a4a8c",
})

# Explicit Wave-4 entries (under the "Wave-4 …" section headers) whose
# descriptions don't happen to mention one of the W3/W4 commit hashes.
# Listed by id so the frontend can still render the green pill.
_NEW_EXPLICIT_IDS = {
    "api_futures_funding_forecast",
    "db_futures_telegram_alert_flag",
    "db_ai_calibrated_predictions_flag",
    "db_ai_calibrated_model_artefacts",
    "db_copy_probation_thresholds",
    "db_copy_probation_state",
    "db_copy_exposure_breakdown",
    "db_copy_cross_module_cap",
}

# Probes that return 0 rows in DRY_RUN mode by design — tagging avoids
# the operator interpreting an empty result as a regression.
_EXPECTED_EMPTY_IDS = {
    "orchestrator_train_report",
    "orchestrator_train_save",
    "db_breaker_active_events",
    "api_breaker_active",
    "db_alloc_current",
    "api_alloc_current",
    "api_alloc_pending",
    "db_copy_probation_state",
    "api_orch_pending_recs",
    "db_orch_recs_summary",
    "db_orch_training_data",
    "db_ai_calibration_sample",
    "db_ai_bandit_state",
    "db_ai_quorum_outcomes",
    "api_ai_quorum_metrics",
    "db_copy_leader_scores_post_refresh",
    "db_ai_calibrated_model_artefacts",
}


def _compute_tags(entry: Dict[str, Any]) -> List[str]:
    """Compute the visible tag list for a single catalog entry.

    Source-of-truth lookup tables above; this function just unions them.
    Result is sorted + deduped + filtered to ALLOWED_TAGS so a typo
    can't ship a mystery pill to the frontend.
    """
    import re as _re
    eid = entry["id"]
    tags: set = set()
    if eid in _MUST_SANITY_SCRIPTS or eid in _MUST_DRY_RUN_PROBES \
            or eid in _MUST_CAP_SEED_PROBES:
        tags.add("must")
    if eid in _P0_MIGRATION_PROBES:
        tags.add("p0")
    if eid in _EXPECTED_EMPTY_IDS:
        tags.add("expected-empty")
    if eid in _NEW_EXPLICIT_IDS:
        tags.add("new")
    else:
        # Commit-hash grep across the description / title / preview blob
        # so future Wave-N additions auto-light-up the moment their
        # commit hash lands in _NEW_WAVE34_COMMIT_HASHES.
        blob = " ".join((
            entry.get("description", "") or "",
            entry.get("title", "") or "",
            entry.get("cmd_preview", "") or "",
        ))
        for h in _re.findall(r"\b[0-9a-f]{7,8}\b", blob):
            if h in _NEW_WAVE34_COMMIT_HASHES:
                tags.add("new")
                break
    # Allow per-entry override via optional "tags" field (additive).
    for t in entry.get("tags", []) or []:
        tags.add(t)
    return sorted(t for t in tags if t in ALLOWED_TAGS)


def _public_catalog_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Strip executor-internal fields (cmd/sql/endpoint) before sending
    the catalog to the client. The client only needs id/title/preview
    to render a button — it never needs the raw command. Tags are
    computed via _compute_tags so the operator sees pill badges."""
    return {
        "id": entry["id"],
        "title": entry["title"],
        "category": entry["category"],
        "kind": entry["kind"],
        "cmd_preview": entry["cmd_preview"],
        "timeout_s": entry["timeout_s"],
        "description": entry.get("description", ""),
        "tags": _compute_tags(entry),
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
