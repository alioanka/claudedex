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
        "title": "Orchestrator: train ML model (report-only)",
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
        "title": "Orchestrator: train + save ML model",
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
    {
        "id": "api_module_dry_run_overview",
        "title": "API: /api/modules effective_dry_run roundup",
        "category": "api",
        "kind": "probe",
        "endpoint": "modules",
        "cmd_preview": "GET /api/modules | .data.modules[*].effective_dry_run",
        "timeout_s": 15,
        "description": (
            "Confirms each module reports an effective_dry_run boolean "
            "in the /api/modules response. If any module is missing the "
            "field, the dashboard UI can't show its DRY/LIVE chip."
        ),
    },

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
        "title": "API: /api/orchestrator/history?hours=72",
        "category": "api",
        "kind": "probe",
        "endpoint": "orchestrator/history?hours=72",
        "cmd_preview": "GET /api/orchestrator/history?hours=72",
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
            "SELECT chain, dex_used, status, COUNT(*) AS n, "
            "  ROUND(AVG(slippage)::numeric, 5) AS avg_slippage, "
            "  ROUND(AVG(gas_used)::numeric, 0) AS avg_gas "
            "FROM trades "
            "WHERE entry_timestamp > NOW() - INTERVAL '24 hours' "
            "GROUP BY chain, dex_used, status "
            "ORDER BY chain, dex_used, status"
        ),
        "cmd_preview": (
            "SELECT chain,dex_used,status,COUNT(*),AVG(slippage),AVG(gas_used) FROM trades"
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
            "SELECT chain, dex_used, COUNT(*) AS open_n, "
            "  ROUND(SUM(amount_in)::numeric, 4) AS total_in, "
            "  MIN(entry_timestamp) AS oldest, "
            "  MAX(entry_timestamp) AS newest "
            "FROM trades WHERE status = 'open' "
            "GROUP BY chain, dex_used ORDER BY chain, dex_used"
        ),
        "cmd_preview": (
            "SELECT chain,dex_used,COUNT(*),SUM(amount_in) FROM trades WHERE status='open'"
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
            "SELECT key_name, "
            "  CASE WHEN encrypted_value IS NOT NULL "
            "       AND length(encrypted_value) > 0 "
            "       THEN 'present' ELSE 'missing' END AS status, "
            "  updated_at "
            "FROM config_sensitive "
            "WHERE key_name IN ("
            "  'FLASH_LOAN_RECEIVER_CONTRACT',"
            "  'FLASH_LOAN_RECEIVER_CONTRACT_ETH',"
            "  'FLASH_LOAN_RECEIVER_CONTRACT_ARB',"
            "  'FLASH_LOAN_RECEIVER_CONTRACT_BASE'"
            ") "
            "ORDER BY key_name"
        ),
        "cmd_preview": (
            "SELECT key_name,status FROM config_sensitive WHERE key_name LIKE 'FLASH_LOAN_RECEIVER%'"
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
            "SELECT chain, status, COUNT(*) AS n, "
            "  ROUND(AVG(NULLIF(gas_cost_usd, 0))::numeric, 4) AS avg_gas_usd, "
            "  ROUND(AVG(NULLIF(slippage_cost_usd, 0))::numeric, 4) AS avg_slip_usd, "
            "  ROUND(SUM(profit_loss)::numeric, 4) AS sum_pnl "
            "FROM arbitrage_trades "
            "WHERE entry_timestamp > NOW() - INTERVAL '24 hours' "
            "GROUP BY chain, status ORDER BY chain, status"
        ),
        "cmd_preview": (
            "GROUP-BY chain,status on arbitrage_trades, 24h, AVG gas_cost"
        ),
        "timeout_s": 15,
        "description": (
            "Per-chain ARB cost roll-up. After 8cf0143 the avg_gas_usd "
            "should differ between chains; the legacy $15 constant "
            "would show identical 15.0 across ethereum/arbitrum/base. "
            "Slippage cost should track entry_usd × default_slippage_pct."
        ),
    },
    {
        "id": "api_arb_settings_get",
        "title": "API: GET /api/arbitrage/settings (min_profit_spread surface)",
        "category": "api",
        "kind": "probe",
        "endpoint": "arbitrage/settings",
        "cmd_preview": "GET /api/arbitrage/settings | grep min_profit_spread",
        "timeout_s": 15,
        "description": (
            "Source for /arbitrage/settings page. After 744ee48 the "
            "engine reads min_profit_spread (UI knob); response must "
            "include the key. 200 + non-empty JSON confirms the GET "
            "handler routes the wave-2 cost knobs back to the page."
        ),
    },
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
            "SELECT status, "
            "  COUNT(*) AS n, "
            "  ROUND(AVG(amount_sol)::numeric, 4) AS avg_amount_sol, "
            "  ROUND(AVG(pnl_usd)::numeric, 4) AS avg_pnl_usd, "
            "  MIN(entry_time) AS oldest, "
            "  MAX(entry_time) AS newest "
            "FROM solana_trades "
            "WHERE entry_time > NOW() - INTERVAL '24 hours' "
            "GROUP BY status ORDER BY status"
        ),
        "cmd_preview": (
            "GROUP-BY status on solana_trades, 24h, AVG amount_sol + pnl_usd"
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
    {
        "id": "api_settings_futures_post_wave2",
        "title": "API: GET /api/settings/futures (wave-2 keys roundtrip)",
        "category": "api",
        "kind": "probe",
        "endpoint": "settings/futures",
        "cmd_preview": "GET /api/settings/futures | grep funding/atr/enforce",
        "timeout_s": 15,
        "description": (
            "Wave-2 settings page must expose skip_long_funding_bps, "
            "atr_sizing_enabled, atr_risk_pct, enforce_isolated_margin. "
            "200 + non-empty JSON confirms the GET handler routes those "
            "keys through FuturesConfigManager.get_*."
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
            "  MAX(written_at) AS last_used "
            "FROM ai_feature_store "
            "WHERE feature_vector->'bandit_v1'->>'template' IS NOT NULL "
            "  AND written_at > NOW() - INTERVAL '14 days' "
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
