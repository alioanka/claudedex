# DASHBOARD Module
## What it does
Operator-facing web UI + REST/WebSocket API for monitoring and controlling every trading module. Runs independently — does NOT require any trading subprocess. Serves auth-gated pages, per-module pause/resume + emergency-stop controls, live position/P&L tables, and the credentials/settings UI.
## Entry point
`modules/dashboard/main_dashboard.py` — launched as a subprocess by `main.py` when `DASHBOARD_MODULE_ENABLED=true`. Engine: `monitoring/enhanced_dashboard.py` (aiohttp app; Socket.IO for live updates). Routes also registered from `monitoring/{module_routes,auth_routes,credentials_routes,analytics_routes,rpc_pool_routes}.py`.
## Key config (DB-backed via `ConfigManager`)
- `host` — bind address (default `0.0.0.0`)
- `port` — HTTP port (default `8080`)
- `DASHBOARD_HTTPS` — env-level; when non-`false`, session cookie sets `secure=True` (MB-28)
- `DASHBOARD_CORS_ORIGINS` — env-level; comma-separated origin allowlist for Socket.IO (MB-26)
- Module enable/disable knobs — write through `config_manager.set(...)` (MB-33 deprecated direct `.env` edits)
## Kill switch
- Global: `logs/.killswitch` — written by `scripts/emergency_stop.py` AND by `/api/bot/emergency-exit` (Phase 2 #7 + MB-31). Read by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`.
- Per-module: `logs/.pause_<module>` — written by `/api/modules/{module}/pause` (admin-gated); deleted by `/resume` (MB-30). The dashboard process itself has no live-write surface, so no `logs/.pause_dashboard` semantics today.
## Logs
`logs/dashboard/` — main, errors (rotating handler). HTTP access logs flow through the aiohttp logger.
## Primary risk-policy gate
None — the dashboard does not execute trades. State-changing endpoints are protected by `auth.middleware.require_auth(require_admin(...))` on `/api/bot/*` and `/api/credentials/*` (MB-28), CSRF middleware in `auth/csrf.py` (MB-27) on every POST/PUT/DELETE/PATCH outside `/api/auth/{login,logout}`, and Socket.IO `connect` validates the session cookie before streaming (MB-26).
## Live-trade readiness
AMBER. Security cluster (MB-26..29, MB-29b) and operational cluster (MB-30, MB-31, MB-32, MB-33) all closed. Pre-prod: enable HTTPS at reverse proxy and set `DASHBOARD_HTTPS=true`; rotate the admin password printed once by `scripts/init_auth.py`.
## See also
- Phase 1 audit reports: `docs/agents/reports/DASHBOARD_*.md` (quant / analyst / backend).
- Canonical engine API: `docs/engines.md`.
