# DASHBOARD Module
## What it does
Operator-facing web UI + REST/WebSocket API for monitoring and controlling every trading module. Runs independently — does NOT require any trading subprocess. Serves auth-gated pages, per-module pause/resume + emergency-stop controls, live position/P&L tables, and the credentials/settings UI.
## Entry point
`modules/dashboard/main_dashboard.py` — launched as a subprocess by `main.py` when `DASHBOARD_MODULE_ENABLED=true`. Engine: `monitoring/enhanced_dashboard.py` (aiohttp app; Socket.IO for live updates). Routes also registered from `monitoring/{module_routes,auth_routes,credentials_routes,analytics_routes,rpc_pool_routes}.py`.
## Key config (DB-backed via `ConfigManager`)
- `host` — bind address (default `0.0.0.0`)
- `port` — HTTP port (default `8080`)
- `DASHBOARD_HTTPS` — env-level; `auto` (default) sets Secure cookies only over HTTPS/X-Forwarded-Proto; `true`/`false` force it
- `DASHBOARD_CORS_ORIGINS` — env-level; comma-separated origin allowlist for BOTH Socket.IO (MB-26) AND the aiohttp-cors HTTP layer (Wave-F6 — replaced the wildcard+credentials default; a `*` entry is filtered out server-side and must never be used)
- Module enable/disable knobs — write through `config_manager.set(...)` (MB-33 deprecated direct `.env` edits)
## Kill switch
- Global: `logs/.killswitch` — written by `scripts/emergency_stop.py` AND by `/api/bot/emergency-exit` (Phase 2 #7 + MB-31). Read by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`.
- Per-module: `logs/.pause_<module>` — written by `/api/modules/{module}/pause` (admin-gated); deleted by `/resume` (MB-30). The dashboard process itself has no live-write surface, so no `logs/.pause_dashboard` semantics today.
## Logs
`logs/dashboard/` — main, errors (rotating handler). HTTP access logs flow through the aiohttp logger.
## Primary risk-policy gate
None — the dashboard does not execute trades. Defense layers on state-changing endpoints: (1) session auth on all non-public routes (`auth/middleware.py`); (2) CSRF double-submit in `auth/csrf.py` (MB-27) on every POST/PUT/DELETE/PATCH outside `/api/auth/{login,logout}`; (3) Wave-F6 RBAC — per-route `require_admin`/`require_operator` wrappers (matrix below) PLUS a central VIEWER write-floor in `auth_middleware_factory` that 403s any mutating request from a viewer session (except `/api/auth/change-password`/`logout`), covering routes registered by files the per-route sweep didn't own; (4) Socket.IO `connect` validates the session cookie before streaming (MB-26). Every RBAC deny/allow emits an audit log line (actor, role, route, outcome). A startup self-test (`auth/route_authz.py`, called from `_on_startup`) logs any mutating route without an explicit gate.
## RBAC matrix (Wave-F6)
Roles: `ADMIN` (everything), `OPERATOR` (trading ops + non-sensitive settings), `VIEWER` (read-only — hard write-floor). Read-only GETs are viewer-accessible unless they leak secrets.

**Admin-only** (infra / lifecycle / secrets / mode changes):
- Module lifecycle: `/api/modules/{m}/{enable,disable,pause,start,restart}`, `/api/modules/{m}/dry-run` POST; `module_routes` `/api/bot/{start,stop,restart,emergency-exit}`, `/api/modules/{m}/{pause,resume}` (MB-28/30)
- `/api/ml/train` (swaps live model artifacts)
- Approvals that change bot behavior/capital: `/api/proposals/{kind}/{id}/{action}`, `/api/orchestrator/recommendations/{id}/{approve,reject}`, `/api/portfolio/allocations/{propose,{id}/approve}`, `/api/circuit-breaker/{id}/clear`
- RPC pool: `POST/PUT/DELETE /api/rpc-pool/endpoints[/{id}]` AND `GET /api/rpc-pool/endpoints` (returns key-bearing URLs)
- Secrets: `/api/credentials/*`, `/api/settings/sensitive/*` (pre-existing); copytrading wallet add/remove + leaders/refresh; `/test-runner` page
- `/users` + user CRUD (`auth_routes`)

**Operator or admin** (capital-impacting trading ops + non-sensitive settings):
- Trade controls: `/api/trade/execute`, `/api/position/{close,modify}`, `/api/order/cancel`
- Per-module ops: `{sniper,solana,futures}` position close / close-all; `{sniper,arbitrage,solana,dex,futures}` `/trading/unblock`; `{arbitrage,dex,copytrading}` `/reconcile`; `/api/portfolio/reset-block`
- Settings writes: `/api/{sniper,arbitrage,copytrading,ai,telegram,advisor}/settings`, `/api/settings/{update,revert,futures,solana}`, `/api/config/{type}` POST, `/api/strategy/parameters` POST, advisor portfolio/holding/sim-close, `/api/copytrading/validate`
- Bounded compute/quota ops: `/api/reports/generate`, `/api/backtest/{run,replay}`, `/api/rpc-pool/{endpoints/{id}/test,test-all,health-check}`

**Any authenticated role**: all read-only GETs, `/api/auth/change-password`, `/api/auth/logout`. Known residual (viewer write-floor only, no explicit gate — outside Wave-F6 file ownership): `module_routes.py` `/api/modules/{m}/{start,enable,disable}` + `/api/modules/reallocate`, `test_runner_routes.py` POST endpoints, `analytics_routes` if any. The startup self-test lists them on every boot.
## ⚠️ Public exposure (operator action required)
docker-compose publishes the control plane on host `0.0.0.0:8080` — on a VPS that is the public internet, and the Wave-F6 ops sweep recorded live attacks (Mirai RCE probe on `/login.cgi`, `/api/.env` fishing, credential-stuffing logins; all held by auth/CSRF). Password auth over plain HTTP is not an acceptable steady state for a trading bot's control plane. Required: firewall 8080 to operator IPs, or bind `127.0.0.1:8080:8080` + SSH tunnel/VPN, or TLS reverse proxy (+ `DASHBOARD_HTTPS`). See the comment above `ports:` in `docker-compose.yml`.
## Information architecture (Wave-F5 final, 2026-07)
One home (`/control-center`; `/` 302s there), one chart page (`/analytics`), one module hub (`/modules`, includes the Manual-Trade panel), one approvals inbox (`/proposals`, absorbed `/orchestrator` + `/allocation`), one settings engine (`/config`). Retired pages 302 to successors — full redirect table in `docs/dashboards.md` §1. `/dex/settings` rebuilt to the 2-tab Settings/Guide pattern with six module-scoped config groups. `/users` now has a real standalone template (was a 500); `/test-runner` is admin-only and off the sidebar; legacy `monitoring/dashboard.py` deleted.
## Live-trade readiness
AMBER → GREEN candidate (pending production verification). Security cluster (MB-26..29, MB-29b) and operational cluster (MB-30, MB-31, MB-32, MB-33) all closed; subprocess discovery bridge (`7808ed0`) makes deployed-bot modules visible; `last_reconcile_at` per-module surface + sticky RESTART OVER-CAP banner (`592cb1b`). Pre-prod: enable HTTPS at reverse proxy and set `DASHBOARD_HTTPS=true`; rotate the admin password printed once by `scripts/init_auth.py`.
## See also
- Phase 1 audit reports: `docs/agents/reports/DASHBOARD_*.md` (quant / analyst / backend).
- Canonical engine API: `docs/engines.md`.
