# ClaudeDex Dashboard Reference

Page map, auth flow, API route table, and Socket.IO event taxonomy
for the dashboard module. Pairs with `modules/dashboard/CLAUDE.md`
(module overview) and `docs/runbook.md` (operational procedures).

## 1. Page map (final IA — Wave-F5 consolidation, 2026-07)

Routes are registered across `monitoring/enhanced_dashboard.py`
(top-level + fallback module pages), `monitoring/module_routes.py`
(embedded-mode module pages), and `monitoring/auth_routes.py` /
`credentials_routes.py` / `analytics_routes.py` / `rpc_pool_routes.py`.

Principles: one home (`/control-center`), one cross-module chart page
(`/analytics`), one module-control hub (`/modules`), one approvals inbox
(`/proposals`), one settings engine (`/config`). Per trading module:
exactly 5 pages. Every retired URL 302s to its successor (no 404'd
bookmarks).

### Sidebar tree

| Entry | Path | Description |
|---|---|---|
| Control Center | `/control-center` | Home (`/` 302s here). Runtime badges, per-module PnL, pause/restart/Go-LIVE, cross-module table, meta decisions. |
| Analytics | `/analytics` | THE cross-module chart/risk page (analytics_routes). |
| Modules | `/modules` | Single enable/disable/pause hub + intel-ops cards + Manual Trade panel (ported from pro-controls). |
| DEX / Futures / Solana / Sniper / Copy / Polymarket | `/<mod>/{dashboard,positions,trades,performance,settings}` | Standard 5-page set. Copy keeps Discovery + Wallet Monitor + Leaders extras; Arbitrage has 4 pages (no Positions by design); Polymarket settings = `/config/polymarket_config`. |
| Financial Advisor | `/advisor/{dashboard,advice,simulations,portfolio,kap,settings}` | 6 pages. |
| AI Analysis | `/ai/{dashboard,sentiment,performance,settings,logs}` | 5 pages. |
| Intelligence & Ops | `/module/<key>` ×16 | Generic panels; settings via `/config/<type>`. |
| Proposals | `/proposals` | Single approvals inbox: advisory proposals + Orchestrator AI recs (`#orchestrator`) + portfolio allocation (`#allocation`). |
| Help | `/help` | Deploy & use guide. |
| All Settings | `/config`, `/config/{type}` | Canonical typed settings engine (docs + audit trail). |
| Telegram / RPC-API / Wallet Balances / Credentials / Simulator / Logs / Backtest Replay | `/telegram/settings`, `/settings/rpc-api`, `/wallet-balances`, `/credentials`, `/simulator`, `/logs`, `/backtest-replay` | System pages. |

Off-sidebar but live: `/settings` (account page, user-avatar menu only),
`/users` (admin user CRUD, standalone template), `/test-runner`
(admin-only dev/QA tool), `/arbitrage/positions` (inline explainer).

### Redirect table (Wave-F5)

| Old URL | → Redirect | Why |
|---|---|---|
| `/` | 302 `/control-center` | four overview pages computed disagreeing portfolio totals |
| `/full-dashboard` | 302 `/control-center` | duplicate chart set |
| `/dashboard` | 301 `/dex/dashboard` | pre-F5 legacy alias |
| `/trades`, `/positions`, `/performance` | 302 `/dex/{trades,positions,performance}` | unprefixed legacy DEX aliases |
| `/reports`, `/dex/reports`, `/analysis`, `/dex/analysis` | 302 `/dex/performance` | legacy DEX pages folded into performance |
| `/backtest`, `/dex/backtest` | 302 `/backtest-replay` | superseded by the maintained replay simulator |
| `/module-control` | 302 `/modules` | duplicate lifecycle controls |
| `/pro-controls` | 302 `/modules` | Manual Trade ported to /modules; rest duplicated |
| `/orchestrator` | 302 `/proposals#orchestrator` | approvals consolidated |
| `/allocation` | 302 `/proposals#allocation` | approvals consolidated |
| `/global-settings` | 302 `/config/allocation_guard_config` | accordion duplicated /config; guard keys live there |
| `/modules/{name}`, `/modules/{name}/details` | 302 `/module/{name}` | old detail page superseded by generic panels |
| `/modules/{name}/configure` | 302 `/config/{name}_config` | template never existed (was a 500) |

Deleted templates: `index.html`, `full_dashboard.html`, `orchestrator.html`,
`allocation.html`, `module_control.html`, `pro_controls.html`,
`global_settings.html`, `reports.html`, `analysis.html`, `backtest.html`,
`module_details.html`, `positions_arbitrage.html` (+ `analysis.js`,
`backtest.js`, legacy `monitoring/dashboard.py`). Added: `users.html`
(fixes the admin `/users` 500).

## 2. Auth flow

```
GET /login -> page sets csrf_token cookie (httponly=false; double-submit pattern)
POST /api/auth/login {username, password}
  on success: sets session_id cookie (httponly=true, secure=DASHBOARD_HTTPS, samesite=Lax)
  on failure: 401

Subsequent requests:
  - GET  /<page>           requires valid session_id (auth_middleware_factory)
  - POST/PUT/DELETE /api/* require session_id AND X-CSRF-Token matching the
                           csrf_token cookie (MB-27 double-submit-cookie)
  - /api/bot/*, /api/credentials/*, /api/settings/sensitive/*, /api/auth/users/*
    additionally require admin role (require_admin, MB-28)

POST /api/auth/logout invalidates the session and clears cookies.
```

The initial admin password is generated by `scripts/init_auth.py` and
printed ONCE to stdout (MB-29b). See `docs/runbook.md` §3 for rotation.

## 3. Settings + Guide tab pattern

Target pattern (arbitrage/advisor style, adopted by the Wave-F5 DEX
rebuild): each per-module settings page has a **Settings** surface
(module-scoped config groups; values flow through
`ConfigManager.set(...)` into `config_settings` — MB-33 deprecated the
`.env`-write fallback) and a **Guide** tab (read-only docs with defaults
and recommended ranges).

Current state per template: `settings_arbitrage.html` and
`advisor_settings.html` have proper Settings/Guide tabs;
`settings_dex.html` was rebuilt in Wave-F5 to six module-scoped tabs
(Trading/Risk/Strategies/Chains/Positions/ML Models) + Guide + an
"App Settings" link-out to `/config`; `settings_futures.html`,
`settings_solana.html` are single long forms;
`settings_sniper.html`, `settings_copytrading.html`, `settings_ai.html`
still inline-append their guides instead of tabbing them (open item,
audit row 16).

## 4. Socket.IO event taxonomy

Handlers are defined in `_setup_socketio` (`enhanced_dashboard.py:1460`).

Server handlers:
- `connect` (`:1463`): validates `session_id` cookie via
  `auth_service.validate_session`; rejects (`return False`) on missing
  or invalid session (MB-26). Then sends `initial_data` to the new sid.
- `disconnect` (`:1495`): logs sid.

Server emits (all from `_broadcast_loop`, ~5s tick at `:6645`):

| Event | Source | Payload | Use |
|---|---|---|---|
| `initial_data` | `_send_initial_data` (`:6630`) | `{portfolio, positions, orders}` | UI bootstrap; emitted once on accepted connect. |
| `dashboard_update` | broadcast loop (`:6683`) | `{portfolio_value, daily_pnl, open_positions, timestamp}` | Live header widgets. |
| `wallet_update` | broadcast loop (`:6763`) | `{balances, total_portfolio, timestamp}` | Wallet-balances widget (per-chain). |
| `performance_update` | broadcast loop (`:6776`) | `{**perf_data, timestamp}` from `db.get_performance_summary()` | Performance widget. |

CORS allowlist: `DASHBOARD_CORS_ORIGINS` env (default
`http://localhost:8080`). Socket.IO routes are excluded from the global
CORS middleware (`enhanced_dashboard.py:594`) and handle CORS internally.

## 5. State-changing API routes

All POST/PUT/DELETE calls require `X-CSRF-Token` (MB-27).

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/auth/login` | none | login |
| POST | `/api/auth/logout` | session | logout |
| POST | `/api/auth/change-password` | session | rotate own password |
| POST/PUT/DELETE | `/api/auth/users[/{id}]` | admin | user CRUD |
| POST | `/api/bot/start` | admin | start trading subprocesses |
| POST | `/api/bot/stop` | admin | stop |
| POST | `/api/bot/restart` | admin | restart |
| POST | `/api/bot/emergency-exit` | admin | flips killswitch + flattens |
| POST | `/api/bot/emergency_exit` | admin | underscore alias (MB-31) |
| GET | `/api/bot/status` | session | `{dry_run, modules, ...}` — feeds the LIVE/DRY-RUN badge (MB-32) |
| POST | `/api/modules/{name}/start,enable,disable` | session | lifecycle |
| POST | `/api/modules/{name}/pause` | session | writes `logs/.pause_{name}` (MB-30) |
| POST | `/api/modules/{name}/resume` | session | deletes the flag file |
| POST | `/api/modules/reallocate` | session | rebalance capital |
| GET/POST | `/api/credentials` | admin | list / add encrypted secret |
| GET | `/api/credentials/stats` | admin | aggregate of `config_sensitive` |
| GET/PUT/DELETE | `/api/credentials/{key}` | admin | per-secret CRUD (read returns metadata, not the value) |
| POST | `/api/credentials/import-env` | admin | one-shot `.env` migration |
| POST | `/api/credentials/validate` | admin | validate all credentials |
| POST | `/api/trades/clear/{module}`, `/api/trades/clear-all` | admin | wipe trades (audit-sensitive) |
| GET | `/api/settings/sensitive/list` | admin | list sensitive keys |
| GET/POST/DELETE | `/api/settings/sensitive[/{key}]` | admin | sensitive-key CRUD |

## 6. See also
- `modules/dashboard/CLAUDE.md` — module-level overview
- `docs/runbook.md` — emergency-stop UI behaviour, killswitch mechanics
- `docs/engines.md` — engine API the dashboard mirrors
- `docs/agents/MASTER_BACKLOG.md` — MB-26 through MB-33 history
