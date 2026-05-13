# DASHBOARD_MODULE — Backend / DevOps Audit

Owner: `backend-devops-expert` (primary). Scope: `modules/dashboard/`, `monitoring/{enhanced_,}dashboard.py`, `monitoring/*_routes.py`, `dashboard/templates/**`, `dashboard/static/**`, `auth/**`, plus the shared infra surface (`config/`, `security/`, `kubernetes/`, `Dockerfile*`, `docker-compose.yml`).

## 1. Executive verdict

The dashboard is **functionally rich but architecturally non-compliant** with the rules in `PLAN.md` and `.claude/agents/backend-devops-expert.md`. It works as a single aiohttp app (`monitoring/enhanced_dashboard.py:85` `DashboardEndpoints`) launched standalone from `modules/dashboard/main_dashboard.py:49`. Auth (bcrypt + TOTP), per-module page routing, RPC pool admin and credentials UI are present, but the implementation breaks the **DB-backed-config rule**, the **secrets-in-DB rule**, and the **per-module-page rule** in many places.

P0 issues (must fix before flipping any module to live):
1. Dashboard writes module enable/disable directly to `.env` (`monitoring/enhanced_dashboard.py:1316-1374`, `_update_env_file`, then `os.environ[key]=value` at L1368). This violates the architecture diagram and bypasses `ConfigManager`/`config_settings`.
2. Session cookie has `secure=False` hard-coded (`monitoring/auth_routes.py:109`), no CSRF token on POSTs (`/api/bot/start`, `/api/bot/stop`, `/api/bot/emergency-exit`, `/api/credentials`, `/api/settings/sensitive`, etc.), and `cors_allowed_origins='*'` on Socket.IO (`monitoring/enhanced_dashboard.py:132`).
3. Default credentials `admin / admin123` are **printed in the login template body** (`dashboard/templates/login.html:92-99`) and re-logged on startup (`enhanced_dashboard.py:321`). The login page is publicly reachable and tells attackers the default.
4. WebSocket / Socket.IO handshake does **not** check the session cookie (`enhanced_dashboard.py:1443-1454`). Any unauthenticated client that reaches port 8080 can subscribe to live trade/position events.
5. 101 `os.getenv(...)` reads in `enhanced_dashboard.py` and 5 in `credentials_routes.py` — every one is a `ConfigManager` migration candidate (see section 4).
6. Encryption key is read from a file `.encryption_key` (`credentials_routes.py:78-92`) which docker-compose mounts read-only at `/app/.encryption_key:ro` (`docker-compose.yml:88`). In k8s the key has **no Secret manifest** — there is no `kubernetes/secret.yaml`; the deployment only references `trading-secrets/{database-url,redis-url,api-keys}` (`kubernetes/deployment.yaml:33-45`).

## 2. Secrets / credentials audit

### 2.1 Credentials capture page (`dashboard/templates/settings_credentials.html`)
- Form fields: `editValue`/`addValue` are `type="password"` (L259, L339) and have visibility toggle. Encryption status checkbox `addIsSensitive` is shown (L365). 
- Submit POSTs JSON to `/api/credentials` (`monitoring/credentials_routes.py:311-341`). Handler tries `SecureSecretsManager.set()` first; if not available, falls back to `_add_credential_to_db()` which Fernet-encrypts using key from file or env (L350-358).
- The DELETE handler soft-deletes by setting `encrypted_value='PLACEHOLDER'` (L482) — correct, value not echoed.
- GET `/api/credentials` lists with `encrypted_value` stripped (L264) — does **not** leak plaintext.
- `api_get_credential` (L280) returns metadata only — correct.
- `api_import_from_env` (L563) loops every key in `CREDENTIAL_MAPPINGS` (L99-150), encrypts and inserts. **Side effect**: leaves the plaintext in `.env` after import; nothing wipes the source. Operator must manually delete from `.env`.
- **Gap**: form does not validate value length / strength; no audit-log call to `security/audit_logger.py` on credential set/delete; `value_hash = sha256` (L361) is used as an integrity check but not as a duplicate guard.

### 2.2 RPC API capture page (`dashboard/templates/settings_rpc_api.html:573`)
- API-key input is `type="password"` (good) but POSTs through `RPCPoolRoutes.add_endpoint` (`monitoring/rpc_pool_routes.py:120+`). I did not see an encryption step before storage in `rpc_api_pool` (the engine stores them as columns). Audit: this is a real **plaintext-API-key-in-DB** risk if the table is dumped without re-encryption. Verify `config/pool_engine.py` writes ciphertext for `api_key`.
- "Test endpoint" button issues a live request — confirms keys work but writes them in any failure log if exceptions are not sanitized.

### 2.3 Encryption key handling
- `get_encryption_key()` (`credentials_routes.py:72-92`) reads `.encryption_key` file, falls back to `ENCRYPTION_KEY` env var. File is bind-mounted RO in compose. **In k8s the file is not provisioned** — no manifest exists, no `Secret`/`projected volume` shipped. Without it the dashboard can list ciphertext but cannot decrypt.
- `EncryptionManager._get_or_create_master_key()` (`security/encryption.py:28-63`) auto-generates a new file if missing. In k8s this means each pod restart creates a fresh key and orphan-encrypts everything. **Must** provision a `Secret` and mount it as `/app/.encryption_key`.
- No key rotation despite `key_rotation_days=30` field (`security/encryption.py:25`) — no rotation routine wired.

### 2.4 Secret storage in `.env`
- `setup_env_keys.py` is the one-time migration utility (per agent doc), but `_update_env_file` and `api_save_*_settings` still **write back to `.env`** at runtime. This is the largest single backwards-compatibility hole.

## 3. Connector / REST / WS audit

### 3.1 HTTP framework
- aiohttp 3.9.1 (`Dockerfile:26`), Socket.IO via `python-socketio==5.10.0`. App is single `web.Application` with two middlewares: a global `error_handler_middleware` (`enhanced_dashboard.py:1498-1518`) and an auth middleware injected from `auth/middleware.py:116`.
- Auth middleware insertion happens during `_on_startup` (`enhanced_dashboard.py:311`). Between server bind and `_on_startup` completion, requests fall through to a `login_placeholder` page (L332). Race window exists for un-auth requests at boot — short, but non-zero.

### 3.2 REST routes
- `_setup_routes` adds ~150 routes (L384-582). CORS is `allow_credentials=True` with default `*` origins (L564-571) — combined with cookie-based auth this is a CSRF wide-open surface. Recommendation: lock `allow_origins` to the actual dashboard hostname.
- Bot-control endpoints (`/api/bot/start`, `/api/bot/stop`, `/api/bot/restart`, `/api/bot/emergency-exit` at L73-76 of `module_routes.py`, also at L477-480 of `enhanced_dashboard.py`) are POST without CSRF tokens. Anyone with a valid cookie can be tricked into stopping the bot.
- `emergency_exit` (`module_routes.py:852-885`) iterates modules, closes positions, then stops. No idempotency token — a double-click can race.
- `/api/settings/sensitive/*` correctly gated by `require_auth(require_admin(...))` (L520-523). Good.
- `/api/credentials/*` is **not** gated by `require_admin` (`credentials_routes.py:182-196`) — any authenticated viewer can list keys.

### 3.3 WebSocket / Socket.IO
- `socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')` (`enhanced_dashboard.py:132`). No session validation in `connect` handler (L1446-1450). The Socket.IO HTTP routes are explicitly **excluded** from auth middleware via the path matcher in `auth/middleware.py:131` (`is_public = request.path.startswith('/static/')` only excludes static, but `/socket.io/` is hit before middleware runs because aiohttp_cors skips it `enhanced_dashboard.py:576`). Net effect: **WS endpoint is reachable without auth**. P0.

### 3.4 SSE
- `/api/stream` route registered (L557) using `aiohttp_sse`. No auth-aware throttling visible; combined with `auth_middleware_factory` it is at least cookie-gated, but no per-client rate limit.

## 4. Configuration audit (every tunable)

Sources matrix (high-level):

| Surface | Where | Source | Verdict |
|---|---|---|---|
| Module enable flags | `_update_env_file` writes `.env` | `.env` + live `os.environ` | **WRONG** — must use `config_settings` (`PLAN.md` rule 6) |
| DB connection (host/port/name/user/pwd) | `main_dashboard.py:86-93` via `os.getenv` | `.env` defaults | Acceptable for infra URL; password should be docker-secret/k8s-secret |
| Health-server ports (FUTURES/SOLANA/SNIPER/ARBITRAGE/COPYTRADING/AI/DEX) | 20+ `os.getenv(*_HEALTH_PORT, ...)` calls (L882, L921, L1138-1168, L1626, L1698, L1941, L1999, L2057, L2117, L2188, L2606, L2630, L4450, L5316-5647, L5671-5766, L6051-6168) | `.env` | Migration candidate — move to `config_settings` (config_type=`DASHBOARD`) |
| `DRY_RUN` | L1623, L7488, L8129, L9951 | `.env` | Should be a DB-backed live-readiness flag |
| `*_MODULE_ENABLED` | L855-857, L1121-1124, L1624, L1696 | `.env` | Currently the **only legitimate** .env value per `PLAN.md` rule 6 |
| `WALLET_ADDRESS`, `SOLANA_WALLET`, `SOLANA_MODULE_WALLET`, `PRIVATE_KEY` | L3743-3801, L3967-4004, L10277-10302 | `secrets.get(...) or os.getenv(...)` | Right-shaped (secrets first, env fallback) but env fallback should be removed once migration is verified |
| `BINANCE_API_KEY`, `BINANCE_API_SECRET` | L3798-3802 | secrets-then-env | Same |
| `HELIUS_API_KEY`, `BIRDEYE_API_KEY`, `SOLANA_RPC_URL` | L9097-9099, L9215-9267, L10301-10302, L3673-3676 | secrets-then-env | Same |
| Solana/EVM RPC URLs (`ETH_RPC_URL`, `BSC_RPC_URL`, `POLYGON_RPC_URL`, `ARBITRUM_RPC_URL`, `BASE_RPC_URL`) | L3748-3752 | `os.getenv` with hardcoded defaults | **Should be `pool_engine.get_endpoint(...)`** per rule 7 — direct env reads are forbidden in module code |
| `INITIAL_BALANCE` | L10340 | `.env` default 400 | Migration candidate (`PORTFOLIO`) |
| Module capital allocations | L1062-1090 reads `config/modules/{module}.yaml` via `yaml.safe_load` | **YAML files** | Conflict with rule "DB > files > env > defaults". Should fall through `ConfigManager` so DB edits stick. |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `TELEGRAM_ADMIN_IDS` | `monitoring/telegram_bot.py:95-97` | `.env` | Migration candidate (`config_sensitive`) |
| `ENCRYPTION_KEY` | `credentials_routes.py:86` + `.encryption_key` file | file > env | Should be Docker/k8s Secret only |
| Bcrypt/2FA params (`session_timeout`, `max_failed_attempts`) | `enhanced_dashboard.py:290-291` | Hard-coded literals (3600, 5) | Migration candidate (`SECURITY`) |
| ConfigMap `trading.json` (max_position_size, max_slippage, min_liquidity) | `kubernetes/configmap.yaml:8-13` | ConfigMap | Bypasses DB. Either delete file or have orchestrator import-once to DB. |
| ConfigMap `security.json` (require_2fa, api_rate_limits) | `kubernetes/configmap.yaml:14-19` | ConfigMap | Same. |

**Count of `os.getenv` calls in the dashboard surface (must all migrate or be justified):**
- `monitoring/enhanced_dashboard.py`: 101
- `monitoring/credentials_routes.py`: 2 (one for `ENCRYPTION_KEY`, one inside import-env loop)
- `monitoring/telegram_bot.py`: 3
- `modules/dashboard/main_dashboard.py`: 5

That is **111 direct env reads** in dashboard-owned code. The architecture diagram (`ENHANCEMENT_PROMPT.md:158-163`) is therefore not enforced.

### 4.1 Settings page DB-backed?
- `/api/settings/all` and `/api/settings/update` (`enhanced_dashboard.py:487-490`) do go through `config_manager`. Good.
- But `_api_module_enable`/`_disable`/`_start` (L1376-1439) bypass it and write `.env`. Bad.
- `api_save_copytrading_settings`, `api_save_sniper_settings`, `api_save_arbitrage_settings`, `api_save_ai_settings`, `api_save_solana_settings`, `api_save_futures_settings` — need to verify all go through `config_settings`. Spot check shows the copy-trading engine reads from `config_settings` (`copy_engine.py:622-657`), so at least that path is DB-backed.

## 5. Auth audit

### 5.1 Login
- `auth/auth_service.py:36-79` — bcrypt with `gensalt()` default cost (12). Acceptable.
- 2FA via `pyotp.random_base32()` (L48) and `_verify_totp` with `valid_window=1` (L303). Good.
- Failed-attempt lock at 5 (`AuthService.__init__` default L29, also enforced L130-134). Lock is **soft** — admin must clear manually; no auto-unlock window.
- No rate-limit on `/api/auth/login` endpoint itself — only the per-user counter. A distributed brute-force across many usernames is uncapped.
- No CAPTCHA / IP-block. The middleware records `ip_address` from `X-Forwarded-For` (`middleware.py:185-188`) **without trust verification** — easy to spoof if not behind a proxy that strips it.

### 5.2 Session
- Cookie: `httponly=True`, `samesite='Lax'`, `secure=False` (`auth_routes.py:108-112`). `secure=False` is dangerous; should be set from a `is_production` config flag, not hard-coded.
- Session token: `secrets.token_urlsafe(32)` (L189) — 256 bits. Good.
- Session expiry: 3600s sliding (`AuthService.session_timeout` default + `expires_at` update on every validate). No absolute max — a long-lived session can be infinitely extended. Add `absolute_expires_at`.
- Logout invalidates server-side via `is_active=FALSE` (L267). Good.

### 5.3 CSRF
- **No CSRF protection** on any state-changing route. aiohttp does not ship middleware for it; nothing in `auth/middleware.py` adds one. Cookie-based auth + CORS `*` = CSRF on every POST. P0.

### 5.4 Role separation
- `UserRole.ADMIN`/`OPERATOR`/`VIEWER` (`auth/models.py`). `require_admin` decorator used on user-mgmt and sensitive-config routes only. **`/api/credentials/*`, `/api/bot/*`, `/api/modules/*/enable|disable|pause|start`, `/api/trade/execute`, `/api/position/close`, `/api/portfolio/reset-block` are NOT admin-gated** — any logged-in viewer can stop the bot or close positions. P0.

## 6. Observability audit

- `monitoring/logger.py` (1323 lines) implements per-module rotating handlers per the agent rules.
- `monitoring/alerts.py` (1505 lines) exists; integration with dashboard is via `alerts_system` param to `DashboardEndpoints` (`enhanced_dashboard.py:97`). When dashboard starts standalone (`main_dashboard.py:144`), `alerts_system=None` — **alerts are not wired in dashboard standalone mode**.
- `observability/prometheus.yml`, `alerts.yml`, `grafana/dashboard.json` are owned by this agent but not actively scraped (no `prometheus_client` counters in dashboard handlers). The Dockerfile installs `prometheus-client==0.19.0` (L131) but it is not used in `monitoring/enhanced_dashboard.py` — search for `Counter(` / `Histogram(` returns nothing.
- No `/metrics` endpoint registered. P1.
- Audit log table writes happen for login/logout only (`auth_service.py:103-176`). Sensitive config changes do **not** call `log_config_change` from any of the `api_save_*` handlers — I verified the audit-log emitter exists (`auth_service.py:330-342`) but it is not invoked by `enhanced_dashboard.py`. P1.

## 7. Resource lifecycle

- DB pool: `DatabaseManager(db_config)` created in `main_dashboard.py:98`, `pool_min=5`, `pool_max=10` (hard-coded L91-92). Fine for one instance; tune via env for prod.
- Wallet/price/simulator caches in `DashboardEndpoints.__init__` (L142-153). No size cap — `_wallet_cache` keys are wallet addresses, bounded; `_simulator_cache` is single-shot. Acceptable.
- `asyncio.create_task(self._broadcast_loop())` (L180) is fired from `__init__` — anti-pattern; should be inside `start()`/`_on_startup`. If `DashboardEndpoints` is instantiated twice (tests, dev reload), it leaks a task per instance.
- `aiohttp.ClientSession` created ad-hoc inside many handlers (`enhanced_dashboard.py:884`, L923, L4450, etc.). Each call opens a fresh session — wasteful, no connection reuse. Recommend a single module-level session created in `_on_startup` and torn down in `stop()`.
- Health-check polling loop (`_fallback_api_modules`) makes 6 sequential outbound HTTP calls per `/api/modules` hit — at 5s timeout each, worst case 30s before the page renders. Move to `asyncio.gather`.

## 8. Profit-leak / loss-leak from infra angle

1. **Emergency-stop racing**: `bot_emergency_exit` loops modules sequentially. If the Solana module is offline (10s health timeout) the EVM positions wait 10s before close. In a flash-crash that is real money. Parallelize.
2. **`.env` writes during runtime** (P0 #1): the trade modules read enable-flags at boot only. The dashboard updates `.env`, *and* `os.environ[key] = value` (`enhanced_dashboard.py:1368`) — but child subprocesses already started will not see it. UI shows "enabled" but the module is still off. Operator thinks the bot is running and walks away.
3. **WS open** (P0 #4): leaks live P&L and position table to passive listeners — strategy-fingerprinting risk for front-runners.
4. **CSRF + no admin gate** (P0 #2, sec 5.4): a malicious link can stop the bot mid-trade, close positions at market, or wipe trade tables via `/api/trades/clear/{module}` (`credentials_routes.py:717`).
5. **`min_out=0` fallback in copy-engine** (`copy_engine.py:330-331`) — infra angle: dashboard config doesn't expose a "force-revert if quote unavailable" toggle, so a quote-API outage triggers full-slippage swaps. Add a dashboard-controlled kill-switch.

## 9. Dashboard quality vs. the `.md` rule

Per `.claude/agents/backend-devops-expert.md` rule 4, each trading module must have five pages plus Settings two-tab + Guide tab. Inventory (present/missing/placeholder):

| Module | dashboard | performance | trades | positions | settings | settings has Guide tab |
|---|---|---|---|---|---|---|
| DEX | **missing** (uses generic `dashboard.html` via `_fallback_dex_dashboard`) | **missing** (uses generic `performance.html`) | **missing** (uses generic `trades.html`) | **missing** (uses generic `positions.html`) | `settings_dex.html` present | **no Guide tab** (only category tabs) |
| Futures | `dashboard_futures.html` | `performance_futures.html` | `trades_futures.html` | `positions_futures.html` | `settings_futures.html` | **no Guide tab** |
| Solana | `dashboard_solana.html` | `performance_solana.html` | `trades_solana.html` | `positions_solana.html` | `settings_solana.html` | **no Guide tab** |
| Sniper | `dashboard_sniper.html` | `performance_sniper.html` | `trades_sniper.html` | `positions_sniper.html` | `settings_sniper.html` | Guide present (L156) |
| Arbitrage | `dashboard_arbitrage.html` | `performance_arbitrage.html` | `trades_arbitrage.html` | `positions_arbitrage.html` | `settings_arbitrage.html` | Guide present (L147, nav-pills) |
| Copy Trading | `dashboard_copytrading.html` | `performance_copytrading.html` | `trades_copytrading.html` | `positions_copytrading.html` | `settings_copytrading.html` | Guide present (L78) |
| AI | `dashboard_ai.html` | `performance_ai.html` | **missing** (no `trades_ai.html`) | **missing** (no `positions_ai.html`) | `settings_ai.html` | Guide present (L425) |

**Per-page rule compliance**: 4/7 modules fully compliant on pages; DEX has zero per-module pages (only the generic ones); AI is missing trades & positions pages.

**Settings/Guide two-tab rule**: 4/7 modules compliant (Sniper, Arbitrage, Copy Trading, AI). DEX, Futures, Solana have **no Guide tab**.

**Dark theme**: `base.html:2` sets `data-bs-theme="dark"` at HTML root. Confirmed default-dark. `themes.css` present, but I did not verify a theme-toggle works.

**Mobile-responsive**: `base.html:5` has the viewport meta, Bootstrap 5.3.2 grid, and `sidebar-overlay` element (L1033) for mobile. Likely usable; no media-query audit done.

**Emergency-stop button**: present in the sidebar of every page that extends `base.html` (L1023-1028), wired to `controlBot('emergency')` which uses native `confirm()` (L1135). Single-click + confirm — meets spec.

**Live P&L / latency / RPC pool / module status widgets**: P&L is broadcast every loop via Socket.IO `dashboard_update` (L6611). Latency widget is **not** present in `base.html`. RPC pool health page exists (`/settings/rpc-api` via `RPCPoolRoutes`). Module-status comes from `/api/modules` (with the .env-reading hack).

**Logs page**: `/logs` registered (L406), template `logs.html` exists. SSE stream `/api/stream` exists but I did not verify cap / filtering — risk of unbounded memory on log spike. P1.

**Dashboard independent of trading modules**: confirmed by `main_dashboard.py` which passes `trading_engine=None, portfolio_manager=None, ...` (L141-150). The fallback routes (`_setup_fallback_module_routes`) provide pages when no `module_manager`. Compliant.

## 10. K8s / Docker

- `Dockerfile` builds the monolithic bot image (CMD `main.py --mode production`). Health probe is a no-op (`HEALTHCHECK ... sys.exit(0)` L178). Replace with `curl -fsS http://localhost:8080/api/bot/status` once auth allows unauth health.
- `dashboard/Dockerfile` exists but is not referenced from `docker-compose.yml`. Dead artifact — clarify if it should be used to split the dashboard out.
- `docker-compose.yml` uses Docker secrets for `db_user`, `db_password`, `redis_password`. Good. Mounts `.encryption_key` read-only — good.
- `kubernetes/deployment.yaml` references `trading-secrets/{database-url,redis-url,api-keys}` but **no `Secret` manifest is committed** (no `kubernetes/secret.yaml`). Add one (sealed-secret or external-secret) before any prod deploy. Port 8080 is exposed via ClusterIP only (good); ingress (`kubernetes/ingress.yaml`) terminates TLS at nginx with cert-manager Let's Encrypt — fine.
- No `NetworkPolicy` — anything in the namespace can reach the bot on 8080. Add a policy that only allows ingress controller.
- Liveness probe `GET /health` (`deployment.yaml:52`) — endpoint **does not exist** in `enhanced_dashboard.py` (search confirms no `/health` route). Pod will be restart-looped under k8s. P0 for k8s deploy.

## 11. Action backlog

| ID | Sev | Owner | Title |
|---|---|---|---|
| DASH-BE-01 | P0 | backend | Remove `_update_env_file` runtime writes; route module enable/disable through `config_settings` + IPC signal to orchestrator. |
| DASH-BE-02 | P0 | backend | Gate Socket.IO `connect()` on a valid session cookie; reject anonymous WS. |
| DASH-BE-03 | P0 | backend | Add CSRF middleware (double-submit cookie or Origin/Referer check) on every POST/PUT/DELETE. |
| DASH-BE-04 | P0 | backend | Force `cookie.secure=True` when `ENVIRONMENT=production`; surface via config. |
| DASH-BE-05 | P0 | backend | Apply `require_admin` to `/api/credentials/*`, `/api/bot/*`, `/api/modules/*/enable|disable|pause|start`, `/api/trades/clear*`, `/api/dex/reconcile`, `/api/portfolio/reset-block`. |
| DASH-BE-06 | P0 | backend | Add `/health` and `/ready` endpoints (unauth-allowed) so k8s probes pass. |
| DASH-BE-07 | P0 | backend | Commit `kubernetes/secret.yaml` template (SealedSecret) for DB/Redis/encryption-key/Telegram. |
| DASH-BE-08 | P1 | backend | Migrate all 101 `os.getenv(...)` in `enhanced_dashboard.py` to `ConfigManager.get(...)` with `.env` fallback for infra-URL keys only. |
| DASH-BE-09 | P1 | backend | Remove default `admin/admin123` from `login.html` and from startup log; require operator to bootstrap via CLI. |
| DASH-BE-10 | P1 | backend | Add Guide tab to `settings_dex.html`, `settings_futures.html`, `settings_solana.html`. |
| DASH-BE-11 | P1 | backend | Create per-module pages for DEX (`dashboard_dex.html`, `performance_dex.html`, `trades_dex.html`, `positions_dex.html`); add `trades_ai.html`, `positions_ai.html`. |
| DASH-BE-12 | P1 | backend | Wire `prometheus_client` `/metrics` endpoint; emit `dashboard_requests_total`, `dashboard_ws_clients`, `dashboard_login_failures_total`. |
| DASH-BE-13 | P1 | backend | Parallelize health-port checks in `_fallback_api_modules` via `asyncio.gather`. |
| DASH-BE-14 | P1 | backend | Move `asyncio.create_task(self._broadcast_loop())` out of `__init__` into `_on_startup`. |
| DASH-BE-15 | P1 | backend | Single module-level `aiohttp.ClientSession` for all outbound calls. |
| DASH-BE-16 | P1 | backend | Cap log SSE stream to last N lines and add server-side filter; otherwise leak memory under log spike. |
| DASH-BE-17 | P2 | backend | Add absolute session-expiry (e.g., 12h) regardless of sliding refresh. |
| DASH-BE-18 | P2 | backend | Replace native `confirm()` for emergency-stop with a typed-text confirmation (`type "STOP" to confirm`). |
| DASH-BE-19 | P2 | backend | Audit-log every `api_save_*_settings` call via `AuthService.log_config_change`. |
| DASH-BE-20 | P2 | backend | Delete `dashboard/Dockerfile` (unused) or wire it into compose and split the dashboard service. |
| DASH-BE-21 | P2 | backend | Decide on `kubernetes/configmap.yaml` — either delete or import-once to DB at orchestrator startup. |
| DASH-BE-22 | P2 | backend | Add `NetworkPolicy` to namespace limiting ingress to nginx-ingress controller. |

## 12. Open questions

- Should the dashboard split into its own pod/container in k8s (the orphan `dashboard/Dockerfile` suggests yes, but `main_dashboard.py` runs inside the monolith currently)?
- Is there an expectation of multi-user concurrent sessions, or single-operator? If single, drop role complexity and harden the single-admin path.
- Are the YAML files under `config/modules/*.yaml` authoritative for capital allocation, or should `config_settings.PORTFOLIO` win? Currently `_fallback_api_modules` reads YAML (`enhanced_dashboard.py:1062-1090`).
- What is the policy for rotating the encryption key file? `security/encryption.py:25` mentions 30-day rotation but no scheduler runs it.
