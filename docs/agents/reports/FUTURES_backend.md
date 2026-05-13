# FUTURES_MODULE — Backend / DevOps Audit

**Author:** backend-devops-expert
**Date:** 2026-05-11
**Branch:** `claude/create-expert-agents-JFSF5`
**Scope:** infrastructure, secrets, configuration, observability, connector quality

---

## 1. Executive verdict — **RED** (must not flip `DRY_RUN=false` for live capital)

The Futures module is the most operationally mature of the trading modules (database-backed config via `FuturesConfigManager`, proper rotating logs, a working aiohttp health server, real ccxt-based engine, dedicated `futures_trades` DB table, Telegram alerting), but it is **not yet production-grade** from a connector and secrets-management standpoint. Three classes of problems prevent a green light:

1. **Two parallel Binance clients exist** and disagree on quality: a hand-rolled `BinanceFuturesExecutor` in `modules/futures_trading/exchanges/binance_futures.py:1` (no recv-window, no server-time sync, no precision via `exchangeInfo`, no rate-limit-header parsing, no 418/429 backoff, no WS user-data stream, no idempotency / `newClientOrderId`, no retry, all-or-nothing on first 200) and a *separate* ccxt-based path inside `core/futures_engine.py:594` actually used by `main_futures.py`. The hand-rolled executor is unsafe and should either be deleted or hardened.
2. **Bybit support is a stub.** `modules/futures_trading/exchanges/bybit_futures.py:84-111` are all literal `placeholder` methods that return `None` / `{'balance': 0.0}`. The orchestrator's CLI flag `--exchange bybit` (`main_futures.py:761`) will silently start a bot that cannot trade.
3. **Secrets path is partially migrated.** `FuturesConfigManager._reload_sensitive_credentials()` (`futures_config_manager.py:284`) does pull encrypted keys from the secrets manager and decrypts Fernet-prefixed `gAAAAAB…` values, **but every downstream call site keeps an `or os.getenv(...)` fallback** (`core/futures_engine.py:572-583, 620-624, 674-685`). `config/futures_config.py:48-52` is even worse — it is plain `os.getenv()` and is the legacy path still imported from elsewhere.

Verdict stays RED until (a) all `os.getenv()` API-key fallbacks are removed from the futures engine, (b) Binance HMAC signing is brought up to mainnet-safe quality, (c) Bybit either implemented or removed from the menu, (d) the connector emits Prometheus metrics, and (e) order send/cancel events flow into `security/audit_logger.py`.

---

## 2. Secrets / credentials audit

### Where each key currently comes from

| Secret | Read site | Source path | Encrypted? |
|---|---|---|---|
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | `core/futures_engine.py:575-576, 582-583`, `config/futures_config.py:48-49`, `config/futures_config_manager.py:215-256, 284-313` | DB `config_sensitive` via `secrets_manager` (preferred) → `.env` fallback | Yes if in DB (Fernet); No in `.env` |
| `BINANCE_TESTNET_API_KEY` / `_SECRET` | `core/futures_engine.py:572-573, 579-580`; `config/futures_config_manager.py:230-238` | Same | Same |
| `BYBIT_API_KEY` / `_SECRET` + `_TESTNET_*` | `core/futures_engine.py:674-685`; `config/futures_config_manager.py:234-238` | Same | Same |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | `core/futures_alerts.py:62-67`; `core/futures_engine.py:449-450` (async path) | Secrets manager → `.env` | Same |
| `DATABASE_URL` / `DB_URL` | `main_futures.py:537-540` via `security/docker_secrets.get_database_url()` → `os.getenv()` | Docker secrets → `.env` | Plaintext in `.env`; file-mounted in Docker |
| `ENCRYPTION_KEY` | `config/futures_config_manager.py:329` (for decrypting `gAAAAAB…`), `main.py:39-48` | `.encryption_key` file → `.env` | The key itself is on disk in cleartext — see Action `FUT-BE-04` |

### Findings

- **FUT-BE-01 (P0): Hard `os.getenv` fallbacks remain throughout the connector code.** Even though `FuturesConfigManager` was specifically built to deprecate this pattern, every async API-key lookup in `core/futures_engine.py:572-583, 620-624, 674-685` ends in `or os.getenv('BINANCE_API_KEY')`. In production this means: if the DB-encrypted secret is missing or fails to decrypt, the engine **silently** uses a possibly-stale `.env` value or `None`. There is no audit event when this fallback triggers. Remove the fallbacks (or at minimum log a `WARNING` with a Prometheus counter `futures_secret_fallback_total{key="…"}` so we know).

- **FUT-BE-02 (P0): `config/futures_config.py` is a dead-weight insecure path.** Lines 14-55 are pure `os.getenv()` reads and look like an earlier-era config that someone forgot to delete. If anything in the repo still imports `FuturesConfig` (legacy code or tests), it will bypass the secrets manager entirely. Delete the file or rewrite it as a thin shim over `FuturesConfigManager`.

- **FUT-BE-03 (P1): The "placeholder" check in `futures_config_manager.py:245, 253, 302` is heuristic.** Strings like `'your_testnet_api_key'`, `'your_mainnet_api_key'`, `'PLACEHOLDER'` are filtered out, but that's a denylist — a typo like `placeholder` (lower-case) or `default_key` slips through. Move this list to `security/secrets_manager.py` so all modules share it.

- **FUT-BE-04 (P1): `.encryption_key` file is plaintext on disk** (`config/futures_config_manager.py:325-327`). Permissions are set to `0o600` in `security/encryption.py:60` only on key creation, not validated at read time. A KMS / HSM envelope path is the eventual right answer; meanwhile add a startup check that asserts `stat().st_mode & 0o077 == 0`.

- **FUT-BE-05 (P2): No rotation visibility.** `config_sensitive` schema (`config/config_manager.py:778-815`) tracks `last_rotated` and `rotation_interval_days`, but the Futures module never reads them. Add a Prometheus gauge `futures_api_key_age_days{exchange="binance"}` and an alert when it exceeds `rotation_interval_days`.

---

## 3. Connector audit — REST/WS quality

### `modules/futures_trading/exchanges/binance_futures.py` — hand-rolled (line-by-line)

| Concern | Finding | File:line |
|---|---|---|
| **HMAC signing** | Plain `hmac.new(secret, query_string, sha256)`. Correct algorithm. **No `recvWindow` ever passed** → on a slow network Binance will reject any signed request with `-1021 Timestamp for this request is outside of the recvWindow`. | `binance_futures.py:91-99, 130-132` |
| **Server time sync** | Not done. `params['timestamp'] = int(time.time()*1000)` assumes local clock is correct. The mainline ccxt path does set `adjustForTimeDifference: True` (`core/futures_engine.py:601`); this hand-rolled one does not. | `binance_futures.py:131` |
| **Idempotency** | No `newClientOrderId` ever sent. A retry on transient network failure can create a duplicate order. | `binance_futures.py:312-328, 369-385` |
| **Rate-limit headers** | Response headers `X-MBX-USED-WEIGHT-1M`, `X-MBX-ORDER-COUNT-1M` are completely ignored. The only throttle is a hard-coded `min_request_interval = 0.1` (`line 68`). | `binance_futures.py:144-149` |
| **418 / 429 backoff** | Not handled. On a 418 ("IP banned") the executor returns `None` and the caller retries forever. | `binance_futures.py:147-149` |
| **Retries** | None. First non-200 response returns `None`. | `binance_futures.py:146-149` |
| **Reconnect / WS** | No WebSocket client. No user-data stream (`listenKey` is never created), so fills are seen only by polling `/fapi/v2/positionRisk`. Polling cadence in `futures_module.py:805` is 10s — fast-moving fills are missed. | n/a |
| **TIF defaults** | No `timeInForce` set on MARKET orders (OK). `STOP_MARKET` set to `closePosition=true` (line 537) without `workingType` (mark vs contract) — Binance defaults to CONTRACT_PRICE which can be gamed during volatile wicks; should pin to `MARK_PRICE`. | `binance_futures.py:529-540` |
| **Position-mode (hedge vs one-way)** | Not configured. Caller passes `reduceOnly=true` but if the account is in Hedge Mode the engine never sets `positionSide` and orders will be rejected. | `binance_futures.py:312-328` |
| **Symbol precision** | `quantity=position_size/100` (`futures_module.py:559`) — magic divisor, not derived from `/fapi/v1/exchangeInfo` `LOT_SIZE.stepSize` / `PRICE_FILTER.tickSize`. Will fail on BTC vs SHIB precision differences. | `futures_module.py:559, 565` |
| **Margin type** | Always forces ISOLATED (`binance_futures.py:310, 366`). Acceptable safety choice but should be configurable through `FuturesLeverageConfig.margin_mode`. | n/a |
| **Mark-price source** | `/fapi/v1/premiumIndex` is fetched but not cached / WS-streamed. Each call is a REST roundtrip. | `binance_futures.py:552-568` |
| **Funding rate** | Public endpoint, OK. No predicted-funding rate (only last). | `binance_futures.py:443-473` |
| **Liquidation polling** | `futures_module.py:816-850` polls every 30 s — far too slow. Liquidation can move 5%+ in under a second. Must subscribe to the user-data WS stream (`ACCOUNT_UPDATE`, `ORDER_TRADE_UPDATE`). | `futures_module.py:820` |

### `modules/futures_trading/exchanges/bybit_futures.py` — placeholder

The entire executor is a stub:

```
binance_futures.py:85-111  → returns None / fake balance
```

Every order method is a no-op. Bybit V5 unified-margin endpoints are not used (and not used because nothing is implemented). Either gut this file and remove `--exchange bybit` from the CLI, or replace with `pybit` (recommended — the docstring even hints at this on line 27).

### Mainline ccxt path (`core/futures_engine.py:556-690`)

This is the *actual* production path under `main_futures.py`. Quality:
- `enableRateLimit: True` (`line 597`) → ccxt handles 429s.
- `adjustForTimeDifference: True` (`line 601`) → handles clock drift.
- `set_sandbox_mode(True)` (`line 606`) for testnet.
- `load_markets()` (`line 609`) → ccxt does pull `exchangeInfo` and applies precision filters. Good.
- Separate `price_client` for mainnet prices when `DRY_RUN` / `testnet` (`line 627`) — clever.
- Still **no WS subscription**, no `newClientOrderId`, no audit-log emission, no Prometheus counter for order outcomes.

### Recommendations

- **FUT-BE-06 (P0)**: Either retire `exchanges/binance_futures.py` or harden it to parity with the ccxt path (server time, recv-window=10s, X-MBX-USED-WEIGHT-1M parse, exponential backoff on 429/418, idempotent `newClientOrderId`).
- **FUT-BE-07 (P0)**: Add Binance user-data WS stream — `listenKey` create + 30-min keepalive `PUT /fapi/v1/listenKey`, with auto-reconnect on disconnect. Reconciliation poll every 60s as a *backstop*, not the primary source.
- **FUT-BE-08 (P0)**: Implement Bybit V5 (`/v5/order/create`, `/v5/position/list`) or remove the option. The current placeholder is a foot-gun.
- **FUT-BE-09 (P1)**: Stop using a magic `quantity/100` divisor (`futures_module.py:559, 565, 731`). Pull `exchangeInfo` once at boot, store per-symbol `quantityPrecision` & `tickSize`, and quantize in a helper.
- **FUT-BE-10 (P1)**: Set `workingType=MARK_PRICE` on every `STOP_MARKET` and `TAKE_PROFIT_MARKET` to prevent wick-driven false triggers.

---

## 4. Configuration audit — where every tunable lives

### Properly DB-backed (via `FuturesConfigManager` → `config_settings`)
Files: `config/futures_config_manager.py:35-167` defines seven Pydantic models, each persisted under a `futures_*` `config_type`. The settings page writes through `update_from_settings_page()` (`line 563`) and the engine reads through `get_general()`, `get_risk()` etc. at construction. **This is the gold-standard path** and the rest of the modules should copy it.

### Hardcoded constants that should be DB-backed
- `FuturesTradingEngine.BINANCE_MAKER_FEE`, `BINANCE_TAKER_FEE`, `BYBIT_MAKER_FEE`, `BYBIT_TAKER_FEE` (`core/futures_engine.py:210-213`) — exchange fee schedules change; move to DB or pull from `/fapi/v1/commissionRate`.
- `DEFAULT_SLIPPAGE = 0.0005` (`core/futures_engine.py:216`) — should be per-symbol via exchange snapshot.
- `tp_levels = [0.02, 0.05, 0.08, 0.12]` and `tp_quantities = [0.25,…]` in the *parallel* `futures_module.py:115-117` — duplicates the DB-backed `tp1_pct…tp4_pct` in `FuturesRiskConfig`. Conflicting sources of truth.
- Polling intervals: `_position_monitoring_loop` 10 s (`futures_module.py:805`), `_liquidation_monitoring_loop` 30 s (`line 820`), `_status_reporter` 120 s (`main_futures.py:692`). Move to `FuturesStrategyConfig`.

### `.env` direct reads (candidates for ConfigManager migration)
Enumerated by `grep -n "os.getenv|os.environ"` across the eight target files. Each line is a candidate for refactor:

| File:line | Variable | Recommendation |
|---|---|---|
| `main_futures.py:525` | `FUTURES_HEALTH_PORT` | OK — infra port, leave in `.env`. |
| `main_futures.py:540` | `DATABASE_URL`, `DB_URL` | Already Docker-secret-aware via `get_database_url()`. OK. |
| `main_futures.py:721` | `CLOSE_POSITIONS_ON_SHUTDOWN` | Should be a UI-controlled `FuturesGeneralConfig` boolean (operator wants to flip without a redeploy). |
| `main_futures.py:762, 787, 790, 796` | `FUTURES_EXCHANGE`, `DRY_RUN` | `DRY_RUN` is OK as `.env` (top-level kill-switch); `FUTURES_EXCHANGE` already in `FuturesGeneralConfig.exchange` — duplicate. |
| `core/futures_engine.py:253, 360` | `FUTURES_TESTNET`, `DRY_RUN` | `DRY_RUN` OK; `FUTURES_TESTNET` is already in DB — env override path is intentional (line 256-266) but it confuses operators. Document precedence in the Settings → Guide tab. |
| `core/futures_engine.py:572-583, 620-624, 674-685` | All exchange API keys (×8) | Remove `or os.getenv(...)` (see FUT-BE-01). |
| `core/futures_alerts.py:63-67` | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Remove `os.getenv` fallback — secrets manager already handles env. |
| `config/futures_config.py:15-52` | 13 reads, *all* should be DB | Delete file (FUT-BE-02). |
| `config/futures_config_manager.py:252, 329` | inside the bootstrap fallback path | Acceptable inside the manager itself. |

**FUT-BE-11 (P1)**: Eliminate the duplicate `FuturesConfig` (single class in `config/futures_config.py`). Twelve env-var reads should be one DB call.

---

## 5. Observability audit

### Logging
- `main_futures.py:38-133` configures three rotating handlers (10 MB × 5 backups each) for `futures_trading.log`, `futures_errors.log`, `futures_trades.log`. **Good.**
- `TradeLogFilter` (`main_futures.py:45-75`) is a string-keyword filter — fragile (case-sensitive on some keywords, breaks if log messages are refactored). Move to a structured-event approach (`logger.info("trade_event", extra={"event_type": "open"})`).
- No JSON-structured logs. `MonitoringConfig.log_format = "json"` (`config/config_manager.py:380`) is configured but the Futures module ignores it and emits human-readable text. Loki / OpenSearch ingestion will be painful.
- Logs are on disk only — no shipper (Filebeat / Promtail / Vector) configured. Recommend a `logs/futures/` → Loki pipeline.

### Prometheus metrics
- `main_futures.py:179-206` exposes `/metrics` over the health server with 11 hand-built metrics. **Working, but flat-format text without HELP/TYPE annotations and not via `prometheus_client`.** Grafana will scrape but the Prom alerting expressions will be brittle.
- `observability/prometheus.yml:17-20` only scrapes `trading-bot:8080`. **The futures health server is on port 8081 (`main_futures.py:525`) and is NOT in the scrape config.** Result: none of the `futures_*` metrics are actually collected.
- `observability/alerts.yml` defines seven generic alerts (HighErrorRate, LowPortfolioBalance, DatabaseDown, RedisDown, HighMemoryUsage, LowWinRate, APIRateLimited) — none of them reference `futures_*` metrics or futures-specific thresholds (consecutive losses, daily PnL drawdown, exchange disconnect).

### Missing metrics (must add)
- `futures_order_send_total{side,symbol,outcome}` (`outcome=ack|reject|error`)
- `futures_order_latency_ms_bucket` histogram
- `futures_ws_reconnects_total` (after FUT-BE-07 ships)
- `futures_rate_limit_weight_used` gauge (parse `X-MBX-USED-WEIGHT-1M`)
- `futures_secret_fallback_total{key}` (track FUT-BE-01)
- `futures_api_key_age_days{exchange}` gauge
- `futures_unrealized_pnl_usd{symbol,side}` gauge — Grafana can build a P&L heatmap

### Alerts (must add)
- `FuturesExchangeDown`: `futures_exchange_connected == 0 for 1m`.
- `FuturesEngineNotRunning`: `futures_engine_running == 0 for 30s`.
- `FuturesDailyLossNear`: `futures_daily_pnl_usd / on() futures_daily_loss_limit_usd < -0.8` for 30s.
- `FuturesConsecutiveLossesHigh`: `futures_consecutive_losses >= 4`.
- `FuturesPositionsStuck`: `time() - futures_last_fill_ts > 600` while orders are open.

### Findings
- **FUT-BE-12 (P0)**: Add futures health endpoint (port 8081) to `observability/prometheus.yml` scrape configs. As of now, all the work the engine does to emit metrics goes into the void.
- **FUT-BE-13 (P1)**: Replace ad-hoc `/metrics` text builder with `prometheus_client.Counter/Gauge/Histogram` and let it expose `/metrics` natively.
- **FUT-BE-14 (P1)**: Add five futures-specific alerts (above) to `observability/alerts.yml`.
- **FUT-BE-15 (P2)**: Switch logging to JSON format using the global `MonitoringConfig.log_format` setting.

---

## 6. Resource / lifecycle audit

- **aiohttp sessions**: `binance_futures.py:73` opens a `ClientSession` in `initialize()`, closes in `close()` — fine. The Telegram alerts module *creates a new session per send* (`core/futures_alerts.py:194`) — wasteful, leaks connection slots; should reuse one.
- **Task cancellation**: `futures_module.py:230-237` cancels `_tasks` and `gather(return_exceptions=True)` — correct shutdown. `main_futures.py:649-659` cancels pending tasks after `FIRST_EXCEPTION`. Good.
- **DB pool**: `main_futures.py:543-548` creates `asyncpg.create_pool(min=1, max=5, command_timeout=60)`. The 5-conn ceiling will throttle a `_load_stats_from_db()` + concurrent settings UI write under load. Set `min_size=2, max_size=10` and add `server_settings={'statement_timeout': '5000'}` so a hung query cannot wedge the engine.
- **`pgbouncer`** is not in `docker-compose.yml` (verified by absence of grep hits); add it once we have more than one writer process.
- **Unbounded structures**:
  - `trade_history: List[Trade]` (`core/futures_engine.py:366`) — never trimmed. In a busy market, will balloon. Use a bounded `deque(maxlen=10000)` or rely on DB.
  - `_price_cache` (`line 382`) — no eviction.
  - `symbol_cooldowns` (`line 369`) — small, fine.
- **DRY_RUN gate at the connector boundary**: NO. `binance_futures.py` will happily POST a signed order regardless of `DRY_RUN`. The gate exists only at the engine layer (`core/futures_engine.py` decides whether to call `_simulate_trade()` vs `_execute_*`). A stray test call would hit mainnet. **FUT-BE-16 (P0)**: add `if os.getenv("DRY_RUN")=="true": return {"simulated": True}` at the top of every `_request` with `signed=True` in `binance_futures.py`.

---

## 7. Profit-leak / loss-leak inventory (infra-driven only)

| # | Leak | Driver | File:line | Annual impact estimate |
|---|---|---|---|---|
| 1 | **Liquidation seen ≥ 30 s late** because we poll instead of subscribing to user-data WS. | `futures_module.py:820` | Worst case = position liquidated at full notional; a single $300 position at 10× = up to $300 loss avoidable by 1-second WS feed. | High |
| 2 | **Stuck fills** — REST poll cadence (10 s for positions, 30 s for liquidation) means a TP fills, price reverses, we don't know we're flat, the trailing stop misfires the next bar. | `futures_module.py:805, 820` | Recurring; each missed-state event costs the trailing-stop edge. | Medium |
| 3 | **Duplicate orders on retry** — no `newClientOrderId`. If the bot retries on `aiohttp.ServerDisconnectedError`, we may end up with 2× position size. | `binance_futures.py:312-328` | One-time tail event; rare but expensive. | Medium |
| 4 | **STOP_MARKET wicked off** — no `workingType=MARK_PRICE`. A 50 ms wick on illiquid alts triggers SL and we exit before recovery. | `binance_futures.py:529-540` | Recurring small loss-leak. | Medium |
| 5 | **No reconnect → forced close on shutdown** — `CLOSE_POSITIONS_ON_SHUTDOWN=true` (`main_futures.py:721`) plus the orchestrator restarting on health-check failure forces market-close at unfavorable spread. | `main_futures.py:721`, `main.py:446-457` | Spread loss × # of restarts. | Low |
| 6 | **Plaintext `.env` keys** — an exfiltrated `.env` file = drained futures wallet (Binance API can withdraw if enabled). Even read-only keys leak open positions. | `config/futures_config.py:48-52`, `core/futures_engine.py:582-583` | Tail risk: catastrophic. | Critical |
| 7 | **Bybit "trade" path is a no-op** — `--exchange bybit` will start a bot that returns success-looking placeholders. Signals are generated but never become orders, while the operator believes they are trading. | `exchanges/bybit_futures.py:84-111` | All-of-Bybit opportunity = 100% leak. | High |
| 8 | **Rate limit IP-ban undetected** — 418 returns `None`, engine retries forever, eventually `RECEIVED 429 from IP, ban for 2h`. We may keep trying with stale prices instead of acknowledging the ban and waiting. | `binance_futures.py:147-149` | Cost of being out of the market during a ban. | Low |
| 9 | **No reconciliation on restart** — `_sync_positions()` exists (`core/futures_engine.py:439`) but `module_positions` table writes from `futures_module.py:972-1003` use the *raw* result dict (e.g. `position.get('size', 0)`), which Binance does not return as `size`. So the DB rapidly drifts from reality. | `futures_module.py:988-996` | Drives wrong TP/SL on next loop. | Medium |
| 10 | **Telegram credentials misroute** — if `TELEGRAM_BOT_TOKEN` is the wrong owner's, every entry/exit broadcasts. Privacy leak, not pure $ loss. | `core/futures_alerts.py:60-68` | n/a (privacy). | Low |

---

## 8. Dashboard integration

Existing pages under `modules/dashboard/templates/`:
- `dashboard_futures.html` ✅
- `performance_futures.html` ✅
- `trades_futures.html` ✅
- `positions_futures.html` ✅
- `settings_futures.html` ✅

The page set required by the dashboard rule is present. **What I cannot verify from this audit** (since I did not open the templates) is whether `settings_futures.html` actually has the two-tab structure (Settings + Guide) — `pm-architect-qa` should confirm in their pass. The `FuturesConfigManager.update_from_settings_page()` method (`config/futures_config_manager.py:563`) is correctly wired with a comprehensive alias map, so the data plumbing is right; only the UI shell needs verification.

- **FUT-BE-17 (P1)**: PM/QA pass to verify `settings_futures.html` Settings/Guide tabs, with default + recommended values. Guide tab must document the `FUTURES_TESTNET` env override precedence (engine line 253-272) because it currently surprises operators.
- **FUT-BE-18 (P2)**: Add a Connectivity Card to `dashboard_futures.html` showing: exchange ws status, rate-limit weight used / cap, last-fill age in seconds, and a "rotate API key" button that calls `set_sensitive_config`.

---

## 9. Action backlog (ranked)

| ID | Priority | Action | Owner |
|---|---|---|---|
| FUT-BE-01 | P0 | Remove all `or os.getenv('BINANCE_*'/'BYBIT_*')` fallbacks in `core/futures_engine.py:572-685`; log `WARNING` + emit `futures_secret_fallback_total` on any fallback | backend |
| FUT-BE-02 | P0 | Delete `config/futures_config.py` (or rewrite as shim over `FuturesConfigManager`) | backend |
| FUT-BE-06 | P0 | Retire or fully harden `exchanges/binance_futures.py` (recv-window, time-sync, idempotent IDs, 418/429 backoff, header parsing) | backend |
| FUT-BE-07 | P0 | Add Binance user-data WebSocket stream w/ `listenKey` keepalive + auto-reconnect | backend |
| FUT-BE-08 | P0 | Implement Bybit V5 or remove `--exchange bybit` | backend |
| FUT-BE-12 | P0 | Add `:8081` futures target to `observability/prometheus.yml` | backend |
| FUT-BE-16 | P0 | Add connector-boundary `DRY_RUN` gate inside `binance_futures._request(signed=True)` | backend |
| FUT-BE-03 | P1 | Move placeholder-token denylist to `security/secrets_manager.py` | backend |
| FUT-BE-04 | P1 | Validate `.encryption_key` file perms at read time; plan KMS path | backend |
| FUT-BE-09 | P1 | Quantize order qty/price via `exchangeInfo` (replace `quantity/100`) | backend |
| FUT-BE-10 | P1 | Set `workingType=MARK_PRICE` on STOP_MARKET / TAKE_PROFIT_MARKET | backend |
| FUT-BE-11 | P1 | Remove duplicate env reads (kill `FuturesConfig`); enforce single source | backend |
| FUT-BE-13 | P1 | Replace ad-hoc `/metrics` text with `prometheus_client` | backend |
| FUT-BE-14 | P1 | Add five futures-specific alerts to `observability/alerts.yml` | backend |
| FUT-BE-17 | P1 | QA pass on `settings_futures.html` Settings/Guide tabs | pm |
| FUT-BE-05 | P2 | Surface `futures_api_key_age_days` gauge from `config_sensitive.last_rotated` | backend |
| FUT-BE-15 | P2 | Switch to JSON-structured logging using `MonitoringConfig.log_format` | backend |
| FUT-BE-18 | P2 | Connectivity Card on `dashboard_futures.html` | backend |
| FUT-BE-19 | P2 | Audit-log every `set_leverage`, `open_long`, `open_short`, `close_position` via `security/audit_logger.py` with a request-id; persist to `audit_events` table | backend |
| FUT-BE-20 | P2 | Add `pgbouncer` to `docker-compose.yml`; raise `asyncpg` pool to (2, 10) with `statement_timeout=5s` | backend |

---

## 10. Open questions

1. Is the hand-rolled `BinanceFuturesExecutor` used anywhere except `modules/futures_trading/futures_module.py`? If not, the simplest fix is to delete the file and have `futures_module.py` consume the ccxt-based engine.
2. What's the intended behavior on Binance IP-ban? Should the orchestrator force a full module restart with a 2-hour cooldown, or should the engine self-quiesce?
3. Are we OK using `.encryption_key` on local disk in production, or do we want a KMS/HSM design before flipping `DRY_RUN=false`?
4. Should the Telegram alerter share a single `aiohttp.ClientSession` for all modules to reduce connection churn?
5. Is there a runbook for an exchange-side credential rotation? Right now `config_sensitive.last_rotated` exists in schema but no operational process uses it.
