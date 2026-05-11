# COPY_TRADING_MODULE — Backend / DevOps Audit (infra angle)

Owner of this report: `backend-devops-expert` (secondary on this module; primary is `market-trading-analyst`). Scope: `modules/copy_trading/{main_copy.py, copy_engine.py}` from an infrastructure standpoint — leader-feed ingestion, secrets, rate-limit/cache discipline, DB schema, resource lifecycle, deployment.

## 1. Executive verdict

The copy-trading engine is **operationally minimal**: one polling loop (`copy_engine.py:572-594`), one outbound API key (Etherscan V2 for EVM) and one Solana RPC, plus Jupiter/Uniswap for execution. It is built honestly — credentials route through `secrets_manager`, slippage-protected swap on EVM, P&L reconciled on SELL — but it is **brittle for production**:

- No leader-feed ingestion (Bybit/OKX/Binance/GMX/Hyperliquid/Birdeye-leaderboard) is wired here. "Discovery" lives only in the dashboard (`enhanced_dashboard.py:9238-9337`), which calls Helius/Birdeye directly with `os.getenv` fallbacks. The engine itself only *executes* against a list of wallet addresses persisted in `config_settings` under `config_type='copytrading_config'` (`copy_engine.py:622-657`).
- Polling-only: 15 s sleep loop, Etherscan `offset=5` per wallet, all EVM wallets serialized in one session per cycle (L719). No WebSocket subscriptions to leader trade events. With 50 wallets that is 50 sequential HTTP round-trips every 15 s.
- Caching of leader trade history is **set-based, in-memory only**: `_known_tx_hashes` and `_known_solana_sigs` grow unbounded across the process lifetime (L537-538). After ~12 h the set hits MBs and the dedupe check is still O(1) but memory leaks.
- No reconnect / backoff state — rate-limit handling sleeps 30 s once and continues, but does not surface to the dashboard or pause monitoring across cycles.
- Wallet pool & leader-score state are **not persisted as first-class tables** — the engine writes back wallet stats by stuffing rows into `config_settings` with `config_type='wallet_stats'` (`copy_engine.py:1239-1243`). This conflates configuration with operational state and prevents proper schema-level indexing/analytics.

P0 infra/secret gaps:
1. The dashboard `api_copytrading_discover` (`enhanced_dashboard.py:9238`) is currently **unauth gated only by the global middleware** and reads `HELIUS_API_KEY`, `BIRDEYE_API_KEY` straight from `secrets.get(...) or os.getenv(...)` (L9261-9267). The fallback to env keeps the keys discoverable through process env if someone gains read access — should be encrypted-DB only after migration.
2. `CopyTradingEngine.dry_run` is computed once at `__init__` from `DRY_RUN` env (`copy_engine.py:527`) — toggling it in the dashboard does not propagate without restart. Operator can think they enabled live trading and actually be in simulation (or vice versa, the more dangerous direction).

## 2. Secrets / credentials audit

| Secret | Source | Verdict |
|---|---|---|
| `ETHERSCAN_API_KEY` | `RPCProvider.get_api_sync('ETHERSCAN_API') or secrets.get('ETHERSCAN_API_KEY')` (`copy_engine.py:519`) | Correct shape — provider first, secrets second. No raw `os.getenv`. Good. |
| `SOLANA_RPC_URL` | `RPCProvider.get_rpc_sync('SOLANA_RPC') or secrets.get('SOLANA_RPC_URL')` (L520) | Good. |
| `HELIUS_API_KEY` | `RPCProvider.get_api_sync('HELIUS_API') or secrets.get('HELIUS_API_KEY')` (L521) | Good. |
| `SOLANA_MODULE_PRIVATE_KEY` | `await self._get_decrypted_key('SOLANA_MODULE_PRIVATE_KEY')` (L184) — secrets_manager async, with Fernet decrypt fallback for `gAAAAAB...` ciphertext (L139-160) | Good. Encryption key loaded from file then env (L143-147). |
| `SOLANA_MODULE_WALLET` | `secrets.get(...) or os.getenv(...)` (L185) | Acceptable; address is non-sensitive. |
| `PRIVATE_KEY` (EVM) | `await self._get_decrypted_key('PRIVATE_KEY')` (L188) | Good. |
| `WALLET_ADDRESS` | `secrets.get(...) or os.getenv(...)` (L189) | Acceptable. |
| `WEB3_PROVIDER_URL` | `RPCProvider.get_rpc_sync('ETHEREUM_RPC') or secrets.get('WEB3_PROVIDER_URL') or os.getenv('WEB3_PROVIDER_URL')` (L193-198) | Good shape. |
| `DATABASE_URL` | `docker_secrets.get_database_url()` falls back to `os.getenv('DATABASE_URL')` (`main_copy.py:109-113`) | Good. |
| `TELEGRAM_BOT_TOKEN` etc. | via `monitoring/telegram_bot.py` (separate audit) | Out of scope here. |

**Direct `os.getenv` calls in the copy-trading module**:
- `copy_engine.py:134` (`_get_decrypted_key` env fallback) — defensible
- `copy_engine.py:147` (`ENCRYPTION_KEY` env fallback) — defensible
- `copy_engine.py:178` (`SOLANA_RPC_URL` fallback when RPCProvider returns None)
- `copy_engine.py:185, 189, 198` (wallet/EVM provider env fallback)
- `copy_engine.py:527` (`DRY_RUN`) — should move to `config_settings`
- `main_copy.py:91, 92, 99, 113` (`ETHERSCAN_API_KEY`, `HELIUS_API_KEY`, `SOLANA_RPC_URL`, `DATABASE_URL`)

Total: ~10 env reads. All are wrapped in fallback chains and most route through `secrets_manager` first. **Cleanest of all modules**, but the env fallback remains a backwards-compat smell. After phase-2 secrets migration, delete every `os.getenv` call here.

**Plaintext logging risk**: no secret value is logged at INFO level. `_load_settings` logs only target counts (L654). Good.

## 3. Connector / REST / WS audit

### 3.1 Outbound APIs used
- **Etherscan V2** — `ETHERSCAN_V2_API = "https://api.etherscan.io/v2/api"` (`copy_engine.py:30`). One key, multi-chain via `chainid` param (L727). Free tier = 5 req/s. The engine fires one request per wallet per cycle, so with 30 EVM wallets cycle ≈ 6 s and pushes rate limit.
- **Solana RPC** — leader monitoring via `getSignaturesForAddress` (L797), tx fetch via `getTransaction` (L976). Hits whatever endpoint `SOLANA_RPC_URL` points to (Helius by default in `.env`).
- **Jupiter Lite API** — `https://lite-api.jup.ag/swap/v1/{quote,swap}` (L22-23). Public, ~600 req/min limit, no auth. For execution only.
- **Uniswap V2 Router** — RPC call to `0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D` (L284). Uses `self.web3_provider`. **No multi-chain awareness in executor** — even though leader monitoring is multi-chain (Ethereum, Base, Arbitrum, BSC, Polygon, Optimism, Avalanche, `EVM_CHAINS` L33-41), the executor always swaps on Ethereum Uniswap V2 (L283-284). **Profit-leak**: a Base-detected trade is mirrored on Ethereum, wrong chain, wrong token contract. P0 logical bug from an infra-routing angle.
- **CoinGecko** — `PriceFetcher` (L47-97). 60 s cache, no API key. Good.

### 3.2 Rate-limit handling
- Etherscan 429 → 30 s sleep, then `continue` (L739-745). Reports rate-limit to `RPCProvider.report_rate_limit('ETHERSCAN_API', ..., 300)`. Good.
- Solana RPC 429 → 30 s sleep + tries to rotate to a new endpoint via `RPCProvider.get_rpc('SOLANA_RPC')` (L803-812). Good.
- No global per-host token bucket — bursty when many wallets share the same chain. Single shared `aiohttp.ClientSession` per cycle (L718) — but a fresh session is created every cycle, no connection reuse.
- No retry-after parsing: when Etherscan returns `{"message": "...rate limit..."}` with HTTP 200, code sleeps only 5 s (L770-772). Should parse `result` for "Max rate limit" hint.

### 3.3 Caching of leader trade history
- `_known_tx_hashes` and `_known_solana_sigs` are **process-local sets**. On restart the engine has no memory of what it already processed — on the first cycle after restart it sees the last 5 txs of each wallet as "new" (filtered only by `int(tx['timeStamp']) > time.time() - 60` L760, or `block_time > time.time() - 120` L836). The time-window filter is the actual dedupe; sets are belt-and-braces. Acceptable, but switching to a Redis-backed set keyed by `wallet:tx` with TTL would prevent unbounded growth.
- No DB-level cache of recent leader signatures. A query to `copytrading_trades` filtered by `source_tx = $1` exists logically but is not used to dedupe before recomputing — small risk of double-recording across restart-races.

### 3.4 WebSocket subscriptions
- **None**. Pure polling. Helius offers a webhook + WebSocket subscription product for wallet activity that would drop latency from up-to-15-s to sub-second and slash Etherscan/RPC calls. Major upgrade path. P1 backlog item.

### 3.5 Reconnect / backoff on data-source outage
- Etherscan failure → `logger.debug` (L775) and continue. No circuit breaker. If Etherscan is down for 1 h the engine retries every 15 s = 240 wasted cycles. Should add exponential backoff with `RPCProvider.report_failure` integration.

## 4. Configuration audit

| Tunable | Source | Verdict |
|---|---|---|
| `target_wallets` (list of `0x...@chain` or Solana base58) | `config_settings` table, `config_type='copytrading_config'`, key `target_wallets` (L623) — JSON, ast literal, or CSV (L632-649) | DB-backed. Good. Multi-format parsing is robust. |
| `max_copy_amount` | Hard-coded `100.0` (L530) | **Not DB-backed.** Migration candidate. |
| `copy_ratio` | Hard-coded `10` % (L531) | **Not DB-backed.** Migration candidate. |
| `_wallet_cooldown_seconds` | Hard-coded `300` (L542) | Migration candidate. |
| `dry_run` | `os.getenv('DRY_RUN')` at engine `__init__` (L527) | **Bad.** Should be `config_settings.dry_run` reloaded per cycle. |
| Poll interval | Hard-coded `await asyncio.sleep(15)` (L591) | Migration candidate. |
| Wallet cooldown 5 min | Hard-coded (L542, L956) | Migration candidate. |
| Etherscan `offset=5` per wallet | Hard-coded (L731) | Migration candidate (controls catch-up depth). |
| `slippage` (EVM copy) | Default arg `slippage: float = 10.0` (L267) — never overridden | **Bad.** Should come from settings page (`settings_copytrading.html:48` has the field at 5%, but it does not flow to `executor.copy_evm_swap`). |
| `slippage_bps` (Solana copy) | Default arg `100` bps (L225) | Same — disconnect between UI and engine. |
| Jupiter `prioritizationFeeLamports` | Hard-coded `5000` (L382) | Migration candidate. |

The `settings_copytrading.html` page surfaces `enabled`, `dry_run`, `max_copy_amount`, `copy_ratio`, `slippage`, `target_wallets` (L23-58). Only `target_wallets` actually wires into the engine — the others sit in the DB and the engine ignores them. **Verify `api_save_copytrading_settings` writes all keys to `config_settings`** (it does, per the dashboard route at `enhanced_dashboard.py:456`) and **fix the engine to read them** in `_load_settings`. Backlog item.

### 4.1 DB schema
- `copytrading_trades` table referenced (L1150-1219). Columns: `trade_id, token_address, chain, source_wallet, source_tx, side, entry_price, exit_price, amount, entry_usd, exit_usd, profit_loss, profit_loss_pct, status, is_simulated, entry_timestamp, exit_timestamp, tx_hash, native_price_at_trade, metadata`. Good shape for P&L reconciliation.
- `copytrading_positions` referenced in `credentials_routes.py:58` for clear-all. Schema not verified here.
- **No `copytrading_leaders` table** — wallet score, win rate, sample size, last-checked-block live nowhere durable. `wallet_stats` rows in `config_settings` (L1239-1243) is a hack. P1: real table with composite index `(chain, address)` and a `last_seen_at` for staleness.
- **No `copytrading_data_source` table** — which APIs were used to discover a wallet, what the score was, when last refreshed. Reproducibility is impossible without this.

## 5. Observability

- Logger uses three rotating handlers (`main_copy.py:38-55`): `copy_trading.log` 10 MB × 5, `copy_trading_errors.log` 5 MB × 3, `copy_trading_trades.log` 10 MB × 5. Compliant with the agent's rotation rule.
- Stats counter every 5 min (`copy_engine.py:596-614`). Good for ops visibility.
- No Prometheus counters. P1: add `copytrading_signals_total`, `copytrading_trades_total{side=}`, `copytrading_pnl_usd`, `copytrading_rate_limited_total{source=}`.
- No alert rules — should add to `observability/alerts.yml`: "no leader cycles for 5 min", "Etherscan rate-limit sustained 10 min", "DRY_RUN flipped".

## 6. Resource lifecycle

- `aiohttp.ClientSession` opened fresh per cycle inside `_monitor_evm_wallets` (`copy_engine.py:718`), `_monitor_solana_wallets` (L790), `_analyze_and_copy_solana` (L980). Should be ONE session on `self.session` held for the engine's lifetime; sub-handlers reuse it. Same pattern as the dashboard issue.
- `_executor.session` IS held for lifetime (`copy_engine.py:169`) — correct here.
- `stop()` (L1248-1256) closes the executor session. Does not close any of the ad-hoc cycle sessions, but those are scoped to `async with` so they self-close. OK.
- `_known_tx_hashes` and `_known_solana_sigs` grow without bound. Trim oldest N entries every cycle (e.g., keep last 10 000).
- Engine's `run()` is a `while self.is_running` loop with broad `except` (L592). No task cancellation handler — a `KeyboardInterrupt` in the main wrapper (`main_copy.py:157`) ends the loop, but nothing handles `asyncio.CancelledError` cleanly inside `run()`.
- No health endpoint. The module is not registered in the dashboard's `_fallback_api_modules` health-port map for copy-trading: see `enhanced_dashboard.py:1158` which uses `COPYTRADING_HEALTH_PORT` defaulting to 8085 — but `main_copy.py` never opens a port. **Health check is dead**; dashboard will always show copy-trading as offline. P0.

## 7. Profit-leak / loss-leak from infra angle

1. **Chain misrouting** (sec 3.1): EVM executor hardcoded to Uniswap V2 / WETH on Ethereum. Multi-chain leader detection feeds it Base/Arbitrum tokens but the swap lands (or fails) on Ethereum. Any "successful" log here is meaningless. P0 logical infra bug.
2. **`min_out=0` fallback** in EVM swap when quote call fails (L330-331). With slippage param defaulting to 10 and `min_out=0` we accept any amount including dust. Sandwich-bot heaven.
3. **`dry_run` snapshotted at boot** (L527): operator toggles in UI, engine keeps simulating, real PnL diverges.
4. **No leader-score / cooldown adjustment by recent leader losses**: the 5-min wallet cooldown is mechanical, not performance-aware. Bleeding wallet keeps getting copied.
5. **Standalone SELL with no matching BUY** (L1183-1201) inserts a row with `entry_usd=0` and `profit_loss=0` — distorts aggregate P&L on the dashboard performance page.
6. **Polling latency**: 15 s loop + 60 s tx-recency window means worst case copy happens 75 s after leader. For meme-coin trades that's a 30-90 % entry-price penalty. Switch to webhook/WS.
7. **No per-wallet daily cap** at the engine level — only the global `max_copy_amount=100` per trade. A leader doing 50 trades in an hour drains 50 × $10 = $500 unsupervised.

## 8. Action backlog

| ID | Sev | Owner | Title |
|---|---|---|---|
| CT-BE-01 | P0 | smartcontract (with backend) | Fix executor to swap on the chain detected by leader monitor — pass chain context from `_execute_evm_copy_trade` through to `copy_evm_swap`. |
| CT-BE-02 | P0 | backend | Open a health-port HTTP server (`/health`, `/stats`) on `COPYTRADING_HEALTH_PORT` (8085) so dashboard status is real. |
| CT-BE-03 | P0 | backend | Reload `dry_run`, `max_copy_amount`, `copy_ratio`, `slippage`, `cooldown_seconds` from `config_settings` every cycle, not at `__init__`. |
| CT-BE-04 | P1 | backend | Add `copytrading_leaders` table (address, chain, win_rate, n_trades, last_seen_at, score, source, last_refreshed_at) and migrate the `wallet_stats` config hack into it. |
| CT-BE-05 | P1 | backend | Replace polling with Helius webhook (Solana) and Etherscan transaction-list-stream / Alchemy notify (EVM). Keep 60 s polling fallback. |
| CT-BE-06 | P1 | backend | Persist `_known_tx_hashes` in Redis with 7-day TTL keyed `copytrading:seen:{chain}:{tx}`. |
| CT-BE-07 | P1 | backend | Add Prometheus counters: `copytrading_signals_total`, `copytrading_trades_total{chain,side}`, `copytrading_pnl_usd`, `copytrading_rate_limited_total{source}`. |
| CT-BE-08 | P1 | backend | Hold one `aiohttp.ClientSession` for the engine lifetime; remove per-cycle session creation. |
| CT-BE-09 | P1 | backend | Trim `_known_tx_hashes`/`_known_solana_sigs` to last 10 000 entries each cycle. |
| CT-BE-10 | P2 | backend | Add exponential backoff with jitter on Etherscan failures; surface circuit-open state to dashboard. |
| CT-BE-11 | P2 | backend | Validate `target_wallets` strictly via `_is_solana_address` / `_parse_evm_wallet` at save time (in the dashboard `api_save_copytrading_settings`) — reject invalid early. |
| CT-BE-12 | P2 | analyst (with backend) | Add per-leader daily-trade cap (e.g., 5 copies/day per wallet) at engine level. |
| CT-BE-13 | P2 | backend | Move `api_copytrading_discover` discovery cache to Redis (key=`copytrading:discover:{type}:{params_hash}`, TTL 5 min) — Birdeye/Helius cost real money per call. |
| CT-BE-14 | P2 | backend | Add `audit_logger` entries for every settings change and every copy-trade execution. |

## 9. Open questions

- Should the engine support Bybit/OKX/Binance copy-leaderboards (CEX-side leaders)? Currently it is on-chain only. If yes, design needs a CEX adapter layer plus credentials per exchange.
- Hyperliquid / GMX leader-tracking? Both have on-chain order books — feasible via The Graph subgraphs, but is the ROI worth the infra?
- Is the 15 s poll interval acceptable for the strategy edge, or do we need sub-second? That decides whether to invest in webhook infra.
- The dashboard's `discovery_copytrading.html` page exists — does it persist discovered wallets back to `config_settings.target_wallets` automatically, or is operator approval required? If automatic, that's a privilege-escalation path (one bad API response can replace the entire wallet list).
- Should `copytrading_trades` be a TimescaleDB hypertable? Trade volume is bursty around meme-coin events.
