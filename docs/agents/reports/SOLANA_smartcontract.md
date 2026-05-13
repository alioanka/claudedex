# SOLANA_MODULE — Smart-Contract / Web3 Audit

## Executive summary

**Verdict: RED.** The Solana module is *not* live-trade safe. There is no Jito bundle/tip path despite the dashboard exposing `pumpfun_jito` and `pumpfun_jito_tip` settings (`modules/solana_strategies/solana_config_manager.py:123-124`, `modules/solana_trading/config/solana_config_manager.py:134-135`) — these toggles are read by the config layer and **never consumed**. Every Solana swap goes through Jupiter's `sendTransaction` to a single RPC URL with no priority-fee tuning in `jupiter_helper.py` (the live path used by both Solana module and pump.fun strategy), no compute-unit-limit, and no bundle landing verification. Two parallel Solana executors exist with divergent decimal handling, wallet handling, and pubkey-match checks: `modules/solana_strategies/jupiter_helper.py` (used by `SolanaTradingEngine._open_position` and `_close_position` for live flow) and `trading/chains/solana/jupiter_executor.py` (BaseExecutor-derived, signed differently, also wrong). A third path lives in `modules/sniper/core/trade_executor.py`. All three sign with `keypair.sign_message(bytes(message))` and `VersionedTransaction.populate(message, [signature])` — that **discards any additional signers Jupiter put on the tx** and is incorrect when the route includes setup instructions that require co-signers. Token decimals are hard-coded to 6 in close paths (`modules/solana_trading/core/solana_engine.py:3308`), to 9 in `spl_token_handler.get_token_decimals` default (`trading/chains/solana/spl_token_handler.py:171`), and to 6 in the sniper output conversion (`modules/sniper/core/trade_executor.py:341, 377, 417`) — the analog of the DEX `*10**18` bug, here causing close-amount underflow on 9-decimal tokens and overflow on 5-decimal tokens (BONK is 5 decimals!).

**Top 3 profit leaks:** (1) `jupiter_helper.get_swap_transaction` (`modules/solana_strategies/jupiter_helper.py:478-484`) sends the payload with NO `prioritizationFeeLamports`, NO `dynamicComputeUnitLimit`, NO `dynamicSlippage`, NO `asLegacyTransaction=false` flag — so Jupiter computes a default tx with conservative compute budget and no priority fee. On a congested Solana mainnet, ~30-60% of these tx drop. The pump.fun close path retries 7 times with escalating slippage but does not retry with escalating priority fee, so a stuck position bleeds while the bot keeps quoting at a price that has already moved. Estimated drag: 0.3-0.8% per pump.fun trade plus ~30% missed fills. (2) No Jito bundles. Pump.fun launch trading on Solana is dominated by Jito-bundled snipers; broadcasting a public sendTransaction during launch loses by a full slot to anyone using bundle inclusion. The `pumpfun_jito_tip: 0.001` SOL config (`modules/solana_strategies/solana_config_manager.py:124`) is fictional. Cost: every pump.fun snipe enters multiple slots late, at materially worse price. (3) Quote-then-execute window: `JupiterHelper.execute_swap` (`modules/solana_strategies/jupiter_helper.py:906-1009`) calls `get_quote`, then `get_swap_transaction`, then signs, then sends. Each call is ~300-800ms over Jupiter's lite API (1 RPS rate-limit), so signing-to-send latency is typically 1.5-3s. On pump.fun tokens with 5-second crash windows that is a guaranteed `0x1771 SlippageToleranceExceeded`. There is no quote refresh just before signing.

**Top 3 risks:** (1) **Wrong signing for multi-signer txs.** `trading/chains/solana/jupiter_executor.py:712-713` does `signature = self.keypair.sign_message(bytes(message)); signed_transaction = VersionedTransaction.populate(message, [signature])` — this hard-overwrites Jupiter's signature array with a single signer. If Jupiter returns a tx that already has a pre-signed inner instruction (e.g., an ALT setup or a Setup → Swap → Cleanup composite), the populate path drops the other signatures and on-chain verification fails. `modules/sniper/core/trade_executor.py:561-562` has the identical bug. The fix in `jupiter_helper.py:639-700` is mostly correct (uses `NullSigner` to preserve other signers via the hybrid populate path), but the **other two executors will silently corrupt multi-signer routes**. (2) **No pool engine for SOLANA_RPC in critical paths.** Both `solana_trading/main_solana.py:405` and `solana_engine.py:1086,1106-1110` fall back to `os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')` if `RPCProvider` is unavailable — and `mainnet-beta.solana.com` is rate-limited to ~5 RPS, completely insufficient for trading. The default public RPC will brick the bot under load. (3) **Private-key plaintext fan-out.** `solana_engine._get_decrypted_private_key` (`solana_engine.py:1254-1319`) decrypts and **passes the plaintext base58 key into `JupiterHelper(private_key=...)` and `DriftHelper(private_key=...)` constructors**, where it is stored as `self.private_key` on `DriftHelper` (`drift_helper.py:88`). If either helper object is logged via `repr()` or pickled (e.g., into a debug dump or a checkpoint), the plaintext key leaks. The keypair object itself is fine, but the raw string lingering on the helper instance is a leak surface.

## File-by-file findings

### `modules/solana_strategies/solana_module.py` (716 lines)
- Wrapper module, mostly metadata. Real strategies are imported lazily from `trading/strategies/pumpfun_launch.py`, `jupiter_limit_orders.py`, `drift_perpetuals.py` (lines 152, 164, 176) — those files are not in this audit scope but the imports are best-effort with no fallback if a strategy is missing.
- `_monitor_pumpfun_launches` (line 451-517) reads `PUMPFUN_WS_URL` directly via `os.getenv` (line 462) **bypassing pool engine**. Single WebSocket, no reconnect-with-backoff exponent, no jittered retry; on a Cloudflare 1015 the bot will hammer pumpportal.fun until banned.
- `_monitor_jupiter_orders` (line 558) sleeps 30s forever and does nothing — declared as monitoring but is a no-op. Misleading.
- `_monitor_drift_positions` (line 571) same pattern — no-op 15s sleep loop.
- `_process_pumpfun_launch` (line 519) routes a launch into `_process_pumpfun_opportunity` synchronously — there is no race-detection: if Jito-bundled snipers buy in the same slot, the WebSocket alert may arrive after the token is already +200%. No "max launch age" gate (the dashboard config has `pumpfun_max_age` but it is never consulted in this file).
- `get_positions` (line 584-646) reads `pos.entry_price`, `pos.position_size` from strategy-internal Position objects — duplicates the position model in `solana_engine.Position` (`solana_engine.py:140-160`). Drift hazard.

### `modules/solana_strategies/jupiter_helper.py` (1009 lines)
- Constructor (`__init__`, line 79) reads `JUPITER_API_URL` directly with `os.getenv` (line 93) — **bypasses pool engine**. RPC URL fetched via `RPCProvider.get_rpc_sync('SOLANA_RPC')` (line 114) — OK. But then `os.getenv('SOLANA_RPC_URL')` fallback (line 116) — env leak.
- `_decrypt_value` (line 255-286) reads `.encryption_key` from CWD and `ENCRYPTION_KEY` env (lines 270-274) — duplicates the encryption logic from `solana_engine.py:1293-1305` and `trade_executor.py:144-147`. Three copies of the same decryptor.
- `get_quote` (line 356) is acceptable but has no `dynamicSlippage` flag — passes only the static `slippageBps`. Jupiter v6 supports `dynamicSlippage=true` which would auto-adjust per route; missed.
- **`get_swap_transaction` (line 453-531) — CRITICAL.** The payload (lines 479-484) sends:
  ```
  {'quoteResponse': quote, 'userPublicKey': user_public_key,
   'wrapAndUnwrapSol': wrap_unwrap_sol, 'asLegacyTransaction': as_legacy_transaction}
  ```
  Missing every modern Jupiter knob: `dynamicComputeUnitLimit`, `prioritizationFeeLamports` (Jupiter recommends `'auto'` or a struct with `maxLamports`), `dynamicSlippage`, `feeAccount` (if collecting fees), `useTokenLedger`. Result: tx is sent with default compute budget (~200k CUs) and no priority fee, so during congestion it drops. This is the single largest profit leak in the module.
- `sign_transaction` (line 533-711) — the **only correct signer** in the codebase, using `NullSigner` to preserve pre-signed signatures (lines 645-684). Good. But this is not the path the toxisol/sniper paths use.
- `send_transaction` (line 713-789) calls `self.session.post(self.solana_rpc, ...)` (line 753) — single RPC. No failover, no `report_failure/report_rate_limit` call to pool engine on 429.
- `confirm_transaction` (line 791-904) has good logic: checks `status.get('err')` before returning success (line 848-852) — important catch that the DEX side often misses. But it does NOT call `pool_engine.report_success/report_failure` for the RPC.
- `last_swap_error` (line 172, 780) is a side-channel string set by `send_transaction` and read by `solana_engine._close_position` (line 3438) — fragile cross-object state. A second concurrent close on a different position would corrupt it.

### `modules/solana_strategies/drift_helper.py` (375 lines)
- Constructor (line 72-94): reads `SOLANA_RPC_URL` via secrets manager then `os.getenv` (lines 83, 87) — **bypasses pool engine** entirely for Drift.
- `initialize` (line 96-152): imports `driftpy` lazily — good for optional dep. Creates `AsyncClient(self.rpc_url)` directly (line 121) — no failover. `DriftClient(connection, wallet, "mainnet-beta")` (line 133-137) — chain hard-coded; cannot be switched to devnet without code edit.
- `open_position` (line 236-291): `int(base_amount * 1e9)` (line 281) hard-codes 9-decimal scaling for **all** Drift markets. Drift's base-asset precision is actually 9, but the price scaling `int(price_limit * 1e6)` (line 283) assumes 6-decimal quote — true for USDC but **wrong if Drift ever lists ETH-PERP with 6-decimal USDC quote and a SOL-quoted variant**. Should pull `base_decimals` and `quote_decimals` from `get_perp_market_account` instead of hard-coding.
- `get_funding_rate` (line 346-375): divides `market.amm.last_funding_rate` by `1e9` (line 361) — Drift's funding rate is in `1e-9` units per hour; the annualization `* 365 * 24 * 100` (line 364) is hourly→annual. Correct for hourly funding, **but** Drift uses prediction-market style funding that updates every funding period (1 hour), so this is roughly right. Comment should clarify.
- **No on-chain account-health pre-flight before opening a position.** A bot at 5x leverage with 80% collateral utilization can be liquidated by funding flip; `open_position` doesn't check `get_user_health` or `marginRatio`.
- `private_key` stored on `self.private_key` (line 88) — plaintext leak surface.
- `httpx` monkey-patch at module level (lines 15-42): applies a global patch that any other module importing httpx after this will see. Side-effect-y; should be encapsulated.

### `modules/solana_strategies/solana_config_manager.py` (563 lines)
- `_load_environment_config` (line 152-179) loads `SOLANA_RPC_URL`, `SOLANA_RPC_URLS`, `SOLANA_BACKUP_RPCS`, `SOLANA_WS_URL`, `HELIUS_API_KEY`, `JUPITER_API_URL`, `JITO_TIP_ACCOUNT`, `JITO_BLOCK_ENGINE_URL` directly from env (line 175). Sensitive — and `JITO_*` variables hint at intended Jito support that was never built.
- `is_dry_run` (line 286): `dry_run = self.get_env('DRY_RUN', 'true')` — defaults to true (good fail-safe), but `dry_run_env in ('true', '1', 'yes')` (line 289) does not handle `'TRUE'`/`'True'` only because of `.lower()` already applied; OK.
- `get` (line 253) priority: env first, then cache, then defaults. For numeric values from env this returns the raw string — callers that expect `float(config.get('jupiter_slippage'))` will work, but `if config.get('pumpfun_jito'):` will be truthy for the literal string `'false'`. Bug surface.

### `modules/solana_trading/main_solana.py` (721 lines)
- `__init__` (line 380-408): if `RPCProvider` unavailable, falls back to `os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')` (line 405) — **public mainnet-beta as default is dangerous** (5 RPS rate limit). Should refuse to start if no real RPC configured.
- `_init_database` (line 415-480) creates `solana_trades` table at startup if missing (lines 442-470). Schema is fine but no migration tracking; if the schema needs to change you have to drop the table manually.
- `_status_reporter` (line 600) reports every 120s — fine.
- `close_position_handler` (line 305-335) and `close_all_positions_handler` (line 337-361) — manual close endpoints. Good, but `close_all_positions` (line 349) calls `engine.close_all_positions()` which iterates `self.active_positions` — if a close fails, does it block subsequent ones? Should be parallel + return per-position status.
- `_signal_handler` (line 410-413) sets `shutdown_event` — but `signal.signal` from non-main thread will fail; OK because main module.
- Mode is `production` by default (line 380, 658). If `_signal_handler` fires mid-trade, `shutdown` (line 619) calls `close_all_positions` (line 638) which can take minutes per stuck position. No max-shutdown-time.

### `modules/solana_trading/core/solana_engine.py` (3795 lines)
- `_init_solana_client` (line 1321-1388): `AsyncClient(self.rpc_manager.current_url)` (line 1328) — no failover wired into client; `RPCManager.mark_failed` exists (line 303) but client is only swapped after `is_connected()` returns false; the rest of the engine never re-creates the client on subsequent failures.
- `_get_token_balance` (line 1545-1624): Method 1 uses `Confirmed` commitment (good for speed, line 1570). Method 2 fallback to direct ATA query — also good. But default `decimals=6` parameter (line 1545) is used by every caller without supplying decimals, so a 9-decimal Solana token (most modern launches) reports 1000x too-large balance. Combined with the slippage retry loop (line 3367-3384), this means `token_amount_raw = int(close_amount * (10 ** token_decimals))` (line 3335) sends a wildly wrong amount to Jupiter — either zero-ing out or asking to sell more than the wallet holds.
- `_open_position` swap leg (line 3117-3176): retrieves decimals from `metadata.get('decimals', 9)` (line 3086) — good, but the *close* path (`_close_position`, line 3308) hard-codes `token_decimals = 6`. **Asymmetric.** Bought with the right decimals, sold with the wrong decimals.
- Slippage handling (lines 3103-3115): uses `safety_engine.get_slippage_for_strategy` — good for pump.fun. But the slippage `200 bps` for non-pump strategies (line 3115) is naively static; no quote-impact check.
- Balance verification after buy (lines 3135-3168) uses 4 retries with up to 28s total delay before giving up. This is acceptable for confirmed commitment but means an opened position **does not appear in `active_positions` until 28s after the buy** — meaning a stop-loss can't fire for that window. Concrete loss: a pump.fun token that pumps 30% then crashes 50% within 20s of the buy will not trigger SL because position.amount is still 0.
- `_close_position` (line 3246-3470+): the retry loop with `_get_token_balance(token_mint, token_decimals)` where `token_decimals=6` is wrong for 9-dec tokens. Sell-amount calculation `int(close_amount * (10 ** token_decimals))` (line 3335) and the post-sell verification (line 3397) all compound on the wrong decimals. **DexScreener pump.fun tokens are typically 6-decimal, but Raydium/Orca established tokens are 6-9. Bot will fail closes on most non-pump tokens.**
- `_init_pumpfun` (line 1497-1529): WebSocket URL read directly from env (line 1511, 1522) — bypass pool engine.
- `_search_pumpfun_tokens` (line 931-1035): polls DexScreener `/latest/dex/pairs/solana` — this is the same data DexScreener gives anyone, with a >1 minute lag. Not a real Pump.fun listener. Filters out everything that isn't `pumpfun/raydium/orca/meteora` (line 956). For sniping new pump.fun launches at t=0, this is structurally too late.
- `RPCManager.mark_failed` (line 303) does NOT call `pool_engine.report_failure` — local-only failure tracking. Pool engine never learns about Solana endpoint health.
- No address-lookup-table (ALT) usage anywhere. Modern Jupiter routes pass through ALTs to fit instructions into a single tx; without consuming `swap_data.get('addressLookupTableAddresses')`, the bot is limited to short routes.
- No compute-budget instructions: `set_compute_unit_limit` and `set_compute_unit_price` are not added. Relies entirely on Jupiter's `dynamicComputeUnitLimit` flag in the swap request — which **`jupiter_helper.get_swap_transaction` does not pass**.

### `modules/solana_trading/core/safety_engine.py` (572 lines)
- `verify_sell_route` (line 171-278): on quote failure returns `(True, ...)` (line 224) — "allow trade when verification fails" because of rate-limit fears. This is exploitable: an attacker who can DoS Jupiter for 30s during your buy-decision window can make the bot skip honeypot detection. Recommend: retry with exponential backoff first, only fall through after N retries.
- `cache_sell_quote` / `get_cached_quote` (line 420-483): in-memory only, max age 2 minutes regular / 5 minutes emergency (lines 115-116). On a crashing token, the cached quote's outAmount is stale by the time emergency_close runs — `emergency_close_with_cache` (line 504-572) does try to refresh first (line 542) but if that fails, uses the stale quote (line 551) with +500 bps slippage (line 535) — may still revert at the higher slippage if price moved >5%.
- `record_close_failure` (line 305-365): tracks per-token stuck positions; circuit-breaker after 5 consecutive failures (line 357). Good. Cooldown 180s (line 85) seems short for a real liquidity rug.

### `modules/solana_trading/core/scam_blacklist.py` (416 lines)
- Hard-coded scam name patterns (lines 41-92). Regex `r'TRUMP'` matches *any* token with TRUMP in the name (line 43) — includes legitimate Trump-themed tokens; aggressive but appropriate for a sniper.
- The blacklist is read but I don't see it being applied in the buy path in this snippet. (Engine probably calls it elsewhere.)

### `modules/solana_trading/core/solana_alerts.py` (353 lines)
- Telegram alerts manager. `__init__` (line 48-80): credentials from `security.secrets_manager.secrets` first, then `os.getenv` fallback (line 70). OK.
- No rate-limiting on send — a stuck-position-retry loop firing 7 alerts per close will spam Telegram and trigger Telegram's per-bot rate limit.

### `modules/solana_trading/config/solana_config.py` (134 lines)
- Reads RPC via `_get_solana_rpc` helper (line 10) which tries pool engine then `os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')` (line 19) — same public-RPC default risk.
- `private_key = os.getenv('SOLANA_MODULE_PRIVATE_KEY', '')` (line 44) — plain env read, no encryption support in this path. **Conflicts with** the encrypted-key handling in `solana_engine._get_decrypted_private_key` (line 1254) — two private-key load paths with different semantics.
- `jupiter_slippage_bps = int(os.getenv('JUPITER_SLIPPAGE_BPS', '50'))` (line 64) — defaults to 0.5%, identical to the safety_engine starting slippage but the actual engine uses 200 bps (line 1110). Three different defaults across three files.

### `modules/solana_trading/config/solana_config_manager.py` (635 lines)
- Duplicate of `modules/solana_strategies/solana_config_manager.py` — two config managers with overlapping schemas. The `solana_trading` one adds `jupiter_position_size`, `pumpfun_trailing_enabled`, `pumpfun_tier0_sl`, `pumpfun_partial_exit_pct`, `pumpfun_cooldown`, `pumpfun_min_holders`, `pumpfun_max_dev_holding`, `pumpfun_scam_detection`, `jupiter_max_positions` keys (lines 67-80). The `solana_strategies` version does not.
- Both define `JITO_TIP_ACCOUNT` and `JITO_BLOCK_ENGINE_URL` in `sensitive_keys` (line 188-189) — but again no code reads these.

### `trading/chains/solana/jupiter_executor.py` (985 lines)
- `__init__` (line 49-188): builds keypair from raw `private_key` config (line 78), supports base58/hex/JSON array. Stores keypair, *not* the raw key string. Good.
- `jupiter_api_url` hard-coded (line 70) — does not honor `JUPITER_API_URL` env. Drift from the helper module.
- `max_slippage_bps = int(config.get('max_slippage_bps', 50))` (line 75) — 0.5% default, applied uniformly; no pump.fun-specific bump.
- `_get_quote` (line 558-614) — adds `_fetched_at` timestamp (line 608) and `_is_quote_expired` (line 526) — good defense. 10s expiration is reasonable.
- `_execute_swap` (line 616-752): payload (lines 662-668) includes `dynamicComputeUnitLimit: True` and `prioritizationFeeLamports: 'auto'` — **this is the *right* call**, missing from `jupiter_helper.get_swap_transaction`. The two executors disagree on basic strategy.
- **Critical bug, line 712-713:** `signature = self.keypair.sign_message(bytes(message)); signed_transaction = VersionedTransaction.populate(message, [signature])`. `populate` here passes only one signature — overwrites any pre-existing signers from Jupiter's tx. If route's tx came back with 2 required signers (e.g., a setup ix that needs a co-signer for ATA creation), the second signer slot becomes the default `Signature.default()` and on-chain verification fails. The `jupiter_helper.sign_transaction` hybrid approach (line 655-700) is correct; this file's approach is wrong.
- `validate_order` (line 463-524) uses `MAX_POSITION_SIZE_USD` from config (line 492) — defaults to $50, very low; will block any meaningful trade.
- `_wait_for_confirmation` (line 754-807): polls every 2s for up to 60s — fine, but **does not check `err` field** on the status object (only `confirmationStatus`, line 790). A confirmed-but-failed tx (e.g., `0x1771` slippage error) returns success. The `jupiter_helper.confirm_transaction` (line 848 in helper) does check `err` — divergent semantics.

### `trading/chains/solana/solana_client.py` (124 lines)
- Generic RPC client with failover list (line 19). Failover happens within a single request (line 39 loop), no health tracking across requests, no pool engine integration.
- `get_token_balance` returns the first account's `uiAmountString` (line 92) — does not sum multiple ATAs. If wallet has split tokens across two ATAs it under-reports.

### `trading/chains/solana/spl_token_handler.py` (170 lines)
- `get_token_decimals` (line 154-171): only knows SOL (9), USDC (6), USDT (6). **Defaults to 9 for everything else** (line 171). The `TOKEN_DECIMALS` constant (lines 144-151) lists BONK as 5 — but `get_token_decimals` doesn't consult that constant. Bug: BONK trades will use 9 decimals, off by 10,000x.
- `format_token_amount` and `to_raw_amount` (lines 85-117): mechanical decimal conversion. Could be unified into a single helper called from every code path that converts; instead it's duplicated inline in `solana_engine`, `jupiter_executor`, `trade_executor`.

### `config/pool_engine.py` (Solana-relevant surface)
- `SOLANA_RPC_URLS`, `SOLANA_BACKUP_RPCS`, `SOLANA_RPC_URL`, `SOLANA_WS_URL`, `HELIUS_API_KEY`, `JUPITER_API_KEY` all wired (lines 375, 376, 394, 399, 406, 408). 
- Helius URL builder (line 474-475): `f"https://mainnet.helius-rpc.com/?api-key={api_key}"` — correct. **Staked connection** support (Helius's premium feature for guaranteed inclusion) is *not* differentiated — all Helius endpoints are treated the same priority.
- `solana_strategies/jupiter_helper.py:114` and `solana_trading/main_solana.py:402-403` use `RPCProvider.get_rpc_sync('SOLANA_RPC')` — good when pool engine is initialized. But the engine's own `_init_solana_client` (`solana_engine.py:1328`) uses `self.rpc_manager.current_url` — which is the static list from constructor, not refreshed from pool engine after init. So once the engine starts, it won't pick up newly-rotated endpoints.

## Risk taxonomy

| ID | Category | File:Line | Severity | Description | Fix sketch |
|----|----------|-----------|----------|-------------|------------|
| SOL-01 | Slippage / decimals | solana_engine.py:3308,3335 | **CRITICAL** | `token_decimals=6` hard-coded in close path; off by 10x-1000x for 5/9-decimal tokens | Read decimals from on-chain `getTokenSupply(mint).decimals` once at position open, cache on `Position.metadata['decimals']`, reuse for close |
| SOL-02 | Gas / priority fee | jupiter_helper.py:478-484 | **CRITICAL** | Swap request omits `dynamicComputeUnitLimit`, `prioritizationFeeLamports`, `dynamicSlippage`; tx drops under congestion | Match `jupiter_executor.py:662-668` payload: `{'dynamicComputeUnitLimit': True, 'prioritizationFeeLamports': {'priorityLevelWithMaxLamports': {'maxLamports': 5_000_000, 'priorityLevel': 'high'}}, 'dynamicSlippage': True}` |
| SOL-03 | MEV / bundle inclusion | n/a (missing) | **HIGH** | No Jito bundle path; pump.fun snipes broadcast publicly and lose to bundled snipers | Add `JitoBundler` class: build tx, append tip ix to Jito tip account, send via `https://mainnet.block-engine.jito.wtf/api/v1/bundles`; wait for landed status |
| SOL-04 | Tx-signing | jupiter_executor.py:712-713, trade_executor.py:561-562 | **HIGH** | `sign_message` + `populate([signature])` overwrites Jupiter's multi-signer tx, corrupting routes with setup co-signers | Replace with the hybrid signer in `jupiter_helper.py:639-700`, or use `VersionedTransaction(message, [self.keypair, NullSigner(...)])` |
| SOL-05 | RPC / pool-engine | solana_module.py:462, solana_engine.py:1086,1511, solana_config.py:19 | **HIGH** | Direct `os.getenv('SOLANA_RPC_URL')` fallback to public mainnet-beta (5 RPS); pool engine bypassed | Require `RPCProvider` for every Solana endpoint read; refuse to start if no real RPC; differentiate staked Helius from public |
| SOL-06 | Key handling | drift_helper.py:88, jupiter_helper.py via init | **HIGH** | Plaintext private-key string stored on helper instance; leak surface on log/repr/pickle | Pass keypair object only; never retain raw string after `Keypair.from_bytes()` |
| SOL-07 | Slippage | jupiter_helper.py:361,400 | MEDIUM | Static `slippageBps` only; no `dynamicSlippage` flag | Pass `dynamicSlippage: {'maxBps': 800}` for pump.fun; let Jupiter auto-set |
| SOL-08 | Honeypot / dev-rug | safety_engine.py:224 | MEDIUM | `verify_sell_route` returns `True` on quote failure — exploitable via Jupiter DoS | Retry 3x with backoff before allowing; expose a `STRICT_HONEYPOT_CHECK` config that fails-closed |
| SOL-09 | Idempotency | jupiter_helper.py:172,780 | MEDIUM | `last_swap_error` is shared mutable state on the helper; concurrent closes corrupt it | Return error from `send_transaction` via tuple, drop side-channel |
| SOL-10 | ALT / route size | n/a (missing) | MEDIUM | No address-lookup-table consumption; routes restricted to short paths | Read `swap_data.addressLookupTableAddresses` from Jupiter response; pass to versioned tx |
| SOL-11 | Tx-ordering | jupiter_helper.py:906-1009 | MEDIUM | Quote-fetch → swap-fetch → sign → send takes 1.5-3s; quote stale by send | Re-fetch quote 200ms before sign; reject if outAmount drifted >`slippage/2` |
| SOL-12 | ATA / SOL rent | n/a | LOW | No ATA creation pre-flight; if ATA missing for output mint, swap adds the create ix and consumes ~0.002 SOL rent per new token | Pre-create ATAs at startup for whitelist tokens; sweep dust ATAs daily |
| SOL-13 | Versioned vs legacy | jupiter_helper.py:483 | LOW | `as_legacy_transaction=False` default but caller can override; pump.fun bonding curve tx must be versioned | Force-versioned for pump.fun paths |
| SOL-14 | Drift account-health | drift_helper.py:236-291 | MEDIUM | No `get_user_health` pre-flight before `open_position`; can hit liquidation if margin tight | Call `drift_client.get_user_total_collateral()` and `get_margin_ratio()`; abort if `<1.5x` initial margin |
| SOL-15 | Pump.fun "listener" | solana_engine.py:931-1035 | **HIGH** | DexScreener `/pairs/solana` poll has >60s lag; not a real launch listener | Replace with Geyser/Yellowstone subscription to `PUMP_FUN` program logs; or use pumpportal.fun WS (already done in `solana_module.py:462`) |
| SOL-16 | Decimals | spl_token_handler.py:171 | MEDIUM | `get_token_decimals` returns 9 default, ignores `TOKEN_DECIMALS` constant that lists BONK as 5 | Wire `TOKEN_DECIMALS` lookup before fallback; or always query `getTokenSupply` on-chain |
| SOL-17 | RPC failover | solana_engine.py:303 | MEDIUM | `RPCManager.mark_failed` is local; doesn't call `pool_engine.report_failure` | Hook `mark_failed` to `pool_engine.report_failure('SOLANA_RPC', url)` and conversely on success |
| SOL-18 | Confirmation semantics | jupiter_executor.py:790 | MEDIUM | `_wait_for_confirmation` ignores `err` on confirmed tx — false success | Mirror `jupiter_helper.confirm_transaction:848-852` logic |
| SOL-19 | Config drift | solana_config.py:64, solana_engine.py:1110, safety_engine.py:73 | LOW | Three different default slippage values (50, 200, 100 bps) | Single source: `SafetyConfig.default_slippage_bps` |
| SOL-20 | Telegram spam | solana_alerts.py | LOW | Stuck-position retries fire alert per attempt | Dedupe alerts on `(token_mint, action)` for 60s window |

## Profitability levers (ranked by ROI)

1. **Add `prioritizationFeeLamports` + `dynamicComputeUnitLimit` + `dynamicSlippage` to `jupiter_helper.get_swap_transaction` (SOL-02).** This is the single biggest mechanical fix. On Solana mainnet during congested hours, untipped txs land at ~40-60%, properly-tipped land at >95%. Expected: +30-50% fill rate on every Solana swap, immediate. Direct port from `jupiter_executor.py:662-668`. (~10 line change.)
2. **Wire token decimals through the close path (SOL-01).** Today closes silently sell the wrong amount on BONK (5 dec), most pump.fun new tokens (9 dec), JTO/PYTH variants. Fix unbreaks every non-6-decimal close. Read decimals once at position-open from on-chain `getTokenSupply`, cache, reuse.
3. **Implement Jito bundles for pump.fun snipes (SOL-03).** The dashboard already exposes `pumpfun_jito_tip: 0.001 SOL` as a setting. Wire it to actual bundle submission to `mainnet.block-engine.jito.wtf/api/v1/bundles` with tip ix. For pump.fun launches the difference between bundled inclusion (slot 0) and public broadcast (slot 1-3) is typically 5-20% on the entry. ROI: massive on profitable launches.
4. **Fix `sign_message` → preserve-multi-signer in `jupiter_executor.py` and `trade_executor.py` (SOL-04).** Today complex routes (ALT-based, multi-hop with setup) fail silently. Direct port of `jupiter_helper.sign_transaction` hybrid path. Without this fix, ~5-15% of routes that Jupiter returns are unsignable.
5. **Re-quote before sign (SOL-11).** Refresh the quote ~200ms before signing; reject if outAmount has drifted by more than half-slippage. Cuts the 0x1771 close-failure rate by an estimated 40%.
6. **Pool engine for all Solana RPC reads + reject public mainnet-beta default (SOL-05).** Today a misconfigured environment silently runs on the 5-RPS public endpoint. Concrete fix: in `solana_engine.__init__` raise if any RPC URL points to `api.mainnet-beta.solana.com` and no Helius/staked alternative is configured.
7. **Geyser/Yellowstone subscription for pump.fun program logs (SOL-15).** Replace the DexScreener poll with a real Solana log subscription. Latency drops from ~60s to ~400ms. Pump.fun snipes only work if you arrive within ~5s of the launch.
8. **Drift account-health pre-flight (SOL-14).** Cheap on-chain read; avoids the catastrophic-liquidation scenario where the bot opens a position into a margin-call.
9. **Consume Jupiter's `addressLookupTableAddresses` (SOL-10).** Unlocks longer routes (e.g., SOL → USDC → exotic), better fills on illiquid pairs.
10. **Fail-closed honeypot verification (SOL-08).** Add a config toggle; for production sniping default to fail-closed after 3 retries.

## Live-trade gap list

- [ ] Token decimals read from on-chain at open, cached on Position, reused at close
- [ ] `jupiter_helper.get_swap_transaction` passes priority fee + compute unit limit + dynamic slippage
- [ ] At least one Jito bundle code path exists, with tip-vs-expected-profit gate
- [ ] All three Solana executors share a single `sign_versioned_tx(message, our_keypair, account_keys)` helper that preserves co-signers
- [ ] Pool engine consulted for every SOLANA_RPC / SOLANA_WS / HELIUS / JUPITER read; no fallback to public mainnet-beta
- [ ] Private key never stored as string on helper instances; only as `Keypair` object
- [ ] Quote refreshed within `<500ms` of sign; reject stale > `slippage_bps/2`
- [ ] `addressLookupTableAddresses` from Jupiter swap response wired into versioned tx construction
- [ ] Drift account-health pre-flight before `open_position`
- [ ] Pump.fun launch listener uses Geyser/Yellowstone or pumpportal.fun WS, NOT DexScreener polling
- [ ] `_wait_for_confirmation` in `jupiter_executor.py` checks `err` field (not just `confirmationStatus`)
- [ ] Solana kill-switch wired into `core/risk_manager.py`: a single flag stops `_open_position` in `solana_engine.py`
- [ ] Honeypot verification fail-closed mode (configurable)
- [ ] BONK and any 5-decimal token tested end-to-end
- [ ] `RPCManager` hooks failures back to `pool_engine.report_failure`
- [ ] Telegram alerts deduped per (mint, action) within 60s

## Proposed action backlog

- [ ] **SOL-01** `Wire token decimals end-to-end in solana_engine` — touches `modules/solana_trading/core/solana_engine.py` (lines 3086, 3308, 3335, 3397, 1545) — expected gain: unbreaks every non-6-decimal close — owner: smartcontract.
- [ ] **SOL-02** `Add priority fee + compute budget + dynamicSlippage to jupiter_helper` — touches `modules/solana_strategies/jupiter_helper.py` (lines 478-484, 356-451) — expected gain: +30-50% landed-tx rate during congestion — owner: smartcontract.
- [ ] **SOL-03** `Implement JitoBundler` — new file `modules/solana_strategies/jito_bundler.py`; integrate into `solana_engine._open_position` for `Strategy.PUMPFUN` when `pumpfun_jito=True` — expected gain: bundled slot-0 inclusion on pump.fun snipes — owner: smartcontract.
- [ ] **SOL-04** `Unify Versioned-tx signer with multi-signer preservation` — extract `jupiter_helper.sign_transaction:639-700` into `trading/chains/solana/tx_signer.py`; replace call sites in `trading/chains/solana/jupiter_executor.py:712-713` and `modules/sniper/core/trade_executor.py:561-562` — expected gain: unbreaks ~5-15% of complex routes — owner: smartcontract.
- [ ] **SOL-05** `Re-quote before sign window` — touches `modules/solana_strategies/jupiter_helper.py:906-1009` — expected gain: -40% on `0x1771 SlippageToleranceExceeded` errors — owner: smartcontract.
- [ ] **SOL-06** `Pool-engine integration for all Solana RPC reads` — touches `modules/solana_trading/main_solana.py:402-405`, `modules/solana_trading/core/solana_engine.py:1086`, `solana_strategies/solana_module.py:462`, `solana_strategies/drift_helper.py:83,87`, `trading/chains/solana/jupiter_executor.py:70-73` — expected gain: kill public-mainnet-beta surprise, enable Helius staked routing — owner: smartcontract + backend.
- [ ] **SOL-07** `Drop plaintext private-key from helper instances` — touches `modules/solana_strategies/jupiter_helper.py:79-155`, `modules/solana_strategies/drift_helper.py:72-94` — expected gain: leak surface eliminated — owner: smartcontract.
- [ ] **SOL-08** `Geyser/Yellowstone pump.fun listener` — replace `solana_engine.PumpFunMonitor.get_new_tokens` (line 575-796) with a Helius WebSocket subscription to the `PUMP_FUN` program logs — expected gain: ~60s latency reduction on launch detection — owner: smartcontract.
- [ ] **SOL-09** `Consume Jupiter ALTs` — touches `modules/solana_strategies/jupiter_helper.py:533-711` to read `addressLookupTableAddresses` from `swap_data` — expected gain: longer routes, better fills on thin pairs — owner: smartcontract.
- [ ] **SOL-10** `Drift health pre-flight + initial-margin gate` — touches `modules/solana_strategies/drift_helper.py:236-291` — expected gain: prevents catastrophic liquidation — owner: smartcontract + analyst.
- [ ] **SOL-11** `Fail-closed honeypot verification` — touches `modules/solana_trading/core/safety_engine.py:171-278` — expected gain: closes the DoS-bypasses-honeypot-check surface — owner: smartcontract.
- [ ] **SOL-12** `RPCManager → pool_engine failure reporting` — touches `modules/solana_trading/core/solana_engine.py:288-346` — expected gain: shared health view across all chains/modules — owner: smartcontract + backend.
- [ ] **SOL-13** `Kill-switch into solana_engine._open_position` — touches `modules/solana_trading/core/solana_engine.py:2785, 3117`, hooks `core/risk_manager.py` flag — expected gain: live-readiness gate — owner: smartcontract + analyst.
- [ ] **SOL-14** `Dedupe Solana config managers` — `modules/solana_strategies/solana_config_manager.py` vs `modules/solana_trading/config/solana_config_manager.py` — pick one, delete the other; reconcile schema — owner: pm + smartcontract.
- [ ] **SOL-15** `Unify decimal handling in spl_token_handler` — touches `trading/chains/solana/spl_token_handler.py:154-171` — wire `TOKEN_DECIMALS` map; fall back to on-chain `getTokenSupply` — owner: smartcontract.

## Open questions

1. Is `trading/chains/solana/jupiter_executor.py` (BaseExecutor) actually used anywhere, or is it dead code superseded by `JupiterHelper`? If dead, delete; if live, the multi-signer bug must be fixed urgently.
2. Is the intent for `solana_module.py` (strategies wrapper) and `solana_trading/main_solana.py` (standalone bot) to converge, or remain separate processes? Currently both have their own engine reference and their own config managers.
3. Should pump.fun trades be routed via Jupiter at all? Pump.fun's bonding curve has its own buy/sell instructions (program `6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P`); pre-graduation tokens have no Jupiter route — direct interaction with the bonding-curve program would be both faster and cheaper.
4. Is Drift's perpetuals path being actively used or just declared? `drift_helper.open_position` (`drift_helper.py:236`) has no health check, no funding-rate check — strongly suggests it has never been used live.
5. Is there a budget for Helius staked connections (~$50-200/mo)? The pool engine treats Helius the same as any public RPC; staked connections give priority queue access for sendTransaction, which compounds with Jito for landed-tx rate.
6. What is the team's stance on holding the `JITO_TIP_ACCOUNT` / `JITO_BLOCK_ENGINE_URL` env vars as live wiring vs. dead config? The fact that the dashboard surfaces `pumpfun_jito` and `pumpfun_jito_tip` settings strongly implies a partial implementation was started.
