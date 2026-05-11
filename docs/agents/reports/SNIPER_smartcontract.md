# SNIPER_MODULE — Smart-Contract / Web3 Audit

## Executive summary

**Verdict: RED.** The sniper module *cannot snipe* in any meaningful sense. Both listeners and both executors are structurally too slow and too unsafe to compete with bundled snipers on either chain. The EVM listener (`modules/sniper/core/evm_listener.py`) polls `eth_getLogs` every loop iteration over the last 5 blocks (`evm_listener.py:96-108`) — block time on Ethereum is 12s, on Base/Arb 0.3-2s, and the inner sniper loop sleeps 0.1s (`sniper_engine.py:279`); a launch detected on Ethereum block N will not enter the buy path until at minimum N+5 blocks (60s) later, by which time the launch is over. Worse, the `FACTORIES` dict (`evm_listener.py:33-37`) lists only Uniswap V2 and SushiSwap mainnet factories — no Base/Arb/BSC/Pancake factories, no Uniswap V3 / V4, no Maverick, no Curve. The `_parse_log` function (line 131) returns `token0` as the "interesting token" (line 117) without checking which side is the non-WETH/non-USDC token. The Solana listener (`solana_listener.py`) is at least multi-AMM (Raydium V4, CPMM, Pump.fun, Orca, Meteora — lines 38-44) and uses `getSignaturesForAddress` + per-tx `getTransaction` with 15s poll interval (line 93) — but Pump.fun launches finish their bonding curves in <30s; this listener is also structurally too late. **There is no WebSocket / Geyser subscription anywhere.** Race-window from launch on chain → buy tx broadcast is on the order of 15-120 seconds; competitive snipers operate at 50-400ms.

The trade executor (`modules/sniper/core/trade_executor.py`) carries the same hard-coded decimal bugs as the Solana module (lines 341, 377, 706), the same wrong VersionedTransaction signing pattern as `trading/chains/solana/jupiter_executor.py` (line 561-562), and an even more dangerous EVM swap that sets **`amount_out_min = 0`** explicitly with the comment `"Accept any amount (risky, but for speed)"` (line 621). Combined with no Flashbots / Mempool-private routing, this makes the EVM sniper a guaranteed sandwich victim on any meaningful launch. Token-safety checks (`token_safety.py`) run BEFORE the buy, sequentially across multiple HTTP APIs (GoPlus, Honeypot.is, RugCheck) — adding 500-2000ms per check (line 154-595). The safety check is the *slowest* part of the pipeline and runs **inside** `_check_filters` (`sniper_engine.py:349`) before the buy is fired; ~2s lost to honeypot checks on every candidate, on top of the listener lag.

**Top 3 profit leaks:** (1) **No real-time listening.** Both chains poll on 100ms/15s intervals over HTTP `getLogs` / `getSignaturesForAddress`. By the time a "new pair" is acknowledged the entry price has moved ≥10x or the rug already happened. Fix: WebSocket subscriptions (`eth_subscribe('logs', {topics:[PairCreated]})` for EVM; Helius/Yellowstone Geyser for Solana). (2) **`amount_out_min = 0` on EVM buys** (`trade_executor.py:621`) — guaranteed sandwich. Every successful EVM snipe is paying maximum extractable slippage to whoever frontruns the buy. (3) **No private mempool / Jito bundle.** Public broadcast of the buy tx on EVM tells the world "the sniper just identified a token"; copy-trade bots react in the same block. Solana side has zero Jito tip path, same story.

**Top 3 risks:** (1) **Honeypot pre-check is not a buy/sell simulation** — it queries GoPlus and RugCheck (external HTTP APIs), which often have stale data or no entry for a 30-second-old launch. RugCheck for Solana (`token_safety.py:441-451`) hits `api.rugcheck.xyz/v1/tokens/{mint}/report` — RugCheck typically does not have a profile for a pump.fun token at t<60s. The check returns no data, `is_honeypot=False`, and the bot buys what may genuinely be a honeypot. There is no on-chain `eth_call` simulation of `swapExactETHForTokens` and an immediate `swapExactTokensForETH` to verify roundtrip — the gold-standard honeypot detector. (2) **`amount_out_min = 0` allows 100% MEV loss.** Sandwich attacker can extract the entire trade, leaving the bot with dust. The comment `"In production, you'd get expected output from router.getAmountsOut"` (line 620) explicitly acknowledges this is a placeholder. (3) **Hard-coded decimals: `result.amount_out, expected_output / 1e6` (`trade_executor.py:341, 417`), `int(amount_in * 1e18)` on EVM sells (line 706).** New launches commonly use 18-decimal (EVM standard), 6-decimal, or 9-decimal Solana tokens; bot's amount_out reporting is wrong by 1000x for many tokens.

## File-by-file findings

### `modules/sniper/main_sniper.py` (267 lines)
- Initialization sequence: Logging (lines 99-125), Pool Engine bootstrap (line 195-203), DB connection (lines 182-188), `ConfigManager.initialize()` (line 211). Pool engine is initialized *after* the first fetch attempt (lines 154-163), so any failure on bootstrap defaults to `os.getenv('SOLANA_RPC_URL')` and `os.getenv('WEB3_PROVIDER_URL')` — bypass surface.
- The `if not solana_rpc and not evm_rpc: logger.warning(...)` (line 168) allows the engine to start with no RPCs at all — it will silently process nothing. Should be a hard error in production mode.
- `StderrToRotatingFile` (lines 24-86) — stderr-redirect-to-file with manual rotation. Reasonable but redundant if logging is configured properly; minor.
- `DRY_RUN` and live mode are not differentiated at this entry point — both engine and trade_executor independently read `DRY_RUN` env (line 213 of `sniper_engine.py`, line 187 of `trade_executor.py`). Drift hazard.

### `modules/sniper/core/sniper_engine.py` (793 lines)
- `__init__` defaults (`sniper_engine.py:114-123`): `dry_run=True`, `trade_amount=0.1`, `slippage=10.0`, `priority_fee=5000`, `min_liquidity=1000.0`. Loaded from DB by `_load_settings` (line 174-236). `priority_fee=5000` is interpreted as Gwei on EVM and Lamports on Solana — **same number, two completely different units** (line 234 of `trade_executor.py`). 5000 Gwei is way over-paying; 5000 lamports on Solana is too low.
- `_load_settings` (line 174-236) reads `take_profit_pct` (line 208) and `stop_loss_pct` (line 210) from DB — but the loop in `_monitor_active_snipes` (line 577) hard-codes `take_profit_pct = 50.0` and `stop_loss_pct = -20.0` (line 580-581) at the start of every iteration. **The DB values are loaded then ignored.**
- `_monitor_new_pairs` (line 251-282): inner loop, `await asyncio.sleep(0.1)` (line 279). Calls `evm_listener.get_new_pairs()` and `solana_listener.get_new_pools()` 10x/sec. The EVM listener polls `eth_getLogs` over the last 5 blocks every call — that is ~50 RPC calls per second to Ethereum. On free-tier Alchemy/Infura that exhausts the credit budget in minutes.
- `_evaluate_target` (line 284-303) routes to `_check_filters` which does the safety check synchronously inside the hot loop. **The hot loop is single-threaded** — a slow GoPlus API call blocks all subsequent target evaluation.
- `_check_filters` (line 305-411): cooldown cache by `token_address` only (line 316) — chain-agnostic, so the same address on different chains collides. Cooldown of 5 minutes (line 138) — fine.
- Test-mode (`test_mode=True`, line 123): relaxes `max_buy_tax` to 50% and `min_liquidity` to `test_mode_min_liquidity` (default $10, line 121) and allows `DANGER` rating (line 376-377). This is a *production* code path — if a misconfigured DB sets `test_mode=true`, the sniper will buy honeypots.
- `_execute_snipe` (line 461-516): calls `executor.execute_buy(token_address, chain, amount_in, slippage, priority_fee)`. No retry, no fallback, no MEV protection wired. If `result.success=False`, the target is marked `failed` and dropped — no escalation.
- `_log_snipe_to_db` (line 518-575): writes to `sniper_trades` table — good. `is_simulated=True` is **hard-coded** (line 564) — even live trades are logged as simulated. Bug.
- `_monitor_active_snipes` (line 577-625): every 1s polls `_get_token_price` (line 600). Solana price via `https://price.jup.ag/v4/price` (line 634) — this is the **deprecated** Jupiter v4 price endpoint; current is `https://api.jup.ag/price/v3` or DexScreener. v4 endpoint returns 404/410 for most tokens now.
- `_get_token_price` for EVM uses `pairs[0].get('priceNative', 0)` (line 650) from DexScreener — first pair is not necessarily highest-liquidity; should use `max(pairs, key=lambda p: liquidity)` like solana_engine does.
- `_exit_position` (line 657-698): same hard-coded TP/SL semantics. No partial exits, no trailing stop. No emergency-exit on rug-detection (e.g., liquidity removal event).
- `_log_exit_to_db` (line 700-782): logs exit but uses CoinGecko-fetched native price for USD conversion (line 708). On a launch crash, the native (SOL/ETH) price has not moved meaningfully — error here is tolerable.
- No kill-switch hook into `core/risk_manager.py`. The sniper can keep buying indefinitely as long as the engine is running.

### `modules/sniper/core/evm_listener.py` (160 lines)
- `FACTORIES` (lines 33-37): only Uniswap V2 mainnet and SushiSwap mainnet. **Missing**: Base Uniswap V3 (`0x33128a8fC17869897dcE68Ed026d694621f6FDfD`), Arbitrum V3, BSC PancakeSwap V2 (`0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73`) and V3, Polygon QuickSwap, Avalanche TraderJoe, Uniswap V4 PoolManager (on multiple chains). Result: 90% of EVM launch volume is invisible to the sniper.
- `__init__` (line 40-57): reads `WEB3_PROVIDER_URL` directly via `os.getenv` fallback (line 57) — bypasses pool engine on fallback.
- `initialize` (line 59-83): single `Web3(Web3.HTTPProvider(...))` with 10s timeout (line 70). No fallback RPC if connect fails — `is_configured=False` and listener silently does nothing.
- `get_new_pairs` (line 85-129): polls `eth_getLogs` from `current_block - 5` (line 98). On Ethereum this scans 60s of history; on Base (~2s blocks) scans 10s. Every poll re-fetches the same 5 blocks until they age out. With the 100ms outer loop (sniper_engine line 279), this hits the RPC 10x/sec for the same data. Caching not implemented.
- Event signature hard-coded for V2 `PairCreated(address,address,address,uint256)` (line 101) — V3 uses `PoolCreated(address,address,uint24,int24,address)`, different signature, NOT caught.
- `_parse_log` (line 131-160): returns `token0` as the interesting token (line 117) — wrong. The interesting token is the one that is NOT WETH/USDC/USDT/DAI. Should check both and pick the non-base token.
- No WebSocket subscription (`eth_subscribe('logs', {topics: [event_sig]})`). The whole listener is HTTP poll-based.
- No mempool monitoring — competitive snipers watch **pending** `addLiquidity` txs to enter ahead of the actual pool creation. This module does not do that. The `data/collectors/mempool_monitor.py` exists but is not wired into the sniper.

### `modules/sniper/core/solana_listener.py` (636 lines)
- `PROGRAM_IDS` (lines 38-44): correct mainnet program IDs for Raydium V4, Raydium CPMM, Pump.fun, Orca Whirlpools, Meteora. Good coverage.
- `_get_enabled_sources` (line 159-175): defaults to `raydium_v4,pump_fun` from `SNIPER_AMM_SOURCES` env (line 161) — Orca/Meteora not enabled by default. Misses Orca-launched tokens.
- `poll_interval = 15s` (line 93) — listener fires every 15s. Pump.fun bonding curves graduate in well under that. The internal sleep between AMM sources is 0.5s (line 258). Effective Pump.fun-detection latency: 0-15s + 0.5s × source_index + per-tx fetch time.
- `_get_recent_signatures` (line 335-384) calls `getSignaturesForAddress(program_id, limit=20)` — returns the 20 most recent signatures regardless of recency. If a program is busy (Raydium V4 is *very* busy), 20 signatures might cover the last ~3 seconds, missing earlier ones. Should use `before` cursor on first poll then `until` cursor on subsequent polls.
- `_check_pool_transaction` (line 386-463): for each new signature does a separate `getTransaction` RPC call. With 5 enabled sources × 20 sigs each = 100 extra RPC calls per poll cycle. Without Helius staked connections this exhausts free-tier credits.
- `_is_pool_init` (line 465-486): matches keywords in `logMessages` — `"initialize2"`, `"create_pool"`, etc. (line 47-53). Brittle: pump.fun uses `"create"` and `"buy"` as keywords (line 50), but `"buy"` matches every pump.fun trade, not just pool inits — this generates massive false-positive volume.
- `_parse_pool_transaction` (line 488-560): extracts token mint from `postTokenBalances` diff (lines 502-519). Reasonable heuristic but misses tokens that don't show in postTokenBalances (e.g., if the buyer also held the token pre-tx). Pool address extraction (lines 530-555) tries account_keys[2..6] for Raydium V4 — heuristic, not guaranteed.
- `_handle_rate_limit` (line 569-584) reports rate-limit to `pool_engine` (line 575), rotates RPC (line 577) — **good**. Pool-engine wired here, but not in the rest of the sniper code.
- No WebSocket / Geyser subscription. The class has a `use_websocket` flag from `SNIPER_USE_WEBSOCKET` env (line 94) but **no code path uses it** — flag is read and ignored.

### `modules/sniper/core/token_safety.py` (594 lines)
- `check_token` (line 114-139): cache TTL 300s (line 99) — same as DEX module's honeypot_checker. Way too long for fresh launches (token can rug in seconds).
- `_check_evm_token` (line 141-299): calls GoPlus → Honeypot.is sequentially (line 157, 240). Each call is ~500-1500ms. Total: 1-3s per EVM token check. **Inside the sniper hot loop.** Should be parallel `asyncio.gather`.
- GoPlus and Honeypot.is endpoints are hard-coded (lines 77-78) — no API key plumbing, no rate-limit handling. Both APIs rate-limit on free tier to ~5 req/min.
- `_check_solana_token` (line 301-411): calls RugCheck.xyz (line 317) — **rugcheck.xyz has no data for a token <60s old**. Honeypot defense fails on the launches that need it most. Should add a `simulate_swap` on-chain fallback: try a buy quote and a sell quote with Jupiter, verify roundtrip > 50%.
- Freeze authority is treated as honeypot (line 369-372) — too aggressive. Many legitimate Solana tokens (USDC itself!) have freeze authority. Should be `warning` not `honeypot`.
- `_calculate_score` (line 477-542): score model has reasonable signals but is purely heuristic. The bigger issue is that the *inputs* (RugCheck for Solana) are missing for new launches.
- No **buy/sell simulation** — the gold-standard honeypot detector. For EVM: `eth_call` with state override to simulate buy then sell. For Solana: `simulateTransaction` with a buy ix and a sell ix bundled — Jupiter even provides a `swapMode: ExactInOut` or simulate endpoint.
- Cache stores SafetyReport by `chain:address.lower()` (line 119) but **`.lower()` on a Solana base58 address is wrong** — Solana addresses are case-sensitive. Different casings will create cache misses (less critical since base58 has one canonical form), but the design is incorrect.

### `modules/sniper/core/trade_executor.py` (822 lines)
- `_get_decrypted_key` (line 111-165): supports secrets manager + Fernet decrypt fallback. Acceptable.
- `initialize` (line 167-210): reads `PRIVATE_KEY` then `EVM_PRIVATE_KEY` (line 174). Wallet address from `WALLET_ADDRESS` or `EVM_WALLET_ADDRESS` (line 180). Two-name fallback is brittle.
- EVM Web3 setup (lines 191-207): `Web3(Web3.HTTPProvider(rpc_url))` direct (line 203). No failover. Reads `WEB3_PROVIDER_URL` or `ETHEREUM_RPC_URL` fallback (line 201).
- `_execute_evm_buy` (line 585-669): **critical bugs**:
  - Line 605: `address=Web3.to_checksum_address(UNISWAP_V2_ROUTER)` — UNISWAP_V2_ROUTER is mainnet Uniswap V2 router. For BSC/Base/Arb/Polygon trades this is **completely wrong** — the router doesn't exist there. Every non-mainnet snipe will revert.
  - Line 621: `amount_out_min = 0` with comment `"Accept any amount (risky, but for speed)"` — **guaranteed sandwich.** Sniffers see the tx in mempool, sandwich it, extract 99% of value.
  - Line 632-639: hard-coded `'gas': 300000`, `'maxPriorityFeePerGas': Web3.to_wei(priority_fee, 'gwei')`. With `priority_fee=5000` (default in sniper_engine), this is 5000 Gwei priority — at $3000 ETH, $0.06 × 300k gas = $18 per tx just in priority fee. Will drain the wallet on a few failed snipes.
  - Line 638: `nonce=self.w3.eth.get_transaction_count(self.evm_wallet)` — uses `latest` block by default; should be `'pending'` to handle in-flight txs. Will collide with self.
  - Line 643: `tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)` — broadcasts to **public mempool**. No Flashbots, no bloXroute, no private relay. The sniper *announces itself* to every MEV bot.
- `_execute_evm_sell` (line 671-757): same issues. Line 706: `amount_tokens = int(amount_in * 1e18)` — hard-coded 18 decimals. WBTC (8 dec), USDC (6 dec), HEX (8 dec), many memecoins (9 or 12 dec) will be sold at wrong amounts. Line 709: `amount_out_min = 0` again. Line 691-692: comment `"First, approve router to spend tokens (if not already approved). In production, check allowance first"` — **the approval call is not implemented**. The `swapExactTokensForETH` will revert on insufficient allowance. EVM sells **do not work**.
- `_execute_solana_buy` (line 284-358): converts SOL to lamports (`int(amount_in * 1e9)`, line 301) — correct. Slippage `int(slippage * 100)` (line 302) — converts a `slippage=10.0` percent to 1000 bps; reasonable but very wide. `amount_out=expected_output / 1e6` (line 341) — **hard-coded 6 decimals** for the output token. Wrong for 9-decimal pump.fun tokens.
- `_execute_solana_sell` (line 360-434): `amount_tokens = int(amount_in * 1e6)` (line 377) — **same hard-coded 6 decimals**. If `amount_in` is the actual UI amount of a 9-decimal token, the bot sells 1000x less than intended. Selling less = position not actually closed.
- `_sign_and_send_solana_tx` (line 495-581):
  - Line 561: `signature = keypair.sign_message(bytes(message))` — wrong API for VersionedTransaction (same bug as `jupiter_executor.py:712`).
  - Line 562: `signed_tx = VersionedTransaction.populate(message, [signature])` — overwrites all other signers.
  - Line 569-570: creates new `AsyncClient(rpc_url)` per call — no client reuse, no connection pool.
- No retry path. No alt-RPC fallback. No bundle / private mempool. The sniper trade executor is essentially a "buy at any cost, hope for the best" path.

### `data/collectors/mempool_monitor.py` (699 lines, head)
- Not wired into the sniper. The sniper would benefit massively from mempool monitoring of `addLiquidity` txs but the `EVMListener` polls events only. The mempool_monitor file uses deprecated `Web3.WebsocketProvider` (line 108 per DEX audit) — would also need fixing.

### `data/collectors/honeypot_checker.py` (1037 lines)
- The sniper uses `token_safety.py` instead of `honeypot_checker.py`. They have overlapping responsibility — `honeypot_checker.py` has Solana support via RugCheck (`honeypot_checker.py:141-142`, `206-280`) with a more careful validation flow. `token_safety.py` duplicates the RugCheck call (line 441-451). Two redundant checkers. The DEX audit already flagged `honeypot_checker.py` cache TTL is 5 minutes — too long for fresh launches; same critique applies wherever it is used.

## Risk taxonomy

| ID | Category | File:Line | Severity | Description | Fix sketch |
|----|----------|-----------|----------|-------------|------------|
| SNIPE-01 | MEV / Slippage | trade_executor.py:621,709 | **CRITICAL** | `amount_out_min = 0` — guaranteed sandwich on every EVM swap | Call `router.getAmountsOut(amountIn, path)` → multiply by `(10000-slippage_bps)/10000` |
| SNIPE-02 | Speed / listening | evm_listener.py:96-108, solana_listener.py:267 | **CRITICAL** | HTTP polling only; 15s+ listener lag; no WebSocket / Geyser subscription | `eth_subscribe('logs', ...)` on WSS RPC for EVM; Helius/Yellowstone Geyser for Solana program-log streaming |
| SNIPE-03 | Chain coverage | evm_listener.py:33-37 | **HIGH** | Only Uniswap V2 + SushiSwap mainnet factories; no V3, no L2s, no PancakeSwap | Per-chain factory map; subscribe to all relevant `PairCreated` + V3 `PoolCreated` events |
| SNIPE-04 | Tx routing / MEV | trade_executor.py:643 | **HIGH** | Public mempool broadcast on EVM; no Flashbots/bloXroute | Route through `eth_sendPrivateRawTransaction` (Flashbots Protect) or `arbitrage_engine.FlashbotsExecutor` |
| SNIPE-05 | Honeypot | token_safety.py:301-411 | **HIGH** | RugCheck has no data <60s; no on-chain buy/sell simulation | Add `simulate_swap` path: Jupiter buy quote + Jupiter sell quote with 50% roundtrip threshold; on EVM use `eth_call` with state override |
| SNIPE-06 | Decimals | trade_executor.py:341,377,417,706 | **HIGH** | Hard-coded 1e6 / 1e9 / 1e18 conversions; wrong for BONK (5), WBTC (8), USDC (6), 9-dec pump.fun tokens | Read `decimals` from `getTokenSupply` (Solana) or ERC-20 `decimals()` (EVM); cache on target metadata |
| SNIPE-07 | Tx signing | trade_executor.py:561-562 | **HIGH** | `sign_message` + `populate([signature])` corrupts multi-signer Jupiter routes | Reuse `jupiter_helper.sign_transaction` hybrid signer; preserve co-signers via `NullSigner` |
| SNIPE-08 | EVM approval | trade_executor.py:691-692 | **HIGH** | `_execute_evm_sell` has no token approval; will revert on first sell | Add `IERC20.approve(router, type(uint256).max)` call; check `allowance()` first |
| SNIPE-09 | Router address | trade_executor.py:45,605 | **HIGH** | UNISWAP_V2_ROUTER is mainnet-only; non-ETH EVM sells will revert | Chain-keyed router map (BSC: PancakeRouter, Base: UniV3 SwapRouter02, etc.) |
| SNIPE-10 | Speed / pipeline | sniper_engine.py:349 | **HIGH** | Safety check runs synchronously inside hot loop; adds 1-3s | Parallelize across candidates; run safety check **after** placing the buy tx (pull buy if safety fails — too late, but at least we have entry) |
| SNIPE-11 | Idempotency | trade_executor.py:638 | MEDIUM | `eth.get_transaction_count(wallet)` uses `latest` block; concurrent txs collide | Use `'pending'` or hoist to shared nonce manager (DEX-03 in DEX audit) |
| SNIPE-12 | Mempool front-run | evm_listener.py | MEDIUM | No mempool watch of `addLiquidity` pending txs — bot reacts to PairCreated event, not the launch intent | Subscribe to pending txs, filter `to=router` with `addLiquidity` selector, pre-stage buy |
| SNIPE-13 | Anti-detection | trade_executor.py | MEDIUM | Single wallet, single RPC — easy to fingerprint as a sniper | Multi-wallet round-robin: rotate signing key per snipe; rotate Flashbots-protect RPC origin |
| SNIPE-14 | TP/SL semantics | sniper_engine.py:580-581 | MEDIUM | DB-loaded TP/SL ignored; hard-coded 50/-20 used | Use `self.take_profit_pct` and `self.stop_loss_pct` (or instance defaults) |
| SNIPE-15 | Jupiter price API | sniper_engine.py:634 | MEDIUM | `price.jup.ag/v4` is deprecated; returns 404/410 | Switch to `api.jup.ag/price/v3` or DexScreener |
| SNIPE-16 | Test mode | sniper_engine.py:336-346 | MEDIUM | `test_mode=True` allows DANGER tokens and $10 liquidity in production code path | Guard `test_mode` behind a DRY_RUN AND non-production assertion |
| SNIPE-17 | Listener filters | solana_listener.py:50,485 | MEDIUM | `INIT_KEYWORDS[PUMP_FUN] = ['create','buy','sell','init']` matches every pump.fun trade, not just creates | Use `'create'` only; or check instruction `data[0]` discriminator for `initialize` ix |
| SNIPE-18 | Pool-engine | sniper_engine.py:213, evm_listener.py:57, trade_executor.py:201,568 | MEDIUM | Multiple `os.getenv` fallbacks for RPC | Hard-require pool engine; fail-fast if not initialized |
| SNIPE-19 | Auto-sell on dev-rug | n/a (missing) | MEDIUM | No detection of LP removal, large dev-wallet sells, or freeze-authority changes | Subscribe to LP token's burn events / token freeze events; emergency-exit on detection |
| SNIPE-20 | `is_simulated=True` hardcoded | sniper_engine.py:564 | LOW | Live trades logged as simulated in `sniper_trades` table | Set from `self.dry_run` |
| SNIPE-21 | Confirmation | trade_executor.py:649 | LOW | `wait_for_transaction_receipt` 60s timeout blocks the snipe loop | Spawn confirmation as separate task; return optimistically with tx_hash |
| SNIPE-22 | Solana TX confirmation | trade_executor.py:570 | LOW | `send_transaction` without polling for landed status; signature returned but tx may never land | Mirror `jupiter_helper.confirm_transaction` post-send check |
| SNIPE-23 | Listener gas | evm_listener.py:96-108 | LOW | 10x/sec `getLogs` over 5 blocks burns RPC credits | Cache last seen block; only fetch (last_block, current] range |

## Profitability levers (ranked by ROI)

1. **Switch to WebSocket/Geyser subscriptions (SNIPE-02).** This is the single ROI lever that makes the sniper viable at all. Without sub-second launch detection there is no point in any other improvement. EVM: `eth_subscribe('logs', {topics: [keccak('PairCreated(...)')]})` via WSS RPC (Alchemy/Quicknode all support). Solana: Helius `RpcWebsocketsClient.programSubscribe(<PROGRAM_ID>)` or Yellowstone gRPC. Latency drop: 15s+ → <500ms.
2. **`amount_out_min` from on-chain quote (SNIPE-01).** Today every EVM snipe is a free lunch for sandwichers. Single line change: `min_out = router.getAmountsOut(amount_in, path)[-1] * (10000 - slippage_bps) // 10000`. Recovers the slippage budget from MEV; expect +5-30% better fills.
3. **Private-mempool routing (SNIPE-04).** Route the buy via Flashbots Protect RPC (`https://rpc.flashbots.net`) or arbitrage_engine's working FlashbotsExecutor (`arbitrage_engine.py:688` per DEX audit). Hides the tx from public mempool until included. Combined with #2, eliminates sandwich risk.
4. **Buy/sell on-chain simulation as honeypot check (SNIPE-05).** Replaces dead-data RugCheck/GoPlus calls with deterministic `eth_call`/`simulateTransaction` of a roundtrip. Catches honeypots that RugCheck won't have profiles for. Marginal cost ~50-100ms, blocking but unavoidable.
5. **Token decimals end-to-end (SNIPE-06).** Today the bot's reported "amount_out" is wrong for any non-default-decimals token. Sells fire wrong size. Read decimals once at acquisition, propagate.
6. **EVM approval + chain-keyed router map (SNIPE-08, SNIPE-09).** Without these, **non-mainnet EVM sells do not work at all**. Once fixed, BSC/Base/Arb sniping becomes possible.
7. **Multi-wallet rotation (SNIPE-13).** Sniper wallets get blacklisted by anti-bot contracts within hours. Rotate across 3-5 wallets, alternate per snipe. Reduces detection rate.
8. **Pull TP/SL from DB (SNIPE-14).** Currently hard-coded values defeat the dashboard.
9. **Mempool pending-tx watch (SNIPE-12).** Watch for `addLiquidity` to a known token-creator address — enter ahead of pool init. Requires WSS + smart filtering, but pays off on big launches.
10. **Parallel safety check (SNIPE-10).** `asyncio.gather(goplus, honeypot_is, rugcheck)` — collapses 1-3s sequential into 500-1000ms parallel. The single largest CPU-bound improvement.

## Live-trade gap list

- [ ] EVM listener uses WSS `eth_subscribe('logs')` on every supported chain
- [ ] EVM listener subscribes to V3 `PoolCreated` in addition to V2 `PairCreated`
- [ ] Solana listener uses Helius/Yellowstone Geyser for `programSubscribe` (or pumpportal.fun WS for Pump.fun)
- [ ] `amount_out_min` computed from on-chain quote on every EVM swap
- [ ] EVM swap routed through Flashbots Protect (or another private relay)
- [ ] Honeypot check uses on-chain simulate-swap roundtrip
- [ ] Token decimals read from on-chain and propagated through buy → tracking → sell
- [ ] EVM sells call `approve(router, max)` if allowance insufficient
- [ ] Per-chain router map (V2/V3 + L2 routers)
- [ ] Solana tx signing preserves all co-signers (shared helper with `jupiter_helper.sign_transaction`)
- [ ] Multi-wallet rotation for the sniper
- [ ] Pump.fun snipe path uses Jito bundle with tip (cross-cuts SOLANA module — SOL-03)
- [ ] DB-loaded TP/SL used; hard-coded defaults removed
- [ ] Sniper kill-switch hooked into `core/risk_manager.py`
- [ ] All `os.getenv('*RPC*')` reads in sniper module removed
- [ ] `is_simulated` field reflects actual DRY_RUN mode
- [ ] EVM nonce uses `'pending'` block or shared nonce manager
- [ ] Listener honors `since-last-poll` cursor; doesn't re-fetch same blocks 10x/sec
- [ ] LP-removal / freeze-authority detection → auto-emergency-exit

## Proposed action backlog

- [ ] **SNIPE-01** `EVM WebSocket subscription` — touches `modules/sniper/core/evm_listener.py` (replace `get_new_pairs` polling with async `subscribe('logs')`) — expected gain: 15s+ latency → <500ms — owner: smartcontract.
- [ ] **SNIPE-02** `Solana Geyser subscription` — touches `modules/sniper/core/solana_listener.py` (replace polling with Helius WS `logsSubscribe` filtered by program) — expected gain: 15s+ latency → <500ms — owner: smartcontract.
- [ ] **SNIPE-03** `On-chain quote for amount_out_min` — touches `modules/sniper/core/trade_executor.py:585-669, 671-757` (add `router.getAmountsOut` call, compute min from slippage) — expected gain: kills sandwich loss on every EVM snipe — owner: smartcontract.
- [ ] **SNIPE-04** `Flashbots Protect for EVM snipes` — touches `modules/sniper/core/trade_executor.py:643` (route through `https://rpc.flashbots.net` or reuse `arbitrage_engine.FlashbotsExecutor`) — expected gain: private-mempool inclusion, anti-frontrun — owner: smartcontract.
- [ ] **SNIPE-05** `On-chain honeypot simulation` — touches `modules/sniper/core/token_safety.py` (add `_simulate_buy_sell` for EVM and Solana) — expected gain: catches honeypots RugCheck/GoPlus miss on fresh launches — owner: smartcontract.
- [ ] **SNIPE-06** `Token decimals through buy → sell` — touches `modules/sniper/core/sniper_engine.py:_execute_snipe`, `trade_executor.py:284-434, 585-757` (read decimals once, cache on target metadata) — expected gain: unbreaks non-6/non-18-decimal trades — owner: smartcontract.
- [ ] **SNIPE-07** `EVM approval + chain-keyed router map` — touches `modules/sniper/core/trade_executor.py:45,605,671-757` — expected gain: unbreaks non-mainnet EVM sells — owner: smartcontract.
- [ ] **SNIPE-08** `Unified Solana tx signer (preserve multi-signers)` — touches `modules/sniper/core/trade_executor.py:561-562` — replace with shared `tx_signer.sign_versioned_with_keypair` (same as SOL-04) — expected gain: complex routes succeed — owner: smartcontract.
- [ ] **SNIPE-09** `Parallel honeypot checks` — touches `modules/sniper/core/token_safety.py:141-411` (asyncio.gather GoPlus/HoneypotIs/RugCheck) — expected gain: -50% safety-check latency — owner: smartcontract.
- [ ] **SNIPE-10** `Per-chain factory map + V3 PoolCreated` — touches `modules/sniper/core/evm_listener.py:33-37, 85-129` — expected gain: 5-10x coverage of EVM launches — owner: smartcontract.
- [ ] **SNIPE-11** `Multi-wallet rotation` — touches `modules/sniper/core/trade_executor.py:167-210` (add `_wallets: List[Keypair]` + round-robin selector) — expected gain: anti-detection — owner: smartcontract + analyst.
- [ ] **SNIPE-12** `Wire DB TP/SL` — touches `modules/sniper/core/sniper_engine.py:577-625` (delete hard-coded constants; use `self.take_profit_pct` / `self.stop_loss_pct`) — expected gain: dashboard control honored — owner: smartcontract.
- [ ] **SNIPE-13** `Hook kill-switch` — touches `modules/sniper/core/sniper_engine.py:_execute_snipe`, `core/risk_manager.py` — expected gain: live-readiness gate — owner: smartcontract.
- [ ] **SNIPE-14** `Pool-engine for sniper RPC` — touches `modules/sniper/main_sniper.py:160-163`, `evm_listener.py:50-57`, `trade_executor.py:194-203,564-568` — expected gain: rate-limit aware failover — owner: smartcontract + backend.
- [ ] **SNIPE-15** `LP-removal / freeze-authority watcher` — new helper in `modules/sniper/core/rug_detector.py`; subscribe to LP-token burn events (EVM) or freezeAuthority changes (Solana) on active snipes — expected gain: auto-emergency-exit before full rug — owner: smartcontract.

## Open questions

1. Is the sniper a production strategy or a research experiment? Several hard-coded test-mode shortcuts (`test_mode=True` allowing DANGER tokens, `amount_out_min=0`, `is_simulated=True` always) suggest the latter. If production, those gates need to be eliminated entirely; if research, the module should be flagged as non-production in the dashboard.
2. Should the sniper share infrastructure with the Solana module (`solana_engine.JupiterHelper`, `safety_engine`) instead of duplicating Jupiter logic in `modules/sniper/core/trade_executor.py`? Today there are three Jupiter executors (helper, executor, sniper trade_executor) with different bugs in each.
3. Pump.fun: pre-graduation tokens have no Jupiter route. Is the sniper expected to interact directly with the pump.fun bonding-curve program for those? Currently `trade_executor.execute_solana_buy` goes through Jupiter, which will return "no route" for pre-graduation pump.fun tokens.
4. EVM coverage: which chains are in scope? `FACTORIES` lists only mainnet — but `chain_data.py` (DEX module) supports ETH/BSC/Base/Arb/Polygon. If the sniper is supposed to cover the same chains, every chain needs its factory + router wired.
5. Is there a budget for Helius staked connection + Geyser stream (~$200-500/mo combined)? Without this, the listener is structurally too slow to compete on launches.
6. Should the sniper use the `arbitrage/arbitrage_engine.FlashbotsExecutor` (which the DEX audit says works) rather than recreate Flashbots logic? Code reuse over duplication.
7. Wallet model: single wallet vs. wallet-pool. Today a single wallet is used for all snipes; once blacklisted by anti-bot contracts (which is common), the bot becomes useless. Decision needed.
