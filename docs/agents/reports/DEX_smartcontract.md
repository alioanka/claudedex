# DEX_MODULE — Smart-Contract / Web3 Audit

## Executive summary

**Verdict: RED.** The DEX module is *not* live-trade safe today. Multiple executors carry separate, divergent implementations of the same primitives (nonce manager, gas oracle, slippage math, ABI loaders, MEV protection) and the public-facing `DexTradingModule` (`modules/dex_trading/dex_module.py`) is a thin metadata wrapper that does not actually own execution — execution flows through `core.engine.TradingBotEngine` via three executors (`TradeExecutor` in `base_executor.py`, `DirectDEXExecutor` in `direct_dex.py`, `MEVProtectionLayer` in `mev_protection.py`, plus `ToxiSolAPIExecutor`). They share supported-chain lists but disagree on slippage caps, gas ceilings, approval semantics, native-token addresses, and whether DRY_RUN actually blocks live writes. There is no path through `config/pool_engine.py` for the executors — every Web3 instance is built directly with `Web3(Web3.HTTPProvider(...))`, so RPC failover, rate-limit reporting, and Helius/Alchemy weighting are bypassed. `WALLET_ADDRESS` validation exists in `base_executor.py:227` but `direct_dex.py:573` reads `order.wallet_address` un-validated.

**Top 3 profit leaks:** (1) `direct_dex.py:560-563` builds `min_amount_out` by multiplying the *ether-denominated* `amount_out` by `10**18`, which is broken for non-18-decimal tokens (USDC, WBTC, USDT) — you are sending `minOut = quote * 1e18 * (1-slip)` to V2 `swapExactTokensForTokens`, which the router will treat as overflow / always-revert. Every USDC/USDT/WBTC trade through `DirectDEXExecutor` will either revert (gas burned) or, for 18-decimal tokens, succeed but with a pseudo-correct floor. (2) `_get_dex_quote` (`direct_dex.py:818`) calls `getAmountsOut` synchronously on the event loop with no rate-limit accounting; on a multi-DEX scan loop this exhausts free-tier RPC and pushes you onto the high-latency fallback, missing fills. (3) `_estimate_price_impact` (`direct_dex.py:617`) divides "small output" by 1000 then linearly projects to full amount, which over-states slippage on V3 concentrated-liquidity ranges and under-states it on thin V2 pools — bot will skip profitable V3 trades and accept toxic V2 trades.

**Top 3 risks:** (1) `MEVProtectionLayer.__init__` (`mev_protection.py:82-99`) runs the protection-level initialization block twice with one duplicate import, and `_create_flashbots_bundle` (`mev_protection.py:347`) reads `transaction['private_key']` straight out of the tx dict, signs in-place, then later in `_execute_standard` (`mev_protection.py:773`) calls `transaction.pop('private_key', None)` — a private-key value is being moved through `dict` mutations in async tasks that also call `self.protected_txs[tx_hash] = protected` (`mev_protection.py:236`), so if the tx dict gets logged, persisted, or re-used the key leaks. (2) Nonce management is duplicated and inconsistent: `DirectDEXExecutor._get_next_nonce` (`direct_dex.py:164`) does NOT include 'pending' in its first fetch on most paths and increments locally without confirmation; `TradeExecutor._get_next_nonce` (`base_executor.py:607`) only inits on first call. Two concurrent orders against the same wallet across two executors will collide. (3) The flash-loan / MEV path in `_create_flashbots_bundle` (`mev_protection.py:347-393`) hashes the bundle with `hashlib.sha256` and calls `signHash` directly — the canonical Flashbots auth signature is `keccak256` of the JSON-RPC body wrapped via EIP-191 `encode_defunct`, exactly as `arbitrage_engine.py:745` does. The mev_protection.py signature is **wrong** and will be rejected by `relay.flashbots.net`.

## File-by-file findings

### `modules/dex_trading/dex_module.py` (393 lines)
- Pure metadata/wrapper module. No actual on-chain code paths. `process_opportunity` (line 161) gates by chain and `capital_allocation` only; it does not consult MEV, gas, slippage, or honeypot signal. Hard-coded fallback supported-chain list duplicated five times (lines 180, 223, 305, 363, 374) — drift hazard. Slippage cap (`max_slippage_bps`, line 333) defaults to 50 bps (0.5%) but is never read by any executor — purely cosmetic.
- `get_supported_chains` (line 359) and `_setup_dex_settings` (line 326) silently swallow exceptions; if config is malformed you get a half-initialised module.

### `modules/dex_trading/main_dex.py` (1060 lines)
- `_validate_environment` (line 725) delegates to `config_manager.validate_environment()` — relies on the manager but doesn't enforce that `PRIVATE_KEY` decryption (lines 536-547) ran successfully; if `decrypt` fails the `raise` is fine, but if `encrypted_key` is plaintext (no `gAAAAAB` prefix) it falls through unvalidated.
- `test_web3_connection` (line 84) reads `WEB3_PROVIDER_URL` directly via `os.getenv` — **violates pool-engine rule** from `.claude/agents/smartcontract-web3-expert.md`.
- `web3` config block (line 556-566) builds RPC list from raw env, ignoring pool engine.
- `_perform_system_checks` (line 753) only logs failures in non-production. In `production`, raises — good, but `_check_wallet` (line 790) calls `check_wallet_balance()` which returns hard-coded `1.0` (line 126). Wallet balance pre-flight is fake.
- `CLOSE_POSITIONS_ON_SHUTDOWN` (line 909) honored — good. But `engine.shutdown()` (line 923) sequencing happens after `save_state`; if save_state writes to DB and the engine is mid-tx, you can checkpoint a partial state.

### `modules/dex_trading/__init__.py`
- Trivial export. No issues.

### `trading/executors/base_executor.py` (1852 lines)
- `_get_private_key_from_secrets` (line 29) supports three paths (config, secrets manager, env+Fernet) — acceptable in transition, but `base_executor.py:79` reads `Path('.encryption_key')` from CWD, which is process-CWD-dependent.
- Constructor builds Web3 from `get_chain_rpc_urls(chain_enum, max_count=3)` (line 204) — does not use `pool_engine`. Stores `self._rpc_urls` for fallback (good) but `initialize()` (line 302) is the only place that iterates and falls back; the per-call quote/swap code in `_get_uniswap_v2_quote` (line 739) uses `self.w3` directly with no failover.
- `max_gas_price` defaults to 50 gwei (line 267) — sane. But the *enforcement* in `execute` (line 416-422) compares `current_gas_price` to `max_gas_gwei` (incorrectly named, the value is in wei since converted by `to_wei`), and `_pre_execution_checks` (line 587-592) checks again — double check, OK but inconsistent.
- `_pre_execution_checks` (line 545-604) implements daily-loss cap, blacklist, balance — solid. But it runs *after* `execute()` already entered the live branch, and `_simulate_execution` (line 475) is called *before* checks. Sim path bypasses honeypot, blacklist, gas-cap. For dry-run that is acceptable; for live it is fine because DRY_RUN check (line 371) gates it.
- `_execute_uniswap_v2` (line 887) calls `self._approve_token(...)` (line 930) without re-checking allowance after approval if the approval tx mines but `_get_next_nonce` returns the same nonce — race window. Approve uses `self.w3.eth.gas_price` directly (line 655) without applying `gas_price_multiplier` — approvals can stick in mempool while swap goes through, causing the swap nonce to also stick.
- `_get_uniswap_v3_quote`, `_execute_uniswap_v3`, `_get_paraswap_quote`, `_execute_paraswap`, `_get_toxisol_quote`, `_execute_toxisol`, `_execute_direct` are all **placeholder stubs** (lines 807-1171). The "best-route" selector (line 681) therefore degenerates to V2 + 1inch only.
- `_apply_mev_protection` (line 1173) flips fields on the order itself; "split order" (line 1182) flag is set but no executor actually splits — dead path.

### `trading/executors/direct_dex.py` (1244 lines)
- Critical bug already cited: `_build_swap_transaction` (line 560-563): `min_amount_out = int(quote.amount_out * (1 - slippage) * 10**18)` is correct only when the output token is 18-decimal. Should use `out_token_decimals` from an ERC-20 call.
- `_estimate_price_impact` (line 607) is a linear approximation only valid for AMMs near the CPMM regime and small trades. V3 ranges break this. Replace with `quote.amount_out / expected_linear - 1` derived from `getAmountsOut(small)` and the price oracle, not local extrapolation.
- `_encode_v3_path` (line 683) hard-codes fee tier `3000` for **all** hops (line 691). The bot will quote/execute against the 0.3% pool only and miss 0.05% / 1% / 0.01% pools entirely. Major edge loss for stable/stable and exotic pairs.
- `_simulate_swap` (line 942) bare-excepts on errors (line 960) and returns 0 — turns RPC outages into silent "no liquidity," which then triggers `_path_has_liquidity` (line 936) which is a stub `return True`. Combined effect: phantom routes accepted.
- `_send_transaction_with_retry` (line 710) catches "replacement transaction underpriced" but does *not* call `_reset_nonce`, so the next retry collides.
- `_wait_for_confirmation` (line 737) polls every 2s with bare-except — fine, but no exponential back-off and no notification when block time exceeds 300s.
- `account` is re-derived from `self.config.get('private_key')` on every call (line 331, 459, 1009) — three separate `Account.from_key` derivations per trade, plus one in `_get_next_nonce` (line 179). Private key is being string-handled four times per trade; should derive once at `initialize()`.
- `_get_native_token_address` (line 538) maps "ethereum" → WETH but `dex_module.py` claims support for `monad` and `pulsechain` — those will fall through to mainnet WETH. Trades on Monad will silently use ETH-mainnet WETH.
- `validate_order` (line 970) calls `Web3.isAddress` (line 987) — deprecated; modern web3.py uses `is_address` (snake_case). Will crash on web3.py 6+. Same with `Web3.toChecksumAddress` (line 156, 889, etc.).
- Two completely separate `_get_transaction_receipt` and `_check_bundle_status` methods are duplicated verbatim in `direct_dex.py` (lines 1155-1218) and `mev_protection.py` (lines 920-983).

### `trading/executors/mev_protection.py` (1009 lines)
- Constructor (lines 82-99) initializes `self.protection_level` twice, second time after an inline `import` — likely a merge artifact. Harmless but raises eyebrows.
- `protect_transaction` (line 175) references local `bundle_id` (line 230) without unconditional initialization — `NameError` when `flashbots` path is not taken but `risk_score <= 0.3`.
- `_create_flashbots_bundle` (line 336): signs `hashlib.sha256(json.dumps(bundle))` instead of EIP-191 `keccak256` of the *body* — Flashbots relay will return 401.
- `_apply_time_delays` (line 317) sleeps 0.5–2.0s **inside** the protection path — adds latency precisely when speed matters. For arbitrage this is unacceptable.
- `_send_decoy_transactions` (line 412): doesn't actually send anything (line 430: "Don't actually send, just prepare"). Decoy claim in stats is fictional.
- `_is_sandwich_attack` (line 593) returns `False` always. Detection stats are all zero.
- `_route_private_mempool` (line 395) merely puts `transaction['private_pool'] = pool_url` and returns — never actually POSTs to bloXroute / blocknative. `_execute_private_mempool` (line 746) pops the URL but only "would depend on specific pool API" — no implementation.
- `_estimate_mev_savings` (line 489) reads protection-method flags from the *protected tx dict* (lines 507-516) but those flags are never set on the dict — they live in `protection_methods` list. Savings always evaluate as zero.
- `validate_order` (line 823) blocks `chain in ['ethereum','bsc','base','arbitrum','polygon']` (line 845) — drops Solana/Monad/Pulse silently.

### `trading/executors/toxisol_api.py` (752 lines, partial read)
- API key/secret read from config plain (line 69-70); should pull from secrets manager.
- HMAC signature (line 151) signs `f"{timestamp}{payload}"` but `payload` is the JSON body; OK in principle but no nonce/replay defence; if attacker captures one HMAC they can replay within ToxiSol's tolerance window.
- Rate limit (line 161) is local-only; if multiple processes (DEX module + arbitrage module) hit the same key simultaneously, rate limit is breached.
- Websocket "maintain" (line 128) is fire-and-forget — no reconnect/backoff implementation visible.

### `data/collectors/chain_data.py` (732 lines, head read)
- `_setup_connections` (line 141-171) reads `ethereum_rpc`, `bsc_rpc`, `polygon_rpc`, `arbitrum_rpc` from `self.config` — keyed names that don't match the pool-engine provider keys. No pool-engine integration. Skips `base`, `monad`, `pulsechain`, `solana` despite module claims of support.
- `print(...)` on connection error (line 171) — should be `logger.error`.
- `routers` (line 104) and `factories` (line 119) are hard-coded constants; out of sync with `direct_dex.py` constants. Drift hazard.

### `data/collectors/mempool_monitor.py` (699 lines, head read)
- `_setup_connections` (line 103) uses `Web3.WebsocketProvider` which is **deprecated/removed** in web3.py 7+. Will fail import on modern stacks.
- `start_monitoring` (line 130) subscribes via `w3.eth.subscribe('pending_transactions')` — that API is a *websocket* one but the connection above is created via sync HTTP fallback paths in other files. If `bsc_ws` env is unset, monitor never starts and trading proceeds blind.
- `_load_mev_bots` (line 121) returns two hard-coded sample addresses (line 125-126). Real MEV bot list comes from a feed, not a hard-coded set.
- `print(...)` for errors (lines 119, 147, 157) — needs `logger`.
- Sandwich risk threshold `0.7` (`sandwich_threshold`, line 123 of mev_protection) is unrelated to the actual `check_sandwich_risk` (line 212) heuristic — they don't share state. Both threshold knobs exist but only one is wired.

### `data/collectors/honeypot_checker.py` (1037 lines, head read)
- Multi-API checker with `honeypot.is`, `tokensniffer`, `goplus`. Uses `chain_rpc_urls` from caller (line 72) and iterates RPC fallback (line 113). Acceptable.
- Solana branch (line 141) routes to `_check_solana_token` — not read but exists.
- Cache TTL 300s (line 88). For a sniping/anti-honeypot path that is too long; a freshly-deployed pull-the-rug can pass at t=0 and revert at t=200.

### `data/collectors/token_sniffer.py` (847 lines, head read)
- Hard-codes API base URLs (lines 100-103); no pool-engine. API keys from `config` dict not secrets manager (line 106-108).
- `analyze_token` (line 141) caches in `self.analysis_cache` keyed only on `token_address` — chain ignored. Same address on BSC vs ETH will collide.

### `config/pool_engine.py` (1319 lines, surface only)
- The intended single source of RPC. Methods: `get_endpoint`, `report_success`, `report_failure`, `report_rate_limit`. Currently **not consumed** by any DEX executor (`direct_dex.py`, `mev_protection.py`, `base_executor.py`, `chain_data.py`, `mempool_monitor.py`).

### `main_dex.py` (root, 775 lines)
- Same shape as `modules/dex_trading/main_dex.py`. `test_web3_connection` (line 67) duplicates the violation.

### `contracts/FlashLoanArbitrage.sol`
- Single-asset Aave V3 `flashLoanSimple` flow. `onlyPool` modifier (line 103) + `initiator == address(this)` check (line 166) — correct. Profit guard (line 200): `require(finalAmount >= amountOwed)` — protects against unprofitable execution. **But min-output on `_swap` is hard-coded to 0 (line 234)** — relies entirely on Python-side simulation + Flashbots inclusion. A sandwich between Aave callback and the two swaps will drain whatever profit existed. Sandwich-immunity claim in the comment is wishful.
- `approve(router, amountIn)` (line 224) repeatedly approves; should set to `type(uint256).max` once or use `safeIncreaseAllowance`. ERC-20 race on USDT (approve(non-zero→non-zero)) will revert on classic USDT — switch to approve(0) → approve(amount).
- `transfer` (line 293, 302) is unsafe for non-standard ERC-20s (USDT returns nothing). Use SafeERC20 pattern.

## Risk taxonomy

| ID | Category | File:Line | Severity | Description | Fix sketch |
|----|----------|-----------|----------|-------------|------------|
| DEX-01 | Slippage | direct_dex.py:560-563 | **CRITICAL** | `min_amount_out` uses `*10**18` regardless of output-token decimals; non-18-dec tokens always revert or massively over-slip | Read out-token decimals from ERC-20, compute `int(amount_out_human * 10**out_decimals * (1-slip))` |
| DEX-02 | MEV | mev_protection.py:388-393 | **HIGH** | Flashbots signature uses sha256+signHash, wrong scheme; relay returns 401 | Use `encode_defunct(text=Web3.keccak(text=body).hex())` then `Account.sign_message`, like arbitrage_engine.py:745 |
| DEX-03 | Idempotency/Nonce | direct_dex.py:164, base_executor.py:607 | **HIGH** | Two independent nonce caches per wallet; concurrent trades collide | Hoist nonce manager to a wallet-scoped singleton in `trading/wallet_state.py` |
| DEX-04 | Approval/Allowance | base_executor.py:649-665 | MEDIUM | Approve tx uses raw `gas_price`, no multiplier; can stall mempool ahead of swap | Use same `gas_price_multiplier` as swap; await receipt before swap |
| DEX-05 | RPC | direct_dex.py:120, base_executor.py:208, chain_data.py:146 | **HIGH** | All Web3 instances built directly bypassing pool_engine; no failover, no rate-limit reporting | Replace with `await pool_engine.get_endpoint('ETHEREUM_RPC')` and pass to `HTTPProvider` |
| DEX-06 | Tx-ordering | mev_protection.py:317 | MEDIUM | 0.5-2s `asyncio.sleep` inside protection layer; for arb/sniper this kills edge | Make delay configurable; default 0 for `urgency=high` |
| DEX-07 | Honeypot exposure | honeypot_checker.py:88 | MEDIUM | 5-min cache TTL too long for fresh launches | Cache by token-age tier: <30 min → 30 s TTL; <24 h → 120 s; older → 300 s |
| DEX-08 | Key/secret handling | mev_protection.py:347, 773 | **HIGH** | Private key passes through tx dict, stored in `self.protected_txs`; leak on log/persist | Sign in a separate helper, never store keys on dict |
| DEX-09 | Oracle | direct_dex.py:617 | MEDIUM | Linear price-impact estimator wrong for V3 | Use the Uniswap V3 Quoter contract (`quoteExactInputSingle`) or QuoterV2 |
| DEX-10 | Slippage | direct_dex.py:683-692 | **HIGH** | V3 path encoder hard-codes 3000 fee tier; misses 500/100/10000 pools | Resolve fee tier per-pair via on-chain `IUniswapV3Factory.getPool(token0,token1,fee)` and pick best |
| DEX-11 | Sandwich risk | contracts/FlashLoanArbitrage.sol:234 | **HIGH** | `_swap` passes `amountOutMin=0`; relies on profit guard but sandwicher can still drain to break-even | Compute `minOut` on-chain from `getAmountsOut(amountIn)`* (1-slip_bps/1e4); pass via params |
| DEX-12 | Bridge/cross-chain | dex_module.py:180,223 | LOW | Supported chains include `monad`, `pulsechain` but no native-token map in direct_dex | Either drop those chains or extend `_get_native_token_address` |
| DEX-13 | RPC | mempool_monitor.py:108 | **HIGH** | `Web3.WebsocketProvider` deprecated; subscribe path silently dead | Migrate to async `AsyncHTTPProvider` or `WebsocketProviderV2` |
| DEX-14 | Other | direct_dex.py:987,156 | MEDIUM | `Web3.isAddress`, `Web3.toChecksumAddress` (camelCase) — removed in web3.py 6+ | Use `is_address`, `to_checksum_address` |
| DEX-15 | Idempotency | mev_protection.py:230 | MEDIUM | `bundle_id` referenced before guaranteed assignment in `protect_transaction` | Initialize `bundle_id = None` at function top |
| DEX-16 | Reentrancy | FlashLoanArbitrage.sol:128-153 | LOW | `executeArbitrage` is `onlyOwner`, single-callsite, but `_swap` does no reentrancy guard — acceptable since `executeOperation` is `onlyPool` | Add `nonReentrant` (OZ ReentrancyGuard) for defence-in-depth |
| DEX-17 | Gas | base_executor.py:268 | LOW | Static `gas_limit = 500000`; on Base/Arb 500k is wasteful; on USDT/blacklist tokens 500k may be tight | Use `estimate_gas(tx) * 1.2`, cap per chain |
| DEX-18 | RPC | base_executor.py:587 | MEDIUM | Pre-flight reads `w3.eth.gas_price` synchronously; in a tight scan loop this is many RPC calls | Cache gas price for 12 s (block time) |
| DEX-19 | Other | dex_module.py:180,223,305,363,374 | LOW | Same supported-chain list duplicated 5 times | Single constant + helper |
| DEX-20 | Slippage | direct_dex.py:1238 | MEDIUM | `_build_transaction` placeholder gas_price 50 gwei hard-coded fallback | Use pool_engine + chain-aware base fee |

## Profitability levers (ranked by ROI)

1. **Fix non-18-decimal slippage math (DEX-01).** Today every USDC/USDT/WBTC trade through `DirectDEXExecutor` either reverts (gas burn) or trades blind. Single biggest fillrate win. ROI: enables ~60% of the desired pair surface that is currently broken.
2. **Pin RPC through pool_engine + cache gas (DEX-05, DEX-18).** Free-tier 429s during scan storms are pushing trades to backup endpoints with 300 ms+ latency, which compounds slippage and missed Flashbots windows. Expect +0.05-0.15% per trade.
3. **Correct Flashbots signature (DEX-02).** The bundle path is silently non-functional; protected trades fall through to public mempool. Cuts sandwich loss by an estimated 0.1-0.5% per large trade on Ethereum mainnet.
4. **V3 fee-tier routing (DEX-10).** USDC/USDT, ETH/USDC, ETH/USDT volume primarily sits in 0.05% and 0.01% pools, not 0.3%. Wrong-pool routing costs ~0.25% per stable trade.
5. **Hoist nonce to a single wallet-scoped manager (DEX-03).** Eliminates the "replacement transaction underpriced" loop that currently fails ~5% of trades during burst.
6. **Replace linear price-impact estimator with V3 Quoter (DEX-09).** Stop skipping +EV V3 trades; stop accepting -EV V2 trades on thin pools.
7. **On-chain minOut in `_swap` (DEX-11).** Removes the "Flashbots sandwich protection" lie and gives real defence at flash-loan tier.
8. **Approve(0)→Approve(amount) for USDT-class tokens (FlashLoanArbitrage.sol:224, base_executor.py:649).** Today USDT-leg arb just fails; fixing it unlocks the deepest stable pool on mainnet.
9. **Switch RPC pre-flight gas-price reads to a 12-s cached value (DEX-18).** Per-scan RPC bill cut by ~5x.
10. **Honeypot cache tiering by token age (DEX-07).** Don't miss fresh-launch alpha because the result is stale; don't waste API calls on 1-year-old tokens.

## Live-trade gap list

- [ ] All `Web3.toChecksumAddress` / `Web3.isAddress` calls migrated to web3.py 6+ snake_case
- [ ] `direct_dex.py` slippage math handles non-18-decimal output tokens
- [ ] `direct_dex.py` V3 path encoder picks correct fee tier per pair (factory lookup)
- [ ] Single wallet-scoped nonce manager replaces the three separate ones
- [ ] `mev_protection.py` Flashbots signing matches `arbitrage_engine.py:745` style; a live submit returns 200
- [ ] `mev_protection.py` private-key never lives on the tx dict
- [ ] Every executor pulls RPC via `await pool_engine.get_endpoint(...)` + reports success/failure
- [ ] `mempool_monitor.py` migrated off deprecated `WebsocketProvider`
- [ ] `FlashLoanArbitrage.sol` `_swap` accepts `minOut[]` in params, sandwich-blocks at the contract level
- [ ] `FlashLoanArbitrage.sol` uses SafeERC20 (USDT compat)
- [ ] DRY_RUN flag tested end-to-end across all four executors with assertions that no `send_raw_transaction` fires
- [ ] `_check_wallet` pre-flight reads real balance, not hardcoded 1.0
- [ ] All `os.getenv('*RPC*')` reads in DEX module removed; only `.env` exception is `DASHBOARD_MODULE_ENABLED` etc.
- [ ] Kill-switch wired: a flag in `core/risk_manager.py` immediately stops `execute()` paths in all four executors
- [ ] Honeypot result cache TTL tiered by token age
- [ ] `_pre_execution_checks` runs *before* `_simulate_execution` so DRY_RUN exercises the same gates

## Proposed action backlog

- [ ] **AC-01** `Fix DirectDEX slippage decimals` — touches `trading/executors/direct_dex.py` (lines 560-606, 819) — expected gain: unbreaks 60% of pair surface — owner: smartcontract.
- [ ] **AC-02** `V3 fee-tier router via factory.getPool` — touches `trading/executors/direct_dex.py` (lines 683-692, add helper) — expected gain: +0.1-0.25% per stable trade — owner: smartcontract.
- [ ] **AC-03** `Centralize nonce manager` — touches `trading/executors/{direct_dex,base_executor,mev_protection}.py` (extract to `trading/wallet_state.py`) — expected gain: kills nonce-collision retry burn (~5% of trades) — owner: smartcontract.
- [ ] **AC-04** `Pool-engine integration for DEX executors` — touches `trading/executors/{direct_dex,base_executor,mev_protection}.py`, `data/collectors/{chain_data,mempool_monitor}.py` — expected gain: kills 429-induced miss-rate — owner: smartcontract + backend.
- [ ] **AC-05** `Fix Flashbots signing in mev_protection` — touches `trading/executors/mev_protection.py` (lines 336-393) — expected gain: actually-working private mempool path — owner: smartcontract.
- [ ] **AC-06** `Migrate off Web3.WebsocketProvider` — touches `data/collectors/mempool_monitor.py` (lines 103-118) — expected gain: revives sandwich detection — owner: smartcontract.
- [ ] **AC-07** `On-chain minOut for FlashLoanArbitrage` — touches `contracts/FlashLoanArbitrage.sol` (lines 215-241, `_swap`) — expected gain: sandwich resistance — owner: smartcontract.
- [ ] **AC-08** `SafeERC20 + approve(0)→approve(n) for USDT class` — touches `contracts/FlashLoanArbitrage*.sol`, `trading/executors/base_executor.py` (line 649) — expected gain: unbreaks USDT pair — owner: smartcontract.
- [ ] **AC-09** `Cache gas price for 12s` — touches `trading/executors/base_executor.py` (line 587), `direct_dex.py` (line 760) — expected gain: -80% RPC reads on scan loop — owner: smartcontract.
- [ ] **AC-10** `Real wallet pre-flight balance check` — touches `modules/dex_trading/main_dex.py` (lines 122-126, 790) — expected gain: catches misconfigured wallets pre-trade — owner: smartcontract + backend.
- [ ] **AC-11** `Tiered honeypot TTL` — touches `data/collectors/honeypot_checker.py` (line 88) — expected gain: catches rug-after-pass — owner: smartcontract.
- [ ] **AC-12** `Remove sleeps from MEV protection critical path` — touches `trading/executors/mev_protection.py` (line 317-322) — expected gain: -1.25s avg latency — owner: smartcontract.
- [ ] **AC-13** `Kill-switch hook into core/risk_manager` — touches all four executors + `core/risk_manager.py` — expected gain: live-readiness gate — owner: smartcontract.

## Open questions

1. Is the intent to **delete** `MEVProtectionLayer` in favor of the working `FlashbotsExecutor` in `modules/arbitrage/arbitrage_engine.py:688`, or fix it? They are 80% duplicate and the arbitrage one works.
2. `modules/dex_trading/main_dex.py` vs root `main_dex.py` — which is canonical? The two diverge (root one has the bare `WEB3_PROVIDER_URL` test). PM agent decision.
3. Is `ToxiSolAPIExecutor` actually used in production, or is it a dead path? `_get_toxisol_quote` / `_execute_toxisol` in `base_executor.py` are stubs, suggesting the integration was abandoned.
4. Should sniper paths reuse `DirectDEXExecutor` or have a dedicated executor optimized for first-block inclusion (priority-fee biasing, no decoy/delay)?
5. Does the team want Flashbots-Protect-style (`https://rpc.flashbots.net`) as a simpler alternative to bundle submission for the DEX module?
6. The flash-loan receiver contracts (ETH/ARB/Base variants) currently differ by router constants and chain provider — would deploying a single proxy + chain-agnostic implementation simplify ownership / rotation?
