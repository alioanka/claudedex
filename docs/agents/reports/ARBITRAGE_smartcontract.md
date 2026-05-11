# ARBITRAGE_MODULE — Smart-Contract / Web3 Audit

## Executive summary

**Verdict: AMBER, leaning RED for triangular and Solana.** Spatial EVM arbitrage (`arbitrage_engine.py`) is the strongest part of the codebase: it uses Aave V3 `flashLoanSimple` via a proper receiver contract (`onlyPool` + `initiator == self`), it has a simulation gate (`checkArbitrage` view) and a profit-vs-gas guard with a 1.3x buffer, and the EIP-1559 gas math is sensible. Flashbots signing is correct (`encode_defunct(keccak(body))`, line 745). However a transcription bug in the canonical DAI mainnet address — present in BOTH `arbitrage_engine.py:169` and `triangular_engine.py:74` (`...EeAdDcB80656c63` vs the real `...EedeAC495271d0F`) — means every DAI leg either reverts (no such token contract exists at that address) or quotes phantom liquidity. Triangular ETH arbitrage is **not atomic** in its fallback path (sends 3 sequential txs when Flashbots is unavailable, line 1042-1075) which is a guaranteed wallet-stuck scenario; Solana arbitrage is honest about this risk (lines 1347-1367) and refuses non-Jito fallback — that is the correct posture.

**Top 3 profit leaks:** (1) The DAI address bug above silently zeroes out the most-liquid stable on Ethereum from the arbitrage universe — any opportunity in pairs involving DAI is invisible or evaluates against a non-existent contract. (2) `_check_arb_opportunity` (`arbitrage_engine.py:1326-1487`) re-quotes both legs serially on every router for every pair every 2 s; for ETH mainnet that is 3 routers × 2 directions × 5+ pairs × 30 rps = quickly throttled, AND it does not refresh the BUY quote after picking BEST_BUY (line 1381) — so when the SELL quote (line 1396) is taken, the BUY price has already moved and `weth_returned > amount_owed` can be a stale-positive. (3) Triangular engine builds three sequential txs (`triangular_engine.py:921-964`) with `tx2`/`tx3` `amountIn=1` placeholders (lines 938, 953), relying on the in-transit balance — that pattern only works inside a flash-loan callback or a custom multicall; as standalone txs, `swapExactTokensForTokens(1, ...)` simply swaps 1 wei and leaves the rest of the prior swap output sitting in the wallet. Net: triangular arb has never executed correctly in non-Flashbots fallback mode.

**Top 3 risks:** (1) Triangular fallback non-atomic execution will permanently strand intermediate tokens; only the `if profit_pct < 0.01` gate (line 1047) limits the bleed. (2) `FlashLoanArbitrage.sol:234` passes `amountOutMin = 0` for both swap legs; the on-chain `require(finalAmount >= amountOwed, ...)` guard (line 200) prevents a *losing* trade from settling, but a sandwicher can still grind profit to break-even because there is no slippage floor on intermediate amounts. (3) Wallet/private-key handling: `FlashLoanExecutor.__init__` (`arbitrage_engine.py:496`) receives the decrypted private key by value and stores it as an instance attribute (`self.private_key`) — that string lives for the lifetime of the process, in heap, in any traceback or `repr`. Same for `FlashbotsExecutor.__init__` (line 694) and `EVMArbitrageEngine` (line 1022). Three-place storage of a hot key.

## File-by-file findings

### `modules/arbitrage/arbitrage_engine.py` (2077 lines)
- `PriceFetcher` (line 25): single-source CoinGecko, 60s cache. Returns `None` on miss (good), but caller in `_log_arb_trade` (line 1882) only warns and skips DB log — silent P&L gap if CoinGecko is flaky. Should fall through to chain oracle (Chainlink ETH/USD on the chain in question).
- `FLASH_LOAN_CONTRACT_ABI` (line 120) is hand-rolled; matches the `executeArbitrage` / `checkArbitrage` signature of `FlashLoanArbitrage.sol`. Good.
- **CRITICAL: DAI address `0x6B175474E89094C44Da98b954EeAdDcB80656c63`** at lines 169 and 1632 — not the canonical DAI (`...EedeAC495271d0F`). Triangular engine inherits this (line 74). Every quote against DAI will either RPC-error (no contract) or quote a different token if some other contract sits at that address.
- `TOKEN_SCAN_AMOUNTS_BASE` (line 461) is the right idea (per-token decimal-aware scan amounts). `_scan_amounts` scaling logic (line 907) ties it to `flash_loan_amount` — good. But comment says the base is "10 ETH" without checking that scan-amount values per token were calibrated to actual L1 vs L2 pool sizes.
- `FlashLoanExecutor._get_next_nonce` (line 513): pulls both confirmed and pending nonces, takes max with tracked. Acceptable, but `_pending_nonce` is reset to `None` only on `'replacement transaction underpriced'` (line 681) — a regular revert leaves the tracker advanced and the next tx skips a nonce.
- `FlashLoanExecutor.execute_arbitrage` (line 571): `gas_limit = 450000` (line 628) is set, EIP-1559 fees built, profit-vs-cost guard with 1.3x buffer (line 633). Strong. But hard-coded `priority_fee = 2 * 10**9` (line 618) — 2 gwei. On Ethereum mainnet during congestion this is too low for next-block inclusion. Should be `max(2 gwei, eth_feeHistory_25th_pct)`.
- `_last_tx_time` rate limit of 12s (line 644-649) is reasonable for L1 but is the same on all subclasses — for Base (2s blocks) and Arbitrum (~250ms), this is throttling profit. Should be `self.chain_config['block_time_s']`.
- `FlashbotsExecutor.send_bundle` (line 712): correct signature scheme via `encode_defunct(text=Web3.keccak(text=body).hex())` (line 745). Builder set defaults to Flashbots only — does not also fan out to BeaverBuild, Titan, rsync — inclusion rate ~30-50% of multi-builder norm.
- `FlashbotsExecutor` uses the *same* `signing_key` as the private key by default (line 697). Should be a separate signing key (the docs call this the "reputation key") to avoid leaking your wallet identity to all builders.
- `EVMArbitrageEngine.__init__` (line 840) reads `rpc_url` from config then falls back to `RPCProvider.get_rpc_sync(self.RPC_PROVIDER_KEY)` (line 864) — partial pool-engine integration, but then re-fallbacks to `os.getenv(self.RPC_ENV_KEY)` (line 870). The order is OK; getting one `rpc_url` and never re-querying means *no failover* once selected.
- `min_profit_threshold` hard-coded `0.003` (line 894) — should be DB-config-driven and chain-specific.
- `_check_arb_opportunity` (line 1326): two-pass query as described. Quotes are stored only in memory in `buy_prices`/`sell_prices` dicts; no timestamp on each → no staleness check. By the time `_execute_flash_swap` runs, both quotes can be 5+ seconds old. Add `_quote_timestamp` and re-validate `checkArbitrage()` immediately before sending.
- `_check_arb_opportunity` line 1472 references `forward_output` and `final_output` in an f-string — those variables are **not defined** in this scope. This `logger.info` will `NameError` on every successful opportunity log. Probably never fired in dry-run because the executable path doesn't reach here; in production it would crash the loop.
- `_execute_with_flash_loan` (line 1650) verifies `weth_checksum in weth_addresses` (line 1706) — good. But `weth_addresses` (line 1699) is a hard-coded set inside the method; should live in `chain_config`.
- `_execute_direct_swap` (line 1739): builds buy_tx + sell_tx with `nonce`, `nonce+1`; `_send_private_transaction` does **not** exist in this class but is referenced in `base_executor.py` — confusion across modules. Direct-swap fallback (line 1856) only sends the buy tx; sell tx is **never broadcast** if Flashbots fails. After fallback, the wallet is now long the intermediate token. Same atomic-leak class as triangular.
- `_log_arb_trade` (line 1863): P&L math hard-codes `FLASH_LOAN_FEE_PCT = 0.0005`, `SLIPPAGE_ESTIMATE_PCT = 0.006`, `GAS_COST_USD = 15.0` (lines 1888-1890). These should come from chain_config and actual receipts.
- `AAVE_FLASHLOAN_ASSETS` (line 1627) hardcodes per-chain allowed flash-loan assets — good. But not consulted in `_execute_with_flash_loan`; only the `weth_addresses` set is checked. Means a non-WETH flash-loan attempt would slip past.

### `modules/arbitrage/triangular_engine.py` (1199 lines)
- `RPCRateLimiter` (line 29) with exponential backoff — good design.
- `TOKENS` (line 68) inherits the **same DAI bug** (line 74). Plus FRAX/LUSD/etc. that may or may not have liquidity on every router being scanned.
- `TRIANGULAR_CYCLES` (line 99) — 22 hard-coded cycles. No detection of cycles that haven't traded recently (dead cycles burn RPC).
- `_check_triangular_opportunity` (line 608): three sequential hops on every router. RPC budget: 3 hops × N routers per cycle × 22 cycles = explodes RPC bill. With rate limiter at 5 rps, a full pass takes minutes; spread will have moved.
- Phantom-spread guard at 10% (line 736) — sensible.
- Dynamic threshold via `_calculate_dynamic_threshold` (line 564) — implementation not read but is invoked. Good.
- `_execute_triangular_swap` (line 870): the **non-atomic fallback is broken**. `tx2`/`tx3` use `amountIn = 1` placeholder (lines 938, 953). The comment says "Will be filled by actual output from tx1" but no actual filling happens. Sequential fallback (line 1042) sends these three txs as-is → tx2 swaps 1 wei, tx3 swaps 1 wei, the intermediate balances from tx1 stay stranded.
- Even the Flashbots path is broken for the same reason: bundles do not magically link outputs. To make a triangular bundle work you need to either (a) use a multicall contract / dedicated triangular contract, or (b) split into individual swaps where each minOut becomes amountIn of the next AND you carefully time them — neither is happening.
- Profit-too-low gate (`profit_pct < 0.01`, line 1047) before sequential is a partial safety net but does not solve atomicity.

### `modules/arbitrage/solana_engine.py` (1954 lines)
- Jupiter quotes + Raydium fallback + Jito bundle execution. Architecture is correct for Solana.
- `JitoClient` (line 294): primary endpoint from `JITO_BLOCK_ENGINE_URL` env or default; rate-limit handling and `_get_next_endpoint` round-robin (line 376). Good.
- `is_available` classmethod (line 342) — tracks rate-limit state class-globally so concurrent `SolanaArbitrageEngine` instances share. Reasonable.
- `tip_lamports` default 10000 (line 1324) is too low; the code itself warns at line 1325-1326. Set the floor to 50000 and the default to a percentage of expected profit (10% of `profit_usd_in_lamports` cap'd at 0.001 SOL).
- `_check_arb_opportunity` (line 1055) compares Jupiter → reverse Jupiter; this measures Jupiter's *spread*, not arbitrage between Jupiter and a direct Raydium/Orca route. A 0.2% threshold can be hit just by Jupiter routing inefficiency on small amounts. Switch to "Jupiter forward quote vs Raydium direct quote" for genuine cross-route arbitrage.
- Quote staleness check (`_is_quote_stale`, line 1177) with `max_quote_age=10` — sound. Refresh logic at line 1216 is good.
- `MAX_REALISTIC_PROFIT = 0.03` (line 1120) phantom-opportunity guard — sound.
- **CRITICAL POSTURE (CORRECT):** explicit refusal to RPC-fallback on bundle failure (lines 1347-1367) — this is the right call. Triangular engine should adopt the same posture.
- `_sign_transaction` and `_get_keypair` (lines 1306, 1374): keypair built fresh per execution. Better than holding the keypair on `self`, but the private key bytes are still on `self.private_key` and copied through `_get_decrypted_key` (line 864).

### `modules/arbitrage/arbitrage_alerts.py` (405 lines)
- Pure Telegram alerts. No on-chain code, no risk surface. `TELEGRAM_BOT_TOKEN` read pattern not audited here (lives in `monitoring/`).
- Pretty-prints amounts in native + USD; no input validation but failure-mode is just a missing alert.

### `modules/arbitrage/main_arbitrage.py` (413 lines)
- Engine bootstrapper. RPC discovery via `RPCProvider.get_rpc_sync(...)` then `os.getenv` fallback (lines 144-151, 173-179, 200-206, 227-233). Partial pool-engine compliance.
- Initializes pool engine *after* manager construction (line 351-359) — order is OK because engines are initialized inside `manager.initialize()` which runs after `Pool Engine initialized for RPC management` log; but `MultiChainArbitrageManager.__init__` does not take pool_engine; engines call `RPCProvider.get_rpc_sync` themselves.
- Telegram registration at line 380 uses `positions_attr='active_positions'` but arbitrage engines do not expose `active_positions` (they are not position-bearing) — falls through silently. Cosmetic.

### `trading/executors/mev_protection.py` (shared)
- See `DEX_smartcontract.md` for full audit. For arbitrage: `arbitrage_engine.py:FlashbotsExecutor` correctly **does not depend on `MEVProtectionLayer`** — it implements its own send/simulate path. That insulates arb from the bugs in `mev_protection.py`. Keep it that way; do not "consolidate".

### `config/pool_engine.py` surface
- Provider keys used in arbitrage: `ETHEREUM_RPC`, `ARBITRUM_RPC`, `BASE_RPC`, `SOLANA_RPC`. Pool engine has them. But `EVMArbitrageEngine` fetches ONE URL and never reports back via `report_success` / `report_failure` / `report_rate_limit`. Without telemetry the pool engine cannot rotate.

### Flash-loan receiver contracts
- `FlashLoanArbitrage.sol` (mainnet, 333 lines): properly gated `onlyOwner` + `onlyPool` + initiator check. `_swap` (line 215) hard-codes `amountOutMin=0` — see DEX-11. `withdrawToken`/`withdrawETH` standard. No pause function — if a bug is discovered, you must redeploy and update env. Add an `emergencyWithdraw(asset)` or a pause flag.
- `FlashLoanArbitrage_Arbitrum.sol`: routers swapped for Arbitrum routers (SushiSwap, Camelot, Zyberswap). Same `_swap` minOut=0 issue. Arbitrum-specific Aave V3 provider hardcoded.
- `FlashLoanArbitrage_Base.sol`: not opened in depth but presumably the same pattern with Aerodrome/BaseSwap routers.
- All three contracts use solc `^0.8.20`. No reentrancy guard, no timelock on `transferOwnership`, no event for ownership transfer.
- ERC-20 calls use raw `IERC20.transfer / approve` (lines 224, 293, 302, 311) — fails for non-bool-returning tokens (USDT-mainnet). USDT will brick the receiver.

## Risk taxonomy

| ID | Category | File:Line | Severity | Description | Fix sketch |
|----|----------|-----------|----------|-------------|------------|
| ARB-01 | Other (data) | arbitrage_engine.py:169, 1632; triangular_engine.py:74 | **CRITICAL** | DAI mainnet address typo: `...EeAdDcB80656c63` instead of `...EedeAC495271d0F` | Single grep+replace; add unit test that does on-chain `code != 0x` check for every constant address |
| ARB-02 | Tx-ordering | triangular_engine.py:921-1075 | **CRITICAL** | Non-atomic 3-hop swap; sequential fallback uses placeholder amountIn=1; bundle path is also broken | Deploy a `TriangularArbitrage.sol` contract analogous to `FlashLoanArbitrage.sol` with `executeTriangular(...)`; refuse non-atomic fallback (mirror Solana posture) |
| ARB-03 | Tx-ordering | arbitrage_engine.py:1856 | **HIGH** | `_execute_direct_swap` fallback only sends buy tx; sell tx is dropped, wallet ends long the intermediate token | Either flash-loan-only mode (refuse non-FL fallback) or send both txs and account for atomicity loss in P&L gate |
| ARB-04 | Sandwich risk | FlashLoanArbitrage.sol:234 | **HIGH** | `_swap(amountOutMin=0)`; sandwicher can grind to break-even | Compute minOut in params, pass per-leg into `executeOperation` |
| ARB-05 | Bridge/cross-chain | FlashLoanArbitrage.sol:224, 293 | **HIGH** | Raw ERC20.approve/transfer; bricks on USDT mainnet | Switch to OZ SafeERC20; approve(0)→approve(amount) pattern |
| ARB-06 | Key/secret handling | arbitrage_engine.py:496, 694, 1022 | **HIGH** | Private key stored as instance attribute in three places, lives in heap for the entire process | Use a lazy unlock pattern; load+sign in a single scope, never assign to `self.private_key` |
| ARB-07 | Oracle | arbitrage_engine.py:1472 | MEDIUM | `forward_output`/`final_output` referenced but not defined; `NameError` on successful opportunity log in live mode | Compute and assign before f-string; add unit test |
| ARB-08 | Oracle/Staleness | arbitrage_engine.py:1326-1487 | **HIGH** | Quotes have no timestamp; gap between BUY-quote scan and SELL-quote scan opens stale-positive window | Tag every quote with `time.time()`; re-call `flash_loan_executor.check_arbitrage_profit` immediately before send |
| ARB-09 | Gas | arbitrage_engine.py:618 | MEDIUM | Priority fee hard-coded 2 gwei; too low for L1 congestion | `max(2 gwei, w3.eth.fee_history(5,'latest',[25])[reward percentile])` |
| ARB-10 | Gas | arbitrage_engine.py:644-649 | LOW | 12s tx-rate-limit hard-coded across all subclasses; throttles L2 | Use `chain_config['block_time_s']` |
| ARB-11 | RPC | arbitrage_engine.py:1044, 870 | MEDIUM | Single RPC URL picked at init; no telemetry to pool engine; no failover on hot path | After every `w3.eth.*` call wrap with pool-engine `report_success/report_failure`; rebind `self.w3` on consecutive failures |
| ARB-12 | MEV | arbitrage_engine.py:697 | MEDIUM | Flashbots signing key == wallet private key; leaks identity to all builders | Generate ephemeral `signing_key` per process (or store separately in secrets manager) |
| ARB-13 | MEV | arbitrage_engine.py:699 | MEDIUM | Single-relay submission (`relay.flashbots.net`); inclusion rate ~30-50% of multi-relay norm | Fan out to BeaverBuild + Titan + rsync in parallel |
| ARB-14 | Honeypot exposure | arbitrage_engine.py — | LOW | Arbitrage engines never call `honeypot_checker`; assumed safe because they trade known assets — true today but a token-set change could expose | Pre-flight check `chain_config['tokens'].values()` once at startup against `honeypot_checker` |
| ARB-15 | Idempotency | arbitrage_engine.py:526, 681 | MEDIUM | `_pending_nonce` only resets on "replacement underpriced"; other revert types leak advanced nonce | Reset on ANY exception in `execute_arbitrage`; refetch on next call |
| ARB-16 | Solana / Tip math | solana_engine.py:1324-1326 | MEDIUM | Default Jito tip 10000 lamports too low | Default to `min(0.001 SOL, profit_usd * 0.10 / sol_price_usd)`; floor 50000 |
| ARB-17 | Solana / Cross-route | solana_engine.py:1075-1094 | **HIGH** | "Arbitrage" = Jupiter forward vs Jupiter reverse — that's Jupiter's spread, not arb | Switch to Jupiter forward vs Raydium/Orca direct; or two competing Jupiter routes (different `excludeDexes`) |
| ARB-18 | Contracts | FlashLoanArbitrage.sol — | LOW | No pause / emergency-stop; bug requires redeploy + env change | Add `pause()` (onlyOwner), check in `executeArbitrage` |
| ARB-19 | Contracts | FlashLoanArbitrage.sol:316 | LOW | `transferOwnership` immediate, no two-step | Adopt OZ `Ownable2Step` pattern |
| ARB-20 | Profitability gate | arbitrage_engine.py:894, 1430 | MEDIUM | `min_profit_threshold=0.003` and `estimated_costs=0.005` are hard-coded; should reflect chain (Arb/Base much lower) | Move both into `chain_config` |

## Profitability levers (ranked by ROI)

1. **Fix the DAI address (ARB-01).** Restores the deepest stable on Ethereum into the arb universe. Single-line fix, immediate value.
2. **Deploy a `TriangularArbitrage.sol` flash-loan-style contract (ARB-02).** Today triangular arb is theatre; with one atomic contract you unlock a class of opportunities (CRV/3pool, frax/3crv, stETH/wstETH/ETH) that pure spatial arb cannot capture. Estimated edge: 0.05-0.2% per cycle, several opportunities/day on Ethereum.
3. **Per-leg minOut into flash-loan receiver (ARB-04).** Most direct sandwich-protection upgrade. On Ethereum mainnet flash loans this can be the difference between 0.3% and 0.05% realized profit per trade.
4. **Refuse non-atomic fallback in arbitrage_engine (ARB-03).** Stops the slow bleed where partial fills strand intermediate tokens.
5. **Switch Solana arb to Jupiter-vs-Raydium-direct, not Jupiter-vs-Jupiter (ARB-17).** Today the bot pays Jupiter's spread to itself; a genuine cross-DEX comparison is where the edge is.
6. **Multi-relay Flashbots fan-out (ARB-13).** Up to 2x inclusion rate at zero cost per attempt.
7. **Quote-freshness re-validate via `check_arbitrage_profit` immediately pre-send (ARB-08).** Eliminates stale-positive trades that consume gas and revert.
8. **Dynamic priority fee tied to `fee_history` (ARB-09).** Better inclusion during congestion when most edge appears.
9. **SafeERC20 + USDT-style approve(0)→approve(n) (ARB-05).** Unlocks USDT triangles and pairs; currently they revert.
10. **Per-chain tx-rate-limit (ARB-10).** On Base/Arbitrum, 12 s per tx is throwing away ~10x throughput.

## Live-trade gap list

- [ ] DAI address corrected and a startup assertion that every token in `chain_config['tokens']` has non-empty `w3.eth.get_code(address)`
- [ ] Triangular engine refuses fallback or has an atomic on-chain executor
- [ ] `_execute_direct_swap` refuses fallback OR sends both legs with reentrancy-safe atomicity (or just deletes fallback and forces flash-loan)
- [ ] FlashLoanArbitrage.sol redeployed with per-leg minOut and SafeERC20
- [ ] Private key never assigned to a long-lived `self.*` attribute
- [ ] Every `w3.eth.*` call wrapped with pool-engine telemetry
- [ ] Flashbots signing key separated from wallet key (env: `FLASHBOTS_SIGNING_KEY`, fall back to ephemeral)
- [ ] Multi-relay fan-out (BeaverBuild, Titan, rsync) implemented in `FlashbotsExecutor`
- [ ] Solana engine switched to real cross-route comparison
- [ ] Solana Jito tip defaulted to 50000 lamports floor or profit-fraction
- [ ] `arbitrage_engine.py:1472` `NameError` fixed (forward_output / final_output)
- [ ] `min_profit_threshold` and `estimated_costs` per-chain in `chain_config`
- [ ] Kill-switch wired: a flag in `core/risk_manager.py` stops `_execute_flash_swap` immediately
- [ ] Pre-flight: on startup, simulate `check_arbitrage_profit` for one known pair per chain; fail boot if it errors
- [ ] Tx-rate-limit per-chain from `chain_config['block_time_s']`
- [ ] `_pending_nonce` resets on any exception, not only "underpriced"

## Proposed action backlog

- [ ] **AC-A01** `Fix canonical DAI address + add startup chain-of-trust check` — touches `modules/arbitrage/{arbitrage_engine,triangular_engine}.py` — expected gain: unblocks DAI pair universe — owner: smartcontract.
- [ ] **AC-A02** `Triangular atomic contract + Python wiring` — touches new `contracts/TriangularArbitrage.sol` + `modules/arbitrage/triangular_engine.py` (delete fallback) — expected gain: unlock triangular edge — owner: smartcontract.
- [ ] **AC-A03** `Per-leg minOut into FlashLoanArbitrage` — touches `contracts/FlashLoanArbitrage*.sol` and `modules/arbitrage/arbitrage_engine.py:FlashLoanExecutor.execute_arbitrage` (new args) — expected gain: -0.2% sandwich loss per L1 trade — owner: smartcontract.
- [ ] **AC-A04** `Refuse non-atomic fallback in EVMArbitrageEngine._execute_direct_swap` — touches `arbitrage_engine.py:1739-1861` — expected gain: kills the partial-fill bleed — owner: smartcontract.
- [ ] **AC-A05** `Quote freshness + pre-send re-simulation` — touches `arbitrage_engine.py:_check_arb_opportunity, _execute_flash_swap` — expected gain: reduce reverted-tx burn rate by ~30% — owner: smartcontract.
- [ ] **AC-A06** `Dynamic priority fee from fee_history` — touches `arbitrage_engine.py:618` — expected gain: better inclusion at congestion peaks — owner: smartcontract.
- [ ] **AC-A07** `Per-chain min_profit_threshold + estimated_costs from chain_config` — touches `arbitrage_engine.py:894,1430` + `CHAIN_CONFIGS` — expected gain: unlock L2 opportunities below 0.5% — owner: smartcontract.
- [ ] **AC-A08** `Per-chain block-time-aware tx rate limit` — touches `FlashLoanExecutor._last_tx_time / 12s gate` — expected gain: ~5-10x throughput on Arb/Base — owner: smartcontract.
- [ ] **AC-A09** `Multi-relay Flashbots fan-out` — touches `FlashbotsExecutor` — expected gain: 2x inclusion rate — owner: smartcontract.
- [ ] **AC-A10** `Solana cross-route arbitrage (Jupiter vs Raydium direct)` — touches `solana_engine.py:_check_arb_opportunity` — expected gain: real Solana edge — owner: smartcontract.
- [ ] **AC-A11** `Jito tip default 50k + profit-share` — touches `solana_engine.py:1324` — expected gain: ~+15% Solana bundle inclusion — owner: smartcontract.
- [ ] **AC-A12** `SafeERC20 + approve(0)→approve(n) in flash-loan contracts` — touches `contracts/FlashLoanArbitrage*.sol` — expected gain: unlock USDT pair — owner: smartcontract.
- [ ] **AC-A13** `Pool-engine telemetry wired in EVM arb engines` — touches `arbitrage_engine.py` post `w3.eth.*` calls — expected gain: surviving RPC outages without dead loops — owner: smartcontract + backend.
- [ ] **AC-A14** `Separate Flashbots signing key + key handling cleanup` — touches `arbitrage_engine.py:FlashbotsExecutor.__init__` + secrets manager — expected gain: identity isolation, smaller key footprint in heap — owner: smartcontract.
- [ ] **AC-A15** `Kill-switch wired into risk_manager` — touches `arbitrage_engine.py:_execute_flash_swap` + `core/risk_manager.py` — expected gain: live-readiness gate — owner: smartcontract.

## Open questions

1. Is `EVMArbitrageEngine._execute_direct_swap` (non-flash-loan path) actually ever expected to run, given that we have a deployed flash-loan receiver on all three chains? If not, delete the whole branch; it is a footgun.
2. Should triangular arbitrage be deprecated entirely in favor of (a) Curve metapool arb via a dedicated `CurveArb.sol`, and (b) Balancer `batchSwap` via Balancer's vault, both of which are atomic-by-construction? Triangular via plain V2 routers is a low-edge / high-failure-mode strategy.
3. The Solana engine carries `RaydiumClient` and `JupiterClient` but only Jupiter is used in the arb path. Is the plan to wire Raydium for cross-route comparison (AC-A10) or to remove dead `RaydiumClient`?
4. Pool-engine integration is partial in arbitrage but the pool engine itself is well-designed. PM agent decision: should the arb module fully convert to `await pool_engine.get_endpoint(...)` on every call (pure but slow), or pre-bind one and only re-query on failure?
5. Is there an existing or planned `flashbots_signing_key` env / secret? Today the wallet key is reused (`signing_key or private_key`, line 697).
6. The `FlashLoanArbitrage` contracts have no event for `transferOwnership` and no `pause()`. Is the team OK with a redeploy-on-bug model, or should an upgrade lever be added?
7. `arbitrage_engine.py:1472` references undefined names — is the live path actually exercised today, or is the engine still permanent-dry-run? (If dry-run-only, the bug is invisible.)
