# ARBITRAGE_MODULE — Trading-Desk / Risk / Live-Readiness Audit

Owner: market-trading-analyst (tertiary; smartcontract-web3-expert is primary, quant-algo-expert second).
Scope: `modules/arbitrage/{arbitrage_engine,triangular_engine,solana_engine,main_arbitrage,arbitrage_alerts}.py` and the cross-module risk plumbing already audited in DEX_analyst.md.

## Executive verdict: RED

The arbitrage module is the most live-exposed surface in the bot — it directly signs, builds, and broadcasts swaps from inside the engine without going through `OrderManager`, `RiskManager.validate_trade`, or `PortfolioManager.can_open_position`. Three findings make it RED:

1. **The cross-module risk plumbing does not gate arbitrage at all.** Every other module routes through `core/risk_manager.py:1047 validate_trade` (which checks circuit breakers) before sending. The three arbitrage engines (`arbitrage_engine.py`, `triangular_engine.py`, `solana_engine.py`) instantiate their own `dry_run` flag (e.g. arbitrage_engine.py:877-883) and call `self.w3.eth.send_raw_transaction` directly (arbitrage_engine.py:670, 1856; triangular_engine.py:1052, 1062, 1069). The circuit breakers configured in `risk_manager.py:285-289` (max consecutive losses, max daily loss, max drawdown) **never see arbitrage P&L**. Hourly gas burn from reverts is also unmonitored at the risk layer.
2. **The atomic-execution fallback is a one-legged trade that guarantees an inventory loss.** In `arbitrage_engine._execute_direct_swap` (arbitrage_engine.py:1739-1861) both buy- and sell-tx are signed (lines 1784, 1803). Flashbots is attempted (line 1818). If Flashbots fails or is unavailable, the code at line 1856 broadcasts ONLY the buy tx to the public mempool — the sell tx is dropped on the floor. The bot ends up holding the bought token with no exit leg, will lose ≥ the bid-ask spread + slippage on the eventual market sell, and is exposed to MEV on the unprotected buy. The same pattern recurs in `triangular_engine._execute_triangular_swap` (triangular_engine.py:1042-1075) where the "sequential" fallback executes legs 1→2→3 but if leg 2 or 3 reverts, leg 1 is already on-chain (line 1057 returns None without unwinding).
3. **No per-hour gas budget kill-switch.** The market-trading-analyst.md explicitly demands: "kill-switch on hourly failed-tx gas > realized profit". Nothing in the three engines implements this. There is a per-pair daily execution cap (arbitrage_engine.py:930; `_max_executions_per_pair_per_day = 5`) and per-cycle daily cap on triangular (triangular_engine.py:329), but those count *attempts* not *gas spent*; on a reverting pair the bot can still spend gas inside the per-day attempt budget without ever realising profit.

Until items 1-3 are resolved, the arbitrage module is unfit for live trading at any size on Ethereum mainnet. On L2 (Arbitrum, Base) the gas-burn risk is smaller (~$0.50-$1 per failed tx, arbitrage_engine.py:422) but item 2 (one-legged inventory loss) is chain-agnostic.

## DRY_RUN propagation audit

Per-engine. Each engine has its own `dry_run` resolution and its own send path.

| Path | File:Line | DRY_RUN-gated? | Notes |
|---|---|---|---|
| `EVMArbitrageEngine._execute_flash_swap` | modules/arbitrage/arbitrage_engine.py:1551 | YES | Early return when `self.dry_run` |
| `EVMArbitrageEngine._execute_with_flash_loan` | modules/arbitrage/arbitrage_engine.py:1650 | INHERITED | Only called from `_execute_flash_swap` which is dry-run-gated upstream. |
| `EVMArbitrageEngine._execute_direct_swap` | modules/arbitrage/arbitrage_engine.py:1739 | INHERITED | Same upstream gate. But internal `send_raw_transaction` at line 1856 has **no second gate**. If the caller mistakenly invokes this with `dry_run=False` and the config later flips, the send fires. |
| `FlashLoanExecutor.execute_arbitrage` | modules/arbitrage/arbitrage_engine.py:571 | NO | This class has no `dry_run` of its own. It is only used when the engine's `dry_run=False` and the executor was initialised (arbitrage_engine.py:1093). One mis-step in the init flow (e.g. database config flip without engine restart) is enough to make live. |
| `TriangularArbitrageEngine._execute_triangular` | modules/arbitrage/triangular_engine.py:824 | YES | Early return when `self.dry_run` |
| `TriangularArbitrageEngine._execute_triangular_swap` | modules/arbitrage/triangular_engine.py:870 | INHERITED | Sequential fallback path (line 1052+) is **not re-gated** |
| `SolanaArbitrageEngine._execute_arbitrage` | modules/arbitrage/solana_engine.py:1196, 1253 | YES | Early return when `self.dry_run` |
| `JupiterClient.get_swap_transaction` calls | modules/arbitrage/solana_engine.py:1284, 1285 | INHERITED | Only called after dry-run check at 1253. |
| `JitoClient.send_bundle` | modules/arbitrage/solana_engine.py (after 1299) | INHERITED | Same upstream gate. |

No completely ungated paths found. The fragility is structural: each engine duplicates the `dry_run = db_config or env_var` resolution (`arbitrage_engine.py:877-883`, `triangular_engine.py:297-303`, `solana_engine.py:820-826`). Three independent sources of truth for the same flag — any one being mis-fetched from the DB flips a single engine to LIVE while the others remain dry. This is a P1 maintainability hazard.

Also note: `arbitrage_alerts.py:213` filters out dry-run tx hashes in alerts (`if alert.tx_hash and alert.tx_hash != "DRY_RUN"`). String comparison; if the dry-run tx_hash format changes anywhere, alerts will leak DRY_RUN trades to operators as real trades.

## Risk-policy coverage

| Control | Status | Evidence |
|---|---|---|
| Per-trade max loss | PARTIAL | Each engine has `min_profit_threshold` (arbitrage_engine.py:894, triangular_engine.py:313, solana_engine.py:830) — but this is a *minimum gross*, not a max-loss bound. The flash-loan path bounds loss to the gas+fee on revert; the direct-swap path is uncapped if the buy lands and the sell doesn't. |
| Per-hour max loss | MISSING | None of the three engines track hourly P&L. `_stats` (arbitrage_engine.py:949, triangular_engine.py:335, solana_engine.py:857) tracks scans/found/executed, not money. |
| Per-day max loss | MISSING in arbitrage; PRESENT only in cross-module RiskManager but **not consulted** by arbitrage |
| Per-hour gas burn budget | MISSING | No tracker for failed-tx gas. |
| Per-symbol/per-pool concentration | PARTIAL | `_max_executions_per_pair_per_day = 5` (arbitrage_engine.py:930); `_max_executions_per_cycle_per_day = 5` (triangular_engine.py:329); `_max_executions_per_pair_per_day = 10` (solana_engine.py:855). These cap *attempts* per pair, not capital deployed. With flash loans the capital is unbounded per attempt. |
| Per-module capital cap | MISSING | The flash-loan amount is read from `config['flash_loan_amount']` (arbitrage_engine.py:901) without any check against a per-module ceiling. Default `10 ETH` (~$25-40k at current ETH) is far above the $10 hard cap enforced for DEX trades in `risk_manager.py:220`. |
| Correlated-drawdown rule (this module vs others) | MISSING | No cross-module rule. If three arbitrage engines lose simultaneously due to mempool conditions, nothing slows DEX module. |
| Oracle-deviation kill-switch | MISSING | The persona file specifically calls this out. No price-feed cross-check; engines use DEX `getAmountsOut` directly which is the manipulation surface. |
| Bridge-time + reorg risk pricing (cross-chain) | N/A | Arbitrage here is intra-chain only; no cross-chain bridging path observed. **Confirm with smartcontract agent.** |
| Emergency-stop hook | MISSING | No analog to `engine.emergency_close_all_positions`. The engines have only `stop()` (e.g. triangular_engine.py:1197) which sets `is_running=False`. In-flight tx do not unwind. |
| Nonce / idempotency on retries | PARTIAL | `FlashLoanExecutor._get_next_nonce` (arbitrage_engine.py:513-527) tracks `_pending_nonce` and uses pending count. But the direct-swap path at arbitrage_engine.py:1765 reads `get_transaction_count` fresh each call without locks; on concurrent triangular+spatial engines on the same wallet, this collides. |
| Position reconciliation on startup | PARTIAL | Arbitrage holds no positions in the usual sense (flash-loan paths exit in the same tx). But the direct-swap one-legged-fallback bug (item 2) leaves inventory; `engine._load_state` is a stub so this inventory is invisible at restart. |
| Stale-heartbeat → flatten/freeze | MISSING | No heartbeat; `_log_stats_if_needed` notices stale spreads (arbitrage_engine.py:1234-1272) but only switches to slow-scan mode — does not freeze. |

## Order/Position lifecycle review

Arbitrage skips most of the standard lifecycle. The flow per engine:

**Spatial (`EVMArbitrageEngine.run` arbitrage_engine.py:1172):**
1. Round-robin pair scan → `_check_arb_opportunity` (line 1326).
2. For each pair: query `getAmountsOut` on all DEXs, compute spread (lines 1357-1399).
3. If `net_spread > min_profit_threshold` (0.3%, line 894): rate-limit check (`_pair_execution_count`), then `_execute_flash_swap` (line 1477).
4. Dry-run: log + DB write, return.
5. Live: gas balance check (line 1564), then either flash-loan path (line 1595) or direct swap (line 1600).
6. Flash-loan path: calls deployed `FlashLoanArbitrage` contract — atomic; if it reverts, you lose only gas.
7. Direct-swap path: signs buy+sell, tries Flashbots, **on Flashbots failure broadcasts only buy** (line 1856).
8. `_log_arb_trade` (line 1863): inserts into `arbitrage_trades` table with cost deductions (0.05% flash fee, 0.6% slippage, $15 gas — hardcoded at lines 1888-1890; values are not chain-aware: Arbitrum/Base gas is ~$1, the $15 estimate massively over-reports realised cost on L2 → P&L underreport).

**Triangular (`TriangularArbitrageEngine._execute_triangular_swap` triangular_engine.py:870):**
- Builds three EIP-1559 txs with nonces N, N+1, N+2 (lines 922-964).
- Attempts Flashbots bundle (line 989). On success, returns receipt of the first tx (line 1017). **Note**: returning the first tx receipt does not prove all three legs landed — Flashbots bundles are atomic per spec but the verification at line 1012-1023 only checks tx1. Tx2 and tx3 status is never confirmed before P&L is booked.
- On Flashbots failure: sequential execution (lines 1052-1075). Tx1 lands → tx2 fails → returns None (line 1066). **Tx1 is now stuck** — no unwind, no alert beyond `_send_error_alert` (line 854).

**Solana (`SolanaArbitrageEngine._execute_arbitrage` solana_engine.py:1196):**
- Better than EVM paths: quote-freshness check (line 1216), profit recheck after refresh (line 1246), SOL balance check for fees (line 1276), Jito bundling.
- Same hardcoded MAX_REALISTIC_PROFIT=3% sanity reject (line 1120) — good guardrail against stale-route phantom spreads.
- Still no integration with cross-module risk_manager.

Desync points:
- D1: Triangular Flashbots success returns `receipt.transactionHash` of tx1 only (triangular_engine.py:1017) — if Flashbots includes tx1 but not tx2/tx3 (rare but possible), code reports success.
- D2: Spatial direct-swap fallback at arbitrage_engine.py:1856 → guaranteed inventory leak as analysed in Executive Verdict.
- D3: Solana `_sign_transaction` partial-sign path (solana_engine.py:1614) creates a tx that needs Jupiter co-signing — if the Jupiter call latency exceeds the quote-staleness window, the sign succeeds on a stale price.
- D4: Per-pair daily counter resets only on first scan after midnight UTC (arbitrage_engine.py:1442). If the bot is offline at midnight, the counter doesn't reset until next opportunity scan — minor.
- D5: `_pending_nonce` in `FlashLoanExecutor` (arbitrage_engine.py:502) is per-engine-instance. Multi-engine concurrent wallet writes (spatial + triangular on ETH) WILL collide; the only safety is that triangular uses fresh `get_transaction_count` each time without tracking — so spatial's `_pending_nonce` increments invisibly to triangular. **Nonce collision risk is real.**

## Profit-leak & loss-leak inventory

| ID | Category | File:Line | Estimated impact | Fix sketch |
|---|---|---|---|---|
| AL-01 | One-legged direct swap (inventory leak) | arbitrage_engine.py:1856 | Whole flash-loan-amount worth of token left stranded per occurrence | Remove fallback or bundle as atomic multicall; if Flashbots fails, abort the trade entirely |
| AL-02 | Sequential triangular fallback half-execution | triangular_engine.py:1052-1075 | Half-route inventory loss on any mid-route revert | Disable sequential fallback entirely (already partially gated at line 1047 with `profit_pct < 0.01`, but 1% can still be wiped by slippage on the partial unwind) |
| AL-03 | Hardcoded $15 gas in P&L (overstates on L2, understates Ethereum at high gas) | arbitrage_engine.py:1890 | P&L misreport ±60% on L2 vs Ethereum | Read realised `gasUsed * effectiveGasPrice` from receipt; store both gross and net |
| AL-04 | Hardcoded 0.6% slippage estimate | arbitrage_engine.py:1889 | Cost misreport when actual slippage diverges | Compute from receipt-reported amountsOut vs quoted |
| AL-05 | No hourly gas-burn budget | (missing) | Unbounded burn on reverting pair | Track `gas_spent_last_60min` per chain; freeze when > realised_profit_60min × 0.5 |
| AL-06 | Multi-engine nonce collision on shared EVM wallet | arbitrage_engine.py:513, 1765; triangular_engine.py:901 | Tx replacement underpriced errors, missed opportunities, possible duplicate sends | Shared nonce manager singleton per wallet+chain |
| AL-07 | Flash-loan amount un-capped by risk policy | arbitrage_engine.py:901 | 10 ETH default vs $10 RiskManager cap = 4000x mismatch | Gate `flash_loan_amount` through RiskManager.max_position_size_usd × leverage_multiplier |
| AL-08 | Stale-spread slow-scan only reduces RPC; doesn't halt | arbitrage_engine.py:1264 | Wasted compute, missed dynamic check failure | Add halt-after-N-hours-of-flat-spread |
| AL-09 | Spatial uses public mempool when Flashbots unavailable | arbitrage_engine.py:1856 | Sandwich MEV (estimated 30-80% of theoretical spread lost) | Require Flashbots-or-no-trade on Ethereum mainnet |
| AL-10 | Solana `arb_slippage_bps=150` (1.5%) is wide | solana_engine.py:836 | Eats ~1.5% of every trade in MEV slop | Tighten to 50-75 bps; retry instead of widen |
| AL-11 | Triangular returns success on tx1 receipt only | triangular_engine.py:1017 | Inflates win rate metrics, hides leg-2/3 failures | Verify all 3 receipts before logging success |
| AL-12 | No price-oracle deviation check | (missing) | Sandwich + oracle-manipulation attacks | Add Chainlink cross-check on each spread before execute; reject if `|getAmountsOut_price - oracle_price| / oracle_price > 50 bps` |

## Live-readiness checklist

- [ ] DRY_RUN honored on every send/order/sign path — **PASS** at first-level gates; **AMBER**: triplicated config resolution and inheritance-only on internal sign/send methods.
- [ ] Per-trade max loss — **FAIL**: the AL-01 / AL-02 inventory-leak paths have no bound.
- [ ] Per-hour max loss — **FAIL**: no tracker.
- [ ] Per-day max loss — **FAIL**: cross-module breakers not consulted.
- [ ] Position reconciliation on startup — **N/A** for flash-loan paths (atomic), **FAIL** for direct-swap inventory leftovers.
- [ ] Idempotent order IDs / nonce management — **PARTIAL**: per-engine nonce tracking; **FAIL** for multi-engine on same wallet.
- [ ] Heartbeat → freeze on stale — **FAIL**: stale-spread is detected (arbitrage_engine.py:1244) but switches to slow-scan, doesn't freeze.
- [ ] Emergency stop wired & reachable — **FAIL**: only per-engine `stop()`; no operator script (same gap as DEX_MODULE).
- [ ] Hourly gas-burn budget (analyst.md explicit) — **FAIL**.
- [ ] Oracle-deviation kill-switch (analyst.md explicit) — **FAIL**.
- [ ] Atomic execution on every multi-leg swap — **FAIL**: one-legged fallback (AL-01) and sequential triangular fallback (AL-02).
- [ ] Receipt status verification before booking P&L — **PARTIAL**: spatial flash-loan path checks initiator/contract success; triangular checks only first of three; Solana checks bundle status.
- [ ] Chain-aware gas cost in P&L — **FAIL**: hardcoded $15.

## Proposed action backlog

- [ ] **AR-01** Remove one-legged fallback in spatial direct-swap. If Flashbots unavailable on mainnet → abort. On L2 → bundle as multicall via deployed contract or abort. Touches `modules/arbitrage/arbitrage_engine.py:1739-1861`. Risk reduced: catastrophic inventory leak. Owner: analyst → smartcontract.
- [ ] **AR-02** Remove sequential triangular fallback. If Flashbots fails or bundle not included, abort and alert. Touches `modules/arbitrage/triangular_engine.py:1042-1075`. Owner: analyst → smartcontract.
- [ ] **AR-03** Add `ArbitrageRiskGate` class consulted by all three engines pre-execution; it reads circuit-breaker state from `core/risk_manager.py` and an hourly gas-burn budget from a new tracker. Touches all three arbitrage engines + a new file `modules/arbitrage/risk_gate.py`. Owner: analyst.
- [ ] **AR-04** Cap `flash_loan_amount` at `min(config['flash_loan_amount'], MAX_FLASH_LOAN_USD / eth_price)` where `MAX_FLASH_LOAN_USD` is a new RiskManager param. Touches `arbitrage_engine.py:898-902`. Owner: analyst.
- [ ] **AR-05** Shared async nonce manager keyed by `(chain, wallet)`. Touches `arbitrage_engine.py:513`, `arbitrage_engine.py:1765`, `triangular_engine.py:901`. Owner: analyst → smartcontract.
- [ ] **AR-06** Add Chainlink/Pyth oracle cross-check in `_check_arb_opportunity`. Reject if `|dex_implied_price - oracle_price| > 50 bps`. Touches `arbitrage_engine.py:1326`, `triangular_engine.py:608`, `solana_engine.py:1055`. Owner: analyst → quant.
- [ ] **AR-07** Hourly-gas-burn kill-switch: track `failed_gas_60min_usd` and `realized_profit_60min_usd` per chain; freeze when ratio < 0.5. Touches new tracker + each engine's `_execute_*`. Owner: analyst.
- [ ] **AR-08** Replace hardcoded $15 gas / 0.6% slippage in `_log_arb_trade` with on-chain receipt-derived values. Touches `arbitrage_engine.py:1863-1995`. Owner: analyst → smartcontract.
- [ ] **AR-09** Verify all three triangular tx receipts (`status==1`) before logging success. Touches `triangular_engine.py:1015`. Owner: analyst.
- [ ] **AR-10** Tighten Solana `arb_slippage_bps` default from 150 → 75 bps; rely on quote-refresh+retry on failure (already in place at solana_engine.py:1216). Touches `solana_engine.py:836`. Owner: analyst.
- [ ] **AR-11** Hooks for `scripts/emergency_stop.py` (from DEX_analyst.md RM-01): each arbitrage engine watches a DB flag and exits cleanly on set. Touches all three engines + `scripts/emergency_stop.py`. Owner: analyst+backend.
- [ ] **AR-12** Wire `arbitrage_alerts` to cross-module `monitoring/alerts.py` and DB breach log so arbitrage breaches show on dashboard alongside DEX. Touches `modules/arbitrage/arbitrage_alerts.py`. Owner: analyst+backend.

## Cross-cutting questions called out in the task

**Runaway gas burn if a leg keeps reverting**: nothing prevents it today. The closest guardrail is the per-pair daily cap (5 attempts / pair / day in spatial, arbitrage_engine.py:930). At Ethereum mainnet 30 gwei × 450k gas × 5 attempts × N pairs (24 pairs on ETH from `ARB_PAIRS_ETHEREUM`) = up to ~$160/day worst case in pure gas with no profit. On L2 the absolute number is smaller (~$2-5/day worst case) but the *ratio* of gas-to-profit can still be 100%+. **Fix: AR-07.**

**Per-hour gas budget**: not implemented. AR-07 introduces one.

**Inventory imbalance after a one-legged fill**: AL-01 (spatial) and AL-02 (triangular) are the only paths that produce one. Resolution today: nothing — the unwanted side stays in the wallet. The bot does not even know it's there because `_load_state` is a stub. **Fix: AR-01 and AR-02.** As a defence-in-depth: a wallet-balance reconciler (DEX_analyst RM-02) should detect any unexpected non-WETH balance and alert.

**Cross-chain arb — bridge time + reorg risk**: I see no cross-chain bridging path in the three engines; each is single-chain. If the smartcontract agent later adds bridging, the reorg risk on Arbitrum/Base/Optimism (~1-hour challenge window on canonical bridges; minutes on third-party) must be priced in as a hold-cost. Flagged but not present today.

**Flash-loan arb — fee + slippage + gas + atomicity**: the Aave 0.05% fee is computed correctly (arbitrage_engine.py:1411). Slippage assumptions are hardcoded at 0.6% per round-trip (line 1889) and not calibrated to pool depth. Gas is hardcoded at $15 (line 1890). Atomicity holds *only* via the `FlashLoanArbitrage.executeArbitrage` contract path (arbitrage_engine.py:1729) — the direct-swap fallback breaks atomicity (AL-01). Recommend: only execute flash-loan path; remove direct-swap entirely (or restrict to chains/pools where multicall+revert-on-loss is deployed).

## Recommended go-live sequence

1. Land **AR-01, AR-02, AR-03, AR-07** as a hard pre-req — these eliminate the catastrophic loss paths and add the explicit hourly gas budget the analyst.md demands.
2. Land AR-05 (nonce manager) before any concurrent live engine — running spatial + triangular on the same wallet without it is a guaranteed collision incident.
3. Stage 1 — paper: all three engines in `DRY_RUN=true` for 1 week. Verify arbitrage_alerts log every "would-have-executed" with realistic on-chain re-quotes, and verify per-pair daily caps reset at UTC midnight.
4. Stage 2 — testnet: Sepolia + Arbitrum Sepolia. Real signing. Real flash-loan contract (verify deployment addresses). Confirm Flashbots-only path works end-to-end with simulated bundle inclusion. Verify AR-01 abort behavior on simulated Flashbots failure.
5. Stage 3 — small live cap on L2 only: Base first (cheapest gas, sufficient liquidity). `flash_loan_amount=0.5 ETH` (~$1.5k notional). `_max_executions_per_pair_per_day=2`. 72h soak. Track gas/profit ratio.
6. Stage 4 — ramp on L2: Arbitrum added; flash_loan_amount → 2 ETH if AR-07 budget shows healthy ratio.
7. Stage 5 — Ethereum mainnet only after Flashbots-only path proven for 2 weeks on L2 and AR-06 oracle cross-check is live. Start at `flash_loan_amount=1 ETH`.
8. Solana arbitrage can run in parallel on its own track; it has the cleanest execution model already (quote-freshness, profit-recheck, Jito), and the catastrophic-leak class of bugs (AL-01/02) does not apply there. Required pre-reqs: AR-03, AR-07, AR-10, AR-11.

## Open questions

- Are `EVMArbitrageEngine`, `TriangularArbitrageEngine`, and `SolanaArbitrageEngine` ever run concurrently on the same wallet? The lack of a shared nonce manager (AR-05) only matters if yes. `main_arbitrage.py` will tell — confirm with smartcontract agent.
- What is the actual deployment status of `FlashLoanArbitrage.sol` per chain? `flash_loan_env_key` (arbitrage_engine.py:399, 411, 423) implies separate contracts per chain; the engine warns loudly (lines 1110-1121) if missing, but the warning is only logged. Suggest: refuse to leave DRY_RUN if flash_loan_executor is None on any enabled mainnet.
- `arbitrage_alerts.py:213` filters by string `"DRY_RUN"` — confirm all three engines emit exactly that token (they do: arbitrage_engine.py:1555, triangular_engine.py:829, solana_engine.py:1258). But this is brittle — recommend a boolean field.
- The wallet that signs arbitrage tx — is it the same wallet that signs DEX trades? If yes, AR-05 must coordinate with `DirectDEXExecutor._get_next_nonce` (direct_dex.py:164) and `TradeExecutor.nonce_lock` (base_executor.py:281). If no, the operator must fund both. Document explicitly.
- Triangular fallback (line 1042) only proceeds if `profit_pct >= 0.01` (1%). Real triangular spreads are rarely > 1% on liquid pools — this gate is mostly closed in practice. Is the fallback ever actually entered? If not, AR-02 is a no-op; if yes, it's catastrophic. Need 30-day log review.
