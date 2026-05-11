# ARBITRAGE_MODULE — Quant / Algorithm Audit

## Executive summary

The arbitrage module has three engines: spatial EVM (`modules/arbitrage/arbitrage_engine.py`, 2077 lines), triangular EVM (`modules/arbitrage/triangular_engine.py`, 1199 lines), and Solana cross-DEX via Jupiter (`modules/arbitrage/solana_engine.py`, 1954 lines), plus a Telegram alerts thin layer (`arbitrage_alerts.py`). The spatial engine has the cleanest economics — it uses an on-chain `checkArbitrage` view function and a Solidity flash-loan receiver (`arbitrage_engine.py:529-569`), gates execution on `expected_profit ≥ gas_cost * 1.3` (`arbitrage_engine.py:633`), and applies a 30%-of-spread slippage haircut elsewhere (`arbitrage_engine.py:1898`). The triangular engine is dramatically weaker: it uses three sequential `getAmountsOut` queries with no execution simulation, hard-codes a `MAX_REALISTIC_SPREAD = 10%` phantom-spread filter (`triangular_engine.py:736`) that papers over deep modeling problems instead of fixing them, and falls back to non-atomic sequential RPC execution when Flashbots is unavailable (`triangular_engine.py:1042-1075`) — a recipe for stuck inventory. The Solana engine is structurally more sound (Jupiter aggregates fee+slippage natively, Jito provides atomicity), but the round-trip profit test compares `reverse_out` to `amount` *without subtracting the Jito tip, priority fee, ATA rents, or refresh slippage* (`solana_engine.py:1101-1104`), and the 1.5% per-leg slippage cap allowed (`solana_engine.py:153`) means a "0.3% threshold" opportunity can be a -2.7% realised trade.

Top three algorithmic gaps: (1) **spread math conflates quote-based round-trip with actual fill round-trip** — `getAmountsOut` is a *deterministic* read of constant-product state, but a real swap moves the curve, and you swap *after* the read so the second leg's curve moved against you (`arbitrage_engine.py:1357-1406`); (2) **no inventory model** — flash-loan path is stateless, but the non-flash fallback (`_execute_triangular_swap`) sequentially fills three legs at market and the engine simply hopes the third leg returns enough; there is no rebalance or hedge if it doesn't; (3) **MEV awareness is binary (Flashbots on/off)** rather than *sized*: a $25k arb trade on a $200k pool moves the price 12.5%, creating sandwich opportunity larger than the spread itself, but trade size is `min_profit_threshold` not size-of-profit-curve-maximum (`arbitrage_engine.py:894`).

Top three validation concerns: (1) `min_profit_threshold = 0.003` and `estimated_costs = 0.005` (`arbitrage_engine.py:894, 1430`) are tuned by hand-comments, never measured against realised executions; (2) the `checkArbitrage` contract simulation runs against *current* on-chain state, while the actual tx executes 1-2 blocks later — there is no slippage estimate from simulated-vs-realised drift; (3) executed trades are logged to `arbitrage_trades` with `entry_price = self._eth_price` (CoinGecko USD reference) (`triangular_engine.py:1126`), not with actual on-chain `amountIn / amountOut`, so live P&L attribution is wrong.

## Existing-edge audit

### EVMArbitrageEngine (spatial, `modules/arbitrage/arbitrage_engine.py`)
- **Alleged edge**: Cross-DEX price discrepancies on V2-style pools (Uniswap/Sushiswap on ETH; Sushi/Camelot/Zyberswap on ARB; Sushi/BaseSwap/SwapBased on Base), captured via Aave flash loan so capital lockup is zero.
- **Costed?** Partially. Aave 0.05% fee is subtracted (`arbitrage_engine.py:1411`). Gas cost is checked against `expected_profit * 1.3` (`arbitrage_engine.py:633`). But slippage uses a flat `estimated_costs = 0.005` (50 bps) (`arbitrage_engine.py:1430`) regardless of pool depth and trade size — the same constant for a $200k Base pool and a $20M Uni V2 pool. MEV/sandwich premium is not modelled at all on Arbitrum/Base where Flashbots is unavailable (`arbitrage_engine.py:1150-1152`).
- **Walk-forward / OOS evidence?** None. There is no backtest harness. The "evidence" is live logs showing 5-minute stats (`arbitrage_engine.py:1219-1230`). With `_max_executions_per_pair_per_day = 5` (line 930), at best 5×N_pairs trades/day land — insufficient for statistical significance.
- **Failure modes**: (a) Phantom liquidity — `getAmountsOut` returns optimistic quotes from thin pools; the code blacklists pairs after 5 failures (`arbitrage_engine.py:1290-1325`), which addresses *route* failures but not *price-impact* failures. (b) The constant slippage haircut over-rejects on deep pools (lost edge) and under-rejects on shallow pools (lost capital). (c) On unknown chain_id, defaults to Ethereum config (`arbitrage_engine.py:1062-1065`) — wrong tokens, wrong AAVE pool.

### TriangularArbitrageEngine (`modules/arbitrage/triangular_engine.py`)
- **Alleged edge**: Cycles A→B→C→A produce profit when product of three forward rates > 1, even when each individual pair is fair-priced (rare, but real).
- **Costed?** Marginally. `_calculate_dynamic_threshold` (line 564) sums `base_threshold + (gas_cost_usd / trade_value) * 1.1`, capped at 5%. The cap itself reveals the design flaw: when trade size is small (`trade_amount_eth = 1.0`, line 315), gas cost dominates the threshold so the engine effectively never fires. When flash loans are configured, the threshold falls — but the engine's *fallback* path (sequential RPC, line 1043-1075) never uses flash loans, so it bears the full gas + capital risk.
- **Walk-forward / OOS evidence?** None. The hard-coded `MAX_REALISTIC_SPREAD = 0.10` (line 736) is the only validation: anything above 10% is rejected as phantom. This is a band-aid, not a model.
- **Failure modes**: (a) **Atomicity loss**: sequential `wait_for_transaction_receipt` between three legs (lines 1056-1073) means a 12-second exposure per leg; price can move by 50+ bps in that window on volatile tokens. (b) **Triangular cycle inversion**: the cycle A→B→C→A is one direction only — but reversed cycle (A→C→B→A) is mathematically equivalent and not scanned, so half of available edge is invisible. (c) **No log-sum stability**: the implementation multiplies three rates directly, not log-sum. For three ratios near 1.0001, you accumulate `1.0001^3 ≈ 1.0003` — fine, but for tiny tokens with prices like `1e-9` the integer division in `getAmountsOut` quantises away the edge.

### SolanaArbitrageEngine (`modules/arbitrage/solana_engine.py`)
- **Alleged edge**: Jupiter aggregator finds best price *across all Solana DEXs*; submit forward quote, then immediately query reverse quote to detect round-trip surplus.
- **Costed?** Underspecified. Jupiter quotes include per-leg fees + slippage (positive), but the reverse-quote-equals-arb logic ignores: priority fees (`prioritizationFeeLamports: 'auto'`, line 229), Jito tip (`tip_lamports = 10000` default, line 1324, which warns but doesn't gate), ATA rent if new token accounts are created, and the *slippage budget* of 150 bps each way (`self.arb_slippage_bps = 150`, line 836). A 0.3% reported profit can become -2.7% if both legs slip to the full 1.5% allowance.
- **Walk-forward / OOS evidence?** None in repo. The author has hard-coded `MAX_REALISTIC_PROFIT = 0.03` (line 1120) and "Note: 10,000 lamports is TOO LOW for competitive arbitrage" warnings (line 1322) — clearly aware that defaults are wrong but defaults remain shipped.
- **Failure modes**: (a) Quote freshness check (`_is_quote_stale`, line 1177) is good — but it refreshes both quotes *separately* (lines 1222 and 1234), so by the time leg-2 is refreshed leg-1 is again ~1 second stale. (b) The cooldown is 300 seconds (line 843) which is fine for spam control but means after a successful arb you can't capture continuation. (c) "Trade amount" defaults to 1 SOL (~$200) (line 831) — at this size, Jito's economic priority is negligible and bundles often get dropped (acknowledged at line 1352-1356).

## Bias & leakage findings

| ID | File:Line | Bias type | Description | Fix |
|----|-----------|-----------|-------------|-----|
| A-01 | `arbitrage_engine.py:1430` | slippage-underestimate | `estimated_costs = 0.005` is a constant; ignores pool depth, trade size, and per-pair fee tier (Uni V3 has 0.01/0.05/0.30/1.0% tiers). | Compute per-pool: `slippage = trade_in / (2 * reserve_in)` for V2 (constant-product approx). Add fee_tier from pool metadata. |
| A-02 | `arbitrage_engine.py:1357-1406` | snapshot mismatch | Forward leg quote uses `getAmountsOut` against current reserves; sell leg quote runs *after* but on still-current reserves (the same block). Actual swap executes 1+ block later when reserves have moved (other traders) and after your *own* forward swap moves them further. | Use the pool's constant-product formula in Python to simulate post-swap state, then quote leg 2 against the simulated post-state reserves, not the pre-state. |
| A-03 | `arbitrage_engine.py:894` vs `:1430` | in-sample tuning | `min_profit_threshold = 0.003` and `estimated_costs = 0.005` are tuned from comments ("typically 0.1-0.5%", "After costs (~0.55%)"). No realised distribution justifies these. | Replay last 30 days of logged opportunities + outcomes; fit threshold to maximise net-of-cost EV per chain/pair. |
| A-04 | `triangular_engine.py:730-736` | snapshot mismatch + bias-band | `profit_pct = (best_hop3_out - amount_in)/amount_in` is product of *three independent* `getAmountsOut` calls against current state. Each call assumes leg-1 already filled when computing leg-2, but on-chain reserves haven't moved. Phantom spreads up to 10% pass the filter. | Same as A-02: simulate post-swap reserves between hops. |
| A-05 | `triangular_engine.py:736` | survivorship via filter | `MAX_REALISTIC_SPREAD = 10%` filter looks principled but is actually masking systematic over-estimation. Real arbs on majors are 1-5 bps. | Replace cap with `expected_realised_profit = simulated_profit - per_leg_slippage_at_size`. |
| A-06 | `triangular_engine.py:1042-1075` | inventory-without-model | Sequential RPC fallback: tx1 succeeds → wait 60s → tx2 fails → engine holds token_b without recovery. No rebalance step, no hedge. | Either drop the fallback (preferred) or implement: on tx2 failure, immediately swap back to token_a at any price; record realised loss; alert. |
| A-07 | `solana_engine.py:1101-1104` | future-feature leak via aggregator | Jupiter's `outAmount` *already includes* its slippage budget, but the round-trip subtracts gross amounts. If both legs use the *full* 150-bps slippage budget the realised P&L can be 3% worse than reported. | Use `quote.otherAmountThreshold` (worst-case fill) for profit math, not `outAmount` (best-case). |
| A-08 | `solana_engine.py:1216-1252` | refresh-cascade leak | Stale-quote handler refreshes leg-1 then leg-2 with two sequential API calls (each ~500ms); by the time leg-2 quote is fresh, leg-1 quote is 0.5-1s stale again. | Refresh as a single combined `getQuote` call with `swapMode: 'ExactIn'` for round-trip in one shot, or accept the existing stale-quote risk and abort. |
| A-09 | `solana_engine.py:1322-1326` | cost-underestimate | `jito_tip_lamports = 10000` default = $0.002. Competitive arbitrage on Solana requires 50,000-500,000 lamports. Tip is *not* subtracted from `profit_pct`. | Subtract `tip_lamports/1e9 * SOL_price` from `profit_usd` before threshold gate. |
| A-10 | `arbitrage_engine.py:633` | gas-buffer asymmetry | Requires `expected_profit > gas_cost * 1.3`. Good for filtering. But gas estimate uses `max_fee * gas_limit` (the *cap*, not the *expected* fee). Wastes opportunities where `base_fee < cap`. | Use `(base_fee + priority_fee) * gas_limit` for expected cost; keep `max_fee` only for cap. |
| A-11 | `triangular_engine.py:954` | leg-2 input bug | `tx2 = router2.functions.swapExactTokensForTokens(1, ...)`: amount-in is hardcoded `1` because "Will be filled by actual output from tx1" — but nothing fills it! In RPC fallback path this would swap 1 wei. | Read tx1 receipt logs to extract actual `amounts[1]` then build tx2 with that as `amountIn`. (Flashbots path is moot since contract isn't called.) |
| A-12 | `arbitrage_engine.py:1421` | tracking inconsistency | `raw_spread = profit / borrow_amount` then `net_spread = raw_spread - 0.005`. `net_spread` is compared to `min_profit_threshold = 0.003`. So a true edge has to be > 0.8% raw, which on V2 pools is rare — the engine is mostly idle by design. | Tune threshold from realised log; likely 0.4-0.6% raw is achievable. |
| A-13 | `solana_engine.py:1067-1068` | wrong scale | When `in_symbol in ['USDC','USDT']`, amount = `int(self.trade_amount_sol * 100 * 1e6)`. That's `1.0 * 100 * 1e6 = 100 USDC`. Way too small for arb significance. | Make stablecoin trade size a config param, not 100× hack. |
| A-14 | `arbitrage_engine.py:1471-1472` | undefined variable | Log line uses `forward_output` and `final_output` which are never defined (only `tokens_bought`, `weth_returned`, `profit`, `amount_in` exist). This is a runtime error path. | Rename to actual variables. |
| A-15 | `triangular_engine.py:325-326` | survivorship of pairs | `TRIANGULAR_CYCLES` is a hand-curated list of 21 cycles; cycles that were ever unprofitable were silently removed (e.g. comments around DEX_MODULE). No mechanism to add/remove cycles based on rolling data. | Auto-discover cycles weekly: scan all triangle permutations of tokens with > $1M TVL on > 2 DEXs. |

## Missing signals / features (ranked by expected lift)

1. **Post-swap pool-state simulation** — implement the constant-product (x*y=k) update *between* hops in Python. Replaces flat slippage haircut with realistic per-trade impact. Expected lift: turns 60-80% of currently-rejected opportunities into +EV trades, plus filters phantom 10% triangular spreads.
2. **Pool reserve / depth feed** — both engines call `getAmountsOut` but never call `getReserves()`. Reserves give you the size-impact curve. Add to `_check_arb_opportunity` once per pair scan.
3. **Block-time / mempool-position estimator** — current latency model is "1 block". On Arbitrum it's 250ms; on Base it's 2s; on Ethereum 12s with mempool delay. Each chain needs its own price-drift-during-inclusion estimate.
4. **Sandwich-attacker tip estimator (per pool)** — Flashbots is binary; the *expected sandwich profit* on a given trade-size/pool-depth combo predicts whether to size up or down. Compute `sandwich_profit_est = trade_size * (1 - (depth/(depth+trade_size))^2)`, compare to attacker's gas; if > 0, reduce trade size until <=0.
5. **Per-hop fee tier auto-detect** — Camelot, SushiSwap, BaseSwap have different fee tiers across pools. The code assumes one fee per router (`flash_loan_fee = 5/10000`, line 1411). Read fee from pool metadata.
6. **Cross-DEX dispersion z-score** — instead of `max(price) - min(price) > threshold`, use z-score of dispersion vs trailing 24h. Spreads only matter when they're unusual; a 0.3% spread that exists every block on a token pair is the *equilibrium*, not an arb.
7. **Reverse-direction cycle scanning** — for every (A,B,C) cycle, also scan (A,C,B). Doubles opportunity surface free.
8. **Funding-window awareness for stable triangles** — `(USDC, USDT, DAI)` arb is driven by curve/balancer rebalances on roughly 4-hour cycles; gate scanning to those windows. Saves RPC.
9. **Tip-vs-spread elasticity (Jito)** — log every Jito bundle's land/fail status; fit `success_rate ~ logistic(tip_lamports)`. Use to set tip dynamically per-opportunity instead of constant 10k.
10. **Bridge-time / capital-lockup model for cross-chain** — currently no cross-chain arb at all, but the codebase mentions it in plans. Bridge time (5-15 min) means capital is exposed to dual-chain price drift; the realised arb needs `expected_spread > volatility * sqrt(bridge_time)`. Required before any cross-chain feature ships.
11. **Reorg-risk discount (L2)** — on Arbitrum/Base, sequencer reorgs are rare but happen. Apply small (~5 bps) reorg-risk haircut to net spread, especially on Base where sequencer is centralized.

## Capital-allocation & sizing review

The arbitrage engines have *no* sizing model. `flash_loan_amount = 10 ETH` (`arbitrage_engine.py:901`) is a single config constant for every pool across every chain. This is fundamentally wrong: a 10-ETH swap on a 50-ETH-deep pool moves price 16%; on a 5000-ETH pool it moves price 0.2%. The current "fix" is `_scan_amounts` pre-scaling (`arbitrage_engine.py:907-910`) which scales *across token decimals* (so 25k USDC ↔ 10 ETH ↔ 0.4 WBTC) but does *not* scale to pool depth.

Required: a `compute_optimal_size(pool_buy, pool_sell, fee_buy, fee_sell)` routine that finds the trade size maximising profit under x*y=k:
```
optimal_size = sqrt(R1_buy * R2_buy * R1_sell * R2_sell * (1-f_buy)(1-f_sell)) - R1_buy
```
…where `R1, R2` are reserves. With AAVE 0.05% fee this typically maximises around 0.5-2% of pool depth, not a fixed 10 ETH.

Kelly does not directly apply (arb is supposedly riskless given flash-loan atomicity), but variance from MEV/sandwich and partial-fill outcomes do exist. Once sized to the per-pool optimum, capital allocation across pools/chains can use risk-parity: `weight_i ∝ 1/realised_variance_i` from logged trade outcomes.

**Cross-module correlation**: spatial arb on ETH and triangular arb on ETH both consume the same ETH wallet's gas budget and compete for inclusion in the same blocks. The Flashbots executors are unaware of each other's pending bundles. At minimum, share a bundle-priority queue.

Inventory model: only the spatial engine via flash loan is truly inventory-free. The triangular non-Flashbots fallback path holds inventory between legs (60+ seconds), and Solana arb holds inventory between Jupiter swap submissions. Need:
1. A per-token "max held inventory" cap.
2. Time-since-leg-1 watchdog that flattens to base token if leg-2 doesn't confirm in N seconds.
3. End-of-day reconciler that auto-rebalances stray tokens accumulated from failed legs.

## ML model health

This module has no ML models directly. It is signal-driven, not learned. However:
1. **`feature_extractor.py` (referenced in audit scope)** is not consumed by arb engines — they have their own price-fetcher and quote logic. There is room to add a "pool-stress" feature from the broader feature store (e.g., spike in failed swaps in a pool → reduce trade size) but it isn't wired.
2. **Tip-success classification** (Jito): a small XGBoost on logged `(tip, gas_at_submit, time_of_day, mempool_competition_proxy, pool_pair) → bundle_landed?` would replace the current static `10k lamports`. Estimated lift: 15-30% increase in successful Solana arb fills.
3. **Phantom-spread classifier**: train a binary classifier on logged `(raw_spread, pool_depth, recent_failed_quotes, time_of_day) → was_realised_profitable?` to replace the current 10% hard cutoff with a probabilistic gate. Would let smaller, real opportunities through and block currently-passing bogus ones.

No model versioning, drift detection, or retrain scripts are needed yet (no models exist).

## Profitability levers (ranked by ROI)

1. **Replace flat 0.5% slippage haircut with per-pool x*y=k simulation** — biggest single edge unlock. Today the engine over-rejects deep-pool opportunities and under-rejects shallow ones. Expected: +20-40% executable opportunities, -30-50% slippage-loss incidents.
2. **Compute optimal trade size per opportunity** (above formula) instead of fixed 10 ETH. Expected lift: 2-4× profit per opportunity on deep pools, 0 partial-fill failures on shallow.
3. **Subtract real Jito tip + priority fee from Solana profit math** (A-07, A-09). Today's "winners" are systematically negative-EV at default tip levels. Either fix the math or raise default tip to 100k lamports.
4. **Reverse-cycle scanning** for triangular — free 2× opportunity surface.
5. **Kill the non-atomic RPC fallback** in `triangular_engine.py:1042-1075`. The expected value of a non-atomic 3-leg execution at sub-1% target spread is negative once you weight by leg-2 failure probability and recovery loss.
6. **Auto-discover triangular cycles** from on-chain pool list weekly; remove cycles with < 1 hit / 30 days; add cycles ranked by historical opportunity rate. Replaces static `TRIANGULAR_CYCLES`.
7. **Per-pool fee detection** instead of hardcoded `5/10000`. Multi-fee-tier pools (Uni V3 0.01/0.05/0.30/1.0% on the same pair) can present 5-25 bps of additional edge.
8. **Tip-success classifier** for Jito (above) — recovers latent arb that's being lost to under-tipping.
9. **Cross-chain capability** (USDC arb between ARB ↔ Base via Stargate/Across) — these spreads run 5-30 bps consistently. Requires bridge model first.
10. **Inventory watchdog + auto-flatten** to recover from non-atomic failures cleanly. Loss-mitigation, not direct profit, but converts a tail-risk loss source into a logged cost.

## Proposed action backlog

- [ ] **QT-A1** Implement `simulate_v2_post_swap(reserve_in, reserve_out, amount_in, fee_bps) -> (new_reserve_in, new_reserve_out, amount_out)` and use it between hops in both `_check_arb_opportunity` and `_check_triangular_opportunity` — touches `arbitrage_engine.py:1357-1406`, `triangular_engine.py:608-800` — expected lift: phantom-spread rejection +90% accuracy — owner: quant.
- [ ] **QT-A2** Add `compute_optimal_arb_size(R1, R2, fee)` and replace fixed `flash_loan_amount` with per-pair size — touches `arbitrage_engine.py:901-911`, `_check_arb_opportunity` — expected lift: 2-4× profit/opportunity — owner: quant.
- [ ] **QT-A3** Subtract Jito tip + priority fee from `profit_pct` before threshold gate in Solana engine — touches `solana_engine.py:1101-1113` — expected lift: removes false positives, hit-rate +30% — owner: quant.
- [ ] **QT-A4** Replace `MAX_REALISTIC_SPREAD = 0.10` hard cap with simulation-based realised-profit estimate — touches `triangular_engine.py:736` — expected lift: unlocks small real arbs currently below noise floor — owner: quant.
- [ ] **QT-A5** Add reverse-cycle scanning: for every `(A,B,C)` in `TRIANGULAR_CYCLES`, also queue `(A,C,B)` — touches `triangular_engine.py:97-128` and run-loop — expected lift: 2× opportunity surface — owner: quant.
- [ ] **QT-A6** Remove non-atomic sequential-RPC fallback in triangular engine; require Flashbots — touches `triangular_engine.py:1042-1075` — expected lift: eliminates tail-loss source — owner: quant + smartcontract.
- [ ] **QT-A7** Use `quote.otherAmountThreshold` (worst-case) instead of `outAmount` (best-case) for Solana profit math — touches `solana_engine.py:1084-1104` — owner: quant.
- [ ] **QT-A8** Dynamic Jito tip from a small XGBoost trained on logged bundle outcomes — touches `solana_engine.py:1322-1336` + new `scripts/train_jito_tip.py` — expected lift: 15-30% land-rate — owner: quant.
- [ ] **QT-A9** Read per-pool fee tier from pool metadata (V3) or fee-on-transfer flag (V2) instead of hardcoded 0.30% — touches `arbitrage_engine.py:1411`, `triangular_engine.py` — owner: quant.
- [ ] **QT-A10** Fix undefined `forward_output`/`final_output` in log line — touches `arbitrage_engine.py:1471-1472` — owner: quant (trivial).
- [ ] **QT-A11** Compute realised P&L from on-chain receipt amounts (`amounts[0]`, `amounts[-1]`) instead of `_eth_price * spread` for DB logging — touches `arbitrage_engine.py:_log_arb_trade`, `triangular_engine.py:_log_trade` — expected lift: enables real EV measurement — owner: quant.
- [ ] **QT-A12** Auto-discover triangular cycles weekly from on-chain pool list — touches `triangular_engine.py:97-128` + new `scripts/refresh_cycles.py` — owner: quant.
- [ ] **QT-A13** Build per-chain "expected price drift during inclusion" model (calibrated from sim-vs-real divergence in logs) and apply as net-spread haircut — touches all three engines — owner: quant.
- [ ] **QT-A14** Implement inventory watchdog: when a leg fails, immediately swap any held intermediate token back to base; log realised loss — touches `triangular_engine.py:1056-1073`, `solana_engine.py:_execute_arbitrage` — owner: quant + smartcontract.
- [ ] **QT-A15** Replace per-engine `min_profit_threshold` constants with values learned from `arbitrage_trades` table: solve `argmax_t E[net_profit | spread > t]` weekly — touches three engines + new `scripts/refit_thresholds.py` — owner: quant.

## Open questions

1. Are the contract's `checkArbitrage` and `executeArbitrage` (`arbitrage_engine.py:529-686`) using the *same* swap path semantics as the engine's `getAmountsOut`? Decimal mismatches at the boundary between Python ints and Solidity uint256 are silent killers — needs an end-to-end fuzz test.
2. Is the wallet that holds non-flash-loan inventory the same wallet that pays gas? If yes, a stuck-leg failure ties up gas funding too.
3. What is the Jito tip success curve in the current market? The "10k is too low" warning suggests it's been seen, but no histogram is logged.
4. Is `arbitrage_trades.entry_price`/`exit_price` ever consumed by any P&L report or dashboard? If yes, it currently displays Coingecko-USD-derived synthetic values, not realised on-chain prices — misleading.
5. For Solana, why is the cooldown 300 seconds (vs 45 on EVM)? Jupiter quotes refresh every 10s; a real edge can reappear within seconds, not minutes.
