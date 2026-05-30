# ARBITRAGE — Wave-2 Audit + Enhancement

**Branch:** `claude/create-expert-agents-JFSF5`
**Owner:** A2 — smartcontract-web3-expert
**Started:** 2026-05-19

## Re-verification of prior MB / P1 items

| ID | What | Verdict | Evidence |
|---|---|---|---|
| MB-03 | DAI mainnet address typo | CLOSED | `arbitrage_engine.py:169` correct checksum `0x6B17...1d0F` |
| MB-04 | One-legged broadcast fallback | CLOSED | `arbitrage_engine.py:1873-1880`; `triangular_engine.py:1070+` |
| MB-05 | Triangular placeholder amountIn=1 | GATED (correct) | `triangular_engine.py:906-911` returns None before any swap build |
| P1-04 | pool_engine RPC sweep | CLOSED | engines pull via `RPCProvider.get_rpc_sync(<chain>_RPC)`; `main_arbitrage.py` initializes PoolEngine + sets it on RPCProvider before any engine init |
| P1-06 | Cross-module risk gate | CLOSED on EVM (`:1599-1608`), Solana (`solana_engine.py:1274+`), and Triangular (`triangular_engine.py:841+`) |

## Residual issues found this wave

### A2-01 — NameError in spatial-opportunity log line (CRASH BUG)
`arbitrage_engine.py:1479` references `forward_output` and `final_output` that were
renamed to `tokens_bought` / `weth_returned` in a prior refactor. Any time
`_check_arb_opportunity` finds a profitable opp, the log line raises `NameError`,
the `except` block swallows it, and the trade is silently dropped. Net effect:
the engine cannot execute *any* spatial-arb when DRY_RUN=false. Priority P0.

### A2-02 — P1-05 hardcoded $15 gas + 30% slippage in PnL
`arbitrage_engine.py:1911-1922` uses `GAS_COST_USD = 15.0` and
`slippage_cost = entry_usd * gross_profit_pct * 0.3` regardless of chain.
Same value applies to Ethereum (where 450K gas at 30 gwei ≈ $30+ usually),
Arbitrum (~$0.20), Base (~$0.15). This poisons the persisted PnL: Base/ARB
trades look unprofitable when they aren't; Ethereum trades look profitable
when they aren't. Priority P0 for live, P1 for DRY-only.

### A2-03 — Pre-execute net-spread filter uses constant 0.005
`arbitrage_engine.py:1437` hardcodes `estimated_costs = 0.005` (0.5%) across
every chain. Should be a per-chain function of (a) flash-loan fee already
deducted, (b) realized average slippage observed on this pair, (c) live gas
quote / borrow-amount-USD. Priority P1.

### A2-04 — `FLASH_LOAN_RECEIVER_CONTRACT_*` read via `os.getenv` only
`arbitrage_engine.py:1096-1098, 1699-1701` use `os.getenv` directly. Per
project rules, sensitive on-chain contract addresses should fall back to
`security/secrets_manager.SecureSecretsManager.get_async()` so the dashboard
"Credentials" page can manage them. `_get_decrypted_key()` already exists at
line 966 — just needs to be called for these keys. Priority P2.

### A2-05 — `SLIPPAGE_ESTIMATE_PCT = 0.006` hardcoded in `_log_arb_trade`
Same root cause as A2-02 but a second magic constant
(`arbitrage_engine.py:1912`). Should derive from realized swap output.
Priority P1.

### A2-06 — No `min_profit_bps` knob; `min_profit_threshold` is class-constant
The settings UI advertises a "Base Profit Threshold (%)" (`settings_arbitrage.html:193`)
that the engine ignores — `EVMArbitrageEngine.min_profit_threshold = 0.003`
is a hard constant set in `__init__`, never reading `config.min_profit_spread`.
Adjusting the UI changes nothing. Priority P1.

### A2-07 — No gas-budget tracker
Currently arbitrage trades can drain wallet gas without an hourly cap.
Operator brief explicitly asks for an hourly gas-budget tracker. Priority P2.

## Enhancements (this wave)

1. Per-chain L2 gas/slippage table — `CHAIN_COST_PROFILE` with
   `(gas_usd_estimate, default_slippage_pct, l2)` per chain, consumed by
   the pre-execution filter and `_log_arb_trade`.
2. Live gas estimation — replace `GAS_COST_USD` with a per-tx wei-gas cost
   converted to USD via current `eth_price`.
3. Hourly gas-budget tracker — `_gas_spent_usd_hour` rolling counter,
   blocks new flash-loan execution when the hourly budget is exhausted.
4. Adaptive `min_profit_bps` curve — observed-gas-spike rate × base; raises
   the threshold when gas is volatile.
5. DB-backed receiver address resolution — `_get_decrypted_key()` for
   `FLASH_LOAN_RECEIVER_CONTRACT_*`.

## Fix log

| Commit | What | Files |
|---|---|---|
| `9e6a7d1` | A2-01: NameError in spatial-arb opportunity log (live trades were silently dropped) | `arbitrage_engine.py`, report seed |
| `744ee48` | A2-06 + per-chain cost-profile foundation: UI knob honored; CHAIN_CONFIGS extended with `flash_loan_gas_limit`, `fallback_gas_gwei`, `default_slippage_pct`, `flash_loan_fee_pct` | `arbitrage_engine.py` |
| `4adcd29` | Cost helpers: `_gas_cost_usd_per_tx` / `_gas_spike_multiplier` / `_adaptive_min_profit_threshold` / `_gas_budget_check_and_charge` | `arbitrage_engine.py` |
| `8cf0143` | A2-02 / A2-03 / A2-05: chain-aware gate + PnL costs replace magic numbers (0.005, $15, 0.006) | `arbitrage_engine.py` |
| `89175d4` | A2-04 + A2-07: receiver address via secrets_manager + hourly gas-budget gate | `arbitrage_engine.py` |
| _next_ | Dashboard knob + CLAUDE.md + report finalize | `settings_arbitrage.html`, `CLAUDE.md`, this file |

## Verification matrix (post-wave)

| Risk surface | Pre-wave | Post-wave |
|---|---|---|
| Live spatial trade broadcast | 100% silent drop (A2-01 NameError) | Path executes; opportunity log shows direction, threshold, gas-mult |
| Pre-execution net-spread filter | Constant 0.5% gas+slip across all chains | `gas_usd / borrow_eth + chain_default_slippage`; threshold adapts to gas spikes |
| Persisted PnL row | $15 false gas + 30%-of-spread fake slippage | Live `_gas_cost_usd_per_tx` + chain `default_slippage_pct` × notional |
| FLASH_LOAN_RECEIVER address | `os.getenv` only | `_get_decrypted_key` -> secrets_manager (DB + Fernet) -> env fallback |
| Cross-module risk gate | Pre-execute call wired (P1-06) | Unchanged — verified still active on EVM / Solana / Triangular paths |
| Runaway gas spend | Bounded only by wallet ETH balance | Hourly USD budget (`gas_budget_usd_per_hour`, default $50) |
| Dashboard `min_profit_spread` knob | Silently ignored — engine used 0.3% constant | Honored; baseline for adaptive curve |
| Triangular path | Gated by MB-05 atomic-receiver guard | Unchanged (per operator brief) |

## Items deferred (out of wave-2 scope)

- Triangular atomic-receiver contract deploy (operator approval required).
- Per-DEX realized-slippage learning to replace the static `default_slippage_pct` (needs ~7 days of `arbitrage_trades` rows to be statistically sound — start collecting now that the PnL row is honest).
- Direct integration with `monitoring/dashboard` to surface `_gas_spend_usd_hour` live tile.

## Final state
All P0/P1 wave-2 items closed. Triangular and Solana engines unchanged this wave. Spatial engine production-ready pending operator LIVE flip + a final smoke on each chain.

