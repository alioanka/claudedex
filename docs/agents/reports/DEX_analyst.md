# DEX_MODULE — Trading-Desk / Risk / Live-Readiness Audit

Owner: market-trading-analyst (secondary; smartcontract-web3-expert is primary).
Scope: `modules/dex_trading/`, `trading/executors/{base_executor,direct_dex}.py`, the trading paths through `core/engine.py`, and the cross-module risk plumbing in `core/{risk_manager,portfolio_manager,decision_maker}.py`.

## Executive verdict: RED

DEX_MODULE must not be flipped to `DRY_RUN=false` in its current state. The module shells out almost everything to `core/engine.py`, which has correct DRY_RUN gates on the *primary* swap path (engine.py:1023, 1213, 1262), but the supporting plumbing has live gaps that defeat the gating:

1. **No emergency-stop script exists.** `scripts/emergency_stop.py` and `scripts/close_all_positions.py` are not in the repo (`scripts/` listing shows neither). The only emergency hook is the in-process method `engine.emergency_close_all_positions()` at `core/engine.py:2893`, which is only reachable from inside a running engine via `modules/dex_trading/main_dex.py:913`. There is no operator-callable kill switch. The market-trading-analyst.md checklist explicitly requires one.
2. **No position reconciliation on startup.** `core/engine.py:2715-2721` `_load_state()` is a hollow `pass` stub. If the process restarts after a real fill, the engine starts with empty `active_positions` and `position_tracker.positions`, while real on-chain tokens sit in the wallet. There is no orphan-position recovery.
3. **Heartbeat path is non-actionable.** `_health_check` (`core/engine.py:2431`) and `_watchdog` (`core/engine.py:2931`) only *log* stale tasks; nothing flattens or freezes on stale heartbeat. `_check_database_health`, `_check_web3_health`, `_check_api_health` at `core/engine.py:2869-2879` always return `{'healthy': True, 'message': 'OK'}` — they are placeholders, not health checks.
4. **DRY_RUN is plumbed via two parallel keys** — `config['dry_run']` (lowercase, used by engine and order_manager) and `config['DRY_RUN']` (uppercase, used by `DirectDEXExecutor:62` and `TradeExecutor:192`). One config mistake or one CLI override path (`main_dex.py:742`, `modules/dex_trading/main_dex.py:1027` both set the env var but don't necessarily land in the right config slot) can leave a sub-executor in LIVE mode while the engine thinks it is dry.

Until items 1-4 are fixed, going live risks orphaned tokens, no kill-switch, and a silent live executor under a dry orchestrator. The trading logic itself (sizing, breakers, alerts) is solid enough to be AMBER — the missing operational scaffolding is what makes it RED.

## DRY_RUN propagation audit

Every code path that signs, sends, or transfers, and whether DRY_RUN gates it before the network call.

| Path | File:Line | DRY_RUN-gated? | Notes |
|---|---|---|---|
| `engine._execute_trade` simulated branch | core/engine.py:1023 | YES | Returns early via `if self.config.get('dry_run', True)` |
| `engine._execute_trade` REAL branch | core/engine.py:1187, 1213, 1259, 1262 | YES | Checked three times before `executor.execute_trade(order)` and the receipt wait |
| `TradeExecutor.execute` | trading/executors/base_executor.py:371 | YES | `_simulate_execution` short-circuit |
| `DirectDEXExecutor.execute_trade` | trading/executors/direct_dex.py:275 | YES | Returns `_simulate_dex_trade` |
| `DirectDEXExecutor.validate_order` balance check | trading/executors/direct_dex.py:1006 | YES | Balance check only outside dry_run |
| `OrderManager.submit_order` | trading/orders/order_manager.py:520 | YES | Synthesises a `FILLED` order; **but this fakes success — the dry-run path stamps `OrderStatus.FILLED` even though no on-chain swap happens; downstream P&L is fictional** |
| `MEVProtection` execution | trading/executors/mev_protection.py:662 | YES | Returns dry stub |
| `JupiterExecutor` Solana path | core/engine.py:262-269 | PARTIAL | `dry_run` is *passed* through the executor config but the Solana wallet balance check at engine.py:988-1004 is *skipped* under dry_run — fine. Need to confirm executor itself honors the flag — out of scope here but flagged for smartcontract agent. |
| `engine.emergency_close_all_positions` → `_close_position` | core/engine.py:2893 | UNKNOWN | The close path invoked from emergency uses the same executor; not explicitly re-gated. If close is invoked while config is LIVE, real sells will fire. That is the desired behavior, but a kill-switch operator may invoke this expecting dry behavior. **Document this.** |

No P0 ungated paths found in DEX_MODULE itself. The P0 concerns are: (a) the `DRY_RUN` key-case bifurcation (lowercase `dry_run` vs uppercase `DRY_RUN`) creates a high-probability config mistake; (b) `OrderManager.submit_order` dry path *invents* a fill price (`order.price or Decimal('100.0')`, order_manager.py:527) — a $100 phantom price will silently corrupt all paper-trading P&L for any order created without an explicit price (which is the market-order default).

## Risk-policy coverage

| Control | Status | Evidence |
|---|---|---|
| Per-trade max loss | PARTIAL | `RiskManager.max_position_size_usd=10.0` (risk_manager.py:220, hard-capped at risk_manager.py:971-972) caps notional — but the *stop-loss* default in engine is a flat 10% per position (`engine.py:1056, 1080, 1308`). Not configurable per-symbol. |
| Per-hour max loss | MISSING | No per-hour aggregation. Circuit breakers check error_rate, slippage, consecutive_losses, drawdown, daily_loss — no hourly bucket. risk_manager.py:285-289 |
| Per-day max loss | IMPLEMENTED | `breaker.max_daily_loss_pct=10` (risk_manager.py:289); `PortfolioManager.daily_loss_limit=0.10` (portfolio_manager.py:127); `_get_current_metrics` tracks daily_pnl (risk_manager.py:1169-1185) |
| Per-symbol concentration | PARTIAL | `RiskManager.check_position_limit` (risk_manager.py:1226) blocks a *second* position in the same token, but it inspects `self.positions` which is set to `{}` at init (risk_manager.py:250) and never populated — this check is effectively a no-op. PortfolioManager has `max_position_size_pct=0.1` (portfolio_manager.py:100) but the same logic is duplicated and inconsistent across the two managers. |
| Per-module capital cap | PARTIAL | `DexTradingModule` checks `self.metrics.capital_used >= self.config.capital_allocation` (dex_module.py:186) but `metrics.capital_used` is only updated in `get_metrics()` (dex_module.py:264) — there's no gate at order-creation time; it relies on the caller having polled metrics. |
| Correlated-drawdown rule (cross-module) | MISSING | `check_correlation_limit` (risk_manager.py:1328) is per-token within the DEX book only, not cross-module. There is no global view of DEX+ARB+Solana+Sniper P&L for correlation. |
| Emergency-stop hook | PARTIAL | `engine.emergency_close_all_positions` exists (engine.py:2893) but is in-process only. No `scripts/emergency_stop.py` for operators. |
| Nonce/idempotency on retries | PARTIAL | `DirectDEXExecutor._get_next_nonce` uses a lock and pending nonce (direct_dex.py:164-197). Reset on error (direct_dex.py:199-208). But `Order.order_id` is a uuid; on retry there is no on-chain idempotency key — if a tx silently lands after a retry, the second tx will fire too. |
| Position reconciliation on startup | MISSING | `engine._load_state` is `pass` (engine.py:2715). No on-chain balance scan vs in-memory positions. |
| Stale-heartbeat → flatten/freeze | MISSING | `_watchdog` only logs; `_health_check` placeholders return OK. No freeze action wired. |

## Order/Position lifecycle review

Signal → decision → sizing → submit → ack → fill → P&L → close, walking from `engine._process_opportunities` to position close, noting desync points:

1. **Signal**: `_monitor_new_pairs` (engine.py:443) discovers pairs from DexScreener. No de-dup across discovery cycles — same pair can re-enter the queue.
2. **Decision**: `_analyze_opportunity` feeds `decision_maker.make_decision` (decision_maker.py:115). The decision is logged in `self.decision_history` (in-memory only).
3. **Sizing**: `RiskManager.calculate_position_size` (risk_manager.py:900) applies Kelly + risk multiplier, hard-caps at `max_position_size_usd`. Good. However `available_balance` comes from `self.wallet_manager.get_available_balance()` (risk_manager.py:917) — this is the wallet balance, not the *deployable* balance after open positions; if positions are open this double-counts.
4. **Submit**: `engine._execute_trade` calls `executor.execute_trade(order)` (engine.py:1259). In LIVE mode it then waits up to 120s for receipt (engine.py:1277). **Desync risk #1**: between `executor.execute_trade` returning and `wait_for_transaction_receipt`, if the process is killed, the tx may land on chain but no `active_positions[token]` entry exists. On restart there is no reconciliation.
5. **Ack/fill**: success branch sets `active_positions[token]=position` under `positions_lock` (engine.py:1316-1318) and schedules an immediate position check (engine.py:1321). The DB save happens after that (engine.py:1399). **Desync risk #2**: if DB save throws, the in-memory position exists with no DB record, and the next restart loses it (since `_load_state` is empty).
6. **P&L update**: `_monitor_positions_with_engine` (engine.py:3017) updates position price every 30s using DexScreener. **Desync risk #3**: DexScreener price ≠ on-chain executable price. Stop-loss decisions made from this price may not be achievable when the real sell tx lands.
7. **Close**: `_execute_position_close` (engine.py:3100) creates a SELL order via `build_order` with 2% slippage tolerance (engine.py:3137). **Desync risk #4**: the closing order uses `position.entry_amount` — if a partial close already happened, this will over-sell. Inspected position_tracker briefly — partial close handling exists but is fragile.
8. **Post-close**: `update_trade_metrics` updates breaker state (risk_manager.py:385). No on-chain confirmation that the sell actually filled at the reported price before P&L is booked.

The blob at engine.py:1294-1399 (success branch in LIVE mode) is also problematic because it does *not* update the `position_tracker` — it only updates `self.active_positions` (engine.py:1316-1318), but the monitoring loop at engine.py:3017 iterates `self.position_tracker.positions`. So LIVE positions opened via engine._execute_trade may never be monitored. Verify with the smartcontract agent — this looks like a P0 bug if confirmed.

## Profit-leak & loss-leak inventory

| ID | Category | File:Line | Estimated impact | Fix sketch |
|---|---|---|---|---|
| PL-01 | Phantom-fill P&L | trading/orders/order_manager.py:527 | Corrupts paper-trade stats, leads to overfitting | Reject dry submit if `order.price` is None; require explicit mark-price |
| PL-02 | Active-positions vs tracker desync (LIVE) | core/engine.py:1316 vs 3034 | Positions opened LIVE may not auto-close on SL/TP → unbounded loss | Add `position_tracker.add_position` in the LIVE success branch |
| PL-03 | Flat 10% stop on every entry | core/engine.py:1056, 1080, 1308 | Mis-sized stops for low-vol stables vs high-vol meme | Use `RiskManager.calculate_stop_loss(risk_score)` (already exists at risk_manager.py:980) |
| PL-04 | Available balance double-counting | core/risk_manager.py:917 | Oversize positions when open positions exist | Subtract `sum(p.value for p in self.positions)` |
| PL-05 | No fee/gas-aware sizing | trading/executors/direct_dex.py:271+ | Sub-$10 positions can lose to gas on Ethereum L1 | Require `expected_pnl > 3×gas_cost` gate at order build |
| PL-06 | 2% slippage tolerance on close | core/engine.py:3137 | Stop-loss exits eat extra 2% in addition to the SL | Tighten to 0.5-1% with retry-on-fail rather than wide-slippage one-shot |
| PL-07 | DEX route cache without TTL | trading/executors/direct_dex.py:96 | Stale routes when LP liquidity moves | Add TTL ≤ 30s |
| PL-08 | Multiple quote queries per trade | trading/executors/direct_dex.py:225 | RPC quota burn, slow execution | Cache quotes 1-2s per (chain, token_in, token_out) |
| PL-09 | No `reduceOnly` on close orders | core/engine.py:3137 (sell order build) | If a buy order races a close, can flip sign | Add an explicit "exit only" flag honored by executor |
| PL-10 | `_apply_mev_protection` is a stub | core/engine.py:2805 | All EVM swaps go to public mempool | Wire actual MEV-protect (or flag for smartcontract agent) |

## Live-readiness checklist

- [ ] DRY_RUN honored on every send/order/sign path — **PASS** for primary paths; **AMBER**: two key cases (`dry_run` vs `DRY_RUN`) split across config, and `OrderManager` synthesises fake fills with `Decimal('100.0')` (order_manager.py:527) which is dangerous.
- [ ] Per-trade max loss — **PARTIAL**: `max_position_size_usd=10` hard cap (risk_manager.py:971), but stop is a flat 10%.
- [ ] Per-hour max loss — **FAIL**: no hourly aggregation.
- [ ] Per-day max loss — **PASS**: risk_manager.py:289 + portfolio_manager.py:127.
- [ ] Position reconciliation on startup — **FAIL**: `_load_state` is a stub (engine.py:2715).
- [ ] Idempotent order IDs / nonce management — **PARTIAL**: nonce lock works (direct_dex.py:164); order_id is UUID, not deterministic.
- [ ] Heartbeat → freeze on stale — **FAIL**: only logs; no freeze.
- [ ] Emergency stop wired & reachable — **FAIL**: in-process method exists; no operator script.
- [ ] Dashboard visibility of circuit-breaker state — **PASS**: `get_circuit_breaker_status` (risk_manager.py:352) returns a dashboard-ready dict.
- [ ] Pre-trade balance + position-limit check before LIVE send — **PASS** (engine.py:1217-1250).
- [ ] Slippage cap on every executor call — **PASS** (order has `slippage_tolerance`; default 5% at engine.py:1200 is wide for non-meme tokens — flag).
- [ ] MEV protection on live mainnet sends — **FAIL**: `_apply_mev_protection` returns the order unchanged (engine.py:2805-2807).

## Proposed action backlog

Ranked by ROI (risk reduction per LOC). Each ≤ 200 lines.

- [ ] **RM-01** Add `scripts/emergency_stop.py` and `scripts/close_all_positions.py` — operator-callable CLI that talks to a UNIX socket / DB flag the engine watches; touches `scripts/`, `core/engine.py` (watcher task). Risk: critical kill-switch gap closed. Owner: analyst+backend.
- [ ] **RM-02** Implement `engine._load_state` to (a) read DB-persisted positions, (b) scan wallet for orphan token balances on enabled chains, (c) refuse to start LIVE if mismatches > tolerance. Touches `core/engine.py`. Risk: orphan positions on restart. Owner: analyst.
- [ ] **RM-03** Add `RiskManager.check_hourly_loss()` mirroring daily logic, wire into `check_circuit_breakers`. Touches `core/risk_manager.py`. Risk: 1-hour drawdown not bounded. Owner: analyst.
- [ ] **RM-04** Unify `dry_run` config key: single canonical path through `ConfigManager`, deprecate uppercase `DRY_RUN` in executor configs, fail fast on mismatch at init. Touches `trading/executors/{direct_dex,base_executor}.py`, `core/engine.py`, `main.py`, both `main_dex.py`. Owner: analyst+backend.
- [ ] **RM-05** Replace `OrderManager.submit_order` dry-run synthetic price (`Decimal('100.0')`) with the mid-quote from `get_best_quote`. Touches `trading/orders/order_manager.py`. Owner: analyst.
- [ ] **RM-06** In `engine._execute_trade` LIVE success branch, also add the position to `position_tracker` so monitoring works. Touches `core/engine.py:1316`. Owner: analyst.
- [ ] **RM-07** Use `RiskManager.calculate_stop_loss(risk_score)` instead of flat 10% in entry plumbing. Touches `core/engine.py:1056, 1080, 1308`. Owner: analyst.
- [ ] **RM-08** Fix `RiskManager` available-balance to subtract open positions. Touches `core/risk_manager.py:917`. Owner: analyst.
- [ ] **RM-09** Implement real `_health_check` and stale-heartbeat freeze action: if any of (DB, RPC, primary-task) > N s stale, set engine state to `FROZEN`, skip new opens, allow closes only. Touches `core/engine.py:2431, 2869-2887`. Owner: analyst.
- [ ] **RM-10** Add `Order.client_order_id` deterministic from `(strategy, token, side, bucket_minute)` so retries are idempotent. Touches `trading/orders/order_manager.py`. Owner: analyst.
- [ ] **RM-11** Per-module capital-cap gate at order-build time (not just metrics polling). Touches `modules/dex_trading/dex_module.py:186` + a hook in engine. Owner: analyst.
- [ ] **RM-12** Tighten exit slippage from 2% to a configurable per-symbol value, retry-on-revert instead of widening. Touches `core/engine.py:3137`. Owner: analyst.

## Recommended go-live sequence

1. Land RM-01 through RM-06 (kill switch, reconciliation, hourly cap, config unification, dry-run fill fix, monitor wiring). These are the minimum bar to even *consider* LIVE.
2. Stage 1 — paper, full coverage: `DRY_RUN=true` for 1 week minimum. Verify daily P&L log matches DB save count, circuit-breaker dashboard updates, alerts fire on synthetic breach.
3. Stage 2 — testnet (Sepolia / BSC testnet): real signing, real RPC, fake liquidity. Verify nonce flow under restart, position reconciliation logic recovers seeded positions.
4. Stage 3 — small live cap: `DRY_RUN=false`, but `max_position_size_usd=5`, `max_open_positions=2`, only on a single chain (recommend Base — cheap gas, decent liquidity). Run 72h. Verify: every fill in DB, every position in tracker, every close has receipt status=1, breakers trip on injected fault.
5. Stage 4 — ramp: double cap and positions every 72h provided no incidents and circuit breakers idle. Cap top-out at `MAX_POSITION_SIZE_USD=100` for the first 2 weeks.
6. Stage 5 — full DRY_RUN-false on all enabled chains, with RM-07 through RM-12 landed.

## Open questions

- Is `OrderManager.submit_order` actually used by the DEX path, or is the engine going around it via `trade_executor.execute_trade`? The dry-run synthetic fill at order_manager.py:527 may be unreachable in practice — confirm with smartcontract agent.
- The Solana executor (`JupiterExecutor`) is initialised at `core/engine.py:271` with `dry_run` in its config — does its internal swap path honor it? Out of scope for this audit; flag for smartcontract agent.
- `RiskManager.positions` is `{}` at init and never populated (risk_manager.py:250). Is the risk manager supposed to be the source of truth, or is it `portfolio_manager.positions`? The two diverge.
- Are there other entry points (e.g., dashboard "manual trade") that bypass `engine._execute_trade`? If yes, RM-01/02/04 must cover them.
