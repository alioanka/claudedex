# ClaudeDex Engines Reference

Cross-cutting engines under `core/` and `config/`. Each module's CLAUDE.md
points here for the canonical API surface and "when to call" guidance.

## 1. `config/pool_engine.py` — RPC pool

**Class:** `PoolEngine` (singleton at `config/pool_engine.py:208`;
`await PoolEngine.get_instance()` at `:238`)

**Public API:**
- `await pool.get_endpoint(provider_type)` at `:551` — single endpoint URL; selects by health + priority.
- `await pool.get_endpoint_with_fallbacks(provider_type, max_fallbacks=3)` at `:603` — ordered list.
- `await pool.report_success(provider_type, url, latency_ms)` at `:695` / `report_failure(...)` at `:763` / `report_rate_limit(...)` at `:651` — every chain client SHOULD report.
- `await pool.run_health_checks()` at `:930` — periodic; orchestrator-only.
- `await pool.shutdown()` at `:1298` — at process exit.

**Provider-type strings:** `'ETHEREUM_RPC'`, `'BSC_RPC'`, `'POLYGON_RPC'`,
`'BASE_RPC'`, `'ARBITRUM_RPC'`, `'OPTIMISM_RPC'`, `'AVALANCHE_RPC'`,
`'SOLANA_RPC'`, plus API providers like `'ETHERSCAN_API'`, `'HELIUS_API'`.

**When to call:**
- Every chain client construction. Never `os.getenv('*_RPC_URL')` directly in module code.
- After every RPC call, report success or failure. This drives the health-weighted rotation.

## 2. `core/risk_manager.py` — Cross-module risk validator

**Class:** `RiskManager` at `core/risk_manager.py:193` (`await rm.initialize()` at `:240`).

**Primary live-gate:**
- `await rm.validate_trade(token_address, amount) -> (bool, reason)` at `:1047` — call BEFORE every live broadcast. ARB engines (P1-06, commit `47998c1`) and AI executor (MB-20 follow-up, commit `a0b8814`) already do this.

**Secondary surface:**
- `await rm.analyze_token(token_address, force_refresh=False)` at `:439` — produces `RiskScore`.
- `await rm.calculate_position_size(opportunity)` at `:900` — Kelly-fraction-style sizing.
- `calculate_stop_loss(risk_score)` at `:980` / `calculate_take_profit(risk_score, market_conditions=None)` at `:1013`.
- `check_circuit_breakers(metrics)` at `:261` / `reset_circuit_breaker(manual=False)` at `:340` / `get_circuit_breaker_status()` at `:352`.
- `update_trade_metrics(trade_result)` at `:385` — feed back actual fills.
- `await rm.check_position_limit(token)` at `:1226`.

**Futures-only sibling:** `modules/futures_trading/futures_risk_manager.py:17`
`FuturesRiskManager.validate_new_position(size_usd, current_positions, available_capital)`
at `:50` — wired at MB-17, dormant until orchestrator injects an instance.

**When to call:**
- `validate_trade` at the LAST gate before order broadcast. Refuse on `(False, reason)`.
- DRY_RUN paths should NOT call `validate_trade` — they don't broadcast.
- Modules without their own DI (SOLANA, SNIPER, COPY) currently rely on local caps; cross-module integration is a P1 follow-up.

## 3. `core/dry_run.py` — DRY_RUN / kill-switch gating

**Pure functions** (no class):
- `should_skip_live(module_dry_run, *, module='', account=None) -> bool` at `:88` — THE primary live-gate. Returns True iff module DRY_RUN, OR global kill switch ON, OR per-module pause flag-file present.
- `resolve_dry_run_env('DRY_RUN', default=True)` at `:74` — safe parse for module init.
- `set_global_kill_switch(value)` at `:65` / `is_global_kill_switch()` at `:70` — process-local.
- `is_module_paused(module)` at `:28` / `set_module_pause(module, paused)` at `:46` — flag-file at `logs/.pause_<module>`.
- `start_killswitch_poller(path='logs/.killswitch', interval=1.0)` at `:130` / `stop_killswitch_poller()` at `:161` — auto-started for every BaseModule subclass via `modules/base_module.py`.

**When to call:**
- Every send/order/sign/broadcast path gates on `should_skip_live` BEFORE the live action.
- Module init parses `DRY_RUN` via `resolve_dry_run_env`.
- `start_killswitch_poller` fires automatically on first BaseModule `start()` — modules don't call it directly.

## 4. `core/units.py` — Cached on-chain decimals + conversions

**Pure functions** (no class):
- `await get_evm_decimals(chain, token_address) -> int` at `:105` (cached after first RPC call).
- `await to_raw_evm(chain, token_address, amount)` at `:142` / `from_raw_evm(...)` at `:148` — human <-> raw uint256.
- `await get_spl_decimals(token_mint)` at `:157` / `to_raw_spl(...)` at `:199` / `from_raw_spl(...)` at `:205`.
- `to_wei(amount)` at `:214` / `from_wei(raw)` at `:219` for native ETH.
- `to_lamports(sol_amount)` at `:224` / `from_lamports(raw)` at `:229` for native SOL.

**When to call:**
- Every place a trade size needs to convert between human-readable and on-chain raw units. NEVER multiply by `10**18` / `1e9` / `1e6` directly — that produced MB-01, MB-06, MB-23.

## 5. `core/portfolio_manager.py` — Capital allocation + position governor

**Class:** `PortfolioManager` at `core/portfolio_manager.py:83` (per-orchestrator instance).

**Public API:**
- `pm.set_dependencies(db, alerts)` at `:150` — DI wiring at startup.
- `await pm.can_open_position()` at `:155` — global position-count + drawdown gate.
- `pm.get_block_reason()` at `:209` / `await pm.manual_reset_block(reason)` at `:271` — observability + operator override.
- `pm.get_available_balance()` at `:316` / `pm.get_max_position_size(chain=None)` at `:346` — sizing inputs.
- `await pm.allocate_capital(opportunities)` at `:383` — multi-opportunity allocator returns per-symbol fractions.
- `await pm.update_portfolio(trade)` at `:487` — post-fill state update.

**When to call:**
- Before opening new positions, gate on `can_open_position`.
- For multi-strategy capital splits, run `allocate_capital`.
- After fills, call `update_portfolio` so subsequent `get_max_position_size` reflects reality.

## 6. `core/decision_maker.py` — Strategy decision pipeline

**Class:** `DecisionMaker` at `core/decision_maker.py:75`.

**Public API:**
- `await dm.make_decision(analysis) -> TradingDecision` at `:115` — main entry. BUY/SELL/HOLD with confidence + size.
- `await dm.evaluate_opportunity(opportunity)` at `:860` / `await dm.validate_decision(decision)` at `:939` — pre-action gates.
- `calculate_confidence_score(signals)` at `:890` / `determine_action(scores)` at `:913` — composable scoring.
- `await dm.update_performance(decision_id, outcome)` at `:818` — feedback loop.

**When to call:**
- DEX / strategy engines call `make_decision` per scan cycle.
- After trade settles, `update_performance` lets the decision-maker tune confidence over time.

## Cross-engine call order (for one trade)

1. `PoolEngine.get_endpoint` — connect.
2. `RiskManager.analyze_token` — score.
3. `DecisionMaker.make_decision` — direction + confidence.
4. `PortfolioManager.can_open_position` — global gate.
5. `RiskManager.calculate_position_size` + `PortfolioManager.get_max_position_size` — size.
6. `core.units.to_raw_evm` / `to_raw_spl` — convert to on-chain units.
7. `should_skip_live(...)` — final live-gate.
8. `RiskManager.validate_trade` — last-mile broadcast gate.
9. Execute via exchange client / DEX router.
10. `PoolEngine.report_success` / `report_failure` — health feedback.
11. `RiskManager.update_trade_metrics` + `PortfolioManager.update_portfolio` — state update.

## See also
- Per-module CLAUDE.md: `modules/<name>/CLAUDE.md`
- Phase 1 audits: `docs/agents/reports/`
- Master backlog: `docs/agents/MASTER_BACKLOG.md`
