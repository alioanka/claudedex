# ARBITRAGE Module
## What it does
Spatial (cross-DEX) and triangular EVM arbitrage with flash-loan funding (Aave V3). Spatial is production-grade; the triangular path is currently gated behind the MB-05 atomic-receiver guard.
## Entry point
`modules/arbitrage/main_arbitrage.py` — launched as a subprocess by `main.py` when `ARB_MODULE_ENABLED=true`. Engines: `modules/arbitrage/arbitrage_engine.py` (spatial), `triangular_engine.py` (gated), `solana_engine.py` (cross-chain helper).
## Key config (DB-backed via `ConfigManager`)
- `flash_loan_amount` — flash-loan size in ETH-equivalent per leg
- `chain_config.tokens` / `routers` / `arb_pairs` — selected via `EVM_DEX_ROUTING` (MB-24)
- `flash_loan_env_key` — name of env var holding the deployed receiver contract address
- `rpc_url` — overrides `PoolEngine` selection when set
- `min_profit_bps` — minimum profit gate before broadcast
## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_arbitrage` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Effect: `should_skip_live` returns `True` -> live-write gates return their `_simulate_*` path.
## Logs
`logs/arbitrage/` — main, errors, trades (rotating handler).
## Primary risk-policy gate
`core.risk_manager.RiskManager.validate_trade(token_in, amount)` called at `modules/arbitrage/arbitrage_engine.py:1600-1604` (P1-06). Injected via `set_risk_manager()` at `:962-964`.
## Live-trade readiness
AMBER. MB-03 (DAI typo), MB-04 (one-legged broadcast), MB-05 (placeholder amountIn) closed; triangular still entry-disabled by atomic-receiver guard pending contract deploy.
## See also
- Phase 1 audit reports: `docs/agents/reports/ARBITRAGE_*.md` (smartcontract / quant / analyst).
- Canonical engine API: `docs/engines.md`.
