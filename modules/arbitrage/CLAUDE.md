# ARBITRAGE Module
## What it does
Spatial (cross-DEX) and triangular EVM arbitrage with flash-loan funding (Aave V3). Spatial is production-grade; the triangular path is currently gated behind the MB-05 atomic-receiver guard.
## Entry point
`modules/arbitrage/main_arbitrage.py` — launched as a subprocess by `main.py` when `ARB_MODULE_ENABLED=true`. Engines: `modules/arbitrage/arbitrage_engine.py` (spatial), `triangular_engine.py` (gated), `solana_engine.py` (cross-chain helper).
## Key config (DB-backed via `ConfigManager`)
- `flash_loan_amount` — flash-loan size in ETH-equivalent per leg
- `chain_config.tokens` / `routers` / `arb_pairs` — selected via `EVM_DEX_ROUTING` (MB-24)
- `flash_loan_env_key` — name of env var holding the deployed receiver contract address (resolved via `security/secrets_manager` first, env fallback)
- `rpc_url` — overrides `PoolEngine` selection when set
- `min_profit_spread` — minimum profit gate before broadcast (accepts fraction `0.005` or percent `0.5`; wired wave-2 A2-06)
- `gas_budget_usd_per_hour` — hourly rolling cap on USD gas spend per engine; new executions are refused once exceeded (default `$50`; wave-2 A2-07)
## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_arbitrage` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Hourly gas-budget gate: `_gas_budget_check_and_charge` (A2-07) — auto-cooldown on USD spend.
- Effect: `should_skip_live` returns `True` -> live-write gates return their `_simulate_*` path.
## Logs
`logs/arbitrage/` — main, errors, trades (rotating handler).
## Primary risk-policy gate
`core.risk_manager.RiskManager.validate_trade(token_in, amount)` called inside the spatial-arbitrage execute path in `arbitrage_engine.py` (search for `# P1-06: pre-execute risk gate`). Injected via `set_risk_manager()` method on the engine.
## Per-chain cost profile (wave-2)
`CHAIN_CONFIGS[chain_id]` carries `flash_loan_gas_limit`, `fallback_gas_gwei`, `default_slippage_pct`, `flash_loan_fee_pct`, `is_l2`. Consumed by:
- `_check_arb_opportunity` net-spread gate (replaces hardcoded 0.5% A2-03)
- `_log_arb_trade` PnL accounting (replaces hardcoded $15 gas / 0.6% slippage A2-02 / A2-05)
- `_gas_cost_usd_per_tx` live gas oracle (1s cached, USD-denominated)
- `_gas_spike_multiplier` adaptive `min_profit_bps` curve (1.0–2.5x baseline)
## Live-trade readiness
AMBER → GREEN candidate (spatial; pending production verification).
Closed pre-wave: MB-03 (DAI typo), MB-04 (one-legged broadcast), MB-05 (triangular gated), secrets_manager wiring (`b20f56a`), pool_engine sweep (`a21ec41`).
Closed wave-2: **A2-01** (NameError crash in opportunity log path — every live trade was silently dropped), **A2-02 / A2-03 / A2-05** (hardcoded gas/slippage replaced with per-chain live profile), **A2-04** (receiver address via secrets_manager), **A2-06** (dashboard min-profit knob now honored), **A2-07** (hourly gas-budget tracker).
Triangular path remains entry-disabled by atomic-receiver guard — explicit scope cut pending contract deploy, not a defect.
## See also
- Phase 1 audit reports: `docs/agents/reports/ARBITRAGE_*.md` (smartcontract / quant / analyst).
- Wave-2 audit: `docs/agents/reports/ARBITRAGE_CAMPAIGN.md`.
- Canonical engine API: `docs/engines.md`.
