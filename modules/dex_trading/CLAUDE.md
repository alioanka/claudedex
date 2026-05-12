# DEX Module
## What it does
Spot trading on EVM DEXes (Uniswap V2/V3, SushiSwap, PancakeSwap) across Ethereum, BSC, Polygon, Arbitrum, Base. Routes through `trading/executors/direct_dex.py` with optional Flashbots MEV protection.
## Entry point
`modules/dex_trading/main_dex.py` — launched as a subprocess by `main.py` when `DEX_MODULE_ENABLED=true`. Engine: `modules/dex_trading/dex_module.py` wraps the shared `trading/trading_engine.py`.
## Key config (DB-backed via `ConfigManager`)
- `max_slippage_bps` — per-trade slippage cap (default 50)
- `max_gas_price` — gwei ceiling for tx submission (default 50)
- `mev_protection` — toggle Flashbots private-bundle path (default True)
- `supported_dexs` — list of router names enabled for routing
- `jupiter_routing` — kept for future SOL-leg routing; currently EVM only
## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_dex` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Effect: `should_skip_live` returns `True` -> live-write gates in `direct_dex.py:64,276,1013` return their `_simulate_*` path.
## Logs
`logs/dex_trading/` — main, errors, trades (rotating handler).
## Primary risk-policy gate
`core.risk_manager.RiskManager.validate_trade(token, amount)` — wired through the shared `trading/trading_engine.py` order-execution path. Per-executor caps (`max_slippage_bps`, `max_gas_price`) enforced inline at `trading/executors/direct_dex.py:88,567`.
## Live-trade readiness
AMBER. MB-01 (decimals) and MB-02 (Flashbots EIP-191) closed. Outstanding: P1-04 `pool_engine` integration for unified RPC selection.
## See also
- Phase 1 audit reports: `docs/agents/reports/DEX_*.md` (smartcontract / quant / analyst).
- Canonical engine API: `docs/engines.md`.
