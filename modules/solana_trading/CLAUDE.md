# SOLANA Module
## What it does
Solana spot trading via Jupiter aggregator with trailing-stop ladder, plus optional pump.fun launch sniping and Drift perp leg. Canonical module dir for all SOL strategies.
## Entry point
`modules/solana_trading/main_solana.py` — launched as a subprocess by `main.py` when `SOLANA_MODULE_ENABLED=true`. Engine: `modules/solana_trading/core/solana_engine.py`. The sibling dir `modules/solana_strategies/` holds helper utilities only (`jupiter_helper.py`, `drift_helper.py`, a duplicate `solana_config_manager.py`); it intentionally has no `CLAUDE.md`.
## Key config (DB-backed via `solana_config_manager.py`)
- `position_size_sol` — base SOL committed per Jupiter entry
- `jupiter_slippage_bps` — Jupiter quote slippage cap (default 50)
- `priority_fee_lamports` — Solana compute-unit priority fee (default 1_000_000)
- `stop_loss_pct` / `take_profit_pct` — Jupiter trailing-stop bounds
- `drift_enabled` / `drift_leverage` — Drift perp leg toggle and leverage cap
- `pumpfun_max_positions` — pump.fun concurrent position ceiling
## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_solana` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Effect: `should_skip_live` returns `True` -> Jupiter execute returns simulated route.
## Logs
`logs/solana_trading/` — main, errors, trades (rotating handler).
## Primary risk-policy gate
Per-module local risk: position-count ceiling + per-strategy SL/TP percent at `modules/solana_trading/core/solana_engine.py` close-path; no cross-module `RiskManager.validate_trade` call yet (P1 follow-up).
## Live-trade readiness
AMBER → GREEN candidate (Jupiter spot; pending production verification). MB-06..MB-10 closed (decimals, co-signers, priority fee, restart reconciliation, DRY_RUN gate); secrets_manager wiring (`8b4ee7d`) and pool_engine sweep (`a21ec41`) landed. MB-15 (Drift hardening) deferred — Drift is a toggleable feature gate, not a blocker for the Jupiter path.
## See also
- Phase 1 audit reports: `docs/agents/reports/SOLANA_*.md` (smartcontract / quant / analyst).
- Canonical engine API: `docs/engines.md`.
