# FUTURES Module
## What it does
Centralized-exchange perp trading on Binance Futures and Bybit V5. Multi-strategy (trend, mean-revert, breakout) with isolated margin and per-position SL/TP.
## Entry point
`modules/futures_trading/main_futures.py` — launched as a subprocess by `main.py` when `FUTURES_MODULE_ENABLED=true`. Engine: `modules/futures_trading/core/futures_engine.py`. Exchange adapters under `exchanges/`. Config schema in `config/futures_config_manager.py`.
## Key config (DB-backed via `FuturesConfigManager`, sectioned by `FuturesConfigType`)
- `general.exchange` — `binance` or `bybit`; selects adapter
- `position.max_positions` / `position.position_size_usd` — concurrent cap + base sizing
- `leverage.default_leverage` / `leverage.max_leverage` — applied per new position
- `risk.stop_loss_pct` / `risk.take_profit_pct` — SL/TP percent on mark price
- `pairs.allowed_pairs` — comma-separated symbol whitelist
- `strategy.*` — per-strategy parameters (RSI, MACD, BB, EMA thresholds)
## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_futures` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Effect: `should_skip_live` returns `True` -> exchange adapter returns simulated order.
## Logs
`logs/futures_trading/` — main, errors, trades (rotating handler).
## Primary risk-policy gate
`FuturesRiskManager.validate_new_position(...)` — wired on the live path post-MB-17. Defined in `modules/futures_trading/futures_risk_manager.py`; called from `core/futures_engine.py` open-position path.
## Live-trade readiness
AMBER → GREEN candidate (pending production verification). MB-16 (init order), MB-17 (margin mode + validate_new_position wiring), MB-17b (Bybit V5 helpers), MB-18 (mark vs last price) closed. Reconcile observability hardened with `last_reconcile_at` + RESTART OVER-CAP detection (`24241e3`); BaseModule reconcile hook (`3981ffd`); Binance↔Bybit position-shape normalizer (`ed350d0`) re-enables liquidation-risk grading for Bybit positions.
## See also
- Phase 1 audit reports: `docs/agents/reports/FUTURES_*.md` (quant / analyst / backend).
- Canonical engine API: `docs/engines.md`.
