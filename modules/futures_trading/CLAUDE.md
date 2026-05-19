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

## Wave-2 changes (campaign 2026-05-19)
- **FUT-RM-01** — `main_futures.py` now merges `FuturesLeverageConfig.max_leverage` and `FuturesPositionConfig.max_positions` into the runtime `FuturesRiskManager`. Pre-wave the subprocess entry path bypassed the dashboard-wrapper fix from `b1b8df9` and silently defaulted to `max_leverage=3`, rejecting every operator-configured ≥5x entry.
- **FUT-RM-02** — `FuturesTradingApplication._assert_runtime_risk_matches_config()` runs at startup and logs an error (or raises when `FUTURES_RISK_ASSERT_HARD=1`) when runtime caps drift from the DB. Catches future regressions of FUT-RM-01.
- **FUT-RM-03** — `tests/unit/test_futures_risk_wiring.py`: 5 cap-propagation tests + 3 DRY_RUN smoke tests + a static-guard pin on the `should_skip_live()` gate around `open_long`/`open_short`.
- **FUT-RM-05** — funding-rate directional entry gate. New `FuturesFundingConfig.skip_long_funding_bps` / `skip_short_funding_bps` (defaults 5 bps ~= 55% APR ceiling). `FuturesRiskManager.should_skip_for_funding(side, funding_rate)` consulted before the validator. `FuturesTradingEngine._get_funding_rate_cached()` uses the mainnet `price_client` with a 300s TTL.
- **FUT-RM-06** — ATR-based per-symbol risk-parity sizing. New `FuturesPositionConfig.atr_sizing_enabled` (off by default), `atr_risk_pct`, `atr_stop_multiplier`. `TechnicalSignals.atr` / `atr_pct` populated in `_get_technical_signals`. `_calculate_position_size` picks the ATR branch when enabled and the ATR reading is usable; otherwise the existing static / dynamic paths run unchanged.
- **FUT-RM-07** — post-fill ISOLATED-margin verification. `FuturesLeverageConfig.enforce_isolated_margin` (default True). `_verify_isolated_or_close()` reads back the position after a live fill and emergency-closes if `margin_type != ISOLATED` (defense-in-depth on MB-17).

## Configuration cheat-sheet (Wave-2 additions)
| Key | Type | Default | What it does |
|---|---|---|---|
| `futures_skip_long_funding_bps` | float | 5.0 | Refuse new longs when funding > N bps per interval. 0 = disabled. |
| `futures_skip_short_funding_bps` | float | 5.0 | Refuse new shorts when funding < -N bps. 0 = disabled. |
| `futures_max_funding_age_seconds` | int | 900 | Stale-data cap on funding gate. |
| `futures_atr_sizing_enabled` | bool | false | Toggle ATR risk-parity sizing. |
| `futures_atr_risk_pct` | float | 1.0 | % of `capital_allocation` risked per trade when ATR sizing is on. |
| `futures_atr_stop_multiplier` | float | 1.5 | Stop distance in ATR units (used by the sizing math). |
| `futures_enforce_isolated_margin` | bool | true | Post-fill ISOLATED-margin verify + emergency-close on mismatch. |
## See also
- Phase 1 audit reports: `docs/agents/reports/FUTURES_*.md` (quant / analyst / backend).
- Canonical engine API: `docs/engines.md`.
