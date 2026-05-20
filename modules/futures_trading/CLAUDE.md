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
| `futures_telegram_emergency_close_enabled` | bool | true | High-priority Telegram alert on every FUT-RM-07 emergency-close. Fail-soft when Telegram is not configured. |

## Wave-4 changes (campaign 2026-05-20)
- **FUT-RM-07b** — `_notify_fut_rm_07_emergency_close()` (futures_engine.py)
  fires a `priority="critical"` Telegram payload (symbol / side /
  intended-vs-actual margin / position size / timestamp / close status)
  via the shared `monitoring.telegram_bot.get_telegram_controller()`
  singleton whenever the FUT-RM-07 verify path detects a CROSS-margin
  fill. Lazy import keeps the engine import-time clean. Gated by
  `FuturesLeverageConfig.telegram_emergency_close_enabled` (default True);
  fail-soft if Telegram is not configured (logs warning, never blocks
  the emergency-close itself).
- **FUT-RM-09b** — new dashboard widget on `dashboard_futures.html` next
  to the FUT-RM-09 hourly chart: per-symbol 24h forward funding-cost
  forecast. Backed by `GET /api/futures/funding-forecast` in
  `enhanced_dashboard.py` which reads the latest snapshot row per
  (symbol, side) from `futures_funding_payments` (migration 029) and
  projects `predicted_usd × intervals_per_window` (default 3 intervals
  of 8h = 24h on Binance/Bybit USDT perps). Returns per-row implied APR.
  Sign convention matches the table: positive = cost to the book.

### Wave-4 commits (this branch)
- `34e6c95` FUT-RM-07b stub: emergency-close Telegram-alert flag plumbing (engine side)
- `68b20fb` Wave-4 stub commit including `FuturesLeverageConfig.telegram_emergency_close_enabled` default
- `6f66608` FUT-RM-07b: wire Telegram alert dispatcher for emergency close
- `142250b` FUT-RM-09b endpoint: `GET /api/futures/funding-forecast`
- `b805626` FUT-RM-09b widget: per-symbol funding-cost forecast panel
- `7b2da32` Minor dashboard widget refinement

## Wave-5 changes (campaign 2026-05-20)

Triggered by operator post-mortem of 4 trades / 25% win rate / -$14.99
realized: 3 SL hits on AAVE/FIL/NEAR shorts (-20% each on 10x), 1 TP1
hit on SOL long (+18%). Audit confirmed two structural causes:
(a) static 2% SL is too tight for 4%-ATR alts at 10x, and
(b) the +4 signal_score bar can clear on a single strong indicator.

- **FUT-RM-15** — multi-indicator confluence gate. New
  `FuturesStrategyConfig.min_signal_confluence_count` (default 2).
  `_scan_opportunities` now requires N of 4 directional indicators
  {RSI, MACD, Bollinger, EMA} to agree with the entry side in addition
  to the signed `min_signal_score`. Volume excluded (confirmer only).
  Rejection logged; `min_signal_confluence_count=0` disables.

- **FUT-RM-16** — ATR-scaled SL/TP per symbol. New
  `FuturesRiskConfig.atr_dynamic_sl_tp_enabled` (default True),
  `atr_sl_multiplier` (1.5), `atr_sl_min_pct` (1.5), `atr_tp_rr_ratio`
  (2.0). `_open_position` computes
  `SL = max(atr_sl_min_pct, atr_sl_multiplier × ATR%)` and rescales
  TP1..TP4 so `TP1 = atr_tp_rr_ratio × SL`. Per-position
  `metadata['dynamic_sl_pct']` is stashed and honored by
  `_check_exit_conditions` so the per-symbol SL is used even after
  the engine restores its static settings for the next entry.

- **FUT-RM-17** — per-symbol consecutive-loss cool-off.
  `FuturesRiskManager` now tracks per-symbol consecutive losses and
  arms a cool-off after `post_loss_cooloff_threshold` (default 2)
  losses on the same pair, refusing new entries on that symbol for
  `post_loss_cooloff_minutes` (default 240 = 4h). Surfaced on
  `FuturesRiskConfig` for settings-page editing. Engine calls
  `should_skip_for_cooloff()` early in `_open_position` and
  `update_on_trade_close(pnl, symbol=...)` in `_close_position`. A
  winning trade clears the cool-off and resets the counter; expiry
  also resets the counter so a single loss after recovery does NOT
  immediately re-arm.

- **FUT-RM-18** — default leverage 10x → 5x. Migration 031 lowers the
  DB-seeded `futures_leverage.default_leverage` from 10 to 5 (only when
  the value is still the legacy 10 — operator customisations survive).
  Pydantic default, env default, and engine fallback all lowered to 5x.
  `max_leverage` stays 20x; FUT-RM-08 per-symbol override table still
  wins for pairs the operator wants pinned higher (e.g. BTC/USDT at 10x).

## Configuration cheat-sheet (Wave-5 additions)
| Key | Type | Default | What it does |
|---|---|---|---|
| `futures_min_signal_confluence_count` | int | 2 | Require N of {RSI, MACD, BB, EMA} to agree before entry. 0 = disabled. |
| `futures_atr_dynamic_sl_tp_enabled` | bool | true | ATR-scaled SL/TP per symbol. False = revert to static. |
| `futures_atr_sl_multiplier` | float | 1.5 | SL = max(atr_sl_min_pct, mult × ATR%). |
| `futures_atr_sl_min_pct` | float | 1.5 | Floor on SL distance (price %). |
| `futures_atr_tp_rr_ratio` | float | 2.0 | TP1 = ratio × SL distance. |
| `futures_post_loss_cooloff_threshold` | int | 2 | Per-symbol consecutive-loss count that arms cool-off. |
| `futures_post_loss_cooloff_minutes` | int | 240 | Cool-off duration in minutes (240 = 4h). |
| `futures_default_leverage` | int | 5 | Lowered from 10 — see migration 031. |

### Wave-5 commits (this branch)
- `e3af3eb` FUT-RM-15: multi-indicator confluence gate (min 2 of 4)
- `8e79b08` FUT-RM-16: ATR-scaled SL/TP per symbol
- `69862c1` FUT-RM-17: per-symbol consecutive-loss cool-off
- `ac7a57b` FUT-RM-17: surface cool-off knobs in FuturesRiskConfig
- `57a6a2c` FUT-RM-18: lower default leverage from 10x to 5x
## See also
- Phase 1 audit reports: `docs/agents/reports/FUTURES_*.md` (quant / analyst / backend).
- Canonical engine API: `docs/engines.md`.
