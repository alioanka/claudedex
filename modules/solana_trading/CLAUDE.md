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
- `adaptive_priority_fee_enabled` (default `False`) — when True, JupiterHelper bids `getRecentPrioritizationFees` p75 (clamped to `adaptive_priority_fee_min_lamports`..`adaptive_priority_fee_max_lamports`, 5s cache). Off keeps legacy structured-dict default.
- `jupiter_quote_max_age_s` (default 10s) — execute_swap refetches quotes older than this before signing (mirrors `jupiter_executor.py` P1 fix).
- `solana_ml_enabled` (default `False`) — when True, `_open_position` runs the lazy-loaded `ml/models/rug_classifier.RugClassifier` and refuses entry above `solana_ml_max_rug_prob` (default 0.40). Fail-soft: no trained model → gate idle (refuse-to-predict, MB-19 pattern).
- `drift_max_leverage` / `drift_max_funding_pct_annual` / `drift_oracle_deviation_max_pct` / `drift_min_oracle_conf_bps` — MB-15 client-side guards inside `DriftHelper.open_position`. Drift `place_perp_order` is gated by all four AND the killswitch / pause flag. Drift stays toggle-OFF by default.
## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_solana` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Effect: `should_skip_live` returns `True` -> Jupiter execute returns simulated route.
## Logs
`logs/solana_trading/` — main, errors, trades (rotating handler).
## Primary risk-policy gate
- Cross-module: `core.risk_manager.RiskManager.validate_trade(token_mint, amount_sol)` called in `_open_position` at `solana_engine.py:~3296` immediately before every Jupiter swap broadcast. Injected via `set_risk_manager()` from `main_solana.py`; engine is fail-soft if `RiskManager` construction fails (logs a warning, continues without the gate). Only entries are gated; exits always allowed.
- Per-module local: position-count ceiling + per-strategy SL/TP percent enforced inside the engine's close-path.
## Live-trade readiness
AMBER → GREEN candidate (Jupiter spot; pending production verification). MB-06..MB-10 closed (decimals, co-signers, priority fee, restart reconciliation, DRY_RUN gate); secrets_manager wiring (`8b4ee7d`) and pool_engine sweep (`a21ec41`) landed. Campaign wave-2 additions:
- **MB-06 residuals (`09a5c85`)** — `_get_token_balance` now resolves decimals on-chain via `core.units.get_spl_decimals` + parsed `tokenAmount.decimals`; the emergency-close path no longer hardcodes `(10 ** 6)` for the Jupiter sell amount.
- **MB-15 Drift hardening (`661cee6`)** — `DriftHelper.open_position` enforces dry-run / killswitch / pause, funding-rate sanity, oracle deviation + Pyth confidence, and a leverage cap. All four fail closed. Drift stays toggle-OFF by default.
- **Adaptive priority fee + quote freshness TTL (`83df4ad`)** — JupiterHelper can bid `getRecentPrioritizationFees` percentile (default p75) and refetches stale Jupiter quotes before signing. Both opt-in; config wired in `8cf0143` bundle.
- **P1-07 ML rug gate (`b1b358f`)** — `_open_position` runs `RugClassifier.predict` when `solana_ml_enabled=True`; refuses entry above `solana_ml_max_rug_prob`. Fail-soft when no trained model.

Outstanding (follow-ups, not blocking):
- Jito bundle path for solana_trading (the JitoClient lives in `modules/arbitrage/solana_engine.py:297`; lift to `trading/chains/solana/jito_bundle.py` and wire as a fourth path alongside JupiterHelper).
- Pump predictor wiring needs a per-token on-line price history buffer; not built yet.
- `pump_predictor.py:170-215` scaler `fit_transform` over full dataset = standardization leakage (independent of the "look-ahead" label characterization in P1-08, which on re-read is a legitimate next-bar binary label, not X→y leakage).

## See also
- Phase 1 audit reports: `docs/agents/reports/SOLANA_*.md` (smartcontract / quant / analyst).
- Wave-2 campaign report: `docs/agents/reports/SOLANA_CAMPAIGN.md`.
- Canonical engine API: `docs/engines.md`.
