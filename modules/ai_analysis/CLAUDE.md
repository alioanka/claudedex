# AI Module
## What it does
LLM-driven sentiment + news analysis pipeline. Produces directional signals from CryptoCompare headlines via OpenAI / Anthropic, optionally executes Binance Futures trades through the canonical futures executor.
## Entry point
`modules/ai_analysis/main_ai.py` — launched as a subprocess by `main.py` when `AI_MODULE_ENABLED=true`. Engine: `modules/ai_analysis/core/sentiment_engine.py` (orchestrates `ai_provider.py` and `ai_trading_engine.py`).
## Key config (DB-backed via `ConfigManager`)
- `ai_provider` — `openai` / `anthropic` / `both`; selects LLM backend
- `confidence_threshold` — minimum LLM confidence to act (default 0.5)
- `trade_amount_usd` — notional per AI-triggered trade (default 50.0)
- `direct_trading` — master enable for live execution (default False; shadow only)
- `take_profit_pct` / `stop_loss_pct` — AI-position bounds (defaults +5 / -3)
- `max_hold_hours` — auto-close timer (default 24h)
- `scaler_path` — currently hardcoded reference in `trading/strategies/ai_strategy.py` (MB-19 follow-up: persist fitted scaler instead of live `fit_transform`)
- `quorum_required` (A6 E1) — when true AND both keys loaded, require both providers to agree on direction + |delta|; failure collapses signal to 0 (default false)
- `quorum_max_disagreement` (A6 E1) — max |s_openai - s_claude| permitted (default 0.4)
- `bandit_enabled` (A6 E3) — toggle prompt-template multi-armed bandit (default false)
- `bandit_epsilon` (A6 E3) — exploration rate for bandit (default 0.1)
- `ai_calibrated_predictions_enabled` (AI-Q-05, Wave 4) — when true, `EnsemblePredictor._predict_from_features` routes the 6 tree-based base classifiers (xgboost_{rug,pump}, lightgbm_{rug,pump}, random_forest, gradient_boosting) through `CalibratedClassifierCV` wrappers loaded from `<model_dir>/calibrated_<name>.pkl` (deliverable wording) or `<name>_calibrated.{pkl,joblib}` (AI-Q-06 trainer naming). Operator flips after first calibration-sample artefact lands. Default FALSE — base booster scores unchanged at default.
## Diagnostics
- `GET /api/ai/diagnostics?hours=24` — read-only "why no trades?" snapshot. Returns `signals_generated` (sentiment_logs in window), `trades_opened` (ai_trades.entry_timestamp in window), `action_rate` (trades/signals), buy/sell/hold split @ effective `confidence_threshold`, `signals_rejected_by_reason` (Counter parsed from the `[ai-skip] reason=<gate>` lines in `logs/ai_analysis/ai.log` — bounded 512KB tail), `recent_skips` (last 20 parsed events), `effective_config` (redacted snapshot of `config_settings` for `ai_config` — no API keys), and `hint` (operator-facing one-liner collapsing the most common config mistakes). Dashboard collapsible panel on `/ai/dashboard` consumes this.
- `[ai-skip]` log convention — every signal that's generated but does NOT open a position routes through `SentimentEngine._record_skip(reason, score, sentiment, symbol, extra)` which emits a fixed-key line:
  `[ai-skip] reason=<enum> conf=<n> sentiment=<+/-n> [symbol=X] [k1=v1 k2=v2]`
  Greppable via `grep -P '\[ai-skip\]\s+reason=\w+' logs/ai_analysis/ai.log`. Known reason enum values: `direct_trading_off` (#1 cause of 0-trades — operator never flipped ON), `confidence_below_threshold`, `zero_sentiment`, `position_exists`, `cooldown_active`, `risk_rejected`, `exchange_unavailable`, `execution_failed`.
## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_ai` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Effect: `should_skip_live` returns `True` -> `AITradeExecutor` returns simulated fill.
## Logs
`logs/ai_analysis/` — main, errors, trades (rotating handler).
## Primary risk-policy gate
`core.risk_manager.RiskManager.validate_trade(symbol, amount_usd)` called in `sentiment_engine.py` immediately before submitting to the canonical `BinanceFuturesExecutor` (search for `# Phase 2 #5 wired self.risk_manager`).
## Live-trade readiness
AMBER → GREEN candidate (pending production verification). MB-19 (load-or-refuse scaler + `ai_feature_store` + training scripts for scaler/rug/pump + 27-feature canonical layout + outcome-backfill on close) end-to-end closed; MB-20 (executor delegation through canonical futures path) and MB-21 (headline sanitisation) closed; secrets_manager already wired. Latch-at-open refinement (`60a3235`) makes re-entry feature-row backfill correct. **A6 wave-2:** multi-provider quorum gate (E1), confidence-calibration table + `/api/ai/calibration` (E2), pinned-template Q-learning bandit + `ai_feature_store` exploration logs (E3) — all disabled-by-default, see `docs/agents/reports/AI_CAMPAIGN.md`. **A6 wave-4:** AI-Q-05 inference-side calibrated booster wrap (`EnsemblePredictor.fit_and_persist_calibration` + `calibrated_predict_proba` + `_load_calibrated_models`, flag `ai_calibrated_predictions_enabled` default FALSE) — commits `16b7dab` + `68b20fb` + `a6c3a89`; quorum observability (`_record_quorum_outcome`/`_persist_quorum_outcome` -> `ai_feature_store.metadata.quorum_outcome` + `GET /api/ai/quorum-metrics?hours=24` + dashboard agreement-rate chart on `dashboard_ai.html`) — commits `c7e4a27` + `50bd9c3`. **A6 wave-5:** "Why no trades?" signal-to-trade diagnostic — `SentimentEngine._record_skip()` emits `[ai-skip] reason=<gate> conf=<n> sentiment=<n>` lines for every rejected signal (`direct_trading_off` / `confidence_below_threshold` / `zero_sentiment` / `position_exists` / `cooldown_active` / `risk_rejected` / `exchange_unavailable` / `execution_failed`); `GET /api/ai/diagnostics?hours=24` joins sentiment_logs + ai_trades + the [ai-skip] log tail (bounded 512KB read) and returns `signals_generated` / `trades_opened` / `action_rate` / buy-sell-hold split / `signals_rejected_by_reason` Counter / `recent_skips` (last 20) / `effective_config` (REDACTED — no API keys) / operator-facing `hint` string; dashboard `/ai/dashboard` "Why no trades?" collapsible panel with color-coded status pill (green firing / red no-trades / amber action-needed / grey idle). Quorum auto-pass (`_quorum_singleprov_logged`): when `quorum_required=true` but only one provider key is loaded, 1/1 agreement is trivially satisfied — single-provider score passes through unchanged with one-time startup log line. Commits `28efc1a` (skip ledger) + `71988d2` (diagnostics endpoint) + `c64890c` (dashboard panel) + `e271ad1` (quorum auto-pass).
## See also
- Phase 1 audit reports: `docs/agents/reports/AI_*.md` (quant / analyst / backend).
- Canonical engine API: `docs/engines.md`.
