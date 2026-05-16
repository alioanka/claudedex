# AI Module
## What it does
LLM-driven sentiment + news analysis pipeline. Produces directional signals from CryptoCompare headlines via OpenAI / Anthropic, optionally executes Binance Futures trades through the canonical futures executor.
## Entry point
`modules/ai_analysis/main_ai.py` — launched as a subprocess by `main.py` when `AI_MODULE_ENABLED=true`. Engine: `modules/ai_analysis/core/sentiment_engine.py` (orchestrates `ai_provider.py` and `ai_trading_engine.py`).
## Key config (DB-backed via `ConfigManager`)
- `ai_provider` — `openai` / `anthropic`; selects LLM backend
- `confidence_threshold` — minimum LLM confidence to act (default 0.5)
- `trade_amount_usd` — notional per AI-triggered trade (default 50.0)
- `direct_trading` — master enable for live execution (default False; shadow only)
- `take_profit_pct` / `stop_loss_pct` — AI-position bounds (defaults +5 / -3)
- `max_hold_hours` — auto-close timer (default 24h)
- `scaler_path` — currently hardcoded reference in `trading/strategies/ai_strategy.py` (MB-19 follow-up: persist fitted scaler instead of live `fit_transform`)
## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_ai` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Effect: `should_skip_live` returns `True` -> `AITradeExecutor` returns simulated fill.
## Logs
`logs/ai_analysis/` — main, errors, trades (rotating handler).
## Primary risk-policy gate
`core.risk_manager.RiskManager.validate_trade(symbol, amount_usd)` called in `sentiment_engine.py` immediately before submitting to the canonical `BinanceFuturesExecutor` (search for `# Phase 2 #5 wired self.risk_manager`).
## Live-trade readiness
AMBER → GREEN candidate (pending production verification). MB-19 (load-or-refuse scaler + `ai_feature_store` + training scripts for scaler/rug/pump + 27-feature canonical layout + outcome-backfill on close) end-to-end closed; MB-20 (executor delegation through canonical futures path) and MB-21 (headline sanitisation) closed; secrets_manager already wired. Latch-at-open refinement (`60a3235`) makes re-entry feature-row backfill correct.
## See also
- Phase 1 audit reports: `docs/agents/reports/AI_*.md` (quant / analyst / backend).
- Canonical engine API: `docs/engines.md`.
