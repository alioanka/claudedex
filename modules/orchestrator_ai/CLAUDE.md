# ORCHESTRATOR_AI Module

## What it does
Reads each trading module's recent DRY_RUN performance + market state,
computes a per-module score, and writes recommendations to the
`orchestrator_recommendations` DB table. The dashboard surfaces pending
recommendations; the operator approves each one explicitly. Approval
flips the corresponding `config_settings.<module>_config.dry_run` row
(or `_MODULE_ENABLED` env).

## Entry point
`modules/orchestrator_ai/main_orchestrator_ai.py` (added when wired
into `main.py`). Engine: `core/orchestrator_engine.py`. Scoring:
`core/performance_scorer.py`. Market state collector reuses the existing
CoinGecko cache from `monitoring/enhanced_dashboard.py:_get_sol_usd_price`.

## Key config (DB-backed via `ConfigManager`)
- `tick_interval_seconds` — scoring tick cadence (default 300 = 5 min)
- `lookback_hours` — performance window for trade aggregation (default 24)
- `min_trades_for_score` — minimum closed trades before a score is emitted (default 30)
- `recommendation_ttl_minutes` — pending recs older than this are auto-superseded (default 60)

## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or
  `/api/bot/emergency-exit`).
- Per-module: `logs/.pause_orchestrator_ai`.
- Effect: poller stops emitting new recommendations; existing rows
  remain in the table for operator audit.

## Logs
`logs/orchestrator_ai/` — main, errors (rotating handler).

## Live-trade readiness
N/A — this module never trades. It only writes advisory rows.
After 30+ days of operator-approval history we revisit auto-action
gated on `confidence > 0.85`.

## See also
- `docs/PHASE3_LIVE_READINESS.md` — roadmap doc
- `migrations/018_orchestrator_recommendations.sql` — table schema
