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
- `min_trades_for_score` — closed-trade floor below which a module gets a
  `not_ready` 'hold' recommendation (NOT silence). Env knob
  `ORCHESTRATOR_MIN_TRADES_FOR_SCORE`, default 5. (Dashboard agent: surface
  this on the orchestrator settings page.)
- `recommendation_ttl_minutes` — pending recs older than this are auto-superseded (default 60)

## Scoring -> recommendation flow (read before touching `run_tick`)
One tick (`orchestrator_engine.run_tick`, called every `tick_interval_seconds`):
1. Run the daily-loss **circuit breaker** first (`circuit_breaker.check_all_modules`).
2. `_supersede_stale_pending` — expire pending rows older than the TTL.
3. For each of the 7 modules in `_MODULE_QUERIES`
   (sniper / arbitrage / copy_trading / futures / solana / dex / ai):
   a. `_collect_module_inputs` — 2 DB reads against that module's
      `*_trades` table: aggregate counts/pnl + a 500-row pnl series for
      Sharpe. Returns `None` only on DB error (module then skipped).
   b. `modules_scored += 1`.
   c. If `closed_trades < min_trades_for_score` -> build a `not_ready`
      'hold' `ModuleScore` (confidence 0, reason "staying DRY_RUN,
      insufficient data"). Otherwise `performance_scorer.score_module`
      produces one of `to_live` / `to_dry` / `disable` / `enable` /
      `hold`, optionally ML-calibrated by `data/orchestrator_ai_model.pkl`.
   d. `_supersede_module_pending(module)` then `_insert_recommendation` —
      every scored module gets **exactly one current** pending row per
      tick; the prior pending row for that module is superseded so the
      table stays bounded to one live row per module (full audit history
      via `superseded_at`).

### Issue 20 fix (Wave-7) — all 7 modules now surface a recommendation
Symptom: `modules_scored=7` but only SNIPER + SOLANA produced
recommendations. Root cause was two silent gates in the old `run_tick`:
(1) `if closed_trades < 5: continue` dropped under-traded modules with no
row at all, and (2) 'hold' verdicts were de-duped to once-per-24h, so a
module that returned 'hold' went dark after its first row. SNIPER/SOLANA
only surfaced because they had enough DRY_RUN volume to clear the
`to_live` thresholds (`_MIN_TRADES_FOR_LIVE_RECOMMENDATION=100`,
`win_rate>=0.55`, `pnl>=$10`, `live_trades==0`). Now: insufficient-data
modules emit a `not_ready` 'hold', and the per-tick supersede-then-insert
guarantees one current row per module — the operator SEES a per-module
recommendation (even "stay DRY_RUN / not ready") instead of silence.

## Positions / execution wallet (issues 6 + 15)
This is a **meta-module**. It holds NO positions of its own and places NO
trades — it only writes advisory rows. There is therefore:
- **No close path / no `close_position` / no `get_positions`.** Dashboard
  agent: do NOT render a close button for the orchestrator; there is
  nothing to close.
- **No execution wallet** (no EVM key, no Solana keypair). It signs
  nothing. Funding is N/A.

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
