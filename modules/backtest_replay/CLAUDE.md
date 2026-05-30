# BACKTEST_REPLAY Module

## What it does
Replays historical trades + orchestrator recommendations under a
chosen "what if I'd done X" strategy, returns counterfactual P&L per
module. Pure compute — never broadcasts, never modifies live state.

## Entry point
**No subprocess.** The replay engine is invoked synchronously via
the dashboard's `POST /api/backtest/replay` handler, which calls
`core.replay_engine.run_replay()`.

## Key inputs
- `start_ts` / `end_ts` — date range to replay (default: last 30 days)
- `strategy` — `approve_all` / `approve_on_confidence` / `never_approve`
- `confidence_threshold` — only used when strategy=approve_on_confidence
- `modules` — optional filter (default: all enabled modules)

## Key outputs
Per module:
- `counterfactual_pnl_usd` — what would have happened with the strategy
- `actual_pnl_usd` — what actually happened (DRY_RUN sim or live, whichever)
- `n_recs_triggered` — how many recommendations the orchestrator emitted
- `n_recs_approved` — how many the strategy would have approved
- `max_drawdown_pct` — worst drawdown over the window
- `sharpe` — per-trade Sharpe over the window

## Kill switch
N/A — this module never writes anything destructive. The endpoint is
read-only over `*_trades` + `orchestrator_recommendations`.

## Logs
Inline via the dashboard logger. No separate log dir.

## Live-trade readiness
N/A — never trades.

## See also
- `docs/PHASE4_BACKTEST_ALLOCATION_BREAKER.md`
