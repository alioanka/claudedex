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
- `strategy_params.live_haircut_usd_per_trade` — Wave-F7 live-window cost
  model (default **0.5** USD/trade): trades falling inside the simulated
  "would have been live" window subtract this flat per-trade estimate of the
  extra live costs DRY_RUN never pays. 0 restores the old cost-free
  counterfactual; the report surfaces `n_live_window_trades` +
  `live_haircut_usd` so zero-haircut runs are visibly optimistic. Flat-USD
  because `TradeRow` carries no notional; a bps-of-notional model is the
  documented follow-up.

## Key outputs
Per module:
- `counterfactual_pnl_usd` — what would have happened with the strategy
- `actual_pnl_usd` — what actually happened (DRY_RUN sim or live, whichever)
- `n_recs_triggered` — how many recommendations the orchestrator emitted
- `n_recs_approved` — how many the strategy would have approved
- `max_drawdown_pct` — worst drawdown over the window
- `sharpe` — per-trade Sharpe over the window

## Wave-F7 honesty fixes (2026-07-10)
External-audit row "Backtest/replay: dry-run PnL understates fees" closed as
far as the data allows, plus a latent defect: `flip_ts` (the simulated
live-flip timestamp) was computed but **never used**, so the counterfactual
always equaled actual (`pnl_delta_usd` ≡ 0 for every strategy). Now the live
window applies the per-trade haircut above. `trade_loader.load_trades` also
drops `metadata.excluded=true` rows (poisoned Solana history mig 140A,
arbitrage triangular phantom fills mig 150A) — a counterfactual built on
fabricated PnL is worthless. Remaining build-out (roadmap, per the audit):
deterministic event replay, bps-of-notional fee/gas/funding model, latency,
partial fills, walk-forward splits.

## Kill switch
N/A — this module never writes anything destructive. The endpoint is
read-only over `*_trades` + `orchestrator_recommendations`.

## Logs
Inline via the dashboard logger. No separate log dir.

## Live-trade readiness
N/A — never trades.

## See also
- `docs/PHASE4_BACKTEST_ALLOCATION_BREAKER.md`
