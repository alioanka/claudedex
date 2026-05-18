# PORTFOLIO_ALLOCATOR Module

## What it does
Computes per-module capital allocation as a percentage of total book,
based on each module's rolling risk-adjusted return. Writes proposals
to `portfolio_allocations` table; operator approves via dashboard.
Never trades.

## Entry point
`modules/portfolio_allocator/main_portfolio_allocator.py` — subprocess
launched by `main.py` when `PORTFOLIO_ALLOCATOR_MODULE_ENABLED=true`.

## Key config (env, will move to DB in follow-up)
- `PORTFOLIO_ALLOCATOR_TICK_INTERVAL` — proposal cadence in seconds
  (default 3600 = hourly)
- `PORTFOLIO_ALLOCATOR_LOOKBACK_HOURS` — performance window for the
  Sharpe input (default 168 = 7 days)
- `PORTFOLIO_TOTAL_BOOK_USD` — total capital under management
  (default 1000 — operator should set this in .env)

## Allocation method
Modified fractional Kelly:
1. For each ENABLED module, compute Sharpe over the lookback.
2. raw_f = max(0, sharpe / max_sharpe_clip)  where max_sharpe_clip=2
3. Clip per-module floor = 5% / ceiling = 40%
4. Reserve = 10% of book always uninvested (gas + emergencies)
5. Normalize so sum of allocations + reserve = 100%

## Kill switch
- Global: `logs/.killswitch` (writes paused)
- Per-module: `logs/.pause_portfolio_allocator`

## Logs
`logs/portfolio_allocator/`

## Live-trade readiness
N/A — never trades. Proposals are advisory until approved.

## See also
- `docs/PHASE4_BACKTEST_ALLOCATION_BREAKER.md`
- `migrations/020_portfolio_allocations.sql`
