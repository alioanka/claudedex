# Phase 4 — Backtest Replay + Capital Allocation + Circuit Breaker

Three deliverables, in order. Each builds on the previous.

---

## 4A. Backtest replay engine

### Goal
Given the full history of orchestrator recommendations + per-module
trades, answer: **"What would my P&L have been if I'd auto-approved
every `to_live` recommendation when it fired?"**

This is the missing decision tool before flipping anything live. Without
it, "should I flip sniper live?" is a vibe; with it, it's a number with
a confidence interval.

### Architecture
```
modules/backtest_replay/
├── __init__.py
├── CLAUDE.md
├── core/
│   ├── replay_engine.py   — main simulator
│   ├── trade_loader.py    — read *_trades within a date range
│   └── strategies.py      — replay strategies (approve-all / approve-on-confidence / never)
└── (no subprocess; invoked via dashboard POST /api/backtest/replay)
```

### Inputs
- Date range: `start_ts`, `end_ts` (defaults: last 30 days)
- Replay strategy: `approve_all` / `approve_on_confidence` / `never_approve`
- Per-module: starting capital, max position size
- Per-module DRY_RUN state at start of window

### Output (JSON + chart)
For each module:
- counterfactual_pnl_usd
- vs actual DRY_RUN pnl
- # recommendations triggered
- # approved (by strategy)
- max drawdown during replay window
- Sharpe over replay window

### Endpoints
```
POST /api/backtest/replay
     {start_ts, end_ts, strategy, modules: [...]}
     → {success, summary, per_module: [...], series}

GET  /api/backtest/replay/{id}   (optional persistence)
```

### Page
`/backtest-replay` — operator picks date range + strategy, clicks
"Run", sees summary table + per-module sparkline chart.

---

## 4B. Per-module capital allocation

### Goal
Replace fixed YAML allocations (`config/modules/*.yaml`) with dynamic
allocation based on each module's rolling Sharpe and drawdown.

### Architecture
```
modules/portfolio_allocator/
├── __init__.py
├── CLAUDE.md
├── core/
│   ├── allocator.py        — fractional-Kelly + floor/ceiling caps
│   └── rebalance_engine.py — runs on tick, writes proposals
└── main_portfolio_allocator.py
```

Subprocess driven by `PORTFOLIO_ALLOCATOR_MODULE_ENABLED=true`.

### Allocation method
Modified fractional Kelly:
- `f* = (sharpe - threshold) / (max_sharpe - threshold)` clipped to `[0, kelly_cap]`
- Module floor: 5% of book per ENABLED module (so a brief drawdown
  doesn't fully deallocate)
- Module ceiling: 40% of book per module (no single-module risk)
- Reserve: 10% of book always uninvested (gas + emergencies)

### DB schema
New table `portfolio_allocations` — one row per (module, ts):
- pct_of_book
- usd_amount (computed from total book at allocation time)
- proposed_by ('allocator' / 'operator')
- approved_at / approved_by
- effective_until

### Endpoints
```
GET  /api/portfolio/allocations            current breakdown
POST /api/portfolio/allocations/propose    trigger a recompute
POST /api/portfolio/allocations/{id}/approve
GET  /api/portfolio/allocations/history?days=30
```

### Page
`/allocation` — pie chart of current breakdown + table with proposed
delta + approve buttons.

---

## 4C. Per-module daily-loss circuit breaker

### Goal
If a module loses > N% in 24h, auto-flip it back to DRY_RUN without
waiting for operator approval. (Waiting for approval during a
drawdown is the wrong time.)

### Architecture
A loop inside `orchestrator_engine.py` (no new subprocess — the
orchestrator already polls every 5 min, this just adds another check
to that tick).

### Logic
```
For each module in LIVE mode:
    pnl_24h = SUM(profit_loss) WHERE NOT is_simulated AND time > NOW() - 24h
    capital = config_settings.<module>.capital_allocated
    pct_loss = pnl_24h / capital * 100
    if pct_loss < -<threshold>:
        UPDATE config_settings.<module>.dry_run = true
        drop logs/.restart_<module>
        INSERT orchestrator_recommendations (recommended='to_dry',
                                              confidence=1.0,
                                              reason='circuit breaker tripped: ...')
```

### DB schema
- New row per module: `config_settings.<m>_config.daily_loss_circuit_breaker_pct`
  (default 5.0%)

### Dashboard surface
- Add a "Circuit Breaker" column to `/orchestrator` trend table
- Big yellow banner at top of `/orchestrator` when any breaker has
  tripped in the last 24h

---

## Commit plan (≤200 LoC each, small commits to avoid timeouts)

### 4A — Backtest replay
- 4A-1 [docs] Phase 4 roadmap (this file)
- 4A-2 backtest_replay package scaffold + CLAUDE.md
- 4A-3 trade_loader.py — read *_trades across date range
- 4A-4 strategies.py — approve-all / approve-on-confidence
- 4A-5 replay_engine.py — main simulator (no DB writes, pure compute)
- 4A-6 dashboard POST /api/backtest/replay
- 4A-7 /backtest-replay page
- 4A-8 Test Runner catalog entries
- 4A-9 unit tests

### 4B — Capital allocation
- 4B-1 portfolio_allocator package
- 4B-2 allocator.py — fractional Kelly logic
- 4B-3 migration: portfolio_allocations table
- 4B-4 rebalance_engine.py + main subprocess
- 4B-5 dashboard endpoints
- 4B-6 /allocation page
- 4B-7 Test Runner entries

### 4C — Circuit breaker
- 4C-1 migration: circuit_breaker_pct rows seeded
- 4C-2 circuit_breaker.py inside orchestrator
- 4C-3 dashboard banner + trend-table column
- 4C-4 Test Runner entries

Total estimate: 22-25 small commits.

---

## Definition of done

- Backtest replay: operator picks "last 30 days, approve_all" and
  sees per-module counterfactual P&L within 10 seconds.
- Capital allocation: operator approves a proposal, allocations are
  written to DB, next module restart reads the new size, all
  visible on `/allocation`.
- Circuit breaker: artificial loss injected into a LIVE module
  triggers auto-flip-to-DRY within one orchestrator tick (5 min).
- All three covered by Test Runner probes (D section for DB checks,
  C section for API probes, B section for the replay script).

---

## Shipped — May 2026 (~22 commits, 959ebe4..5e50253)

### 4A — Backtest replay
- `modules/backtest_replay/`:
  - `core/trade_loader.py` — schema-aware reader over 5 *_trades tables + orchestrator_recommendations
  - `core/strategies.py` — 4 strategies (approve_all / approve_on_confidence / never_approve / operator_replay) + registry
  - `core/replay_engine.py` — pure-function counterfactual simulator with Sharpe + max DD + sampled equity curve
- `POST /api/backtest/replay`, `GET /api/backtest/strategies`
- `/backtest-replay` page (date pickers, strategy dropdown, summary cards, per-module delta table with unicode sparklines)
- 16 unit tests in `tests/unit/test_backtest_replay.py`

### 4B — Capital allocation
- `migrations/020_portfolio_allocations.sql` — proposals + approvals + audit
- `modules/portfolio_allocator/`:
  - `core/allocator.py` — modified-Kelly with floor (5%)/ceiling (40%)/reserve (10%)
  - `core/rebalance_engine.py` — collect Sharpe inputs, propose, persist
  - `main_portfolio_allocator.py` — subprocess wrapped by `PORTFOLIO_ALLOCATOR_MODULE_ENABLED`
- `GET /api/portfolio/allocations[/current]`, `POST /api/portfolio/allocations/propose`, `POST /api/portfolio/allocations/{id}/approve`
- `/allocation` page (stacked bar with reserve, pending table with override input, full history)
- 11 unit tests in `tests/unit/test_portfolio_allocator.py`

### 4C — Daily-loss circuit breaker
- `migrations/021_seed_daily_loss_breakers.sql` — per-module threshold rows + `circuit_breaker_events` audit table
- `modules/orchestrator_ai/core/circuit_breaker.py` — runs as a step inside `orchestrator_engine.run_tick`
- 4 side-effects on trip: flip DB dry_run, audit row, recommendation row, restart flag
- `GET /api/circuit-breaker/{events,active}`, `POST /api/circuit-breaker/{id}/clear`
- Global amber banner in `base.html` — visible on every dashboard page

### Total artifact count this Phase
- 4 migrations (018-021)
- 3 new modules (backtest_replay, portfolio_allocator, orchestrator_ai already existed)
- 3 new pages (backtest_replay, allocation; orchestrator already existed)
- 27 unit tests (16 backtest + 11 alloc; orchestrator scorer's 15 already existed)
- 13 new Test Runner catalog entries (4A: 1, 4B: 4, 4C: 4, plus 4 from earlier orchestrator history work)
