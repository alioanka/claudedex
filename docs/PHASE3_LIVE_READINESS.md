# Phase 3 — Per-Module DRY_RUN ↔ LIVE + AI Orchestrator

Roadmap doc for the May-Jun 2026 push toward live trading. Tracks
work-in-flight; updated as commits land.

---

## Goal

Three deliverables, in order:

1. **All modules collecting accurate DRY_RUN data** so the operator can
   compare per-module performance before flipping any to live.
2. **Per-module DRY_RUN ↔ LIVE toggle** — flip individual modules to
   live trading while others continue paper-trading. Process-wide
   `DRY_RUN` stays as the default; per-module override (env or DB)
   wins.
3. **AI/ML orchestrator** — read each module's DRY_RUN performance +
   live market state → recommend toggle actions (enable / disable /
   dry / live). Advisory-only initially; operator approves each
   action via dashboard.

---

## Architecture

### Precedence for "is this module live?"

```
1. logs/.killswitch present       → ALL modules forced to DRY_RUN
2. logs/.pause_<module> present   → that module DRY_RUN
3. config_settings.<module>_config.dry_run  (DB)  → if set, wins
4. <MODULE>_DRY_RUN env var       → if set, wins over global
5. DRY_RUN env var (process-wide) → fallback default
6. hardcoded default True         → safe-by-default
```

### New helper

`core.dry_run.resolve_module_dry_run(module, *, db=None) -> bool`

- Encapsulates the precedence above
- Cheap to call (one stat() + one env read + one optional DB query)
- All `main_<module>.py` entry points use this instead of raw env reads

### DB schema

`config_settings` already exists; add `dry_run` key per module:

```sql
INSERT INTO config_settings (config_type, key, value, value_type)
VALUES ('sniper_config', 'dry_run', 'true', 'bool')
ON CONFLICT DO NOTHING;
```

Dashboard `/api/modules/{module}/dry-run` POST writes this row.

### AI/ML orchestrator (new module)

`modules/orchestrator_ai/` — runs as its own subprocess. Each tick:

1. Read `*_runtime_stats` + closed `*_trades` rows from each enabled module.
2. Compute per-module score:
   - Sharpe over rolling 7d window
   - P&L vs gas spent ratio
   - Win-rate trend (last 100 vs prior 100)
   - Detection-latency p95 (sniper only)
3. Read market state (BTC + ETH 24h delta from CoinGecko cache).
4. Score modules; emit recommendations to `orchestrator_recommendations`:
   - `recommended_action`: `enable` / `disable` / `to_dry` / `to_live`
   - `confidence`: 0..1
   - `reason`: short string
5. Dashboard surface: `/orchestrator` page shows pending recs;
   operator clicks "Approve" → flip the module's enable / dry_run flag.

Initially the orchestrator does NOT auto-act. After 30 days of
operator-approval history we revisit auto-action gated on
`confidence > 0.85`.

---

## Commit plan

Small commits (≤200 LoC each), pushed individually so the operator
can pull incrementally:

### A. Per-module DRY_RUN override
- A1 `[backend] core.dry_run: resolve_module_dry_run() helper` ← in flight
- A2 `[backend] main_sniper: use resolve_module_dry_run`
- A3 `[backend] main_arbitrage: use resolve_module_dry_run`
- A4 `[backend] main_copy: use resolve_module_dry_run`
- A5 `[backend] main_futures: use resolve_module_dry_run`
- A6 `[backend] main_solana: use resolve_module_dry_run`
- A7 `[backend] main_dex: use resolve_module_dry_run`
- A8 `[backend] main_ai: use resolve_module_dry_run`

### B. Dashboard surface for per-module dry_run
- B1 `[backend] POST /api/modules/{module}/dry-run`
- B2 `[backend] settings UI toggle on each module page`
- B3 `[backend] /api/modules now includes effective_dry_run per row`

### C. Test Runner additions
- C1 `[backend] test_runner: per-module data-collection readiness probes`
- C2 `[backend] test_runner: per-module trade-count + open-position probes`
- C3 `[backend] test_runner: per-module config_settings probes`
- C4 `[backend] test_runner: module-enable readiness (env + creds)`
- C5 `[backend] test_runner: AI orchestrator readiness probes`

### D. AI/ML orchestrator scaffolding
- D1 `[backend] modules/orchestrator_ai/: package scaffold`
- D2 `[backend] orchestrator_recommendations table + migration`
- D3 `[backend] performance scorer per module`
- D4 `[backend] market state collector (CoinGecko cache reuse)`
- D5 `[backend] /orchestrator dashboard page (advisory view)`
- D6 `[backend] approve-recommendation POST handler`

### E. Module enable-all readiness audit
- E1 `[docs] per-module DRY_RUN persistence audit` (this commits a
      checklist of each module's DRY_RUN gate + trade-table write
      paths so the operator knows what they're enabling)

---

## Definition of done

- All 7 trading modules enabled with `<MODULE>_MODULE_ENABLED=true`
  and `<MODULE>_DRY_RUN=true` flip independently of the global
  `DRY_RUN=true`.
- Each module persists trades to its `*_trades` table with
  `is_simulated=true` (or equivalent flag).
- /test-runner Section D probes confirm trade counts are growing
  for every enabled module.
- /orchestrator page shows non-zero per-module scores after 24h of
  data collection.
- Operator can flip a single module to LIVE via dashboard without
  affecting other modules.

---

## Shipped — May 2026 (25 commits, f9c43e8..c1c07c0)

### A. Per-module DRY_RUN
- `core.dry_run.resolve_module_dry_run(module, db_row_value=...)` —
  precedence DB > `<MODULE>_DRY_RUN` env > `DRY_RUN` env > default True.
- All 7 `main_*.py` entry points use it.
- `<MODULE>_DRY_RUN` env override works for every module.

### B. Dashboard surface
- `GET/POST /api/modules/<m>/dry-run` — read + flip DB row.
- `POST /api/modules/<m>/restart` — drop logs/.restart_<m> flag;
  main.py picks up within 5s and restarts the subprocess.
- `/api/modules` includes `effective_dry_run` per module.
- `/full-dashboard` Module Overview shows DRY/LIVE chip per module.

### C. Test Runner additions
- 13 new catalog entries covering per-module DRY_RUN flags,
  trade-count growth, P&L split simulated-vs-live, orchestrator
  table presence, pending-rec count, history endpoint, ML training-
  data view + 2 trainer entries in B-section.

### D. AI orchestrator (advisory, end-to-end)
- `migrations/018_orchestrator_recommendations.sql` + `019_orchestrator_labels.sql`.
- `modules/orchestrator_ai/`:
  - `core/performance_scorer.py` — 5-signal weighted score (win-rate,
    P&L, Sharpe, volume_factor, regime_signal). 15 unit tests pass.
  - `core/orchestrator_engine.py` — collect → score → write recs.
    Reads per-trade pnls (cap 500/module) so Sharpe fires on real
    data. Optional ML calibration on confidence when the trained
    model is on disk.
  - `core/market_state.py` — CoinGecko BTC/ETH 24h cache with
    stale-but-existing fallback on network failure.
  - `core/ml_trainer.py` — sklearn-free logistic regression on
    operator-approval labels; writes `data/orchestrator_ai_model.pkl`
    + JSON sidecar. Skip-with-reason when <30 examples.
  - `main_orchestrator_ai.py` — subprocess entry; cadence configurable
    via `ORCHESTRATOR_TICK_INTERVAL`/`LOOKBACK_HOURS`/`REC_TTL_MINUTES`.
- `main.py` registers the orchestrator subprocess under
  `ORCHESTRATOR_AI_MODULE_ENABLED`.
- Dashboard:
  - `GET /api/orchestrator/recommendations` (filterable by status/module).
  - `GET /api/orchestrator/history?hours=N` (per-module score timeseries).
  - `POST /api/orchestrator/recommendations/{id}/{approve|reject}` —
    approve flips the DB row + drops the restart flag in one click.
  - `/orchestrator` page: status tabs, rec cards with approve/reject,
    score-trend table with unicode sparklines, auto-refresh every 60s.

### Operator workflow (end-to-end)

1. Enable all modules in DRY_RUN via .env.
2. Wait ~5-10 min for first orchestrator tick. Recs land in the table.
3. Visit `/orchestrator` — see "Sniper → to_live, conf=0.78".
4. Click Approve. Dashboard writes config_settings + drops restart flag.
5. main.py picks up flag within 5s, restarts sniper subprocess.
6. Sniper re-bootstraps with `dry_run=false` from DB row, starts
   trading live. Total wall-clock click → live: ~10-15s.
7. After 30 days of approval/reject decisions, run
   `python -m modules.orchestrator_ai.core.ml_trainer` (or click the
   Test Runner button). Engine auto-loads the pkl on next tick and
   uses it to calibrate confidence on future recs.
