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
