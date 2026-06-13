# ClaudeDex Operations Guide

Day-to-day operation of a deployed stack. Assumes `docs/DEPLOYMENT_GUIDE.md`
has been completed. Per-module reference: `docs/MODULE_CATALOG.md`. Incident
procedures: `docs/runbook.md`. Dashboard internals: `docs/dashboards.md`.

## 1. Dashboard tour

`http://<host>:8080` (the only published port). Log in at `/login` — the
initial admin password is printed ONCE by `scripts/init_auth.py`.

| Page | What you do there |
|---|---|
| `/` | Module-status tiles + the emergency-stop button (top of every page) |
| `/modules` | Per-module enable / disable / pause / resume grid |
| `/positions`, `/trades` | Live open positions; cross-module trade history |
| `/performance`, `/analytics` | P&L charts, aggregated portfolio metrics |
| `/settings` (+ per-module settings pages) | Every DB knob; each module's template has a **Settings** tab (writes to `config_settings`) and a **Guide** tab (what each knob does) |
| Control center (`/api/control-center/overview`, `/api/performance/cross-module`, `/api/meta/decisions`) | Cross-module health + meta_controller decision stream |
| `/wallet-balances` | Per-chain on-chain balances |
| `/logs` | Live log tail, filterable by module/level |
| `/credentials` | Encrypted credential management (admin) |
| `/backtest` | Counterfactual replay (backtest_replay, read-only) |

Status badges: modules with a bound health server (see port map in root
`CLAUDE.md`) get live `/health` probes; probe-only modules (sniper, arbitrage,
ai, copy_trading) and the serverless meta modules (orchestrator_ai,
portfolio_allocator) show DB-heartbeat-based badges instead — a stale badge
there means "no recent heartbeat row", not necessarily "crashed". Confirm with
`/logs` before reacting.

## 2. Shadow-running a module and reading its panel

Standard pattern for anything new (and for every expansion-wave module, which
all default to shadow/advisory):

1. Enable it: `.env` flag + restart, or the `/modules` grid. Leave its DB
   `shadow_mode=true` / `DRY_RUN=true` defaults alone.
2. Let it run days, not hours. All output rows are written with
   `is_simulated=true` (or module-equivalent), so nothing it records is money.
3. Read its output where it lives:
   - Trading modules: `/positions`, `/trades`, per-module dashboard pages.
   - Advisor: `/api/advisor/performance` panel + `/api/advisor/simulations`.
   - Meta layers: control-center + `/api/meta/decisions`.
   - Expansion modules without a dashboard panel yet (tca, treasury,
     warehouse, catalyst, options_vol, yield, clmm, basis, stat_arb,
     smart_money, param_tuner): query their output tables directly
     (`docs/MODULE_CATALOG.md` lists them) and watch
     `logs/<module>/<module>_errors.log`.
4. Judge skeptically: simulated fills assume no market impact and optimistic
   liquidity. A module that looks marginal in shadow will be worse live.

## 3. Approving advisory proposals

Nothing advisory self-executes. Approval surfaces:

| Producer | Pending rows in | Approve via |
|---|---|---|
| orchestrator_ai | `orchestrator_recommendations` | `POST /api/orchestrator/recommendations/{id}/approve` (or `/reject`) |
| portfolio_allocator | `portfolio_allocations` | `POST /api/portfolio/allocations/{id}/approve` |
| meta_controller | `meta_decisions` | Informational stream (`/api/meta/decisions`); actuation only via its own autopilot flag |
| regime_allocator | `regime_allocation_proposals` (pending = `approved_at IS NULL AND superseded_at IS NULL`) | **No dashboard surface yet** — review/approve via SQL; optional `mirror_to_portfolio_allocations` knob (default false) |
| param_tuner | `param_proposals` (status `pending`) | **No dashboard surface yet** — apply by setting the proposed value in the module's Settings tab, then mark the row `operator_applied` via SQL |
| advisor | `advisor_advice` | Human reads advice; nothing to approve into execution |

Autopilots (all default OFF in DB, all limited to pause files only):
`meta_autopilot_enabled`, `sentinel_autopilot_enabled`, and param_tuner's
`auto_apply_enabled` (config values only, risk keys hard-excluded). Treat
flipping any of these as a change-control event: announce it, watch the
module's actions table (`sentinel_actions`, `param_proposals`,
`meta_decisions`) for the first week.

## 4. The safe path to LIVE: shadow → DRY_RUN → LIVE

Per module, never skipped, never reordered:

1. **Shadow / advisory** (default): module records simulated rows only.
   Exit criterion: ≥1-2 weeks of sane output, flat error log, plausible
   simulated economics after costs.
2. **DRY_RUN**: for trading modules, full signal + sizing + risk-gate pipeline
   with broadcast suppressed by `core/dry_run.should_skip_live`. Exit
   criterion: simulated P&L you would accept live, plus the pre-LIVE
   punch-list in `docs/DEPLOYMENT_GUIDE.md` section 9 (DEX
   `live_max_execute_retries=1`, SNIPER `safety_check_enabled=true`, dual-flag
   modules need BOTH `shadow_mode=false` AND `live_execution_enabled=true`).
3. **LIVE, smallest size**: flip the module's flag(s) in its Settings tab,
   with position-size knobs at minimum. Watch the first live fills against
   quotes (execution_quality's TCA tables exist for exactly this).
   Scale only after measured live costs match the shadow assumptions.

Reality check per module class:

- **Live-capable today**: dex, futures (entries suppressed by default —
  re-enabling is a strategy decision, not a flag flip), solana, sniper,
  arbitrage, ai (`direct_trading`), copy_trading, polymarket, stat_arb,
  options_vol (BUY legs only).
- **Cannot go live regardless of flags**: yield_treasury, clmm_lp, basis_desk
  (live paths intentionally not built — flags terminate in
  `live_path_not_implemented`), intent_solver (parked scaffold, zero live
  code), and all read-only modules. Do not interpret their shadow/live knobs
  as a promise.

## 5. Emergency stop, kill switch, pause

Three escalation levels, weakest first:

1. **Pause one module** — `/modules` grid pause button, which writes
   `logs/.pause_<module>`. Entries freeze; the module can still exit
   positions. Resume deletes the file. (Sentinel/meta autopilots use this
   same mechanism and clear only their own files.)
2. **Global kill switch** — `logs/.killswitch`, written by the dashboard
   emergency-stop button (`POST /api/bot/emergency-exit`, admin) or
   `python scripts/emergency_stop.py` from a shell (works even if the
   dashboard is down). Every module polls it via
   `core.dry_run.start_killswitch_poller`; all live broadcast paths refuse
   while it exists. Remove the file deliberately, after you understand why
   you set it.
3. **Stop the stack** — `docker compose stop trading-bot`. Note open exchange
   positions (futures, stat_arb live) remain open at the venue while the bot
   is down; prefer level 2 + manual flattening over a blind stop.

## 6. Per-module quick reference

Flag = `.env` enable; gates = what stands between it and money. Ports/tables:
`docs/MODULE_CATALOG.md`.

| Module | Day-to-day check | Live gate(s) | Watch out |
|---|---|---|---|
| dex_trading | `/dex/dashboard`, `logs/dex/` | DRY_RUN + killswitch/pause at broadcast + RiskManager fail-closed | Set `live_max_execute_retries=1` before LIVE |
| futures_trading | `/futures/*` pages | DRY_RUN + `FuturesRiskManager` + `entries_suppressed=true` default | Momentum stack audited unprofitable; don't re-enable entries without a redesign |
| solana_trading | `/solana/*` pages | DRY_RUN + RiskManager (entry-only); Drift/pump.fun sub-flags OFF | pnl clamps hide outliers — check raw `solana_trades` when curious |
| sniper | `/settings` sniper tab, `logs/sniper/` | DRY_RUN + safety filters; flip `safety_check_enabled=true` pre-LIVE | DRY_RUN exit P&L is modeled, not measured |
| arbitrage | `logs/arbitrage/`, near-miss log lines | `live_execution_enabled=false` + DRY_RUN + RiskManager + gas budget | Reverted tx = gas burnt; receipts confirmed before booking (mig 108) |
| ai_analysis | `/settings` AI tab, sentiment logs | `direct_trading=false` + DRY_RUN + RiskManager + LLM budget | LLM spend: check `core/llm_budget` consumption |
| copy_trading | copy panel, `copy_leader_scores` | BUY-only caps, probation, exposure cap, RiskManager | Leader scores carry survivorship bias; v3 discovery is shadow-first (mig 135) |
| polymarket | `polymarket_trades` (simulated) | dual flags + DRY_RUN + RiskManager | Binary resolution risk; Gamma rate limits stale the feed |
| stat_arb | `stat_arb_spread_state`, suggestions | dual flags + DRY_RUN + RiskManager + Bybit keys | Pair-break tail risk; hard z-stop is the only backstop — never widen it |
| advisor | `/api/advisor/*` panels | none (advice-only) | Sim caps per channel; confidence penalty on provider disagreement |
| dashboard | always-on | n/a | Admin auth + CSRF; HTTPS flag behind a proxy |
| meta_controller | `/api/meta/decisions` | autopilot OFF; pause files only | Calibration is retrospective — don't read it as forecast skill |
| orchestrator_ai | recommendations API | operator approval required | `not_ready` HOLD rows are normal for thin samples |
| portfolio_allocator | `portfolio_allocations` | operator approval required | Proposals only; nothing executes them |
| regime_allocator | `regime_snapshots` | advisory; mirror knob default false | Tick skips on dual kline-source failure |
| sentinel | `sentinel_anomalies` | autopilot OFF; own pause files only | futures/solana/copy lack heartbeats in v1 (silent-module blind spot) |
| param_tuner | `param_proposals` | proposals-only; `auto_apply_enabled=false` | Shadow rewards overfit — accept only out-of-sample winners |
| execution_quality | `tca_scorecards` | none (read-only) | Use it as the gate for any execution-touching change |
| treasury | `treasury_snapshots`, gas alerts in errors log | none (observe-only) | `gas_low` + open LIVE positions = exit-gas starvation, act fast |
| market_data_warehouse | `market_candles` row counts | none | Disk growth; retention knobs |
| catalyst_calendar | `catalysts` | none | Static macro list expires 2026-12-31 |
| options_vol | `options_vol_suggestions` | dual flags + DRY_RUN + RiskManager + premium budget; SELL never live | Premium caps are the budget — verify monthly counter |
| yield_treasury | `yield_treasury_advice` | live path NOT BUILT | Low upside by design; advice rows are HOLD-heavy and that is correct |
| execution_gateway | `execution_gateway_sends` audit | library; per-chain private-send flags default false | No caller wired yet — an empty audit table is expected |
| clmm_lp | `clmm_shadow_positions` | no live code in v1 | Fee APR without IL accounting is fake — read `net_usd`, not fees |
| intent_solver | `intent_fill_opportunities` | PARKED, zero live code | Research feed only; do not plan capital around it |
| basis_desk | `basis_carry_suggestions` | live wiring not implemented | Carry math assumes both legs fill at quoted costs |
| smart_money | `smart_money_signals` | none (read-only) | Crowded signals are soft-capped; coverage is EVM-only in v1 |
