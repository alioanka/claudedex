# Wave-F7: GPT-5.6 audit — per-module fix completeness report

**Date:** 2026-07-10. **Branch:** `claude/friendly-ramanujan-nMWNv` (base `d8e48c3` + Wave-F7 commits below).
**Why this report exists:** the operator said *"I cannot see all the fixes for GPT's concern for each module. Almost for every module it found something."* The Wave-F6 adjudication (`docs/agents/wave-f6/06_gpt56_adjudication.md`) triaged the audit's 20-item priority list but did NOT walk row-by-row through the audit's **Module-by-module assessment** (12 rows) and **Log review** (31 namespaces). This report does exactly that: every row, quoted, verdict against CURRENT code, evidence, and — where a gap was genuinely open — the fix applied in this wave.

**Critical context (applies to every row):** the audit reviewed commit `e054a50` — the OLD default branch, ~40 Wave-F5 commits behind — and logs from that old deployment. `FALSE_POSITIVE_STALE` below means "only true on e054a50; already absent on this branch when the audit was published."

**Verdict legend:** `ALREADY_FIXED` (closed by a prior wave, cited) · `FIXED_F7` (closed by this wave, commit cited) · `PARTIAL` (part fixed, residual named) · `STILL_OPEN` (recorded follow-up with owner) · `FALSE_POSITIVE_STALE` · `BY_DESIGN` (deliberate, documented trade-off) · `OUT_OF_SCOPE` (roadmap/architecture, not a bug) · `OPERATOR` (needs operator action, not code).

---

## Part A — Module-by-module assessment (12 rows)

### A1. DEX — audit verdict "Blocked"
> "uses a legacy `core/engine.py` path, duplicate entrypoints (`main_dex.py` … different hashes), no final `validate_trade()` call, and allows an unknown contract to proceed with a warning … 58 reported trades [vs] 84 successful … impossible to reconcile."

| Claim | Verdict | Evidence |
|---|---|---|
| No final `validate_trade()` call | FALSE_POSITIVE_STALE | Live-readiness wave wired it fail-closed at broadcast (`core/engine.py:~1626`); adjudication #6 confirmed. The *separate* defect that `validate_trade` could never return True (AttributeError on `wallet_manager.get_available_balance`) was fixed in F6 (fail-soft `self.get_available_balance()`, `core/risk_manager.py`). |
| Duplicate `main_dex.py` entrypoints | ALREADY_FIXED | Root `/main_dex.py` deleted in F6 (hygiene commit; `main.py:524` launches the `modules/dex_trading/` copy). |
| Unverified contract proceeds with warning | ALREADY_FIXED | F6 `e8a543a`: `_final_safety_checks` step 6 is DRY_RUN-aware — warn-only in DRY_RUN, **fail-closed in LIVE**; DB knob `trading.block_unverified_contracts` (mig 147, default true). |
| Impossible trade counters (successful > total) | ALREADY_FIXED | F6 `536b9b2`: `successful_trades` no longer incremented at OPEN; means "closed at profit" only. |
| Negative PnL / no edge | ALIGNED | DRY_RUN-only; the honest blocker is below. |
| (residual) DEX shows 0 entries | STILL_OPEN — recorded | Collector-side: `data/collectors/dexscreener.py` discovery scans global Solana-dominated lists; EVM chains discover ~0 new pairs. Named residual #1 in the F6 final report. Owner: smartcontract-web3-expert. NOT fixable by config. |

### A2. Futures — audit verdict "Blocked"
> "its current circuit breaker holds new entries while leaving three positions open … 1,426 simulated trades; 43.6% win rate; total PnL -$25.90 … **Estimated funding is explicitly not applied to net PnL.**"

| Claim | Verdict | Evidence |
|---|---|---|
| Funding not applied to net PnL | **FIXED_F7** (`5342625`) | It was genuinely cosmetic: F5 stashed `est_funding_usd` in `Trade.metadata` with an explicit "NOT applied to net_pnl" log line. Now `_close_position` **debits the estimate from `net_pnl` for SIMULATED closes** (LONG pays +funding, SHORT pays −; intervals = hold/8h). LIVE closes untouched — the exchange settles real funding; an estimated debit would double-count. `metadata` keeps `est_funding_usd`/`funding_applied`/`net_pnl_before_funding` for attribution. **Measurement note:** post-F7 DRY_RUN PF is not directly comparable to earlier windows — flagged in `modules/futures_trading/CLAUDE.md`. |
| Breaker holds entries, leaves positions open | BY_DESIGN | The consecutive-loss breaker blocks NEW entries; existing positions stay monitored and exit via SL/TP/time/manual. Force-closing open positions on a breaker trip would convert a pause into a realized loss — deliberate. F6 mig 109 separately guarantees LIVE positions are never paper-closed while orders are blocked. |
| Losing / no edge | ALIGNED | Matches our own posture: geometry re-tuned migs 139 (rr=1.0 etc.) + 150 (hold 240→480); verdict deferred to 2+ weeks fresh DRY_RUN. Separate from portfolio-wide risk = roadmap (typed reservation service). |

### A3. Solana trading — audit verdict "Blocked"
> "the shared `RiskManager(config={})` is broken … **Drift injects a fake `+12%` funding rate in DRY_RUN** … 69.3% 'win rate', but PF 0.64, Sharpe -21.58 … paper accounting is not credible; remove synthetic funding and require transaction-level fill replay."

| Claim | Verdict | Evidence |
|---|---|---|
| RiskManager permanent veto | ALREADY_FIXED | F6 core-safety fix (adjudication #6). |
| Drift fake +12% funding in DRY_RUN | **FIXED_F7** (`1ad2322`) | Verified still injected at `solana_engine.py:3770` (`signal+2%` = 12%/yr when chain read unavailable). Kept as a wiring check (unreachable in LIVE; `_init_drift` strips Drift when chain is unreachable) but now every trade-log record off that branch is tagged **`simulated_funding: true`** and the log line states it is not carry evidence. Important scoping fact found during verification: Drift wiring-check entries go ONLY to the trade log (`_log_trade`) — they never reach `solana_trades` or any promotion scorecard, so the "inflates paper PnL" half was already structurally impossible via the DB path. |
| Contradictory paper metrics (PF 0.64 vs +1249 SOL) | ALREADY_FIXED | F5 fake-PnL chain closed (PriceValidator authority, quorum-gated jumps, mig 140A back-tag) + F6 entry-side corroboration (`5e365c4`). Wave-F7 extends the exclusion to the meta layers (A9/A10) which still counted the tagged rows. |
| Transaction-level fill replay | OUT_OF_SCOPE | Roadmap (audit Phase 1/2); consistent with adjudication #20. |

### A4. Sniper — audit verdict "Blocked"
> "its entrypoint does not start the global-kill poller … default active-position cap of 500 is much too high … every recent analyzed candidate fails the low-BSR gate (0% pass)."

| Claim | Verdict | Evidence |
|---|---|---|
| No kill poller | FALSE_POSITIVE_STALE | `modules/sniper/main_sniper.py:86` `start_killswitch_poller()` — re-verified this wave. |
| 500-position cap | ALREADY_FIXED | Mig 140B: 25-position cap, 0.05 SOL size, `sniper_max_daily_loss_usd`, entry cooldown. |
| 0% BSR pass rate | ALREADY_FIXED | F6 mig 148 `sniper_bsr_fallback_mode` (default `skip_gate`): a down Birdeye BSR source skips ONLY that gate; all other safety gates run. |
| WSS pipeline unusable data / silent listener | ALREADY_FIXED (mechanism) + OPERATOR | F5 `_startup_supervisor` self-healing listener + loud `listener_status=rpc_auth_failed`; multi-key Helius rotation. Detects nothing until the operator supplies real Helius keys (`docs/RPC_API_KEYS_GUIDE.md`). |

### A5. Arbitrage — audit verdict "Blocked"
> "no global kill/pause integration, a nonfunctional empty-config risk manager, and sequential Solana sends remain … zero opportunities/executions … disable scans with no supported liquid universe."

| Claim | Verdict | Evidence |
|---|---|---|
| No kill/pause gate | FALSE_POSITIVE_STALE | `arbitrage_engine.py:~3021` + `solana_engine.py:~870` gate via `should_skip_live(..., module='arbitrage')`; poller at `main_arbitrage.py:366` (adjudication #8). |
| Broken risk manager | ALREADY_FIXED | F6 core-safety fix (shared with all modules). |
| Sequential Solana sends (two swaps, inventory risk) | BY_DESIGN (gated) + OUT_OF_SCOPE | Still non-atomic (`solana_engine.py:~1374` RPC fallback when Jito bundle fails) — but live execution is default-OFF (`live_execution_enabled`, migs 097/108) and the standing verdict is **PARK IT** (F6: V2-only legs vs ~65bps cost floor). Atomic receiver contract = roadmap precondition for ANY live multi-leg path. No code change. |
| Phantom paper PnL (+3250, 100% win) | ALREADY_FIXED + **FIXED_F7 residual** (`690b8e0`) | F6 `cc87b95` books triangular DRY fills net-of-gas + self-tags `metadata.excluded=true`; mig 150A back-tags history. **Gap found this wave:** the control-center overview + cross-module series queries in `monitoring/enhanced_dashboard.py` never gained the exclusion filter — the phantom +3250 was still served on the operator's main decision surface. Both queries now filter (mirrors the existing solana filter in the same SQL maps). Per-page `/api/arbitrage/*` diagnostics aggregates still include raw rows — recorded follow-up (backend-devops). |
| Zero opportunities / idle scans | ALIGNED | Honest signal is the hourly `SPREAD DISTRIBUTION` line (F5); PARK IT verdict stands. |

### A6. Copy trading — audit verdict "Blocked"
> "the live risk gate fails closed; EVM quote failure can broadcast `minOut=0` … three [leaders] dead for 18-33 days; zero … copies."

| Claim | Verdict | Evidence |
|---|---|---|
| minOut=0 broadcast | PARTIAL (mostly fixed) + BY_DESIGN residual | `copy_engine.py:697-718`: LIVE **BUY fails closed** on quote failure. **SELL deliberately proceeds** with `min_out=0` at a CRITICAL log — a panic exit must not be blocked by a flaky quote. Documented trade-off (adjudication #7). Optional hardening recorded: last-known-price floor with generous tolerance instead of 0 (owner: smartcontract-web3-expert; touches a panic-exit path, needs careful review — not rushed in this wave). |
| Dead leaders / zero copies | ALREADY_FIXED (mechanism) + OPERATOR | Leader lifecycle mig 092; discovery + shadow-sim UNCONDITIONALLY enabled in F6 mig 149 (the F5 conditional flip had silently failed); per-key Helius budgets + honesty. Watch for `[discovery-v3] sweep:` lines on the new deployment. |
| Helius fallback rate-limited | ALREADY_FIXED (honesty) + OPERATOR | F6 dedup-by-account-key + cadence cuts (rps 8→2, poll 15→30s). Real headroom requires distinct extra Helius accounts. |

### A7. AI analysis — audit verdict "Blocked"
> "Direct exchange execution exists behind local gate … its `RiskManager(config={})` makes live entry reject … keep LLM strictly as a feature/annotation source."

| Claim | Verdict | Evidence |
|---|---|---|
| RiskManager reject | ALREADY_FIXED | F6 core-safety fix. |
| Kill poller missing | ALREADY_FIXED | F6 added `start_killswitch_poller()` to `main_ai.py:169` (re-verified this wave). |
| LLM must stay advisory / bounded | ALIGNED | Execution rules are deterministic (signal thresholds, per-trade notional ceiling mig 111, `core/llm_budget` caps + per-module reservations); the confirmation strategy (mig 139) is filter-only. Independent model-risk approval process = roadmap. |

### A8. Dashboard — audit verdict "Blocked"
> "highest-risk service because it is publicly exposed and over-authorized."

| Claim | Verdict | Evidence |
|---|---|---|
| Over-authorized (RBAC gap, ~45 routes) | ALREADY_FIXED | F6 `0796806`/`d2b1de5`/`8b8e323`: `require_operator`/`require_admin` on every mutating route + global VIEWER write-floor + route-authz startup self-test. |
| Wildcard CORS + credentials | ALREADY_FIXED | F6 `c6b50fc`: `DASHBOARD_CORS_ORIGINS` allowlist, never `*`+credentials. |
| Public HTTP on 0.0.0.0:8080 | OPERATOR | Code cannot fix a public bind decision: firewall/VPN/TLS-reverse-proxy port 8080 (F6 final report, operator action #3). |
| Duplicated log output volume | STILL_OPEN (minor) — recorded | Transport-level duplicate lines noted by the audit; cosmetic/disk-cost only. Owner: backend-devops (logging handler audit). |

### A9. Orchestrator AI — audit verdict "Advisory only"
> "its database aggregates are not authoritative and include simulation assumptions. It emits `to_live` recommendations without real-fill validation. … Suggested Solana `to_live` while the module's own paper PF/Sharpe are strongly negative."

**FIXED_F7** (`eb36aad`). Verified genuinely open on current code: `_collect_module_inputs` counted `metadata.excluded=true` rows — the exact poisoned Solana history (mig 140A) and arbitrage phantom fills (mig 150A) that F5/F6 excluded from the *dashboards* but not from the *scorer*. That is mechanically how "Solana to_live on negative PF" happened. Fixes:
- Shared `EXCLUDED_ROW_FILTER` applied to both the aggregate and the Sharpe-series queries (all 7 trade tables have `metadata JSONB` — verified against migs 006/008/009/010 + the legacy `trades` schema).
- Every `to_live` reason now states **"PAPER evidence only (no real-fill validation); confirm execution quality (TCA) and net-of-cost edge before approving."** The advisory-only + explicit-operator-approval design is unchanged and correct; replacing the promotion rule with live-quality evidence gates (TCA-fed) = recorded roadmap.

### A10. Portfolio allocator — audit verdict "Advisory only"
> "It allocates by module-level rows, not actual consolidated wallet, margin, correlated exposure, or liquidity. … redesign around a central portfolio ledger before execution authority."

**OUT_OF_SCOPE (roadmap) — honestly classified, not a bug.** The module is documented ADVISORY-ONLY (`portfolio_allocations` proposals; no feedback loop executes them; fractional-Kelly with 5-40% bounds + 10% reserve), which the audit itself acknowledges ("no execution authority"). Allocating from module-level rows is a *known limitation of an advisory proposal generator*, not a defect — the fix is the central portfolio ledger + typed pre-trade reservation (adjudication #20 roadmap), which must precede ANY execution authority. Two concrete notes: (1) it inherits the Wave-F7 excluded-row filter indirectly — its inputs read `orchestrator_recommendations.metrics`, now computed on cleaned aggregates; (2) reserve variance 9.98-21.42% flagged by the audit is the Kelly output range, not drift.

### A11. Backtest / replay — audit verdict "Needs build-out"
> "not supervised and has no runtime log. Existing comments acknowledge dry-run PnL understates fees."

**FIXED_F7** (`57e4a48`) for what the data supports; rest honestly roadmapped:
- **Latent defect found during verification (worse than the audit's claim):** `flip_ts` — the simulated live-flip timestamp — was computed and then never used; the counterfactual ALWAYS equaled actual (`pnl_delta_usd ≡ 0` for every strategy). The replay was decorative.
- Now: trades in the simulated "would-have-been-live" window take a `live_haircut_usd_per_trade` debit (strategy_params, **default 0.5 USD/trade**; 0 restores old behavior). Report surfaces `n_live_window_trades` + `live_haircut_usd` so a zero-haircut run is visibly optimistic. Flat-USD because `TradeRow` deliberately carries no notional; bps-of-notional model = recorded follow-up.
- `trade_loader.load_trades` now drops `metadata.excluded=true` rows — a counterfactual built on fabricated PnL is worthless.
- "Not supervised / no runtime log" = BY_DESIGN (synchronous dashboard invocation, documented in its CLAUDE.md). Deterministic event replay, latency, partial fills, walk-forward = roadmap (matches adjudication #20).

### A12. Solana strategies — audit verdict "Consolidate"
> "it is not registered by the current supervisor as an independent strategy module … remove duplicate strategy ownership and expose one tested execution interface."

**FIXED_F7** (`d6f83c2`) + BY_DESIGN. `modules/solana_strategies/` is documented as helpers-only (`jupiter_helper.py`, `drift_helper.py` — the live SDK surface the solana engine imports); not being a supervisor-registered module is intentional. The genuine duplicate-ownership item — `solana_config_manager.py`, a stale copy of the canonical `modules/solana_trading/config/solana_config_manager.py` (backlog SOL-14) — was proven importer-free by repo grep and deleted per dead-code policy. Remaining SOL-14 half (three parallel Jupiter execution paths: `jupiter_helper` / `jupiter_executor` / sniper `trade_executor`) = recorded consolidation follow-up (smartcontract-web3-expert).

---

## Part B — Log review (31 namespaces)

| # | Namespace | GPT's assessment (condensed quote) | Verdict | Evidence / action |
|---|---|---|---|---|
| 1 | root orchestrator | "Reports 30 services … many absent from source. Deployment/source drift; memory budget fragile." | FALSE_POSITIVE_STALE + OUT_OF_SCOPE | All "absent" dirs exist on this branch (adjudication #1) — GPT audited `e054a50`, the drift was its own stale checkout. Restart-latch class fixed in F5 (RC-D1 + `logs/.restart_<module>`). Release identity (SHA/digest at startup) + memory budget = roadmap. |
| 2 | dashboard | "Authentication working, but public HTTP exposure unacceptable." | OPERATOR | See A8. Firewall/VPN/TLS proxy on 8080 — F6 operator action list #3. |
| 3 | pool_engine | "All Helius endpoints rate limited; starvation fallback intentionally reuses cooled-down endpoints. Capacity incident." | PARTIAL → **GOVERNOR FOLLOW-UP** | F6: dedup-by-account-key (one real account behind five names) + cadence cuts; F5: multi-key rotation. RECORDED for the RPC-governor agent (owns `config/pool_engine.py` — not touched this wave): (a) never serve a STARVED/cooled endpoint for *broadcast/execution* calls; (b) budget-aware per-provider scheduler, market-data vs execution quota split. Real headroom = operator adds distinct Helius accounts. |
| 4 | dex_trading | "Mismatch between total/success/failure counters. KPI/ledger integrity failure." | ALREADY_FIXED | F6 `536b9b2` (win double-count). Unified fill/ledger schema = roadmap. 0-entries residual = collector-side (A1). |
| 5 | futures_trading | "Strategy paused but not de-risked; not profitable." | BY_DESIGN + FIXED_F7 | Breaker semantics deliberate (A2). Funding honesty closed this wave (`5342625`). Profitability = A/B window in flight. |
| 6 | solana_trading | "Simulated fills, artificial Drift funding, repeated emergency exits, negative risk-adjusted metrics. Paper results not deployable evidence." | ALREADY_FIXED + FIXED_F7 | Fake-PnL chain F5/F6; Drift synthetic funding tagged this wave (`1ad2322`); meta layers stop counting excluded rows (`eb36aad`). Agreed paper ≠ deployable evidence — shadow-first posture. |
| 7 | sniper | "Massive WSS reject/no-result counts; 0% current pass rate." | ALREADY_FIXED + OPERATOR | A4: BSR fallback (mig 148), self-healing listener, hard caps (mig 140B). Needs real Helius keys. |
| 8 | arbitrage | "High scan counts, no opportunities, stale prices, Arbitrum no liquidity. Disable idle scans." | ALIGNED (PARK IT) | A5. Live OFF; hourly `SPREAD DISTRIBUTION` is the judge; F5 RPC-infra rotation fixed the stale-price/blacklist poisoning class. |
| 9 | copy_trading | "3/4 leaders dead; zero copies; fallback poll rate limited." | ALREADY_FIXED + OPERATOR | A6: mig 149 unconditional enable; budget/cadence honesty. Watch discovery liveness lines. |
| 10 | ai_analysis | "47 simulated trades … do not promote an LLM-led execution path." | ALIGNED | A7. Advisory LLM + deterministic bounded execution; budget caps. |
| 11 | orchestrator_ai | "Recommended Solana live despite contradictory module metrics. Scoring data contract is unsafe." | **FIXED_F7** | A9 (`eb36aad`): excluded-row filter + paper-evidence caveat. |
| 12 | portfolio_allocator | "Keep advisory until ledger is unified." | OUT_OF_SCOPE (roadmap) | A10. Already advisory-only by construction; inherits cleaned inputs. |
| 13 | advisor | "Repeated Kronos OHLC schema failures; ~1 GB logs; source missing here." | ALREADY_FIXED + STALE | Kronos OHLC synthesis fixed F6 (`c66d14c`); `modules/advisor/` exists on this branch (stale checkout). Log-volume budget = ops follow-up. |
| 14 | basis_desk | "Correctly rejecting, but no current yield." | NO_ACTION | That IS the designed outcome — honest rejection of negative net carry. Do not loosen thresholds (audit agrees). |
| 15 | catalyst_calendar | "Only one static-macro item; external feeds return zero. Degraded data source masquerades as coverage." | BY_DESIGN (honest) + OPERATOR | Verified: sources warn-once on death, degrade to empty (never fabricate), macro schedule EXPIRES 2026-12-31 with explicit "honesty over fake coverage" note; FMP source is key-gated OFF. Richer paid feeds = operator/roadmap choice. The "masquerade" claim does not hold against current code — degradation is logged and documented. |
| 16 | clmm_lp | "Not live evidence; source missing." | FALSE_POSITIVE_STALE + ALIGNED | `modules/clmm_lp/` exists; SHADOW-ONLY with `live_path_not_implemented` terminal skip; carry-note (low/negative edge until TCA + HODL benchmark) stands. |
| 17 | execution_gateway | "Claimed implementation path is absent. Do not assume central execution protection exists." | FALSE_POSITIVE_STALE + ALIGNED | `modules/execution_gateway/gateway.py` exists (mig 127). The honest half we already document ourselves: it is a LIBRARY, "no module wired to it yet" — wiring modules onto it = roadmap (adjudication #20). |
| 18 | execution_quality | "Useful post-trade layer; must feed hard pre-trade limits." | ALIGNED (roadmap) | TCA read-only by design (mig 120). Feeding TCA scorecards into pre-trade gates = roadmap; also now the explicit approval criterion named in every orchestrator to_live caveat (A9). |
| 19 | intent_solver | "CoW API HTTP 403 every hour. Disabled/degraded integration." | ALREADY_FIXED | F5: browser UA + hourly dead-source backoff + mig 143 poll 60s→3600s (was 28k calls). Module is a PARKED scaffold by verdict. |
| 20 | market_data_warehouse | "Healthy but far too narrow for broad portfolio/model claims." | ALIGNED (scope) | By-design v1 scope (free sources, mig 123). Broaden series universe = roadmap; no code defect. |
| 21 | meta_controller | "Autopilot false; calibration is 0/3 forward decisions matched. No evidence for autonomous actuation." | ALIGNED + FIXED_F7 | The 0/3 calibration is the module *honestly reporting its own miss rate* — the designed self-audit. Autopilot stays default OFF. This wave: `_collect_track` excludes phantom rows (`eb36aad`) so decisions/calibration never score fabricated PnL. |
| 22 | notifications | "Verify payload redaction and alert escalation separately." | RECORDED (no defect claimed) | Audit asked for a verification, not a fix. Recorded: redaction + escalation audit of `monitoring/telegram_bot` / alerts payloads (owner: backend-devops). |
| 23 | options_vol | "Research signal only; no execution proof." | ALIGNED | ADVISOR by design; SELL legs hardcoded record-only; BUY legs behind full dual-flag chain (mig 125). |
| 24 | param_tuner | "Correctly non-autonomous; require experiment governance." | ALIGNED | `auto_apply_enabled=false`; risk/live/secret keys hard-excluded; F5 retargeted knobs; out-of-sample acceptance requirement documented (mig 129/143). |
| 25 | polymarket | "Emits score/momentum observations; source missing; do not grant capital." | FALSE_POSITIVE_STALE + ALIGNED | `modules/polymarket/` exists. F5 added signal-quality knobs + forward-outcome tracking (mig 137); shadow-first — no live consideration until the outcome scorecard proves edge. |
| 26 | regime_allocator | "Research overlay only; validate regimes out of sample." | ALIGNED | ADVISORY-ONLY proposals (migs 118-119); out-of-sample validation = documented acceptance bar. |
| 27 | sentinel | "Current cycles see no anomalies, despite a retained critical record of sniper 100% rejection. Alert state needs lifecycle/acknowledgement and source identity." | **FIXED_F7** (`b8718c9`, mig 152) | Genuinely open: `sentinel_anomalies` had no terminal state — a weeks-old critical was indistinguishable from an active one. Mig 152 adds `resolved_at` + `acknowledged_at/by` + open-rows partial index; engine auto-resolves OPEN rows quiet longer than `anomaly_auto_resolve_minutes` (240, 0 disables; rows KEPT for audit); refire dedup refreshes only unresolved rows (re-fire after resolve = new incident); pre-mig fail-soft fallback. Bonus fix: `_window_pnl` (loss-velocity/corr-drawdown inputs) now excludes `metadata.excluded=true` rows — arbitrage triangular DRY fills self-tag excluded at insert and were feeding the loss detectors. Dashboard ack route (columns are schema-ready) = follow-up (backend-devops, RBAC operator-gated). |
| 28 | smart_money | "Ethereum ingestion stays at zero due to RPC `eth_getLogs`; Base works but produces zero signals. Replace provider and measure signal value." | PARTIAL + OPERATOR | F5 added the hourly WARN when eth_getLogs is blocked (visible, not silent). Provider replacement = operator key/plan choice + governor capacity work; signal-value measurement = the module's own forward-return design. |
| 29 | stat_arb | "Add cointegration stability, borrow/funding, and fill model before a pilot." | ALIGNED (roadmap) | SHADOW-FIRST; live path wired but dual-flag gated OFF (mig 132). The named models are the documented pilot preconditions. |
| 30 | treasury | "No wallet address discovered, so every tick is idle. Inoperative." | ALREADY_FIXED + OPERATOR | F5: `main_treasury.py` now calls `secrets.initialize(pool)` before derivation (the 20-day zero-snapshot outage); addresses DERIVED from PKs with one-shot WARNING on failure (`treasury_engine.py:105-216`, re-verified this wave). Needs the PK secrets present — operator config. |
| 31 | yield_treasury | "No fresh treasury snapshot and zero candidates. Inoperative downstream dependency." | ALREADY_FIXED (dependency) | Reads fresh `treasury_snapshots` only; self-heals with no change of its own once treasury (row 30) writes snapshots. Live deposit path remains NOT BUILT by design (mig 126). |

---

## Part C — What Wave-F7 changed (commits, all on this branch, small batches, compiled per commit)

| Commit | Change |
|---|---|
| `eb36aad` | orchestrator_ai + meta_controller: `EXCLUDED_ROW_FILTER` on all scoring queries (aggregates + Sharpe series, dry+live tracks); `to_live` paper-evidence caveat. |
| `b8718c9` | sentinel: anomaly lifecycle — **mig 152** (`resolved_at`, `acknowledged_at/by`, open-rows partial index, `anomaly_auto_resolve_minutes`=240 seed), auto-resolve pass, resolved-aware refire dedup (pre-mig fail-soft), phantom-PnL exclusion in `_window_pnl`. |
| `57e4a48` | backtest_replay: live-window cost haircut (`live_haircut_usd_per_trade`, default 0.5) — also fixes the latent `flip_ts`-unused defect (delta was always 0); excluded-row filter in `trade_loader`; `n_live_window_trades`/`live_haircut_usd` surfaced. |
| `1ad2322` | solana: Drift DRY_RUN synthetic funding tagged `simulated_funding: true` in trade log + explicit not-evidence log line. |
| `5342625` | futures: estimated funding APPLIED to SIMULATED `net_pnl` at close (LIVE untouched); attribution metadata; comparability note. |
| `690b8e0` | dashboard: control-center arbitrage overview + cross-module series exclude `metadata.excluded=true` (the F6 "resets toward 0" promise now actually delivered on the operator's decision surface). |
| `5eb5670` | docs: orchestrator_ai + meta_controller CLAUDE.md. |
| `d6f83c2` | dead-code: deleted `modules/solana_strategies/solana_config_manager.py` (duplicate, zero importers — SOL-14 half). |
| (follow-on) | docs: solana CLAUDE.md duplicate-mention fix; this report. |

**Safety audit of this wave:** no live/paid flag flipped; no gate loosened — every change is stricter or purely-honester (funding debit only on simulated closes; haircut only in the simulated live window; filters only EXCLUDE fabricated rows; sentinel auto-resolve only stamps a timestamp, rows kept; deletion had zero importers). Mig 152 is idempotent (`ADD COLUMN IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` / `ON CONFLICT DO NOTHING`) and collision-free (151 left to the RPC-governor wave). Every touched file `py_compile`d per commit.

**Measurement notes for the operator (honest caveats):**
1. Futures DRY_RUN PF/PnL now carries estimated funding — post-F7 windows are NOT directly comparable to the mig-139/150 baseline; judge the geometry A/B with this in mind.
2. Orchestrator/meta/sentinel scores will shift once phantom rows stop counting — that is the fix working, not a regression.
3. Control-center arbitrage PnL should now genuinely reset toward ~0.

## Part D — Recorded follow-ups (not fixed this wave)

**RPC-governor agent (owns `config/pool_engine.py` / `config/rpc_provider.py` / `monitoring/rpc_pool_routes.py` — deliberately untouched here):**
1. Refuse STARVED/cooled-down endpoints for broadcast/execution calls specifically (adjudication #12's targeted candidate).
2. Budget-aware per-provider scheduler; separate market-data vs execution quotas; request coalescing.

**Other owners:**
3. Arbitrage per-page `/api/arbitrage/*` aggregates: add excluded-row filter (backend-devops; diagnostics surface). 
4. Sentinel dashboard ack route on the schema-ready `acknowledged_*` columns (backend-devops; operator-gated RBAC).
5. Copy LIVE-SELL bounded `min_out` floor instead of 0 (smartcontract-web3-expert; panic-exit path — design review first).
6. Drift perp-appropriate RiskManager gate (quant; spot-liquidity check structurally blocks perp signals — F6 residual #3).
7. Solana position-monitor cadence ~40s → 90-120s (solana owner; Helius load — F6 residual #2).
8. DEX EVM pair discovery chain-scoped collector source (smartcontract; F6 residual #1 — DEX stays idle until fixed).
9. Backtest replay bps-of-notional cost model (needs notional columns in `trade_loader`).
10. Jupiter execution-path consolidation (SOL-14 remainder) + dashboard duplicate-log-output audit + notifications redaction/escalation audit (backend-devops).
11. Roadmap cluster (adjudication #20, unchanged): release identity at startup, typed pre-trade reservation/central ledger, execution-gateway adoption, data-quality SLOs + DEGRADED state, non-root container + pinned images + lockfile.

## Part E — Verdict counts

**Module table (12 rows, per-claim):** ALREADY_FIXED (prior waves): DEX×4, Sniper×3, Copy×2, AI×2, Dashboard×2, Advisor-class fixes — 15 claims. FIXED_F7: 6 claims across Futures, Solana, Arbitrage-surface, Orchestrator AI, Backtest/replay, Solana-strategies. FALSE_POSITIVE_STALE: 4 (DEX validate_trade, sniper poller, arb gates, "missing" dirs). BY_DESIGN documented: 3 (futures breaker, copy sell-exit, seq-arb-gated). OUT_OF_SCOPE roadmap: 3 (portfolio ledger, fill replay, replay build-out). STILL_OPEN recorded: 2 (DEX collector discovery, dashboard log duplication) + OPERATOR: 3 (ingress, Helius keys, treasury secrets present).

**Log table (31 namespaces):** FIXED_F7 3 (orchestrator_ai, sentinel, meta-controller-input) · ALREADY_FIXED/prior-wave 8 · FALSE_POSITIVE_STALE 5 · ALIGNED/BY_DESIGN/NO_ACTION 10 · OUT_OF_SCOPE/roadmap 2 · OPERATOR-dependent 2 · GOVERNOR follow-up 1 (pool_engine).

Nothing GPT flagged remains both *unfixed* and *unrecorded*: every row above is either closed with cited evidence, deliberately documented as a design trade-off, assigned to a named follow-up owner, or requires an operator action already on the F6 action list.
