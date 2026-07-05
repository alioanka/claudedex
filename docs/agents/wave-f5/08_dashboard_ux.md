# Wave-F5 / 08 — Dashboard Information-Architecture Audit

Read-only analysis. Evidence: `dashboard/templates/base.html` (sidebar, lines 749–1344),
route registrations in `monitoring/enhanced_dashboard.py` + `monitoring/module_routes.py` +
`monitoring/{analytics,auth,credentials,rpc_pool,test_runner}_routes.py`, and 77 full-page
screenshots in `screenshots/` (all reviewed). Date: 2026-07-05.

---

## (a) Current sidebar tree — per-page verdict

Handler file is `monitoring/enhanced_dashboard.py` (ED) unless noted. MR = `monitoring/module_routes.py`.

### Top-level overview pages (4 overlapping "home" pages)

| Sidebar entry | Route | Template | Verdict |
|---|---|---|---|
| Control Center | `/control-center` | `control_center.html` | WORKS. Per-module runtime cards (Today/7D/All PnL, pause/restart/Go-LIVE/disable), cross-module table, intel-ops table, meta decisions. Best of the four. Weakness: all 16 intel-ops modules show STATUS "UNKNOWN". |
| Full Dashboard | `/full-dashboard` | `full_dashboard.html` | WORKS but redundant: 15+ charts, several "No data yet" (Fee Analysis, Slippage Impact), absurd "+10000.00%" P&L stat. Duplicate of `/` + `/analytics`. |
| Analytics | `/analytics` | `analytics_routes.py` → `analytics.html` | WORKS. Per-module drill-down + risk metrics. Recent-trades entry/exit all $0.00; Sharpe/VaR mostly 0. |
| Main Overview | `/` | `index.html` | WORKS but WRONG numbers: "TOTAL P&L $8,938,964.61" (vs $882 on full-dashboard, $1,982 on analytics), "TOTAL PORTFOLIO $0.00" while the shared header says $1,282.06; Wallet Balances all $0.00. Ten charts duplicating full-dashboard/analytics. |
| Modules | `/modules` | ED fallback → `modules.html` (MR variant exists too) | HALF-BROKEN: "No Modules Found — no trading modules registered" empty state contradicting "Active Modules 1/0"; capital-allocation bar empty; the useful part (16 intel-ops cards) duplicates control-center. |

Four pages compute the same portfolio KPIs from different queries and **all disagree**. This is the single worst operator-trust problem on the dashboard.

### Trading module groups

| Group | Pages in sidebar | Verdict summary (screenshots) |
|---|---|---|
| DEX Trading | dashboard, positions, trades, performance, **backtest, reports, analysis**, settings (8 pages — only module with 8) | dashboard WORKS-degraded (red "DEX engine data unavailable" banner, wallet cards $0.00); positions: closed rows have $0.00 entry/exit but real P&L; trades WORKS; performance WORKS but corrupt stats (Daily Vol 5,502%, Annual Vol 105,133%, Calmar 85,813). backtest/reports/analysis: legacy DEX-scoped pages (`backtest.html`, `reports.html`, `analysis.html`), no screenshots captured; superseded by `/backtest-replay` and `/dex/performance`. settings: see section (d). |
| Futures | 5 std pages | All WORK with plausible data; funding-cost chart empty. Settings: single long form, DRY/LIVE banner, no tabs, no Guide. Healthiest module group. |
| Solana | 5 std pages | WORKS but data bugs: positions AGE column "NaNm" on every row; trades P&L% "+2000.00%" on every row (clamp artifact); performance Max Drawdown −134.5% (impossible); dashboard ROI 11,763%. Settings: long form, no tabs. |
| Sniper | 5 std pages | Self-contradictory: dashboard P&L −$4,898 with a RISING equity sparkline; performance −$536.56 with a FALLING equity curve and 1,000 trades; trades page shows **0 of 0** trades. Settings: long form + full inline "Complete User Guide" appended (no tabs). |
| Arbitrage | 4 pages (no Positions — intentional; `/arbitrage/positions` route renders an inline-HTML explainer) | All WORK. Dashboard "Cumulative P&L" chart is blank while the identical chart on performance renders. Profit Factor renders "∞". Settings: **the best pattern in the app** — tabs General / Chains / Advanced / **Guide** / Setup, all populated. |
| Copy Trading | 7 pages (std 5 + Wallet Discovery + Wallet Monitor) | All WORK. "Top Performing Wallets" widget appears on dashboard AND performance AND wallets. Settings: form + enormous inline guide (not a tab). Orphan extra: `/copytrading/leaders` (`leaders_copytrading.html`) is routed but NOT in the sidebar. |
| Polymarket | 2 pages: `/module/polymarket` (generic panel, bespoke nav) + `/config/polymarket_config` | Both WORK; the only module already on the target "generic panel + generic settings" pattern, and the only `/module/*` panel with a real status badge ("DRY RUN" vs everyone else's "UNKNOWN"). |
| Financial Advisor | 6 pages | dashboard/kap/portfolio/settings WORK (settings has proper **Settings/Guide tabs**). **`/advisor/advice` is HARD-BROKEN**: red "Error: Unexpected token 'N' … `"try_low": NaN` … is not valid JSON" — backend serializes NaN. `/advisor/simulations` partially broken by the same NaN-JSON bug (top tiles all "—"). |
| AI Analysis | 5 pages (dashboard, sentiment, performance, settings, OpenAI Logs) | WORK but idle-looking: all signals HOLD/score 0, "Last Analysis 184 min ago", Sentiment-Trend/Quorum/Signal-Distribution charts empty. Sentiment page duplicates dashboard gauge + trend chart. `/ai/logs` not screenshotted. Settings: form + inline guide; contains a stat trio (75 trades / 41.3% / $18.30) repeated on 3 of its own pages. |
| Intelligence & Ops | 16 × `/module/<key>` (generic `module_panel.html`) | All 18 render cleanly and identically; empty-state correct. Populated: meta-controller, regime-allocator, sentinel, execution-quality, market-data-warehouse, catalyst-calendar, smart-money, options-vol, stat-arb, clmm-lp, param-tuner. Fully empty (module off): treasury, yield-treasury, basis-desk, intent-solver, execution-gateway. Every status badge = "UNKNOWN". Settings is a header button → `/config/<type>`. |

### Ops / bottom section

| Entry | Route | Template | Verdict |
|---|---|---|---|
| Proposals | `/proposals` | `proposals.html` | WORKS. Param-tuner + regime + meta + copy-leader approval inbox. Overlaps `/orchestrator` and `/allocation`. |
| Help | `/help` | `help.html` | WORKS (header status pill stuck on "Loading…"). Duplicates the module catalog again. |
| Module Control | `/module-control` | `module_control.html` | WORKS. 7 enable/disable/pause/start cards. **Fully duplicated by Control Center.** |
| Pro Controls | `/pro-controls` | `pro_controls.html` | WORKS. Start/stop toggles (dup) + **unique**: Manual Trade panel + PANIC SELL ALL. |
| Global Settings | `/global-settings` | `global_settings.html` | WORKS. Allocation-Guard budgets (unique) + a raw config-key accordion that duplicates `/config`. |
| All Settings | `/config`, `/config/{type}` | `settings_generic.html` | WORKS. The best settings surface in the app: every config_type, typed inline edit, per-key docs + audit trail. Should become the canonical settings engine. |
| Telegram Alerts | `/telegram/settings` | `settings_telegram.html` | WORKS, unique job. Keep. |
| RPC/API Config | `/settings/rpc-api` | `rpc_pool_routes.py` → `settings_rpc_api.html` | WORKS (64 endpoints, health colors), unique job. Keep. |
| Wallet Balances | `/wallet-balances` | `wallet_balances.html` | Renders but ALL balances $0.00 / "No balance data" — the data feed is dead, page currently useless. |
| Secure Credentials | `/credentials` | `credentials_routes.py` → `settings_credentials.html` | WORKS, unique job (vault + trade-record clearing). Keep. |
| Trade Simulator | `/simulator` | `simulator.html` | WORKS. Dry-run monitor + validation. Overlaps equity/PnL/win-rate charts with `/` and `/full-dashboard`. |
| Logs | `/logs` | `logs.html` | Not screenshotted; route live. |
| Test Runner | `/test-runner` | `test_runner_routes.py` + `test_runner.html` | WORKS (dev/QA tool; arguably should be admin-only, not main nav). |
| Orchestrator AI | `/orchestrator` | `orchestrator.html` | WORKS ("Not ready" states dominate). Approve/Reject dup with /proposals. |
| Backtest Replay | `/backtest-replay` | `backtest_replay.html` | WORKS; Delta all +$0.00 and Max-DD values corrupt (sniper 217,344%). |
| Allocation | `/allocation` | `allocation.html` | WORKS. Approval UI overlaps /proposals; budgets overlap /global-settings. |

---

## (b) Orphan routes & templates (exist in code, NOT in sidebar)

| Route | Handler/Template | State | Disposition |
|---|---|---|---|
| `/settings` | ED:5241 → `settings.html` (644 lines) | Reachable only from the user avatar menu (base.html:733). Account/password page + module cards; duplicates the Account tab inside settings_dex.html. | Keep as the ONE account page; strip everything except account/security. |
| `/dashboard` | ED:1170 | 301 → `/dex/dashboard` | Fine — this is the redirect pattern to replicate. |
| `/trades`, `/positions`, `/performance`, `/reports`, `/backtest`, `/analysis` | ED:1174–1187 | Unprefixed legacy aliases of the DEX pages (same handlers). | Convert to 301s; remove double registration. |
| `/modules/{name}`, `/modules/{name}/details` | MR:49–50 → `module_details.html` | Alive but unreachable from nav; superseded by `/module/<key>` panels. | Delete + 302 → `/module/{name}`. |
| `/modules/{name}/configure` | MR:51 → `module_configure.html` | **Template does not exist** (and the fallback `error.html` doesn't exist either) → guaranteed 500. Dead. | Delete route. |
| `/users` | `auth_routes.py:38` → opens `dashboard/templates/users.html` | **File does not exist** → 500 for admins. | Build a minimal page or delete the route; today it's a landmine. |
| `/copytrading/leaders` | ED:1672 → `leaders_copytrading.html` | Routed, templated, never linked. | Either add to Copy Trading submenu or fold into `/copytrading/discovery`. |
| `/arbitrage/positions` | ED:12883, inline HTML | Deliberate explainer page (arb has no open positions). | Fine; leave unlinked. |
| `/__routes__` | ED:1165 | Debug endpoint. | Gate behind admin or remove in prod. |
| — | `positions_arbitrage.html` | **Zero references anywhere** (handler renders inline HTML instead). | Delete file. |
| — | `monitoring/dashboard.py` (old `add_get('/')`, `/dashboard`) | Legacy dashboard class; `modules/dashboard/main_dashboard.py` imports only `enhanced_dashboard.DashboardEndpoints`. | Dead module — delete after a grep-sweep for stragglers. |

Sidebar pages with **no screenshot captured** (couldn't be verified visually; verify before/after any change): `/dex/settings` (see §d — audited from code), `/dex/backtest`, `/dex/reports`, `/dex/analysis`, `/ai/logs`, `/copytrading/discovery`, `/logs`.

---

## (c) Duplication map

1. **Overview × 4:** `/`, `/full-dashboard`, `/analytics`, `/control-center` all render portfolio-KPI cards + equity/PnL charts, and **their totals disagree** ($8.9M vs $882.06 vs $1,982.86). Equity Curve appears on `/`, `/full-dashboard`, `/analytics`, `/simulator`; P&L-by-module / win-rate-by-module bars on `/`, `/full-dashboard`, `/simulator`.
2. **Module lifecycle controls × 4:** `/modules`, `/module-control`, `/pro-controls`, `/control-center` each expose enable/disable/pause/start for the same 7 modules (plus per-panel Pause/Resume on `/module/<key>`).
3. **Per-module dashboard vs performance:** every trading module repeats its cumulative-P&L curve and win-rate donut on both pages (arbitrage's dashboard copy is even broken while the performance copy works). Copy-trading "Top Performing Wallets" appears on 3 pages; AI sentiment gauge + trend chart on 2; AI stat trio on 3.
4. **Config editors × 5:** `/config` (canonical), `/global-settings` accordion, `settings_dex.html`'s General/Portfolio/API/Monitoring/Features tabs, `/settings` account page, `/credentials` (also reachable via settings_dex "Sensitive" tab). Same `config_settings` rows, five UIs.
5. **Approvals × 3(+1):** `/proposals`, `/orchestrator`, `/allocation` all have Approve/Reject queues; `/control-center` also shows the meta-decisions table.
6. **Intel-ops module list × 3:** `/control-center`, `/modules`, `/help` each render the 16-module catalog.
7. **Simulation/backtest × 3:** `/simulator`, `/backtest-replay`, `/dex/backtest` (legacy).

---

## (d) DEX settings page (`/dex/settings` → `settings_dex.html`) — tab-by-tab

14 tabs, driven by `dashboard/static/js/settings.js` filling `#<category>-settings-form` from `GET /api/settings/all` (which returns EVERY config_type in the DB; categories without a matching container are silently dropped).

| Tab (`data-category`) | State | Problem |
|---|---|---|
| Account | works | App-global password change — does not belong on a DEX page; duplicates `/settings`. Save/Revert header buttons hidden on this tab (correct) but the page title still says "DEX Settings". |
| 📦 Modules | **BROKEN — permanent spinner** | Container is `#modules-container`, but settings.js only fills `#<cat>-settings-form`; **no JS anywhere targets `modules-container`** (repo-wide grep). "Loading modules..." never resolves. Duplicates `/modules` + `/module-control` anyway. |
| General | populated (5 keys) | App-global, not DEX. Belongs in `/config`. |
| Portfolio | populated (13 keys) | App-global-ish; overlaps allocation-guard on `/global-settings`. |
| Risk | populated (17 keys) | Legitimately DEX (engine reads `risk_management.stop_loss_pct` etc.). Keep. |
| Trading | populated (16 keys) | DEX. Keep. Contains a hardcoded "Wave-13: seed `min_vol_liq_ratio` via DB" advisory banner — stale doc rot baked into the template. |
| Strategies | populated (27 keys) | DEX. Keep. |
| Chains | populated (28 keys) | DEX/chain. Keep. |
| Positions | populated (10 keys) | DEX. Keep. |
| API | populated (4 keys) | App-global; overlaps `/settings/rpc-api` + `/credentials`. |
| Monitoring | populated (4 keys) | App-global. |
| ML Models | populated (7 keys) | DEX. Keep. Has a second hardcoded Wave-13 "seed these keys via migration" banner + a `setTimeout(1500ms)` ensemble-status probe. |
| Features | populated (4 keys) | App-global feature flags. |
| Sensitive | works | Full duplicate of `/credentials` (encrypted-secret CRUD) sitting on a module page. |

Net: of 14 tabs, **1 is hard-broken (Modules)**, **7 are app-global content masquerading as DEX settings** (Account, Modules, General, Portfolio, API, Monitoring, Features, Sensitive), and only 6 (Risk, Trading, Strategies, Chains, Positions, ML Models) are actually DEX. There is **no Guide tab** (arbitrage and advisor have one). This template predates the per-module settings pattern — it is the old app-global settings page relabeled "DEX Settings", which is exactly the operator's "total mess" complaint.

---

## (e) Target information architecture

### Principles
- One overview. One control surface. One approvals inbox. One settings engine. Per module: exactly 5 pages (trading) or panel+settings (advisory/intel).
- Every removed URL 302s to its successor — no 404'd bookmarks.
- Settings pages standardize on the **arbitrage/advisor pattern**: two tabs — **Settings** (module-scoped config_types via the `settings_generic.html` engine) + **Guide** (the doc content that copy/AI/sniper currently inline-append).

### Target sidebar
```
Control Center            /control-center      (home; '/' 302s here; absorbs Manual-Trade + Panic-Sell from pro-controls)
Analytics                 /analytics           (the ONE cross-module chart/risk page)
── Trading modules (identical 5-page set) ──
DEX | Futures | Solana | Sniper | Arbitrage | Copy Trading | Polymarket
    Dashboard / Positions / Trades / Performance / Settings(2 tabs)
    (Copy keeps Discovery + Wallet Monitor as its 2 extra pages; Arbitrage Positions stays the explainer)
── Advisory ──
Financial Advisor (6 pages, fix NaN bug) · AI Analysis (drop /ai/sentiment into dashboard; keep 4)
Intelligence & Ops → 16 × /module/<key>  (fix UNKNOWN badge; header Settings → /config/<type>)
Proposals                 /proposals           (absorbs /orchestrator + /allocation approval queues as tabs)
── System ──
App Settings              /config              (canonical; absorbs /global-settings allocation-guard card)
RPC/API · Telegram · Credentials · Wallet Balances · Logs · Simulator · Backtest Replay · Help
(Account page /settings stays user-menu-only; Test Runner moves behind admin flag)
```

### DELETE / REDIRECT plan (file-level)

| # | Action | Route(s) | Template / file | Migration note |
|---|---|---|---|---|
| 1 | REDIRECT | `/` → 302 `/control-center` | delete `index.html` | Header portfolio widget already global; move Recent-Trades-all-modules table into control-center if missed. |
| 2 | DELETE+REDIRECT | `/full-dashboard` → 302 `/analytics` | delete `full_dashboard.html` | Unique widgets worth salvaging into /analytics: Chain ROI, Fee Analysis, Drawdown (once fed). |
| 3 | DELETE+REDIRECT | `/module-control` → 302 `/control-center` | delete `module_control.html` | Control-center cards already do enable/pause/restart/Go-LIVE. |
| 4 | DELETE+REDIRECT | `/pro-controls` → 302 `/control-center` | delete `pro_controls.html` | FIRST move Manual Trade + PANIC SELL ALL widgets into control_center.html (they are unique). |
| 5 | DELETE+REDIRECT | `/modules` → 302 `/control-center` | delete `modules.html` (both ED fallback + MR copies) | Capital-allocation summary moves to `/allocation`-tab inside /proposals. |
| 6 | DELETE+REDIRECT | `/global-settings` → 302 `/config` | delete `global_settings.html` | Port the Allocation-Guard card into `/config` as a pinned card (or into the Proposals→Allocation tab). |
| 7 | REBUILD | `/dex/settings` | rewrite `settings_dex.html` | Two tabs only: Settings (config_types: trading, risk_management, strategies, chain, position_management, ml_models — rendered by the settings_generic engine) + Guide. Drop Account/Modules/General/Portfolio/API/Monitoring/Features/Sensitive tabs; kill both hardcoded Wave-13 banners. |
| 8 | REDIRECT | `/trades` `/positions` `/performance` `/reports` `/backtest` `/analysis` → 301 `/dex/*` | — | Replace the duplicate `add_get` registrations with one redirect handler. |
| 9 | DELETE+REDIRECT | `/dex/reports`, `/dex/analysis` → 302 `/dex/performance` | delete `reports.html`, `analysis.html` (+ `analysis.js`) after salvaging any unique chart | Cuts DEX from 8 pages to the standard 5 (+1 while backtest lives). |
| 10 | DELETE+REDIRECT | `/dex/backtest` → 302 `/backtest-replay` | delete `backtest.html`, `backtest.js` | backtest_replay is the maintained simulator. |
| 11 | MERGE | `/orchestrator`, `/allocation` → tabs inside `/proposals` | keep templates initially; then fold into `proposals.html`; old URLs 302 `/proposals#orchestrator` / `#allocation` | One approvals inbox; score-trend + history tables come along. |
| 12 | DELETE | `/modules/{name}`, `/modules/{name}/details` → 302 `/module/{name}`; `/modules/{name}/configure` → 302 `/config/{type}` | delete `module_details.html`; remove dangling `module_configure.html` + `error.html` references in `monitoring/module_routes.py` | These render-or-500 today. |
| 13 | FIX-OR-DELETE | `/users` | `auth_routes.py:38` opens nonexistent `users.html` → 500 | Either ship a minimal admin users page or remove the route. |
| 14 | DELETE | — | `positions_arbitrage.html` (zero refs), `monitoring/dashboard.py` (legacy, unimported) | Pure dead code. |
| 15 | LINK-OR-FOLD | `/copytrading/leaders` | `leaders_copytrading.html` | Add to Copy submenu or merge into discovery page. |
| 16 | STANDARDIZE | AI/Copy/Sniper settings guides | `settings_ai.html`, `settings_copytrading.html`, `settings_sniper.html` | Move each inline "Complete User Guide" into a Guide tab (arbitrage pattern); shrinks page weight massively. |
| 17 | DEDUP within modules | dashboard vs performance charts | all `dashboard_*.html` | Dashboard keeps ONE compact PnL sparkline + live status; full chart set lives only on performance. Fixes arb's broken duplicate chart by deletion. |
| 18 | DEMOTE | `/test-runner`, `/__routes__` | — | Admin-gate; remove from main sidebar. |

Rollout: one commit per row (≤200 lines each), redirects land in the same commit as the deletion. Sidebar edits in base.html last, after all redirects exist.

### Data-integrity bugs surfaced by the audit (separate backlog, not IA)
- `/advisor/advice` + `/advisor/simulations`: backend emits `NaN` in JSON (`try_low`, `y_price`) — hard page break; serialize with `NaN→null`.
- Cross-page P&L disagreement (`/` $8.9M vs others) — the four overview pages use different aggregation queries; after consolidation only control-center + analytics remain, both must share one aggregation API.
- Solana: "NaNm" AGE, +2000.00% clamp displayed raw, −134.5% drawdown. Sniper: dashboard vs performance vs trades mutually contradictory (−$4,898 / −$537 / 0 trades). DEX: $0.00 entry/exit on closed positions; 105,133% volatility. Wallet-balances page entirely $0.00 (dead feed). `/module/*` status badge always "UNKNOWN"; `/help` header pill stuck "Loading…".
