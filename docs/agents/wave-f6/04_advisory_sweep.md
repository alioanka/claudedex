# Wave-F6 — Advisory Sweep (COPY_TRADING / ADVISOR / AI / POLYMARKET)

READ-ONLY post-F5 validation. Repo `/home/user/claudedex`, branch `claude/friendly-ramanujan-nMWNv`.
Logs cover **2026-07-07 → 2026-07-09** (~3 days; Wave-F5 deployed ~07-07, so this is entirely post-F5). All four modules DRY_RUN.

Legend for F5-fix verdicts: **ENGAGED** = fix's log line / behaviour observed; **NOT DEPLOYED** = fix code present but never fired (gated off in DB or path not reached); **BROKEN** = fired but wrong.

---

## 1. COPY_TRADING  (`logs/copy_trading/`)

### stderr triage — the 9.1 MB file is 98% one line
`stderr.log` = 61,305 lines; `copy_trading.log` = 1,354 lines. Level buckets in stderr:

| logger / level | count | note |
|---|---|---|
| PoolEngine WARNING | 59,827 | **the bloat** — Helius rate-limit spam |
| CopyTradingEngine INFO | 744 | 5-min stats heartbeats |
| CopyTradingEngine WARNING | 599 | 578 = "Solana RPC rate limited in fallback poll" |
| PoolEngine INFO | 124 | |
| **ERROR / CRITICAL** | **0** | clean — no exceptions in 3 days |

Per-day stderr lines: 07-07 = 5,755 · 07-08 = 29,859 · 07-09 = 25,691. The growth is pure Helius rate-limit WARN. Per-day `Rate limited: HELIUS_API` counts: 1,242 / 6,679 / 5,715, spread evenly across all 5 rotating keys (~2,730 each). The multi-key rotation IS working (5 distinct endpoints: `SOL #1..3`, `Helius API #5`, `HELIUS_API_KEY`) but **the aggregate free-tier call rate still saturates every key** — pacing is set to only 8 req/s (`Copy: Helius outbound pacing set to 8 req/s`, one line, engine default `copy_helius_rps=8.0` at `copy_engine.py:2901`).

### F5-fix verdicts
| F5 fix | Verdict | Evidence |
|---|---|---|
| Discovery v3 sweep (mig 135/141) finds NON-tracked wallets | **NOT DEPLOYED** | Zero `[discovery-v3]` / `CopyDiscoveryV3` lines in either log. Sweep is gated behind `copy_auto_discovery_enabled` (`copy_engine.py:1801`); mig 141 only flips it to true `WHERE value` is still the default string — evidently the stored value was not the exact default (or mig 141 unapplied), so `_maybe_run_discovery_v3` returns immediately. **Discovery never ran.** |
| Shadow-copy simulator (`copy_shadow_sim_enabled`) | **NOT DEPLOYED** | Zero `CopyShadowSim` lines. Same gate story (`copy_engine.py:1829`). |
| Helius per-key daily budget + 429 rotation | **PARTIAL** | Rotation across 5 keys confirmed; NO `copy_helius_budget` / "budget" line appears (budget counter lives in discovery path, which never ran). |
| Leader lifecycle / dead-leader flagging | **ENGAGED** | 11× `DEAD LEADERS 3/… — suggest removing`; 11× `Leader refresh complete: N active, N dead-flagged`; reconcile drift warnings present. |
| Reconcile (leader holdings drift, advisory) | **ENGAGED** | 9× "reconciled — no holdings drift" + 8× `[reconcile] reason=leader_exited … advisory; not auto-closed`. |

### Activity / quality
- Only **4 target wallets** loaded; 3 of them **DEAD** (no on-chain activity 3+ days) → effectively 1 live leader.
- **EVM Copies: 0 | Solana Copies: 0** for all 708 stat windows. Zero copy activity over 3 days.
- Net: the module is idling on a near-empty, mostly-dead wallet list, and the one feature meant to fix that (discovery v3) never turned on.

### New issues (ranked)
1. **[HIGH] Discovery v3 + shadow sim never engaged** — the headline F5 deliverable. Root cause is the conditional `UPDATE … WHERE value = <default>` in `migrations/141_copy_discovery_helius_quota.sql:90-96`; if the row was ever hand-edited the flip is skipped. Fix: operator sets `copytrading_config.copy_auto_discovery_enabled=true` + `copy_shadow_sim_enabled=true` in DB directly, or re-run 141 unconditionally. Verify by watching for `[discovery-v3] sweep:` (`discovery_v3.py:963`).
2. **[MED] Helius rate-limit WARN floods stderr (98% of 9.1 MB)** — pacing 8 req/s is still too high for 5 pooled free keys under the 33-wallet fan-out. `copy_engine.py:2901` reads `copy_helius_rps` default 8.0. Fix: lower `copytrading_config.copy_helius_rps` to ~2-3, and/or demote the repeated PoolEngine "All HELIUS endpoints rate-limited" to DEBUG after first-per-window in `config/pool_engine.py` (it already demotes the *counted* line but not the "using least-penalized" line).
3. **[LOW] Dead leaders never auto-removed** — engine flags but by design won't prune (`copy_engine.py` DEAD LEADERS warning). With 3/4 dead the module is inert; operator must curate `target_wallets`.

---

## 2. ADVISOR  (`logs/advisor/`)

### stderr / error triage
`advisor_errors.log`: 454 / 731 / 626 lines per day (07-07/08/09). Dominant buckets:

| message | count | verdict |
|---|---|---|
| `kronos_forecaster … Inference error: klines missing OHLC columns; have ['close']` | **1,650** | **NEW BUG** (see below) — 420/690/540 per day |
| `telegram_notifier` rate-limit / Bad Gateway (429, 502) | 93 | transient, self-retrying |
| `BISTAnalyzer cycle summary: … failed ALL sources (fonoloji/borsapy/yfinance)` | 55 | KOZAL/KOZAA/BETAE/ISKUR/GOLDA persistently unfetchable |
| `[llm_budget] … denied … falling back to rule-based` (advisor ceiling 60, kap 30) | ~7 | budget caps engaging as designed |
| `fonoloji HTTP 451 (legally restricted)` short-circuit | 3 | breaker firing correctly, once/day/symbol |

### F5-fix verdicts
| F5 fix | Verdict | Evidence |
|---|---|---|
| Midas `await` bug fixed → fund advice revived | **ENGAGED** | `midas_funds` is in every `[advice] Cycle #… markets=[…,'midas_funds']`; 13 `Opened sim … channel=midas_funds` (IJC) on 07-09. Midas is producing advice + sims again. |
| Per-channel sim-cap no longer blocks PUBLISHING | **ENGAGED** | `Cycle #54 sim-cap demotions (advice published, sim skipped): bist x102 …` — advice still published while sim skipped. Working as intended. |
| NaN-safe JSON + entry-price finite guard (mig 138) | **ENGAGED (code)** | Guard live at `core/portfolio_engine.py:135`; no NaN persistence errors in logs. |
| Fonoloji HTTP-451 circuit breaker | **ENGAGED** | breaker fires once/day/symbol then short-circuits (no budget burn). |
| BIST source swap (borsapy baked in) + yfinance fallback | **PARTIAL / DEGRADED** | Chain is wired (fonoloji→borsapy→yfinance) but 5 symbols (KOZAL, KOZAA, BETAE.IS, ISKUR.IS, GOLDA.IS) fail **ALL THREE** sources 14-22×/day. yfinance itself times out (`Failed to get ticker 'KOZAA.IS' … Operation timed out`). yfinance is *reachable* for AKBNK (sim #164 opened) but not these tickers. |
| KAP bounded re-queue of budget-starved UNCLASSIFIED | **NOT OBSERVED** | Zero `re-queue`/`UNCLASSIFIED`/`budget-starved` lines. KAP classifier runs fine (1-2 classified/cycle, alerts sent) and only 3 kap budget-denials/period, so the re-queue path simply wasn't exercised — cannot confirm it works, but no backlog symptom either. |

### Activity / quality
- **Advice PUBLISHED per day: 1,767 / 3,023 / 2,799** (14 / 22 / 19 cycles). Publishing is healthy and well above 0/day — the F5 "publishing revived" goal is met.
- Markets active: crypto, us_equities, fx, bist, midas_funds + discovery ("gems").
- Sim positions advancing (#157 → #174).

### New issues (ranked)
1. **[HIGH] Kronos forecaster fires 1,650 inference errors (`klines missing OHLC columns; have ['close']`)** — Kronos requires open/high/low/close (`core/kronos_forecaster.py:358-361` raises when any of OHLC absent) but some analyzers hand it a **close-only** DataFrame. The BIST/midas analyzers *synthesise* missing OHLC from close (`analyzers/bist.py:593`, `:1002`) but the frame reaching Kronos via `advice_engine.py:264` (`result.extra['klines_df']`) is not synthesised for whichever source returns close-only. Fix: synthesise OHLC (or skip Kronos) before `self.kronos.predict(klines)` at `core/advice_engine.py:265`, mirroring the bist.py:593 synthesis, OR make `_slice_klines` in `kronos_forecaster.py:358` fall back to `close` for o/h/l when they're missing. Currently every such symbol spams an error and gets a null Kronos signal.
2. **[MED] 5 BIST symbols unfetchable across all 3 sources** — KOZAL/KOZAA/BETAE.IS/ISKUR.IS/GOLDA.IS fail fonoloji(451)+borsapy+yfinance(timeout), 14-22×/day. These are in the `bist50` universe (`core/analyzers/universes.py:124,131`). Fix: either drop the chronically-dead tickers from the default universe or add a per-symbol "all-sources-dead" cooldown so they aren't retried every cycle.
3. **[LOW] Telegram 429/502 (93 events)** — self-retrying; consider a longer backoff in `core/telegram_notifier.py` if operator sees dropped alerts (2 "Failed after N attempts").

---

## 3. AI  (`logs/ai_analysis/`)

### stderr / error triage
`ai_errors.log` **empty**; `stderr.log` ERROR/CRITICAL = **0**. Only meaningful WARNING: 1× `cryptocompare SKIPPED: no CRYPTOCOMPARE_API_KEY` (fires once, key-gated as designed — F5 401-spam fix ENGAGED). stderr per day 1,181/2,032/1,614 = normal INFO volume.

### F5-fix verdicts
| F5 fix | Verdict | Evidence |
|---|---|---|
| LLM budget daily cap 150 + per-module reservations (ai=40) | **ENGAGED** | 3× `[llm_budget] denied … module=ai … shared pool exhausted. total=150/150 used_by_module=40` — one per day. AI consumes exactly its 40 reservation, then the shared 150 pool blocks further paid calls → rule-based fallback. Cap + reservation both honoured. |
| Real analyses/day (not budget-starved to zero) | **ENGAGED** | 242 `sentiment cycle tick` over 3 days (63/96/83); **86** actual `Claude API Request` (6/40/40) — 40/day = the reservation ceiling, hit cleanly. Claude model `claude-haiku-4-5-20251001`. |
| Confirmation filter (mig 139, filter-only) | **ENGAGED** | 103 `[ai-skip] reason=confirmation_not_met conf=… sentiment=… symbol=… min_conf=…` (BTC 35, ETH 39, SOL 29). Filter is actively skipping entries, never adding. |
| SHORT `profit_loss` sign fix + per-signal position cap | **ENGAGED** | 47× `AI Signal LONG … (cap=N, open=N)`, 2× `[ai] per-signal cap reached (N/N); not opening more`. Cap enforced. |
| CryptoCompare key-gated (401 spam killed) | **ENGAGED** | single once-only skip warning; no 401 loop. |
| claude_api.log spend | present | 200 KB, request+prompt+response logged; 40 paid req/day post-budget. |

### Activity / quality
- Symbols fixed to BTC/ETH/SOL, `max_positions=3`, `per_signal=1`, trade amount $50, DRY_RUN.
- **Total trades plateaued at 47** (43→47 over the window) — only ~4 new positions in 3 days; the confirmation filter + per-signal cap + 3-position ceiling mean most cycles open nothing (`44× [ai] No new positions opened this cycle`). This is *by design* (filter-only, tight caps) but worth noting the module is now very conservative.

### New issues (ranked)
1. **[LOW/OBSERVATION] AI is near-inert on new positions** — 44/242 cycles explicitly opened nothing; net +4 trades/3d. Not a bug (filter + caps working) but if operator wants signal, `ai_confirmation` min_conf and the 3-position ceiling are the knobs. No code fix.
2. **[LOW] AI hits its 40-call reservation every day then falls back to rule-based for the rest of the day** — expected under the shared 150 cap, but means late-day cycles never get LLM sentiment. If AI is the priority consumer, raise its reservation in `core/llm_budget` module-reservation map. No defect.

---

## 4. POLYMARKET  (`logs/polymarket/`)

### stderr / error triage
`polymarket_errors.log` empty; `polymarket.log` = `stderr.log` (identical, 538 KB). ERROR/CRITICAL = **0**. Only 2 WARNING total: 1× `Gamma fetch failed (/markets): TimeoutError` + its `No markets fetched` — a single transient Gamma timeout. Loggers: 5,518 PolymarketModule INFO, 2 WARN. Very clean.

### F5-fix verdicts
| F5 fix (mig 137) | Verdict | Evidence |
|---|---|---|
| Signal-quality gates cut the ~1,700/day noise | **ENGAGED** | Total signals now **583 / 651 / 675 per day** (momentum + new_market) — roughly a **60% cut** vs the F5-cited 1,700/day. Gates (`momentum_min_liquidity_usd=10000`, `min_volume_24h_usd=5000`, `min_move_frac=0.05`, `min_score=0.3`) applied at `main_polymarket.py:224-231`. |
| YES+NO<1 arb **false positives** eliminated (edge cap) | **ENGAGED** | **Zero `[arb]` signals in 3 days.** The `arb_max_edge_bps=500` too-good-to-be-true cap + `min_arb_edge_bps=100` + `fee_gas_buffer_bps=100` (`main_polymarket.py:193-197`) killed the phantom risk-free arbs entirely. |
| Real `new_market` age check (`new_market_max_age_hours=24`) | **ENGAGED** | 201 `[new_market] … dir=WATCH` signals (22/67/112/day), age-gated at `main_polymarket.py:230`. |
| Flip cooldown (`momentum_flip_cooldown_minutes=30`) | **ENGAGED (code)** | `_flip_suppressed` wired at `main_polymarket.py:234`; no direct log line but momentum flip volume is bounded. |
| Forward-outcome marking (`polymarket_price_snapshots` / `polymarket_signal_outcomes`, LATE) | **RUNNING, UNVERIFIABLE from logs** | `_save_snapshots` + `_mark_outcomes` called every cycle (`main_polymarket.py:246-247`); zero insert-failure errors → they ran, but **neither emits a success/heartbeat log line**, so I cannot confirm rows were written or any signal was marked LATE. Needs a DB check or a stats log line. |

### Activity / quality
- Shadow-mode confirmed: `engine loop starting (shadow_mode=True, live_execution_enabled=False)`.
- Signal mix: 1,708 momentum (868 NO / 840 YES) + 201 new_market over 3 days. No arb.
- Momentum still ~560/day — reduced but the single largest signal source; many are repeat `dir=NO` on the same markets (throttle exists at `_throttled`).

### New issues (ranked)
1. **[MED] Outcome-marking is a black box in logs** — `_mark_outcomes` / `_save_snapshots` (`main_polymarket.py:245-247`) log only on error. The whole point of F5's forward-outcome tracking is the edge-proof scorecard, but there is **no log evidence any signal was marked LATE or any snapshot written**. Fix: add a per-cycle INFO summarising `snapshots_last_cycle` (stat already tracked at `main_polymarket.py:277`) and count of outcomes marked, so operators/agents can validate the edge pipeline without DB access.
2. **[LOW] Momentum still ~560/day, heavily repeated** — gates cut volume 60% but momentum on liquid markets re-fires each cycle (868 NO / 840 YES). If still noisy for the operator, raise `momentum_min_score` (0.3) or lengthen `_throttled` window. Not a defect.
3. **[LOW] No periodic stats/heartbeat line** — the `self.stats` dict (`main_polymarket.py:119-120`) is never logged. A 5-min heartbeat (like copy_trading's) would make markets_seen / arb_signals / momentum_signals / snapshots observable.

---

## Cross-module summary of F5-fix verdicts
| Module | ENGAGED | NOT DEPLOYED / DEGRADED | New bugs |
|---|---|---|---|
| COPY | leader lifecycle, reconcile, key rotation | **discovery v3 + shadow sim (never turned on)** | Helius WARN flood (9.1 MB), rate saturation |
| ADVISOR | Midas revived, sim-cap publish, 451 breaker, budget caps, publishing 1.7-3k/day | BIST 5-symbol all-source failure; KAP re-queue unobserved | **Kronos 1,650 OHLC errors/window** |
| AI | budget 150+reservation 40, confirmation filter, per-signal cap, cryptocompare key-gate | — | none (module very conservative: +4 trades/3d) |
| POLYMARKET | noise -60%, arb FPs → 0, new_market age gate, shadow mode | outcome-marking runs but **unverifiable from logs** | outcome pipeline is a logging black box |
