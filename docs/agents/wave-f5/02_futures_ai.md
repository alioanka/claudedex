# Wave-F5 Report 02 — FUTURES + AI Performance Diagnosis

Analyst: market-trading-analyst (read-only wave). Evidence window: **2026-06-15 21:49 → 2026-07-05 19:03 UTC** (20 days of committed runtime logs, all DRY_RUN). Parsers used to generate every number below are in the session scratchpad (`parse_futures.py`, `parse_ai.py`); method: per-trade PnL reconstructed from `Daily PnL` deltas in `logs/futures_trading/futures_trades.log` (691/691 closes matched to opens), AI PnL from BUY/SELL pair reconstruction in `logs/ai_analysis/ai.log` (24/24 paired). No code was changed.

---

## 1. FUTURES — quantified diagnosis

### 1.1 Headline numbers (log-derived, 20-day window)

| Metric | Value |
|---|---|
| Trades closed | 691 (700 opened; 9 still open) |
| Net PnL | **-$171.15** (dashboard cumulative `Total PnL` -$178.81 incl. open-window drift; dashboard all-time 1000-trade view: -$146.64, PF 0.76, Sharpe -1.56) |
| Win rate | 39.1% |
| Profit factor | **0.55** (gross win $209.99 / gross loss $381.14) |
| Avg win / avg loss | +$0.78 / -$0.91 → **break-even win rate 53.8%**, observed 39.1% |
| Fees paid | ≈ **$56** (avg $0.0816/trade × 691; 0.08% round-trip taker on $135/$67.50 notionals) — **33% of the net loss** |
| Funding | **$0 simulated** — zero `funding` lines in either log. DRY_RUN PnL excludes funding entirely; live would be worse. |
| Best / worst day | +$11.80 (07-05) / -$32.98 (07-01, tripped the $30 daily loss limit) — 15 of 20 days negative |
| Consecutive-loss streaks | up to **20**; consecutive-loss circuit breaker tripped **31 times** in 20 days |

### 1.2 Exit-reason decomposition — the whole loss is the stop-loss bucket

| Exit reason | n | share | win% | PnL | avg |
|---|---|---|---|---|---|
| **SL Hit** | 149 | 21.6% | 0% | **-$250.90** | -$1.68 |
| time_limit (240 min) | 495 | 71.6% | 50.7% | +$69.36 | +$0.14 |
| Signal (reversal) | 43 | 6.2% | 34.9% | +$6.49 | +$0.15 |
| TSL Hit | 4 | 0.6% | 100% | +$3.90 | +$0.98 |
| **TP1..TP4 full exit** | **0** | 0% | — | — | — |

**Zero take-profit exits in 691 trades.** Only 23 TP1 partials and 4 TP2 partials ever fired. The book's realized payoff is: 21.6% of trades lose the full 1.5% stop; the rest are a 4-hour coin flip worth +$0.14 average after fees.

By hold time: every trade closed under 4h is effectively an SL (win rate 0-2%, net -$256); trades that survive to the 4-8h time-limit window are net +$74. By side: LONG -$98.24 (37.8% win), SHORT -$72.91 (40.3%) — losing symmetrically, i.e. no directional regime excuse; the geometry itself is broken.

### 1.3 Why the geometry is broken (root cause #1)

Runtime config (from log lines + `modules/futures_trading/config/futures_config_manager.py`):
- FUT-RM-16 logs `ATR%=0.21–0.55 -> SL=1.50% TP1=3.00% (R:R=2.0)` on essentially every entry. Observed 15m ATR% is 0.21-0.55, so `atr_sl_multiplier × ATR% = 0.3–0.8%` — **the `atr_sl_min_pct=1.5` floor binds ~100% of the time**, and TP1 is rescaled to `2.0 × 1.5% = 3.0%`.
- `max_hold_minutes = 240`. Over 16 fifteen-minute bars, diffusion is ≈ 4×ATR ≈ 0.8-2.2% of price. The SL sits ~1-1.5σ of the 4h horizon; TP1 sits ~2-3.5σ. **TP1 is statistically unreachable before the time limit** — hence 0 TP exits — while the SL is hit 21.6% of the time at full size.
- The advertised "R:R = 2.0" is fictional under a 4h cap: realized R:R is **risk 1.5% to collect a ~+0.1% expected time-limit drift**.

This re-confirms the Wave-24 verdict in `modules/futures_trading/CLAUDE.md` ("SL/TP payoff geometry implies break-even win rate above observed; structurally unprofitable") — the recovery-wave re-enable changed the numbers (SL 1.2→1.5, TP1 1.8→3.0, 4h cap) but reproduced the same class of defect in mirror image: before, TP was too close relative to costs; now it is too far relative to the hold window.

### 1.4 Secondary loss factors (evidence-ranked)

2. **Fee churn on edgeless entries** — 691 trades × $0.082 ≈ $56. 35 trades/day at PF 0.55 is paying the exchange for noise. The volume gate was demoted to diagnostic-only in Wave-13; entries now routinely fire at **0.09-0.41× reference volume** (dead tape — e.g. BTC short at `Volume: 0.09x`, `futures_trading.log` 2026-07-04 23:30:26).
3. **Counter-momentum entries at indicator extremes** — SHORT signals trigger with RSI already 30-33 (oversold: DOGE 32.7, SOL 31.2, AAVE 31.2, LINK 31.1, LTC 30.6 in the last log window alone). Shorting into oversold with a 1.5% stop is how the SL bucket fills. `min_signal_score` is running at **±3 in DB** while the code default is 4 (`futures_config_manager.py:278` — "Increased from 3 for better entries" never made it to the DB row).
4. **Session clustering** — losses concentrate in two UTC windows: **03:00-06:59 = -$66.20** (162 trades, 33% win) and **13:00-16:59 = -$113.41 incl. h13 -$27.00, h15 -$15.97** (US open volatility). Hours 08-12 + 17 are net positive. ~74% of the net loss comes from ~40% of the trading day.
5. **FUT-RM-27 works — but leaks re-entries.** The rolling gate benched the worst symbols (DOT/SUI/ZEC/NEAR/FIL/AVAX have 0-14 trades in the window; benched at startup and mostly stayed out). But mid-tier bleeders cycle back in after the 24h bench: AAVE (-$30.55), ADA (-$28.63), JUP (-$22.73), ETH (-$21.08), LINK (-$19.84) were each benched 3-4 times and still accumulated the top-5 losses. Only **LTC is net positive (+$17.96, 57.1% win over 56 trades)**.
6. **Funding-carry v2 and the FUT-RM-05/19 funding inputs are dark** — zero funding-related log lines in 70k lines. Either `futures_funding_carry_enabled` is off (fine, default) or the funding fetch silently returns nothing; either way DRY_RUN PnL contains no funding drag and the funding gates are untested by live data.

### 1.5 FUTURES tuning proposals (config key → current → proposed)

All keys are DB rows via `FuturesConfigManager` (sections in parentheses); no live flags involved; every change is DRY_RUN-testable.

| # | Key (section) | Current | Proposed | Expected effect |
|---|---|---|---|---|
| F1 | `atr_tp_rr_ratio` (futures_risk) | 2.0 | **1.0** | TP1 = SL = 1.5% ≈ reachable within 4h. Converts part of the 495-trade time-limit coin flip into real TP exits; the 4-leg ladder finally participates. At the observed distribution, TP1 at 1.5% would have fired on roughly the top-quartile of time-limit closes. |
| F2 | `max_hold_minutes` (futures_risk) | 240 | **480** | Alternative/complement to F1: gives TP1 a 32-bar window (2-2.5σ). The >=8h bucket in this sample was 9/9 winners (+$11.18); 4-8h bucket was the only other net-positive bucket. Pick F1 *or* F2 first, not both at once, so attribution stays clean. |
| F3 | `enable_trailing_stop` / TSL distance (futures_risk; dashboard "Trailing Stop Distance" = 1.5) | armed late (4 exits in 691) | **arm at +0.75% price (0.5×SL), trail 0.75%** | TSL exits were 4/4 winners avg +$0.98 — the best per-trade expectancy of any exit path. Locking half-SL when a trade is +0.75% converts time-limit give-backs into keeps. Hard cap stays (no dynamic widening — working-rule compliant). |
| F4 | `min_signal_score` (futures_strategy) | 3 (DB) | **4** (restore code default) | Cuts marginal score-3 entries. Score-3 entries are the ones firing at RSI 31-33 / 0.1-0.4x volume. Fewer trades directly reduces the $56 fee bleed and SL count. |
| F5 | new key `entry_hours_utc_block` (futures_strategy; needs ~15-line guard in `_scan_opportunities`, `modules/futures_trading/core/futures_engine.py` near the FUT-RM-20 candle throttle) | n/a | **block 03:00-06:59 UTC** | Removes the worst session (-$66 in sample, 33% win). Cheap, reversible, measurable within a week at 35 trades/day. |
| F6 | `rolling_gate_bench_minutes` (futures_risk, mig 088) | 1440 | **2880** | AAVE/ADA/JUP/ETH/LINK each got re-benched 3-4×; doubling the bench halves their re-entry bleed (~-$25/window at observed rates). |
| F7 | `rolling_gate_max_win_rate` (futures_risk, mig 088) | 0.45 | **0.48** | Break-even win rate at current realized payoff is 53.8%; benching below 48% instead of 45% benches the bleeders one cycle earlier. Revisit after F1-F3 change the payoff. |
| F8 | volume gate (futures_strategy, demoted Wave-13) | diagnostic-only, ref 0.80x | **hard gate at 0.25x** | Wave-13 demoted it because 0.80x blocked 100% of signals (live range 0.17-0.76x). Reinstating at the *bottom* of the observed range only blocks the truly dead tape (0.09-0.24x entries seen in logs) instead of everything. |
| F9 | RSI sanity bound (futures_strategy; small change at signal assembly in `_scan_opportunities`) | none | **no SHORT if RSI<35, no LONG if RSI>65** | Directly targets the shorting-into-oversold pattern that dominates the SL bucket. FUT-RM-21 blocks counter-*trend*, not counter-*extension*; this closes the gap. |
| F10 | honesty item, not a knob | funding not simulated | add estimated funding accrual to DRY_RUN position PnL (engine close path) | Prevents a false-green: at 3×8h intervals/day and typical 1-3 bps, an always-on 5-10 position book eats a further ~$0.5-1.5/day that today's ledger never sees. |

Sequencing: F4+F5+F8 (entry hygiene, zero-risk) first week; then F1 *xor* F2 plus F3 (geometry) second week; F6/F7 anytime; F9 with the geometry batch; F10 before any GREEN/live conversation. Do **not** flip live on this module — with PF 0.55 the checklist's profitability precondition is nowhere near met; this remains a DRY_RUN tuning program.

---

## 2. AI module — quantified diagnosis

### 2.1 The funnel is starved at the source

From `logs/ai_analysis/ai.log` (36.5k lines, 2026-06-15 → 07-05):

| Funnel stage | Count | Rate |
|---|---|---|
| Sentiment cycles (15-min) | 1,910 | — |
| Real LLM analyses (provider=anthropic) | **30** | **1.6% of cycles** |
| Cycles returning `provider=none, score=0.00, cost=$0` | 1,880 | 98.4% |
| `[ai-skip] reason=zero_sentiment` | 1,880 | matches 1:1 |
| `[ai-skip] reason=confidence_below_threshold` | 1,886 | the zero-analyses again (conf 0.00 < 0.35) |
| `position_exists` / `cooldown_active` skips | 30 / 18 | |
| Trades opened (window) | 24 | ~1 basket of 3 per real signal day |

**Root cause (P0): the bot-wide shared LLM budget is set to 10 calls/day and the Advisor drains it within ~25 minutes of midnight UTC.** Evidence:
- `logs/.llm_budget.json` = `{"day": "2026-07-05", "count": 10}`.
- `logs/advisor/advisor.log` 2026-07-05 00:24:35: `[llm_budget] Daily paid-LLM budget reached (10/10 for 2026-07-05, last kind=advice_rationale)`.
- Every one of the 30 successful AI analyses is timestamped **00:04–00:37 UTC** — the module only wins the race for the first 1-3 of the day's 10 calls, then `core/llm_budget.try_consume` (`modules/ai_analysis/core/ai_provider.py:477`) returns False for the next ~23.5h and `analyze_sentiment` falls through to the `provider='none'` branch (`ai_provider.py:419-428`).
- The cap resolves to 10, not the code default 25 (`core/llm_budget.py:42` `_DEFAULT_DAILY_MAX=25`; resolution order explicit `cap=` → env `BOT_LLM_DAILY_MAX_CALLS` → config `llm_daily_max_calls`/`advisor_llm_daily_max_calls` → 25) — so someone set 10 via env or config.
- Cost reality check: the 30 real calls cost ~$0.001 each (claude-haiku-4-5, ~410 in / ~140 out tokens per `claude_api.log`). The module's own `daily_budget_usd=$10` has never been >0.1% consumed. The 10-call cap protects ~**$0.01/day** while reducing the module to one stale reading per day.

### 2.2 Secondary defects

2. **Dead news source still being polled** — `cryptocompare returned HTTP 401 (API key required)` on **every cycle for 20 days** (~1,900 warnings). Wave-13 claimed "dead news source dropped"; it was not. Headline supply is CoinDesk-only (10 headlines, largely static intraday), so even with budget the intraday signal would be low-variance.
3. **Signal design: one market-wide score → 3 correlated positions.** Every trade day opens LONG or SHORT **BTC+ETH+SOL simultaneously at $50 each** on the same score — that is one $150 bet with 3x line-item concentration, taken once per day at 00:04-00:12 on headlines already ~daily-stale by construction.
4. **Exit geometry repeats the futures mistake at daily scale** — TP=6% / SL=-3% / max_hold=24h on majors with ~1.5-2.5% daily vol: 12 of 24 trades expired at exactly 24.0h; SL fired on the 06-18 and 06-23 baskets; TP never fired. The `Signal Reversal` exit (min_score 0.5) *cannot* fire intraday because there is no intraday analysis (budget starvation) — the exits are structurally blind.
5. **Confidence threshold is a dead knob at current provider behavior** — Wave-13 lowered 0.50→0.35, but realized confidences are quantized at 0.72-0.78 and scores at ±0.15/±0.35/±0.72. The threshold change did nothing; the binding constraint was and is the budget gate. (Score 0.15 days correctly don't trade; ±0.35/±0.72 days always trade.)
6. **Cross-source confirmation filter is OFF** — zero `confirmation` lines in ai.log; `ai_confirmation_signal_enabled` (mig 102, filter-only, no LLM spend) was never enabled.

### 2.3 Realized AI performance (window)

24 trades: **net -$9.29 on $50 notionals (-0.77%/trade), win rate 33.3%**, avg hold 19.3h. LONG baskets -$11.41 (12 trades), SHORT baskets +$2.12 (12). Symbols indistinguishable (BTC -$3.84 / ETH -$2.65 / SOL -$2.81) — as expected when all three mirror one score.

**P0 dashboard reporting bug: AI Performance shows Total P&L +$18.30; the truth is negative. SHORT-trade P&L signs are inverted in the dashboard.** Verifiable from the trade-history table vs the log: 25.06 SHORT SOL 68.23→64.13 is **+$2.94** in reality (price fell, short wins) but rendered **-$3.00**; 01.07 SHORT SOL 73.68→76.02 is **-$1.68** in reality but rendered **+$1.59** (same inversion on the ETH/BTC legs). Every "all-time 75 trades, 41.3% win, +$18.30" aggregate on `/ai/performance` is untrustworthy until the sign convention in the dashboard's AI P&L computation (ai_trades read path in `modules/dashboard/enhanced_dashboard.py`) is fixed. Flagging to the dashboard/backend agent.

### 2.4 AI tuning proposals

| # | Key / location | Current | Proposed | Expected effect |
|---|---|---|---|---|
| A1 | `BOT_LLM_DAILY_MAX_CALLS` (env) or `llm_daily_max_calls` (config) | 10 (shared, advisor-drained) | **150** | At measured ~$0.001/call ceiling cost is ~$0.15/day. Restores ~96 AI cycles/day + advisor's 10-50. This single change un-bricks the module; nothing else matters until it lands. |
| A2 | `core/llm_budget.py` enhancement | one global counter | **per-module reservation** (e.g. `llm_daily_max_calls_ai`, `_advisor`) so the advisor cannot starve AI regardless of totals | Removes the midnight race permanently; ~20 lines, keeps the global backstop. |
| A3 | news sources (`sentiment_engine.py` fetch list) | CryptoCompare 401 every cycle + CoinDesk | **remove or key the CryptoCompare call** | Kills 1,900 junk warnings; if keyed, roughly doubles headline variance so intraday analyses aren't cache-identical. |
| A4 | `ai_confirmation_signal_enabled` (mig 102) | false | **true** | Filter-only tape-math confirmation; can only skip entries, never add. Free safety on the restored intraday flow. |
| A5 | basket structure (`sentiment_symbols` / sizing in `sentiment_engine.py`) | BTC+ETH+SOL × $50 on one score | **BTC only × $50** (or per-symbol scores if A1 lands) | Same information content, one-third the correlated exposure. |
| A6 | `stop_loss_pct` / `take_profit_pct` / `max_hold_hours` (ai_config) | -3 / +6 / 24h | **-2 / +2.5 / 48h**, and make Signal-Reversal the primary exit once intraday analyses exist | Matches the barrier distances to majors' realized daily vol so exits are informative rather than 24h-timer coin flips. |
| A7 | `confidence_threshold` | 0.35 | leave; instead gate on **min abs sentiment 0.35** explicitly (already the de-facto behavior) | Documents reality; the 0.50→0.35 Wave-13 change was a no-op and should not be credited/blamed in future tuning. |

### 2.5 Verdict inputs
- Do not judge the AI strategy's alpha on this sample: 20 days at one stale reading/day is not a test of the sentiment thesis. Fix A1/A2/A3, run 3-4 weeks of true 15-min cadence in DRY_RUN, then evaluate against the same PF/win-rate bar as futures.
- The +$18.30 dashboard figure must not be used in any go/no-go or meta_controller/orchestrator scoring until the SHORT sign inversion is fixed — if `orchestrator_recommendations` or `meta_decisions` consume `ai_trades`-derived PnL, verify which sign they see (risk of a wrong ACTIVATE).

---

## 3. Enhancement notes (cross-module)
- **Shared-resource starvation is a new failure class**: `logs/.llm_budget.json` is the first cross-module contended resource that silently degrades one module (AI) because of another's (advisor) consumption pattern. Sentinel has a "silent module" anomaly check — it did not fire because AI kept heartbeating while producing zeros. Consider a sentinel rule: "module produced 0 non-degraded signals for 24h while ticking" (the `provider=none` / `zero_sentiment` ratio is directly greppable).
- **DRY_RUN cost honesty**: neither futures (funding) nor AI (slippage on $50 market orders) simulates the full live cost stack. Before any GREEN verdict, both ledgers need funding/slippage lines, or live results will undershoot paper by a predictable margin.
- **Dashboard trust**: two independent reporting defects surfaced in one review (AI short-sign inversion; futures dashboard "Total P&L" widows differing across pages). A small reconciliation job — dashboard aggregates vs log-ledger recompute, alert on >5% divergence — would catch this class permanently.
- The futures per-symbol table (dashboard 1000-trade view) agrees with the 691-trade log window on ordering (LTC best; AAVE/ADA/ETH/LINK worst), which validates the FUT-RM-27 tiering data source itself — the gate's inputs are sound, only its thresholds are too forgiving.
