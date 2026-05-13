# DASHBOARD Module — Quant / Math-Truth Audit

**Auditor:** quant-algo-expert (20+ yrs)
**Date:** 2026-05-11
**Question being answered:** *Does the math the dashboard SHOWS match the math the engines DO?*
**Scope:** `monitoring/analytics_routes.py`, `monitoring/performance.py`, `monitoring/enhanced_dashboard.py` (performance/charts/backtest/copytrading endpoints), `core/analytics_engine.py`, `core/pnl_tracker.py`, plus `dashboard/static/js/{analytics,performance,backtest}.js` and templates `analytics.html`, `performance.html`, `backtest.html`, `reports.html`.

---

## 1. Executive verdict

The dashboard is honest about **trade-level** P&L for the DEX module but **misleading** about every risk-adjusted metric and **broken** in several places where templates show numbers the API never produces. Specifically:

- There are **at least three different `sharpe_ratio` implementations** in the codebase, each annualising with a different factor (252 vs 365) and each operating on a different units convention (per-trade ROI vs daily $-PnL). They produce numerically different Sharpe values for the same trade set, depending on which surface the user looks at.
- The dashboard's **backtest is dishonest**: `monitoring/enhanced_dashboard.py:7259-7322` replays *closed historical trades* but recomputes a fake P&L using `roi = (exit - entry)/entry` × 10 % of initial balance — **no fees, no gas, no funding, no slippage, no MEV cost** — and the backtest payload still hard-codes `sharpe_ratio: 0, sortino_ratio: 0` placeholders (lines 7318-7319) that the front-end then renders as "Sharpe = 0.00".
- The **copy-trading P&L surface mixes dry-run and live trades** (`enhanced_dashboard.py:8701-8732` has no `is_simulated = false` filter).
- The **drawdown shown is per-realized-trade only**, not per-snapshot-of-equity-including-open-positions. The bot can be in a 50 % MTM hole and the dashboard reads 0 %.
- **No sample-size warnings, no confidence intervals.** Any metric with N<30 is shown with two decimal places and no caveat. That's how you ruin a discretionary trader who trusts the screen.

Verdict: **C+** for the read-only DEX view, **D** for the risk surface, **F** for the backtest tool, **D** for copy-trading P&L. None of this is unfixable; see DASH-Q-01..15.

---

## 2. Architectural map (where each number comes from)

The dashboard pulls from **two** parallel stacks. The user does not know which one is rendering on which page.

| Page / Endpoint | Backend file | Calculation engine | Storage |
|---|---|---|---|
| `/analytics` page | `monitoring/analytics_routes.py` | `core/analytics_engine.py` | live DB query (`trades`, `futures_trades`, `solana_trades`) |
| `/api/performance/metrics` | `monitoring/enhanced_dashboard.py:4077` | inline pandas (no AnalyticsEngine) | live DB query (`trades` only) |
| `/api/performance/charts` | `monitoring/enhanced_dashboard.py:4212` | inline pandas | live DB query (`trades` only) |
| `/api/backtest/run` | `monitoring/enhanced_dashboard.py:7217` | inline pandas | replays `trades` rows |
| `/api/copytrading/stats` | `monitoring/enhanced_dashboard.py:8676` | inline SQL | `copytrading_trades` |
| `PerformanceTracker` (in-memory) | `monitoring/performance.py` | own stats module | `data/performance.db` (SQLite, parallel store) |
| `PnLTracker` (in-memory, embedded in engines) | `core/pnl_tracker.py` | own stats module | in-process only |

That is **six** Sharpe ratios computed by **four** different formulas across **three** annualisation conventions. The dashboard surfaces all of them under the same label.

---

## 3. Per-metric math audit

### 3.1 Sharpe ratio

**Where it is shown:**
- `dashboard/templates/performance.html:42-43` (`#sharpeRatio`)
- `dashboard/templates/performance.html:194-195` (`#sharpeRatioDetail`)
- `dashboard/templates/analytics.html:222-223` (`#sharpe-ratio`)
- `dashboard/templates/backtest.html:130-131` (`#backtestSharpe`)

**Formulas in code (4 of them):**

(a) `core/analytics_engine.py:699-715` — used by `/analytics`:
```
excess = daily_pnl_$ − (rf / 252)        # rf = 0.02 hard-coded line 702
sharpe = mean(excess) / std(excess) * sqrt(252)
```
Operates on **dollar daily PnL**, not %-return. Annualises by **252**.

(b) `monitoring/enhanced_dashboard.py:4123-4128` — used by `/api/performance/metrics`:
```
daily_returns = df['profit_loss'].resample('D').sum()
sharpe = mean(daily) / std(daily) * sqrt(365)        # NOT 252
```
**$-units, annualised by 365.** Risk-free rate dropped silently.

(c) `core/pnl_tracker.py:265-295`:
```
daily_rf = 0.02 / 365                     # line 282
sharpe = (mean(daily_$) − daily_rf) / std(daily_$)
       × sqrt(365)                        # line 293
```
**Daily $-PnL minus a $-RF of 0.02/365 ≈ $0.0000548. The risk-free rate is denominated wrong** — it should be subtracted from a return ratio, not from a dollar amount. The whole term is numerically negligible but mathematically wrong.

(d) `monitoring/performance.py:368-377`:
```
returns = [pnl / entry_value for t in trades]   # per-trade ROI
sharpe = (mean − rf) / std * sqrt(252)
```
**Per-trade returns, not daily.** This conflates trade-frequency with returns and is the worst of the four because it produces meaningful Sharpe numbers from very few trades.

**Truth check:** standard finance definition is `Sharpe = (mean(R) − Rf) / σ(R)` on **calendar-time returns at one frequency**, then annualise by `√(periods/year)`. Crypto trades 24/7, so `365` is defensible, but the codebase mixes 252 and 365. **Pick one.** The 0.02 RF is fine but must be of the same convention as the returns (decimal/yr).

**Verdict:** Sharpe shown anywhere on the dashboard is **unreliable by ≈√(252/365) ≈ 0.83x or its inverse**, depending on the route. None of them properly de-fee or de-gas the numerator.

### 3.2 Sortino ratio

**Shown:** `analytics.html:198-199`, `performance.html:198-199`, `backtest.html:134-135`.

**Formulas:**
- `analytics_engine.py:717-739`: numerator uses excess returns, denominator uses `np.std(returns[returns<0])` — standard.
- `enhanced_dashboard.py:4145-4147`: `downside_std = daily_returns[daily_returns<0].std()` — same shape, **annualised by 365**, RF ignored.
- `performance.py:476-478`: numerator is `period_return` (cumulative !), denominator is `stdev(negative_returns)` — **denominator divides only by N_negative, not by N_total**. The Pedersen / Sortino-Satchell paper requires `√(Σneg²/N_total)`. This implementation underestimates downside deviation when `N_negative < N_total`, producing **inflated Sortino** for sparse-loss strategies — exactly the strategies most likely to look attractive on this dashboard. Critical.
- `pnl_tracker.py:297-334`: uses `Σneg² / N_total` (line 324) — **correct** definition.

So **two of three Sortinos are correct, one (`performance.py`) is over-stated**. The dashboard does not tell you which is rendering.

### 3.3 Calmar ratio

**Shown:** `performance.html:202-203`, `analytics.html` (no), `backtest.html` (no).

**Formulas:**
- `analytics_engine.py:307-310`: `calmar = float(total_pnl) / max_dd` — **dollar PnL divided by fractional drawdown**. Unit mismatch: numerator is $, denominator is unitless. Wrong.
- `enhanced_dashboard.py:4149-4150`: `(daily_returns.mean() * 365) / (max_drawdown / 100)` — units OK but numerator is dollar/year, denominator is fractional, → dollar/year/fraction. Wrong.
- `pnl_tracker.py:336-360`: `annualized_return($ / yr) / max_drawdown($)` — units OK but the conventional Calmar is **annualised %-return / max-DD %**, both dimensionless. So even when the units cancel, the numerical value is not comparable to Bloomberg's Calmar.

**Verdict:** every Calmar surfaced on the dashboard is unit-inconsistent. The numbers it produces are pseudo-quantities. Do not rely.

### 3.4 Max drawdown

**Shown:** `performance.html:49-50` and many others.

**Formulas:**
- `analytics_engine.py:741-772`: peak-to-trough on the *equity curve of closed trades only*. Open positions invisible.
- `enhanced_dashboard.py:4131-4138`: `expanding().max()` on `equity = initial + cumsum(pnl)`. Same blindness to MTM of open positions.
- `enhanced_dashboard.py:7286-7291` (backtest): same.
- `performance.py:776-799`: same — equity curve = list of values, peak-to-trough.
- `pnl_tracker.py:199-207`: peak-to-trough on `current_capital` updated *only when trade closes*.

**Common bias:** the bot can be MTM −50 % on 5 open positions and the dashboard prints `max_drawdown = 0 %` because no trade has yet been *closed* at a loss. This is **survivorship at the equity-curve level**. The dashboard's "current drawdown" indicator is therefore lying during the worst moments — exactly when it matters.

### 3.5 Win rate

**Shown:** `performance.html:99-100`, `analytics.html:214-215`, `dashboard_copytrading.html`.

**Formulas:**
- `analytics_engine.py:268-273`: `len(pnl>0) / total_trades`. Trades with `pnl==0` are counted as losses. Borderline OK; affects break-even trades only.
- `enhanced_dashboard.py:4109-4110`: same — `losing_trades = df[df['profit_loss'] <= 0]`. So PnL = 0 counts as a loss.
- `enhanced_dashboard.py:8717-8730` (copy): more careful — requires `entry_usd > 0 AND exit_usd > entry_usd` for a "win". Excludes standalone sells from numerator AND denominator. **This is correct** and is the only place this is done.

**Bias:** partial closes. If a position is partially closed at +10 % and the remainder at −5 %, the dashboard counts it as **two trades** with split P&L. There is no notion of per-position rollup. The win rate inflates with the number of fragments. (`copy_engine.py:1148-1181` does correctly merge, but the DEX trades table does not.)

### 3.6 Profit factor

**Formula:** `Σ wins / |Σ losses|`. Same definition in all four places — consistent. Note all four use the "cap at 9999.99" hack (`pnl_tracker.py:373`, `enhanced_dashboard.py:4120`, `performance.py:348`). That is fine for JSON, but the dashboard renders **9999.99** without a "∞ / no losses" label, which mis-signals to the user that the strategy has ≈10000x ratio.

### 3.7 ROI

**Shown:** `performance.html:35-37`.

**Formula:** `enhanced_dashboard.py:4141-4142`: `ROI = total_pnl / initial_balance × 100`. `initial_balance` comes from `self.config_mgr.get_portfolio_config().initial_balance`.

**Bias:** `initial_balance` is the **config-time starting capital**, not the **balance at the start of the displayed timeframe**. So if a user has been running 6 months and looks at "7d ROI", the denominator is still 6-month-old initial balance. The shown ROI for 7d is therefore not 7-day ROI — it is "cumulative PnL since forever / initial balance". Mislabelled.

### 3.8 VaR-95 and CVaR-95

**Shown:** `analytics.html:257-258`.

**Formula:** `core/analytics_engine.py:819-841`:
```
index = int((1 - confidence) * len(returns))
var   = abs(sorted_returns[index])
cvar  = abs(sum(tail[:index]) / len(tail))
```

**Issues:**
- `returns` here is **daily PnL in dollars**, not %-return. So VaR is in $-units. The dashboard displays it as `formatCurrency(...)` which is consistent. OK.
- `index` is computed from *number of historical days observed*, not from a fitted distribution. So with N=15 daily observations, VaR-95 takes the 0th element (no interpolation, no Cornish-Fisher). For small samples this is meaningless — but no min-N gate.
- CVaR has a divide-by-`len(tail)` (line 839) where `tail = sorted_returns[:index]`. If index = 1, tail has 1 element, and CVaR == VaR. No degenerate-case guard.
- **No 99 % CVaR** shown in the UI despite `var_99` being computed (`analytics_engine.py:447`). Dead variable.

### 3.9 Annual volatility

**Shown:** `analytics.html:261-262`, `performance.html:187-188`.

**Formula:**
- `analytics_engine.py:450-452`: `np.std(daily_returns) * sqrt(252)` (dollar units).
- `enhanced_dashboard.py:4152-4154`: `daily_returns.std() * sqrt(365) * 100` (dollar units × 100).

The `* 100` in (b) treats it as if `daily_returns` were a decimal fraction — but it is in *dollars*, not a fraction. So the rendered "Annual Volatility = 1234 %" is meaningless. **This is a unit bug.**

### 3.10 Equity curve

**Shown:** `analytics.html:285`, `performance.html:62`, `backtest.html:114`.

**Formulas:**
- `/analytics`: `analytics_engine.py:658-697`, plain cumulative sum of `pnl` per trade. Closed trades only. Indexed by `T1, T2, …` (`analytics.js:194`), so the x-axis is **trade ordinal**, not **time**. A 5-minute scalp and a 5-day swing are equidistant on the chart. Misleading.
- `/api/performance/charts`: `enhanced_dashboard.py:4248-4276`, also cumulative-sum, but x-axis is `exit_timestamp` (correct time). **Two different visualisations of "equity" with two different x-axes** under similarly-named pages.
- backtest: same scheme as performance/charts.

### 3.11 Cumulative / daily P&L attribution

**By strategy:** `enhanced_dashboard.py:4285-4289` groups by `strategy` column with fallback to `metadata`. This is consistent with the trades log. Good.

**By symbol:** not shown anywhere. There is no per-token breakdown.

**Realized vs unrealized:** `analytics_engine.py:354` hard-codes `realized_pnl = total_pnl` and never populates `unrealized_pnl` from open positions. The dashboard `data['unrealized_pnl']` is therefore always 0 from this route. The other route (`/api/performance/metrics`) doesn't even expose unrealized.

### 3.12 Fees, gas, funding

- DEX trades store `gas_fee` (used as `fees` in `analytics_engine.py:600`). It is subtracted in `analytics_engine.py:278` to produce `net_pnl`. **Good** for DEX.
- Futures store `fees` separately and have a `net_pnl` precomputed.
- **No funding subtraction** anywhere for futures display. Funding cost is a major P&L driver in perp markets and the dashboard ignores it.
- **No slippage cost** (theoretical fill vs actual fill) anywhere. Slippage is a frictionless miracle in our analytics.
- Copy trading: `gas_fee`/`fees` is **not stored** in `copytrading_trades` at all. The schema in `copy_engine.py:1187-1200,1205-1219` has columns for `entry_usd, exit_usd, profit_loss, profit_loss_pct` but no fee column. So copy P&L is **gross**, not net.

### 3.13 Backtest math (the rosy version)

The most dangerous piece on the dashboard. `enhanced_dashboard.py:7217-7326`:

```python
position_size_per_trade = balance * 0.1                  # 10% fixed
roi = (exit_price - entry_price) / entry_price           # gross %
simulated_pnl = position_size_per_trade * roi            # NO fees, no slippage, no gas
balance += simulated_pnl
```

Then it serves to the front-end:
```
sharpe_ratio: 0,        # placeholder
sortino_ratio: 0,       # placeholder
```
(lines 7318-7319) which the JS at `backtest.js:99-100` happily renders as numeric values.

**Issues:**
1. **No fee / gas / slippage / funding deduction.** The backtest is structurally optimistic.
2. **Fixed 10 % position size** — does not match the live position sizer (which uses Kelly + max_position_size_pct).
3. **Strategy filter** uses a JSONB path `metadata->'strategy'->>'name'` (line 7234) — fine, but the live engine writes strategy differently in different modules. So the backtest can be empty for strategies that exist live, or vice versa.
4. **Sharpe / Sortino are zero placeholders.** UI shows 0.00 → user thinks "no risk-adjusted edge". Or thinks the strategy is meaningless. Either way wrong.
5. **Returns/initial_balance ROI is correct** (line 7279), but again no friction.

**Verdict:** the backtest tool **overstates strategy edge** by exactly the fee+slippage+gas+funding stack. For DEX-style trades this is 0.5–2 % per round trip in friction. Over 100 trades that is 50-200 % of return wiped out from the rosy version.

### 3.14 Time-zone handling

Mixed. `pnl_tracker.py` uses local `datetime.now()` (e.g. line 17 not shown but `record_trade` accepts whatever `exit_time` the caller gives). `analytics_engine.py:419,442` uses `datetime.now()` (naive local), but trades from DB come back with timezone-aware timestamps. There are isoformat / Z-strip hacks in `analytics_engine.py:332-334` and `enhanced_dashboard.py:4210`. Risk: **off-by-N-hours** in daily bucket boundaries depending on which side of UTC the deployment runs.

**Verdict:** not catastrophic, but DASH-Q-13.

### 3.15 Currency / quote ambiguity

- DEX `usd_value` is whatever the engine wrote at trade time. The price source is not stored alongside it.
- Solana: `pnl_sol`, `pnl_usd`, `sol_price_usd` are all in `solana_trades` (good — see `analytics_engine.py:633-641`).
- Copy: uses CoinGecko spot at the moment of trade detection (`copy_engine.py:50-97`) — but no `coingecko_timestamp` is stored. If CoinGecko is stale by 30 s and the token moved 5 %, our PnL is mis-reported.
- The dashboard does **not** display "USD / USDC / USDT" tag next to any dollar value.

---

## 4. Bias inventory

| Bias | Where it lives | Severity |
|---|---|---|
| Survivorship at equity-curve level (open MTM losses invisible) | every max_drawdown impl | **Critical** |
| Look-ahead in backtest | None at signal level; backtest only replays already-closed trades, so trivially no look-ahead but also trivially no out-of-sample | High (false confidence) |
| P-hacking | None visible; no parameter search UI exists | None |
| Currency-of-quote | every $ value; no tag | Medium |
| Survivorship at strategy-name level | strategies that were renamed silently drop out of the strategy-breakdown chart (`enhanced_dashboard.py:4239-4244`) | Low |
| Time-zone | mixed UTC/local | Medium |
| Sample-size | no N<30 warning anywhere | High |
| Annualisation drift | 252 vs 365 mixed (Sharpe (a) uses 252; (b)(c)(d) use 365) | High |
| Dry-run pollution | copy-trading stats includes `is_simulated = true` | Critical |
| Partial-close inflation | per-position rollup missing in DEX/futures | Medium |

---

## 5. Attribution audit (does the dashboard tell you *why*?)

| Cut | Implemented? | Match underlying trade log? |
|---|---|---|
| By strategy | yes (`/api/performance/charts`) | Yes, modulo strategy-renames silently splitting buckets |
| By symbol | **no** | — |
| By time-of-day | **no** | — |
| By chain | partial — only via module-level routing | Yes for the DEX/Sol/Futures split |
| By leader (copy) | computed per-call in `enhanced_dashboard.py:8363-8396` but **not displayed** on `/analytics` | — |
| By holding-period bucket | **no** | — |
| Realised vs unrealized | partial — unrealized always 0 | **No** |

The dashboard is **aggregate-first**. A user cannot answer "is the edge from one token? one leader? one hour of day?" without dropping out to SQL.

---

## 6. Profitability levers (what these audit fixes are worth, ranked)

1. **Make Sharpe consistent across surfaces (one formula, one annualisation)** — restores user trust, reveals strategies that are actually negative-Sharpe. (DASH-Q-01)
2. **Compute drawdown from MTM equity (open + closed)** — the biggest hidden risk indicator. (DASH-Q-02)
3. **Add fee/slippage/funding to the backtest** — kills false-positive strategies before they go live. (DASH-Q-03)
4. **Add `is_simulated = false` filter to all live P&L aggregations** — currently dry-run polluted. (DASH-Q-04)
5. **Add per-position rollup** to avoid partial-close inflation of win-rate. (DASH-Q-05)
6. **Fix Calmar units** — currently a pseudo-quantity. (DASH-Q-06)
7. **Per-symbol and per-time-of-day attribution** — biggest discoverability lever. (DASH-Q-07)
8. **Replace Sharpe/Sortino placeholders in backtest with real values.** (DASH-Q-08)
9. **Sample-size warnings (N<30 ⇒ greyed out + tooltip).** (DASH-Q-09)
10. **Fix annual-vol *100 unit bug.** (DASH-Q-10)

---

## 7. Action backlog

| ID | Title | Files | Est. lines | Risk |
|---|---|---|---|---|
| DASH-Q-01 | Unify Sharpe: extract one helper in `core/pnl_tracker.py`, call it from `analytics_engine.py:699`, `enhanced_dashboard.py:4123`, `performance.py:368` | 4 files, replace dupes | ~80 | low |
| DASH-Q-02 | Drawdown on MTM equity: include open-position unrealized PnL in `equity_curve` | `analytics_engine.py:658-697`, `enhanced_dashboard.py:4131-4138` | ~80 | medium |
| DASH-Q-03 | Backtest must subtract `gas_fee, slippage_bps, funding` per trade | `enhanced_dashboard.py:7259-7322` | ~60 | medium |
| DASH-Q-04 | Add `WHERE is_simulated = false` to copy PnL aggregation, and a UI toggle | `enhanced_dashboard.py:8701`, copy dashboard tpl | ~20 | low |
| DASH-Q-05 | Per-position rollup: group `trades` rows by `parent_position_id` before win/loss counting | `analytics_engine.py`, `enhanced_dashboard.py` | ~100 | medium |
| DASH-Q-06 | Fix Calmar units: numerator and denominator both fractional, both annual | `analytics_engine.py:307`, `enhanced_dashboard.py:4149`, `pnl_tracker.py:336` | ~30 | low |
| DASH-Q-07 | Per-symbol & per-hour breakdowns in `/api/performance/charts` | `enhanced_dashboard.py:4212-4317`, `analytics.js` | ~150 | low |
| DASH-Q-08 | Compute real Sharpe/Sortino in backtest | `enhanced_dashboard.py:7304-7322` | ~40 | low |
| DASH-Q-09 | N<30 warning on every ratio metric | both JS files | ~60 | low |
| DASH-Q-10 | Annual vol unit fix (`* 100` only if returns are fractional) | `enhanced_dashboard.py:4152-4154` | ~5 | low |
| DASH-Q-11 | Show `is_simulated` flag in trades tables; default-exclude from PnL | tpl + JS + SQL | ~50 | low |
| DASH-Q-12 | Surface `var_99` in the UI (it's computed and discarded) | `analytics.html`, `analytics.js` | ~10 | low |
| DASH-Q-13 | Use timezone-aware UTC consistently | `analytics_engine.py`, `pnl_tracker.py` | ~40 | medium |
| DASH-Q-14 | Tag every $-value with quote currency (USDC default), show in tooltip | front-end | ~30 | low |
| DASH-Q-15 | Show "Net of fees" vs "Gross" toggle on Sharpe / PF / ROI | both JS + API | ~60 | low |

---

## 8. Open questions

1. **Which Sharpe is "the" Sharpe?** The team needs to pick a convention (252 vs 365, $ vs %, with/without RF). My recommendation: 365 (crypto), per-day return-fraction, no RF (set 0). Document in `pnl_tracker.py` as the source of truth.
2. **Should the dashboard compute MTM equity at refresh?** If yes, we need a price-feed dependency on every refresh. Acceptable latency budget?
3. **Backtest semantics:** is the backtest meant to be a "what if we'd had different sizing on the same trade timeline" (current code) or a true "replay raw market data through the strategy" (much bigger lift)? These are very different products.
4. **Partial-close model:** is each row in `trades` truly a complete round-trip, or are some rows partial closes of the same position? The schema does not enforce a `parent_position_id`.
5. **Initial balance for ROI denominators.** Should the dashboard show ROI vs *start-of-window* balance, or vs *all-time-start* balance? The current code shows the latter mislabelled as the former.
6. **Should the dashboard surface confidence intervals (bootstrap Sharpe CI etc.) or just point estimates?** With small N this matters a lot.

---

## 9. Summary

The dashboard is a **competent read-only display** of the closed-trade log, with **four parallel Sharpe implementations**, **multiple unit bugs**, a **dishonest backtest** (no fees / no slippage / placeholder Sharpe), and a **drawdown number that lies during the worst moments** (MTM blind). The math the dashboard SHOWS does **not** match the math the engines DO in at least three load-bearing places: Sharpe (different formulas), drawdown (no MTM), and backtest P&L (rosy).

Highest-impact next change: **DASH-Q-01 + DASH-Q-02 + DASH-Q-03** as one PR — unify Sharpe, MTM-aware drawdown, friction-aware backtest. That moves the dashboard from "marketing slide" to "decision tool."
