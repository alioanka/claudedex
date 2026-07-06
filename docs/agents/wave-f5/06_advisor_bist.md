# Wave-F5 / 06 — Financial Advisor: per-market outage diagnosis + BIST price-source replacement

Date: 2026-07-05. READ-ONLY analysis. Evidence: `logs/advisor/advisor.log` (11,058 lines, window 2026-07-03 04:48 → 2026-07-05 18:28), `logs/advisor/advisor_errors.log` (5,408 lines, window 07-04 → 07-05), `logs/dashboard/dashboard.log`/`dashboard_errors.log`, screenshots `screenshots/screencapture-38-242-251-156-8080-advisor-*.png`, code under `modules/advisor/`, git history.

**Operator complaint decoded:** the module is not one big breakage but FIVE independent defects that compound: (1) a Midas analyzer `await` bug kills all fund advice; (2) the BIST scan universe is 88% garbage tickers from Fonoloji's `/stocks/list`, and Fonoloji's price endpoints went HTTP 451 on 2026-07-01; (3) the per-channel sim caps are saturated (75/75) and the risk gate rejects the ADVICE itself when the cap is full, so **0 advice/day is published for every market** (this is the real "no advice for US / metals / FX / BIST"); (4) the advice-history and simulations APIs emit literal `NaN` in JSON, so both pages die with a parse error; (5) KAP's LLM classification stage is permanently budget-starved and unclassified stamps are never retried.

---

## (a) Per-analyzer failure diagnosis (error counts from `advisor_errors.log`, 2-day window)

| Analyzer | Errors | Distinct message | Verdict |
|---|---|---|---|
| `BISTAnalyzer` | **4,397** | `All BIST data sources failed for '<SYM>'…` | Universe garbage + Fonoloji 451 (below). Watchlist tickers are FINE. |
| `MidasFundsAnalyzer` | **960** | `'coroutine' object has no attribute 'data_source_status'` | **100% outage — missing `await` bug** |
| `USEquitiesAnalyzer` | 45 | `yfinance returned no data for HONA` | Healthy; only the discovery gem HONA is a dead ticker |
| `FXAnalyzer` | 0 | — | Healthy (yfinance majors/metals OK) |
| `CryptoAnalyzer` | 0 | — | Healthy (ccxt binance OK) |
| `advisor.data.fonoloji` | 4 | daily budget 80% / EXHAUSTED (3000/3000 both days) | Budget burned on dead chart calls |
| llm_budget | 1 | `Daily paid-LLM budget reached (10/10 …, last kind=advice_rationale)` | Starves KAP LLM stage (see c) |

### MIDAS FUNDS — total outage, one-line-per-call-site fix
`modules/advisor/core/analyzers/midas_funds.py:951` declares `async def _build_nav_result(...)` but **all four call sites return it without `await`** — lines **352** (`_analyze_fonoloji`), **471** (`_analyze_tefas_crawler`), **593** (`_analyze_tefasfon`), **734** (`_analyze_tefas_scrape`). So `_analyze_fonoloji` returns a coroutine object; `analyze()` line 269 does `result.data_source_status` → `AttributeError` → caught at :317 → `_error_result` for EVERY fund. 960 errors = 10 watchlist funds (`TPP,TP2,AIS,PHE,PRY,EKF,IJC,CPT,YJK,RTG`) × 3 horizons × 32 cycles. Every fallback path is equally broken, so no data source change can revive Midas — only the `await`s. Introduced in commit `3ac0bc2` (the method was born async and never awaited); it has silently produced `_error_result`s ever since (`analyze()` swallows the exception, so nothing crashed).

### BIST — three stacked causes
1. **Fonoloji price endpoints dead (HTTP 451 since 2026-07-01).** The logs cannot show the 451 directly because `fonoloji_client.py:481-482` logs unexpected statuses at **DEBUG** (`logger.debug("[fonoloji] %s unexpected HTTP %d.")`) — only 401/429/503 get WARN/INFO lines. Corroborating evidence: (i) 8× `[fonoloji] /stocks/<T>/chart HTTP 429; honoring retry-after` lines show chart calls are being made and never yield data; (ii) the daily local budget hits 3000/3000 on both 07-04 and 07-05 (budget spent re-fetching a dead endpoint — 451 responses are never cached, so every symbol retries every cycle); (iii) non-price Fonoloji endpoints still work: `/stocks/list` returns 3,273-3,368 tickers, `/screener` gems return 46 candidate rows (`discovery.bist … Fonoloji gems: 46 candidate rows`). This exactly matches the operator's report (price/chart = 451, list endpoints = items with empty price fields). Live re-verification from this analysis container was blocked by the egress proxy (fonoloji.com not allowlisted).
2. **Universe garbage floods the analyzer.** `advisor_bist_universe` auto-resolves to the Fonoloji live list (`activation.effective_universe_mode` + key present). `universes._fonoloji_bist_list` filters with `_BIST_TICKER_RE = ^[A-Z][A-Z0-9]{2,5}$` (`universes.py:53`) — which happily passes the **Midas-tradable US symbols** the list also contains (`AAPLO`, `AAPLUS`, `ABBVN`, `ADBEUS`, `AAOIUS`, `AC4`, `ADU`…). The cap is 60 (`advisor_universe_max`), filled **watchlist-first then alphabetically from the list head** (`universes.py:223-231`), so every cycle = 7 real watchlist tickers + 53 A-prefixed garbage: log `[advice] bist universe scan: 7 watchlist -> 60 symbols`. All 53 fail all three sources (Fonoloji 451 → borsapy not installed → yfinance `.IS` no such ticker) → 4,397 warnings/2 days (58 distinct symbols × 3 horizons × ~32 cycles). Real BIST names beyond the watchlist (THYAO is IN the watchlist; but e.g. AKBNK, EREGL) are **never even attempted**.
3. **Watchlist BIST tickers actually succeed** — `MGROS,CUSAN,THYAO,SASA,CANTE,FENER,HEKTS` appear in **zero** error lines: after Fonoloji returns None they fall to `_fetch_yfinance` (`bist.py:280-294`, `.IS` suffix) which works from this German VPS. So BIST is *degraded*, not dead — the reason the operator sees no BIST advice is (c) below plus the broken pages.
   - `borsapy` is the designed first fallback (`bist.py:259`) and is in `requirements.txt:101`, but **is not installed in the runtime image** (`python3 -c "import borsapy"` → `ModuleNotFoundError`; every error message says "borsapy not installed").

### Cross-cutting publish blocker — why EVERY market shows "no advice"
59 of 60 cycles in the window end `Cycle #N complete: 0 advice(s) published.`; exactly ONE advice in ~2.6 days. Cause: `[advice] mark-to-market: 75/75 open sims marked` every cycle — 5 channels × `advisor_sim_cap_per_channel`=15 are saturated. With "Auto-open sim position per published advice" ON (settings screenshot), `risk_engine.should_publish` gate #4 (`modules/advisor/core/risk_engine.py:127-133`) rejects the **advice itself** (`if result.sim_enabled and open_sim_count >= cap: return False`) — not just the sim. Rejections log at DEBUG (`advice_engine.py:268-270`) → invisible. Proof of mechanism: the single publish (Cycle #407, 07-04 02:09) came one hour after the only sim close in the window (`Closed sim #141: BTC/USDT … reason=expired`, 07-04 01:05) freed one crypto slot, and immediately opened sim #152 re-filling it. Discovery is equally blocked: every pass logs `15 candidate(s) across 3 market(s); 0 discovery advice(s) published` (gems channel also full) — hence "New Gems" shows only the 27 stale crypto rows (latest 02.07), i.e. "gems only for crypto".

---

## (b) Advice-history + Simulations pages "empty with errors"

**Primary root cause (visible in the screenshots): invalid JSON — literal `NaN` tokens.**
- `/advisor/advice` renders: `Error: Unexpected token 'N', …"try_low": NaN, "entr"… is not valid JSON` → the response contains `"entry_low": NaN`.
- `/advisor/simulations` renders: `Error: Unexpected token 'N', …"y_price": NaN, "curr"… is not valid JSON` → `"entry_price": NaN, "current_price"…`.

Chain: rows in `advisor_advice` (`entry_low/entry_high/target_price/stop_price`) and `advisor_sim_positions` (`entry_price/current_price`) contain float `NaN` (Postgres stores `'NaN'::float8`). The endpoints — `api_get_advisor_advice` (`monitoring/enhanced_dashboard.py:18297`, route :1702) and `api_get_advisor_simulations` (:18543, route :1705) — serialize with Python's default `json.dumps` (`allow_nan=True`), emitting bare `NaN`, which is **not valid JSON**; the browser's `res.json()` throws `SyntaxError`, the templates' catch blocks (`dashboard/templates/advisor_advice.html:150-152,209-211`; `advisor_simulations.html:257,273-276`) paint the red error and the tables stay empty. The performance panel on the same simulations page loads fine because `/api/advisor/performance` aggregates server-side and leaks no NaN.

NaN provenance: there is **no finiteness guard anywhere** (`grep isnan|isfinite` over `levels.py`, `advice_engine.py`, `portfolio_engine.py` = zero hits). `levels.horizon_levels` (`modules/advisor/core/analyzers/levels.py:170-172`) guards `close <= 0` — which is `False` for `NaN`, so a NaN close/ATR propagates into entry/target/stop, gets persisted, and later poisons sims at open and at mark-to-market.

**Secondary cause (intermittent):** every request to these two endpoints captured in `logs/dashboard/dashboard.log` returned **401** (`"GET /api/advisor/advice?limit=200" 401`) — the `session_id` cookie is issued with `max_age=3600` and `secure=True`-by-default over the plain-HTTP deployment (`monitoring/auth_routes.py:107-114`; `auth/middleware.py:169-186`), and is never re-issued while the DB session slides. A dashboard left open >1h flips to an error state on the next auto-refresh. `dashboard_errors.log` (all 140 lines) contains **no advisor tracebacks** — handlers and schema (migs 058/076/077) are correct; the sims handler even self-heals a missing `channel` column (:18586-18592).

---

## (c) KAP — "thousands of unclassified items"

**Not a backlog, not a stopped worker — a permanently-starved classifier that stamps and never retries.**
- The worker runs fine: `classifier_worker.py` — interval 60s (`:50,:98-102`), batch 100 (`:51`, used at `:170` in the ONLY `get_unclassified(limit=100)` call). Ingestion ≈ 50-160 disclosures/day vs theoretical 144k/day throughput; the log shows **161 ingested / 160 classified** over the window, 1:1.
- But "classified" mostly means **stamped `event_type='UNCLASSIFIED'`, `classifier_stage='unclassified'`, confidence 0** (exactly what the screenshot shows: 200 rows, most `UNCLASSIFIED / unclassified / 0%`, a minority `GENERAL ASSEMBLY / rule / 90%`):
  1. **Stage-1 rule taxonomy rarely matches** the Turkish subjects — only ~3/160 directional matches in the window (`classifier.py:512-529`).
  2. **Stage-2 LLM is budget-starved**: `classifier.py:340-342` calls `try_consume(kind="kap_classify")` against the ONE shared bot-wide daily counter (`core/llm_budget.py`, file `logs/.llm_budget.json` = `{"day":"2026-07-05","count":10}`). The cap is set to **10** and `advice_rationale` (crypto/US analyzers) consumes all 10 within the first hour of each UTC day (`Daily paid-LLM budget reached (10/10 …, last kind=advice_rationale)` at 00:24 on 07-05). KAP effectively makes **zero** LLM calls, ever.
  3. **No retry path**: `kap_store.get_unclassified` (`kap_store.py:279`) `LEFT JOIN … WHERE c.disclosure_id IS NULL` — a row with an UNCLASSIFIED stamp is permanently "done". Weeks of ingestion → thousands of permanent UNCLASSIFIED rows.

---

## (d) BIST price-source replacement (Fonoloji /stocks price+chart = HTTP 451 since 2026-07-01)

Delayed-data acceptability: **yes** — the advice cycle is hourly, levels are daily-bar heuristics (ATR-14 on daily closes), horizons are 1d-2y. 15-min-delayed or EOD data is fully adequate. Note KAP-driven sims and mark-to-market also only need a recent last-price.

### Evaluation
| Source | Verdict | Notes |
|---|---|---|
| **Yahoo Finance `.IS`** (yfinance / `query1.finance.yahoo.com`) | **PRIMARY** | **Already proven working from THIS German VPS** — the BIST watchlist tickers fall back to `_fetch_yfinance` today and produce zero errors. Broad BIST coverage (`THYAO.IS`, `MGROS.IS`, …), ~15-min-delayed intraday + EOD daily bars. Unofficial/undocumented; community-observed limits ≈ 360 req/h and 429s after ~1k rapid calls/IP ([yfinance #2128](https://github.com/ranaroussi/yfinance/issues/2128), [AlgoTrading101](https://algotrading101.com/learn/yahoo-finance-api-guide/)). At ≤60 symbols/cycle with a 3-6h TTL cache that is ~200-500 calls/day — comfortable. |
| **borsapy** ([saidsurucu/borsapy](https://github.com/saidsurucu/borsapy), Apache-2.0) | **FALLBACK** | TradingView WebSocket feed, ~15-min delayed without credentials, yfinance-like API. **Already integrated** as the first fallback (`bist.py:259`, `_fetch_borsapy`) and already in `requirements.txt:101` — it is simply **not installed** in the runtime image. `pip install borsapy` activates it with ZERO code change. Caveats: README says personal/educational use; unofficial TradingView protocol can break. |
| Midas unofficial WP-JSON | Tertiary, spot-only | `POST https://www.getmidas.com/wp-json/midas-api/v1/midas_table_data` returns a 15-min-delayed table of all BIST rows (price/bid/ask/high/low); `…/midas_stock_time` gives historical closes ([community docs](https://github.com/prosman/doviz-borsa-kripto-api/blob/master/README.md)). No auth, but undocumented/fragile — usable as a cheap last-price for KAP sims, not as an OHLCV backbone. |
| Stooq | **Rejected** | No BIST coverage found (covers PL/US/DE/UK/HU/JP; no `.is` symbols surfaced). |
| TradingView raw protocol | Rejected | Same feed as borsapy with none of the packaging; use borsapy. |
| investing.com scraping | Rejected | Cloudflare + ToS-hostile; fragile. |
| Matriks (paid) | Deferred | Already a stub opt-out (`bist.py:213-214`); only if real-time becomes a requirement. |

### Exact endpoint shapes
- Yahoo chart (what yfinance wraps): `GET https://query1.finance.yahoo.com/v8/finance/chart/THYAO.IS?range=6mo&interval=1d` → `{chart:{result:[{meta:{currency:"TRY",regularMarketPrice,…}, timestamp:[…], indicators:{quote:[{open[],high[],low[],close[],volume[]}]}}]}}`. Requires a browser User-Agent + cookie/crumb dance — **use the `yfinance` lib (already a dependency, already wired in `_fetch_yfinance`)** rather than raw HTTP.
- borsapy: `import borsapy as bp; df = bp.Ticker("THYAO").history(period="1y")` → pandas OHLCV (bare ticker, no `.IS`).

### Integration plan (adapter seam already exists — no new abstraction needed)
The seam is the **source chain inside `BISTAnalyzer`**: `data_source_status()` (`bist.py:171`) + `_analyze_internal()` (`bist.py:225`) trying `_fetch_fonoloji` (:382) → `_fetch_borsapy` → `_fetch_yfinance`, with preference decided centrally by `modules/advisor/core/data/activation.py` (`prefer_fonoloji_bist`, :60-70). A new provider = one `_fetch_<name>()` returning the shared signals dict + one branch in the chain + a `data_source_status` case + (optionally) an activation rule. Steps:
1. `pip install borsapy` into the runtime image (add to the Docker build; it is already in requirements.txt) — instant working fallback.
2. **451 circuit breaker in `fonoloji_client._fetch_remote`** (`fonoloji_client.py:474-482`): treat 451 like 401 — WARN loudly ONCE (`[fonoloji] HTTP 451 (legally restricted) on /stocks/* — BIST price data disabled by provider`) and short-circuit further `/stocks/*/chart|/price` calls for the UTC day. This stops the 3000/day budget burn on dead endpoints (today the daily budget exhausts by ~16:00 and then even the WORKING endpoints — screener, movers, fund NAV — are refused cache-only service).
3. Flip preference so yfinance is tried first for BIST price series while Fonoloji-451 persists: either operator sets `advisor_fonoloji_auto_prefer=false` (blunt: also demotes universe/Midas/gold preference) or — better, code — make `prefer_fonoloji_bist` consult the circuit breaker from step 2. Keep all NON-price Fonoloji uses (they still work): `universes.py:132 stock_list`, `bist_discovery.py:261 stock_movers` / `:267 screener_bist`, `bist.py:427 stock_recommendations` / `:461 market_digest`, `midas_funds.py:385 fund_history` (NAV — verify post-451; unknowable from logs while the await-bug masks it), `fx.py:222-239 gold_live/market_live`.
4. Fix the universe: set `advisor_bist_universe='bist50'` (curated list in `universes.py:63-77`) OR harden `_fonoloji_bist_list` to drop the Midas-US rows (e.g. reject `*US`-suffixed codes / require the row's market field to be BIST / validate against yfinance once daily). Randomize or score the fill order instead of alphabetical head-fill (`universes.py:223-231`) if the live list is kept.
5. Complete Fonoloji price call-site inventory (all go through `FonolojiClient`): price series `bist.py:401 stock_chart` (**the broken one**); `stock_price()` (`fonoloji_client.py:272`, the "cheap KAP price path") currently has **no runtime caller** — KAP sims price via the BIST analyzer, so fixing the analyzer chain fixes KAP sims too.

---

## (e) Fix list, ranked

| # | Pri | Fix | Where | Effort |
|---|---|---|---|---|
| 1 | P0 | Add `await` to the four `_build_nav_result(` returns — revives ALL Midas fund advice | `modules/advisor/core/analyzers/midas_funds.py:352,471,593,734` | 4 words |
| 2 | P0 | Stop rejecting ADVICE when the channel sim-cap is full — publish with `sim_enabled=False` instead of `return False` (or gate only the sim open). Restores advice flow for US/FX/metals/BIST/crypto + gems immediately. Consider also auditing why 75 sims sit open (max-hold long=365d) | `modules/advisor/core/risk_engine.py:127-133`, caller `advice_engine.py:264` | small |
| 3 | P0 | NaN-safe JSON: sanitize floats (`math.isfinite` → `None`) in `api_get_advisor_advice` + `api_get_advisor_simulations` (or a shared `json_response(dumps=partial(json.dumps, allow_nan=False, default=_nan2null))`); PLUS guard at the source: `levels.horizon_levels` reject non-finite close/vol, `portfolio_engine` refuse NaN entry/mark prices; one-off DB cleanup of `'NaN'` rows | `monitoring/enhanced_dashboard.py:18297,18543`; `levels.py:170`; `portfolio_engine.py` | small |
| 4 | P1 | Fonoloji 451 circuit breaker + WARN-once (451 currently invisible at DEBUG) + stop burning the 3000/day budget on dead `/stocks` price calls | `modules/advisor/core/data/fonoloji_client.py:474-482` | small |
| 5 | P1 | BIST source swap: `pip install borsapy` in the image (already in requirements.txt:101) + prefer yfinance/borsapy over Fonoloji for price series while 451 persists (activation rule or operator flag) | Docker image; `activation.py:60-70` | small |
| 6 | P1 | BIST universe: `advisor_bist_universe='bist50'` (config-only, today) or filter Midas-US rows out of the Fonoloji list; kills ~2,200 error lines/day and the wasted quota | config / `universes.py:53,168` | config / small |
| 7 | P1 | KAP: raise/split the LLM budget (10/10 is eaten by advice_rationale before KAP runs — give `kap_classify` its own budget or raise `BOT_LLM_DAILY_MAX_CALLS`), add bounded re-queue of `classifier_stage='unclassified'` rows, and extend the Stage-1 Turkish taxonomy (only ~2% rule hit-rate) | `core/llm_budget.py`; `kap_store.py:279`; `classifier.py`/taxonomy | medium |
| 8 | P2 | Dashboard session: re-issue the sliding `session_id` cookie / scheme-aware `secure` flag (same class as the CSRF fix in `76698e4`) — removes the intermittent 401 "Error" states after 1h | `monitoring/auth_routes.py:107-114` | small |
| 9 | P2 | Demote/aggregate the per-symbol BIST failure WARN (4.4k lines/2 days) to one summary line per cycle; log discovery/publish reject reasons at INFO once per cycle so "0 published" is explainable from logs | `bist.py:296-306`; `advice_engine.py:268` | small |

**Sources** (BIST alternatives research): [yfinance rate-limit issue #2128](https://github.com/ranaroussi/yfinance/issues/2128) · [AlgoTrading101 Yahoo Finance API guide](https://algotrading101.com/learn/yahoo-finance-api-guide/) · [MarketXLS Yahoo API guide 2026](https://marketxls.com/blog/yahoo-finance-api-ultimate-guide) · [borsapy (GitHub)](https://github.com/saidsurucu/borsapy) · [borsapy on PyPI](https://pypi.org/project/borsapy/0.3.0/) · [Midas unofficial endpoints (community README)](https://github.com/prosman/doviz-borsa-kripto-api/blob/master/README.md) · [Yahoo THYAO.IS quote](https://finance.yahoo.com/quote/THYAO.IS/) · [Twelve Data BIST coverage (paid alt)](https://support.twelvedata.com/en/articles/5749822-borsa-istanbul-bist)
