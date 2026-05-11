# FUTURES_MODULE — Quant Audit

Auditor: `quant-algo-expert`
Date: 2026-05-11
Scope: signal/strategy/sizing math layer of `modules/futures_trading/`. Risk is owned by `market-trading-analyst`; this report focuses on edge claims, biases, leakage, indicator math, sizing, and live-readiness from a quant perspective.

---

## 1. Executive verdict — RED

The futures module ships a working **single-venue, single-asset CEX execution engine** with TP1-4 / trailing-SL / signal-score plumbing that is structurally fine. But:

- **There is no edge.** The signals are a sum of 5 textbook indicators on closed candles, scored as integers, with NO regime gate, no walk-forward validation evidence, and no out-of-sample Sharpe/hit-rate measurement (`core/futures_engine.py:1023-1029`). With taker fees of ~4 bps per side on Binance and SL=1.2% / TP1=1.8%, a 54% hit-rate claim in the config docstring (`config/futures_config_manager.py:79-82`) is **back-of-envelope arithmetic, not a backtest**.
- **The funding-arbitrage strategy is mathematically incomplete.** It checks gross funding APR but never subtracts the two taker fees (entry + exit on both legs), borrow costs, or the funding-interval discount (`strategies/funding_arbitrage.py:69-127`). It also hard-codes 8h funding intervals (`:72`, `:210`), which is wrong for Bybit (USDT perps generally 8h, but on some symbols and on Binance occasionally 4h, and some markets dynamically retune). The "negative funding" branch admits it can't short spot but still emits a `LONG perp` signal that has **directional risk, not arb** (`:114-127`).
- **The hedge strategy uses no statistical pairing** — no correlation, no cointegration, no rolling beta. It just shorts a flat dollar fraction of DEX exposure when DEX drawdown crosses a static threshold (`strategies/hedge_strategy.py:62-114`).
- **The trend-follower hard-codes 3% SL / 8% TP and leverage 1–3 inside the strategy file** (`strategies/trend_following.py:94-108`), bypassing the database-driven config that the rest of the engine reads. Two sources of truth.
- **The TP percentages are leveraged-PnL-adjusted incorrectly in one path and not in the other** — the engine multiplies `pnl_pct * leverage` to display `unrealized_pnl_pct` (`core/futures_engine.py:872-874`), then **un-leverages** to compare against `stop_loss_pct` and `take_profit_pct` (`:902-904`), but `_calculate_tp_levels` builds TP prices directly from `price_pct` (1.8%, 3.5%, …) without leverage adjustment (`:1369-1409`). Net effect: SL/TP are price-move thresholds, which is correct — but the docstring and config comments repeatedly describe percentages as P&L-on-margin (`config_manager.py:79-83`). The reader of the config will set the wrong numbers.

I cannot, under any honest reading of the code, recommend flipping `DRY_RUN=false`. The execution plumbing is OK; the edge layer is not there yet.

---

## 2. Edge audit, by strategy

### 2.1 Trend-following / indicator-score engine (`core/futures_engine.py`)
**Claimed edge.** 5-indicator confluence (RSI, MACD, volume ratio, Bollinger position, EMA-9/21 cross) summed into integer score ∈ [−10, +10]; entry when `|score| ≥ min_signal_score` (default 4) and three quality filters pass (`futures_engine.py:1023-1097`).

**Reality.**
1. **No alpha decay correction.** RSI / MACD / BB / EMA are all derived from the *same* close series — they are mechanically correlated. Summing them double-counts trend (EMA-cross + MACD-histogram + BB-position above middle all fire together in an uptrend), so the score effectively has ~2 degrees of freedom, not 5. The "minimum 4 of 10" threshold is therefore meaningless as a statistical test.
2. **Look-ahead is absent here** (good): `closes[-1]` is the last *closed* candle and signals reference `closes[-2]` etc. (`futures_engine.py:1249-1264`). But:
3. **Bollinger "breakout = STRONG_BUY"** (`:1206-1208`) and **EMA-cross detected by re-computing EMA on `closes[:-1]`** (`:1228-1229`) — this is a 2x EMA recomputation per cycle and is O(N²) but not biased.
4. **MACD signal line is faked.** `signal_line = macd_line * 0.9` (`:1308`) is not the 9-period EMA of MACD; it is a one-point shrinkage. `histogram = macd_line - signal_line = 0.1 * macd_line`. Every "MACD histogram > 0" decision is then **identical to "MACD line > 0"** — the signal-line filter does nothing. This is a quant bug, not just style.
5. **Volume signal only fires on `volume_ratio > 2.0` AND aligned with MACD** (`:1182-1187`). Because MACD histogram is fake (see #4), the volume contribution collapses to "MACD > 0 AND vol > 2x". The score's "5 indicators" is closer to 3.
6. **MTF is a placeholder** that always returns `LONG / 0.6` (`futures_module.py:421-428`). Anywhere this is wired in (not currently in `futures_engine.py`'s loop, but in `futures_module.py`'s `_combine_signals` weighting), 10% of the decision is a constant bullish prior.

**Verdict.** Indicator-confluence trend-followers on liquid majors at 15m candles have published Sharpe ≈ 0.5–0.8 *before* fees on long lookbacks. With Binance taker 4 bps × 2 + slippage ≈ 10–13 bps round-trip, you need average winners > ~15 bps net of TP-pct distribution. The current setup *can* be profitable but only with a regime filter and proper signal-line MACD. None of that is in place.

### 2.2 Funding-rate arbitrage (`strategies/funding_arbitrage.py`)
**Claimed edge.** Capture funding when |rate| > `min_funding_rate` (default 1 bp) and annualized > `min_apr` (default 10%).

**Reality.**
- **APR formula assumes 8h funding for every venue, every symbol** (`:72, :210`). Binance USDT-M is 8h for most, but special funding (BTCUSDT-DOMUSDT and some new listings) can be 4h. Bybit perps are 8h. This is OK as an approximation for majors but **the constant `1095` is wrong** as a generic figure if you trade a 4h-funding symbol — you'll under-state APR by 2x and skip real edge, or take a "fake" 8h opportunity that's actually 4h.
- **No fee / slippage subtraction.** A 1 bp funding-rate threshold means **you need to round-trip the long-spot + short-perp pair for under 1 bp**. Binance spot taker is 10 bps, perp taker is 4 bps. Round trip cost = ~28 bps. Holding for *one* 8h funding period at 1 bp = 1 bp gross gain vs 28 bps cost = guaranteed loss. `calculate_expected_profit` (`:185-220`) computes `profit_per_period = size * |funding_rate|` and **never subtracts execution cost**. This is the single biggest quant bug in the module.
- **The "negative funding" branch opens a directional long perp with no spot short** (`:114-127`). Comment says "Note: Shorting spot is harder, so might just long perp". That is not arbitrage — that is a long. Mis-labelled and dangerous.
- **No price-difference profit/loss tracking** during the hold. If perp diverges from spot during the 8h hold, the carry can be wiped out by mark-to-market on the spread.
- **No cross-exchange combo** (the rubric asked: SHORT-perp-on-Binance + LONG-perp-on-Bybit when funding differs). Not implemented anywhere.

**Verdict.** Pure paper code. Disable in config (`futures_funding.funding_arbitrage_enabled = False` is the default — `config_manager.py:166` — that is the only thing saving you).

### 2.3 Hedge strategy (`strategies/hedge_strategy.py`)
**Claimed edge.** When DEX positions are drawn down > 10%, open futures short of total DEX exposure × 80%.

**Reality.**
- **Pair selection: none.** `dex_positions` are treated as a single fungible exposure regardless of underlying token; the hedge is whichever symbol the caller picks (presumably BTCUSDT, but there is no logic). Hedging a basket of meme-coins with BTCUSDT has beta ≠ 1 by an order of magnitude.
- **Hedge ratio: static 0.8** (`:44`). No rolling beta of DEX basket vs the hedge instrument. No Kalman filter. No OLS. Just a fixed scalar.
- **Z-score reset / unwind rule: ad-hoc.** Removal triggers on `pnl_pct > 0` OR `volatility < 30` (`:188-196`). The volatility threshold "30" is unit-free in the code — could be percent, could be ATR-points; this is a unit bug waiting to happen.
- **No cost accounting.** Opening + closing a BTCUSDT hedge on Binance is 8 bps round-trip + funding paid during hold. Hedging a 10% drawdown that recovers same-day means you pay the bid-ask twice for a hedge that delivered zero protection.

**Verdict.** Wrong abstraction. A hedge of a DEX basket needs (a) a beta estimator, (b) a notional sizing rule = `beta * basket_notional`, (c) a hedge-instrument liquidity check. None present.

### 2.4 Trend-following strategy class (`strategies/trend_following.py`)
**Claimed edge.** Price > SMA50 > SMA200 + RSI > 50 + MACD histogram > 0 for longs; reverse for shorts. Trend-strength score 0–1 from MA-spread, RSI extreme, and volume ratio.

**Reality.**
- **Stops/TPs are hard-coded** in the strategy file at 3% / 8% (`:94, :107`) and not surfaced to config. The engine uses its own config-driven SL/TP. Which one runs depends on which caller invokes which — this is a stale class likely dead-coded; verify with `pm-architect-qa`. Tagged `FUT-Q-DEADCODE`.
- **Trend-strength volume term:** `min((volume_ratio - 1) * 0.4, 0.4)` (`:184-187`). Volume ratio < 1 contributes 0 — fine. But it caps at 40% of total strength on a single noisy input.
- **`_determine_trend` is path-dependent on SMA200 alignment** — for a fresh listing or any symbol with < 200 candles loaded, SMA50 and SMA200 will both be `0` and the condition `price > sma_50 > sma_200` collapses to `price > 0 > 0` → False, suppressing all signals. Not a bug per se, but **silently disables the strategy for new symbols** with no warning logged. Tag `FUT-Q-COLDSTART`.

### 2.5 Sizing (`futures_engine._calculate_position_size`, `futures_risk_manager.calculate_position_size`)
**Claim.** Dynamic sizing based on signal-score normalized to `[min_position_pct, max_position_pct]` of capital × leverage (`futures_engine.py:1323-1367`).

**Reality.**
- Uses `getattr(signals, 'combined_score', 3)` (`:1344`) — but `TechnicalSignals` dataclass has **no `combined_score` attribute** (`:62-110`); the score is computed externally. Therefore `signal_score` is **always 3**, the default, and dynamic sizing collapses to a fixed mid-range `(min + max) / 2` regardless of actual signal strength. **Dynamic sizing is silently disabled** — verify by inspecting logs for "Signal=3" appearing on every trade. Tag `FUT-Q-01` (highest profitability lever).
- `futures_risk_manager.calculate_position_size` (`:214-250`) implements **Kelly-like risk sizing**: `position = capital * risk_pct / stop_loss_pct * leverage`. This is OK as a fixed-fraction-of-capital-at-risk rule but it is **NOT Kelly-capped** — there is no `min(Kelly, max_kelly_fraction)`. With a 54% win rate and 1:1.5 R:R, full Kelly is roughly `2p - 1 - (1-p)/b` ≈ 23% — way above the 2% risk_per_trade default, so the bot is under-betting. That's fine *defensively*, but the docstring claims "Kelly-fraction-capped" implicitly via the conservative defaults. State it explicitly.
- **`get_adjusted_leverage`** drops to 1x after 3 consecutive losses (`futures_risk_manager.py:312-313`) — good. But **never raises** leverage back unless wins start.

### 2.6 Multi-TP / trailing-SL state machine (`futures_engine._check_tp_levels`)
**Claim.** TP1 (40%) → SL → entry (breakeven), TP2 (30%) → trailing-SL active, TP3/TP4 (20% / 10%) ride the trail.

**Reality.** Logic is correct (`:1411-1479`). One subtle issue:
- After TP1 hits, `trailing_stop_price` is set to `entry_price` (`:1459`), but `trailing_stop_active = False`. The exit-condition check then routes through "Breakeven SL" branch (`:917-918`) by checking `abs(tsl_price - entry_price) < 0.0001`. Floating-point tolerance is tight; if the engine ever rounds `entry_price` (e.g., from exchange fill at `position.entry_price = float(order['average'])` — `:1701`), the check will fall through to "SL Hit" instead. Cosmetic in P&L but pollutes exit-reason analytics. Tag `FUT-Q-LOGGING`.
- **`_partial_close_position` does not update `notional_value` consistently with the new size** — it uses `position.size * position.entry_price` (`:1532`) for the remainder, but the original notional was `original_size * entry_price`. Correct. But it does NOT prorate `fees_paid`. Subsequent close will subtract the full entry fee from a partial position, slightly over-counting fees. <0.5% PnL impact but not zero.

### 2.7 Exchange-specific micro
- **Binance fees: maker 2 bps, taker 4 bps** (`futures_engine.py:210-213`). Correct as of late 2024. **Bybit fees: maker 1 bps, taker 6 bps** (`:212-213`). Bybit taker for VIP-0 USDT-M perp is 5.5 bps with BLP discount, 6 bps without — close enough.
- **No maker rebate path.** All orders are `create_market_order` (`:1519, :1689, :1774`) — pure taker. The cited Binance maker rebate (2 bps) is never captured because the bot never quotes limit orders. Tag `FUT-Q-MAKER`.
- **`set_leverage` is called per-order on Binance and is not idempotent-safe** — `binance_futures.py:307` issues `set_leverage` before *every* `open_long` / `open_short`. Binance throttles `/fapi/v1/leverage` at ~50 req/min. Under burst trading you can get a 429 and lose the entry. Cache leverage state per-symbol.
- **Bybit executor is a stub** (`exchanges/bybit_futures.py:84-111`). `open_long`, `open_short`, `close_position` are placeholder `return None`. Anyone enabling Bybit in config will get silent no-op trades. Tag `FUT-Q-BYBITSTUB` — critical safety bug if a user toggles `bybit_config.enabled = True`.
- **Funding interval per exchange:** hard-coded 8h in funding-arb math (`:72`), no `/fapi/v1/fundingInfo` lookup. Binance has switched some symbols to 4h. Tag `FUT-Q-FUNDINGINTERVAL`.

### 2.8 Sharpe / hit-rate evidence
- `pnl_tracker.PnLTracker` is initialized and updated on each close (`futures_engine.py:393-396, :1822`). Sharpe / Sortino / Calmar / max-DD are surfaced in `get_stats()` (`:1959-1967`). That is good infrastructure.
- **There is NO backtest entry point.** No `scripts/backtest_futures.py`, no `tests/backtest_*` for this module. The Sharpe number you see in dashboard is *live since boot*, sample size O(few-trades), and meaningless until you have ≥ 100 trades.

---

## 3. Bias inventory

| Bias | Where | Mechanism | Severity |
|---|---|---|---|
| **Look-ahead** | `_get_technical_signals` | All indicators use `closes[-1]` (last *closed* candle from `fetch_ohlcv`). No bias detected. | Clean |
| **In-sample tuning** | `min_signal_score=4`, `tp1_pct=1.8`, `sl_pct=1.2`, weights in `futures_module._combine_signals` (40/30/20/10) | All hand-picked, no documented walk-forward search. Comment block at `config_manager.py:73-83` cites a "54% win rate" but no out-of-sample evidence. | **High** |
| **Leakage — funding-arb label** | `funding_arbitrage.py:55-127` | The strategy treats the *current* `funding_rate` (which is the predicted/observed rate for the *next* funding event on Binance per `/fapi/v1/premiumIndex`) as if it is the rate that will be *paid*. On Binance, the rate at time T is the rate for the funding event at the *next* funding boundary; you do receive that exact rate if you hold over the boundary — so this is **not strictly leakage**. But if the bot uses `last historical funding rate` from `/fapi/v1/fundingRate?limit=1` (`binance_futures.py:454-460`), that returns the *last paid* rate, which is **stale**. Confirm which the engine wires. | Medium |
| **Survivorship** | `pairs_list = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']` | Manually-curated majors. No survivorship bias *in the strategy*, but any back-of-envelope hit-rate claim derived from majors will be over-optimistic vs the full perp universe. | Low |
| **p-hacking** | TP grid `[1.8, 3.5, 6, 10]` and size split `[40, 30, 20, 10]` | No noted-down "we tried X grids on Y data; this won". Reads like product intuition. | Medium |
| **Cold-start regime bias** | `_determine_trend` requires SMA200 | Symbols with <200 bars get no signal — silently. | Low |
| **Mainnet price client in DRY_RUN** | `_get_ticker` uses `price_client` (mainnet) (`futures_engine.py:1893-1896`) even on testnet | **Not a bias, but a fidelity issue**: positions get filled at *mainnet* prices in dry-run mode. Backtest analogue would over-report perf if mainnet prices are tighter than the venue you'll actually trade on. Acknowledge in stats. | Low |

---

## 4. Feature audit

| Feature | Informativeness | Redundancy | Scale | Note |
|---|---|---|---|---|
| RSI(14) | Medium on 15m majors | Correlated with EMA-cross | 0–100, fine | OK |
| MACD-hist | **Broken** — signal line ≠ EMA9(MACD) (`:1308`) | — | — | FIX |
| Volume ratio (5-bar vs 20-bar SMA) | Medium | Used as gate AND scorer | Unbounded → clip needed for ML | OK with clip |
| BB position | Low — collapses to "above/below middle" most of the time | Highly correlated with EMA-spread | OK | OK |
| EMA(9/21) cross | High when fresh, low when established | Same source as MACD | OK | OK |
| `price_change_1h` momentum filter | Useful gate (`:1070-1079`) | — | OK | OK |
| Trend SMA20 vs SMA50 | Medium | Same series | OK | OK |
| `volatility` (passed in from outside) | Not computed — `get_adjusted_leverage` passes `volatility=50` constant (`futures_module.py:537`) | — | — | **Stub** |
| Funding rate | High when fees < edge — currently unmodelled | — | OK | Need fee/slip subtraction |
| Spot–perp basis | High for arb | — | OK | Computed but only used as gate, not as carry attribution |

---

## 5. Model audit
There is no ML *training* path in this module (despite `futures_module.py:_get_ml_signal` calling `self.ml_model.predict`). The `ml_model` is injected from outside and is the same ensemble used by the AI/DEX layers; see `AI_quant.md`. Locally the futures engine treats it as a black box; if the upstream `predict` is unhealthy (no models loaded, scaler unfit), `_get_ml_signal` returns `None` and the trade falls back to indicator score alone. There is **no health check** on the ML model before relying on it. Tag `FUT-Q-MLHEALTH`.

---

## 6. Profitability levers (ranked)

| Rank | Lever | Expected ROI | Effort | Tag |
|---|---|---|---|---|
| 1 | **Fix `combined_score` attribute** so dynamic sizing actually scales with signal strength | +20–40% expected return on winners | XS | `FUT-Q-01` |
| 2 | **Fix MACD signal line** to real EMA9(MACD-line) | Restores 1 independent signal axis; ~+10% Sharpe | S | `FUT-Q-02` |
| 3 | **Subtract round-trip taker fees + slippage from funding-arb decision** before opening; disable strategy until cross-leg fee math is added | Saves ~28 bps/trade × N trades | S | `FUT-Q-03` |
| 4 | **Add a regime filter** (e.g., BTC realized vol > X → reduce leverage; choppy regime → block trend trades) | +0.3–0.5 Sharpe in backtest | M | `FUT-Q-04` |
| 5 | **Walk-forward backtest harness** with realistic slippage + fee + funding model and store results in DB | Enables every other lever to be validated | L | `FUT-Q-05` |
| 6 | **Replace fixed-pct trailing with ATR-based trail** | +5–15% expected on runners | S | `FUT-Q-06` |
| 7 | **Beta-hedge the DEX basket** using rolling 30d OLS beta of DEX-portfolio-returns vs BTCUSDT futures returns | Hedge effectiveness from ~0.5 → ~0.85 | M | `FUT-Q-07` |
| 8 | **Cache leverage state** per symbol; skip `set_leverage` if already set | Avoids 429s on burst | XS | `FUT-Q-08` |
| 9 | **Add maker-only mode** with limit-order quoting at top-of-book ± 1 tick; fallback to taker after Xs | Captures 2 bps × 2 = 4 bps per round-trip | M | `FUT-Q-09` |
| 10 | **Health-gate ML model** in `_get_ml_signal` — if `model.is_fitted == False`, force confidence to 0 | Prevents silent garbage signals | XS | `FUT-Q-10` |

---

## 7. Live-trading hygiene

- **Model staleness:** N/A locally; depends on upstream ensemble retrain cadence (see AI report).
- **Retrain cadence:** N/A — no in-module training.
- **Signal latency:** `scan_interval_seconds = 30` (`config_manager.py:144`). On 15m candles this is fine, but ticker is fetched from a **mainnet** client even in testnet (`futures_engine.py:1893-1896`); under heavy market this can drift up to a few seconds, but on 15m timeframe the alpha decay is negligible.
- **Scaler-fit-on-live-row:** **not in this module** (no in-module ML scaler). Risk lives upstream in `AIStrategy` — see AI report.
- **Floating-point exit-reason classification:** `abs(tsl - entry) < 0.0001` will misclassify after exchange-fill rounding. Cosmetic.
- **Bybit silent no-op** — critical (see § 2.7).
- **`futures_module._combine_signals` weighted score uses `> 0.7` threshold AND raw weighted sum** (`futures_module.py:481-492`); since weights sum to 1 and individual confidences are <= 1, the sum is in [0,1] — so a 70% threshold against the weighted-confidence is a 70%-confidence rule, defensible but tighter than expected given the 40/30/20/10 mixture.

---

## 8. Proposed action backlog

| ID | Action | Files | Effort |
|---|---|---|---|
| FUT-Q-01 | Compute `combined_score` and attach to `TechnicalSignals` so dynamic sizing receives it | `core/futures_engine.py:1023-1029, 1323-1367` | XS |
| FUT-Q-02 | Replace MACD signal-line shortcut with 9-period EMA of macd-line history | `core/futures_engine.py:1291-1311` | S |
| FUT-Q-03 | Funding-arb: subtract `2 * taker_fee` (both legs) and slippage est from `expected_profit_per_period`; require net ≥ `min_profit_per_period_bps`; remove "long perp only" branch | `strategies/funding_arbitrage.py:55-220` | S |
| FUT-Q-04 | Add regime classifier (e.g., realized-vol bucket on BTC 1h returns) as a top-level gate before any entry | `core/futures_engine.py:984-1116` | M |
| FUT-Q-05 | Build `scripts/backtest_futures.py` consuming the same `_get_technical_signals` + slippage/fee/funding model; persist results | new file | L |
| FUT-Q-06 | ATR-based trailing stop: replace `trailing_stop_distance` % with `k * ATR(14)` | `core/futures_engine.py:849-864` | S |
| FUT-Q-07 | Implement rolling-OLS beta for DEX-basket hedge; size = `beta * basket_notional` | `strategies/hedge_strategy.py` | M |
| FUT-Q-08 | Cache `(symbol, leverage)` state and skip redundant `set_leverage` | `exchanges/binance_futures.py:182-213, 306-307` | XS |
| FUT-Q-09 | Add `quote_mode='maker'` config flag; quote LMT @ best±1 tick, taker fallback after `maker_timeout_ms` | `core/futures_engine.py:1685-1705` | M |
| FUT-Q-10 | `_get_ml_signal` returns None unless `self.ml_model.is_fitted and (now - model.trained_at) < max_staleness` | `futures_module.py:308-346` | XS |
| FUT-Q-11 | Wire real Bybit executor or hard-fail at init when `bybit_config.enabled=True` and executor is stub | `exchanges/bybit_futures.py:84-111` | M |
| FUT-Q-12 | Look up `/fapi/v1/fundingInfo` to get per-symbol funding interval; use it in APR math | `strategies/funding_arbitrage.py:72,210` | S |
| FUT-Q-13 | Document calibration: log `Daily PnL` vs predicted EV; alert if running |EV − realized| > 50% | `core/futures_engine.py:1859` | S |
| FUT-Q-14 | Make `trend_following.py` reference engine config (single source of truth) or remove if dead code | `strategies/trend_following.py:94-108` | XS |
| FUT-Q-15 | Floating-point exit-reason: change `abs(tsl - entry) < 0.0001` to a fraction of price (`< entry_price * 1e-6`) | `core/futures_engine.py:917, :926` | XS |

---

## 9. Open questions for analyst / PM

1. Is `trend_following.py` actually invoked anywhere in production, or is the engine's inline indicator code the only live path? (Tag dead-code if not.)
2. Confirm Binance funding-rate source: `/fapi/v1/premiumIndex` (next predicted) vs `/fapi/v1/fundingRate?limit=1` (last paid). The arb math needs the former.
3. Is there any historical futures-trades dataset we can replay against for a walk-forward Sharpe estimate? `futures_trades` table exists (`futures_engine.py:476-488`) — how many rows?
4. Why is `min_signal_score` 4 and not 5 in `config_manager.py:154` when both the docstring says "4-5 balanced, 6+ conservative"? Was this chosen by trial?
5. Does `core/pattern_analyzer.py` (referenced by `futures_module._get_pattern_signal` `:374-408`) have any documented hit-rate?
6. Was the "54% win rate" figure in `config_manager.py:79-82` derived from real trades or assumed? If real, on what asset / period / fee schedule?

---

End of report — `FUTURES_quant.md`.
