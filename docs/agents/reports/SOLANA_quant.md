# SOLANA Module — Quantitative / Algorithmic Audit

**Author:** quant-algo-expert (Claude)
**Date:** 2026-05-11
**Scope:** Signal generation, scoring, ML, feature engineering, position sizing, and exit logic for the Solana module (`modules/solana_strategies/`, `modules/solana_trading/`, `analysis/`, `ml/models/`).

---

## 1. Executive Summary

The Solana module is overwhelmingly a **rule-based filter + parametric trailing-stop strategy**. It looks for new tokens (Pump.fun / DexScreener feeds), passes them through a hard-coded screen of liquidity/volume/holder thresholds, sizes positions with a multiplicative heuristic, then manages exits via a six-tier trailing-stop ladder. The sophisticated ML stack present in `analysis/` and `ml/models/` (PumpPredictor, RugClassifier, VolumeValidator, TokenScorer) is **not wired into the live Solana execution path**. `solana_engine.py` never imports `analysis.token_scorer`, `analysis.rug_detector`, or `ml.models.*`; the only "intelligence" applied at buy-time is the regex-based `scam_blacklist.py` and a hard-coded threshold pyramid.

This is the single largest quant problem with the module: there is **no learned signal** anywhere in the live path. The screen is binary, the position size scales with win-rate but not with predicted edge, and exit timing is set by hard constants. The ML models exist as dead code from a previous DEX-side build.

The strategy nonetheless has some real strengths: the trailing tier ladder is well-conceived for power-law memecoin payoffs (Tiers 0.5 → 5 with peak-relative stops at high tiers), the rapid-decline circuit-breaker is well-tuned, and the auto-blacklist on crash is the right feedback loop. The biggest profit lever is **replacing the binary rule-based screen with a calibrated ML score that gates the buy and modulates position size**, plus fixing two material biases (look-ahead in pump-predictor labels, slippage-free PnL reporting).

**Top profit lever:** Wire `ml/models/rug_classifier.py` + a re-trained `pump_predictor.py` into `_open_position()` at `solana_engine.py:2823`; replace the 20+ hard-coded boolean filters with a probability threshold + Kelly-fraction sized bet.

**Top bias risk:** The pump-predictor in `ml/models/pump_predictor.py:163-190` builds its label from `future_price > current_price * 1.10` using DataFrame indices, with no temporal split — straight look-ahead. The Tier 0.5 trailing rule was also retro-fit to specific historical tokens (`nip`, `BTC`, `GAS`, `THRT` in inline comments at `solana_engine.py:2527`, `2713`), strongly suggesting in-sample tuning.

---

## 2. Existing-Edge Audit (per signal / model)

### 2.1 Pump.fun new-token screen (`solana_engine.py:575-781`, `_open_position` checks at 2826-2960)
- **Source of edge:** filter out obvious garbage (sub-$3k liquidity, single-character symbols, scam-name regex, sells > 2× buys in 5m).
- **Realised edge:** unknown. The module logs a win-rate (`solana_engine.py:1844`) but no per-feature attribution exists. There's no A/B test of the filters; you cannot tell which threshold contributes alpha vs. which is theatre.
- **Verdict:** survival filter, not alpha generator. It mainly rejects rugs that would have lost 90 %+; the marginal effect on positive-EV tokens is unmeasured.

### 2.2 Pump.fun entry filters (`solana_engine.py:683-758`, `2826-2960`)
The pre-buy gauntlet:
- `liq < 3000` → reject (`:700`)
- `vol/liq > 50` → reject (wash) (`:694`)
- `vol/liq < 0.5` → reject (dying) (`:706`)
- `price_change_24h < -10` → reject (`:689`, `:2935`)
- `price_change_5m < -5` → reject (`:724`)
- `sells_5m > 2× buys_5m` → reject (`:733`)
- `sells_1h > 1.5× buys_1h` → reject (`:741`)
- `buys_5m < 3` → reject (`:746`)
- `makers_1h < 10` → reject (`:754`)
- `5000 < mcap < 10M` window (`:2919-2924`)
- Holder count proxy via `makers_1h` (`:2906`)
- Dev holding ≤ 10 % (`:2913`)

**Issue:** these thresholds are unjustified. None reference a study or backtest. They are very tight and likely throw away a lot of borderline-EV tokens. The `vol/liq > 50` rule is good (wash detection); the `vol/liq < 0.5` rule is reasonable; the `price_change_24h < -10` rule may be too aggressive — many pump.fun memes have legitimate -15% intra-bonding-curve dips before reversing.

### 2.3 Position sizing (`solana_engine.py:1637-1703`)
`calculate_dynamic_position_size` multiplies the base size by:
- `signal_multiplier = 0.5 + signal_strength` (range 0.5–1.5)
- `volatility_multiplier = clip(1/vol, 0.5, 1.5)`
- `trend_multiplier = 0.8 + trend_strength*0.4` (range 0.8–1.2)
- `win_rate_multiplier = 0.5 + win_rate` (range 0.8–1.2 for win_rate 0.3–0.7)
- `exposure_multiplier = max(0.5, 1 - exposure*0.5)`

This is a **heuristic stack with no Kelly grounding and no ruin-probability awareness**. For pump.fun bets where ruin probability per trade is 30–60 %, you need fractional-Kelly with a ruin cap, not a 50–150 % multiplicative envelope. The `signal_strength` input on the only path that uses it (`_scan_jupiter_opportunities`, `:2690`) is computed as `min(1.0, abs(momentum)/2.0 + 0.3)`, where momentum is the price change vs. last 9 ticks — almost pure noise on liquid SOL pairs. The Pump.fun path **does not call this function at all** (`:2789` passes the raw `buy_amount_sol`), so dynamic sizing is wasted on Pump.fun.

### 2.4 Exit timing — trailing stop ladder (`solana_engine.py:2254-2574`)
Six-tier ladder:
| Tier | Trigger (peak gain) | SL gain | Partial exit |
|------|---------------------|---------|--------------|
| 0    | <20%                | −20% (entry) | none |
| 0.5  | 20–40%              | +10%    | 30%   |
| 1    | 40–80%              | +20%    | 25%   |
| 2    | 80–150%             | +40%    | 25%   |
| 3    | 150–400%            | 65% of peak | 20% |
| 4    | 400–1000%           | 75% of peak | 20% |
| 5    | 1000 %+             | 80% of peak | remainder |

This is **the strongest single piece of quant work in the module**. The structure correctly handles memecoin power-law payoffs: take chips off the table early to fund optionality on a 100× tail. The hand-tuned constants (`+10%` lock at 20% peak, `+20%` at 40% peak) are aware of the 5-second monitoring gap and pump.fun crash speeds.

**Weaknesses:**
- Constants are global; should be per-token-risk-band (a low-mcap meme dies faster than a $5 M meme).
- The `early_volatility_window = 120` second SL widening (`:2452-2457`) is a sensible patch but is also unmeasured.
- The "gap loss" warning at `:2501-2506` is the right diagnostic but no action is taken to size around it.

### 2.5 Rapid-decline circuit breaker (`solana_engine.py:2313-2397`)
- `decline_pct >= 40` between two 5s checks → emergency exit + auto-blacklist (`:2329`).
- `decline_pct >= 30` AND held < 5 min → emergency exit + blacklist (`:2357`).
- `decline_from_peak >= 60` → emergency exit (`:2388`).

This is a good defensive mechanism and the auto-blacklist feedback loop (`scam_blacklist.on_rapid_crash_detected`) is exactly the right learning loop for a meme bot. However: the emergency-close trade is recorded with **assumed PnL** (`pnl_pct = -50`, `pnl_sol = -0.5 * value_sol` at `:2076-2077`) rather than the actual fill — this directly biases reported PnL.

### 2.6 SafetyEngine sell-route verification (`safety_engine.py:171-278`)
Honeypot check works by attempting a sell quote pre-buy. Two problems:
- On rate-limit / `quote == None`, the function **allows the trade** (`:222-224`) and explicitly comments "Conservative approach: Allow the trade." That is the *opposite* of conservative — it ignores the signal entirely. Real honeypots will fail to quote.
- The expected-output sanity check at `:252` uses a hard-coded `100` for the SOL price (`actual_usd = actual_sol * 100`). Should pull `self.sol_price_usd`.

### 2.7 Scam blacklist regex (`scam_blacklist.py:41-92`)
Static regex of celebrity/scam patterns (`TRUMP`, `ELON`, `100X`, etc.). Effective for the obvious cases. Cost: zero false-negatives for new patterns the regex doesn't know about, since list never auto-learns from on-chain rug data — only from the rapid-decline detector. **The blacklist should also be populated from `dev_analyzer.py` outputs** (creator wallets that have previously rugged), which is currently not wired in.

### 2.8 Jupiter / Drift helpers
- `jupiter_helper.py`: pure execution. No signal logic. **No multi-route comparison or price-improvement check** — Jupiter is treated as a single source of truth.
- `drift_helper.py:346-375`: funding-rate fetch exists but is never consumed by any trading logic. Drift perpetuals are configured but the entire `_scan_drift_opportunities` body is two log lines (`solana_engine.py:2712-2715`).

### 2.9 TokenScorer / RugDetector / PumpPredictor / DevAnalyzer / VolumeValidator
**Not invoked anywhere from the Solana live path.** Files exist with substantial detail (TokenScorer at `analysis/token_scorer.py:101-180` has a full composite-scoring flow; RugClassifier at `ml/models/rug_classifier.py:46-95` has a sensible 35-feature set including LP-burn %, freeze authority, dev wallet activity), but `grep` confirms `solana_engine.py` does not import any of them. The same Black-box ML stack used by the DEX module is **completely orphaned** from the Solana module.

---

## 3. Bias & Leakage Findings

| # | Location | Bias / Leakage | Severity |
|---|----------|----------------|----------|
| B1 | `ml/models/pump_predictor.py:184-188` | `prepare_sequences` labels by `future_price > current_price * 1.10` using a single DataFrame; no walk-forward; no temporal embargo — **look-ahead** | High |
| B2 | `ml/models/pump_predictor.py:362` | `train_test_split` by 80/20 of array index — fine for random data, **catastrophic for time-series** if data is sorted | High |
| B3 | `analysis/pump_predictor.py:352` | "prediction_accuracy" formula `1 - abs(actual - predicted)/predicted` — silently divides by predicted magnitude; favours small predictions | Med |
| B4 | `solana_engine.py:2076-2089` | Emergency-close trade records a **hard-coded −50% PnL** rather than actual fill; biases reported metrics negative-of-truth | High |
| B5 | `solana_engine.py:683-758` filter constants | Inline comments name specific tokens (`nip`, `BTC`, `GAS`, `THRT`, `:2527`, `:2713`) — clear in-sample tuning | High |
| B6 | `analysis/token_scorer.py:642` | Confidence uses `np.mean` over partial-data factors; if one is `1.0` it inflates confidence even with missing fields | Low |
| B7 | `solana_engine.py:3157` | `actual_entry_price = (amount_sol * sol_price_usd) / actual_tokens` ignores fee + slippage on the buy leg, so reported entry is too low → **inflated PnL** on winners | Med |
| B8 | `_save_trade_to_db` (`:1758`) records PnL **gross of execution gas/priority fee** — only counts Jupiter fee field which is rarely populated | Med |
| B9 | `_get_token_price` (`:2803`) caches DexScreener price 5 s (`:371`); during fast-moving moments SL/TP triggers on cached not actual price | Low |
| B10 | `safety_engine.py:255` | Hard-coded SOL price `100` in honeypot ratio check — under-estimates expected output vs. SOL appreciation | Low |
| B11 | `analysis/pump_predictor.py:209` | Volume-spike formula divides recent (last 5) by mean of last 50 — survivorship bias when token has just listed (no 50-period history) | Med |
| B12 | `dev_analyzer.py:88-100` | Loads a `known_ruggers` set in-memory; never persisted or refreshed from chain — survivorship of stale data | Med |

The pump-predictor labelling (B1, B2) means **any reported model accuracy from `analysis/pump_predictor.py` should be considered fraudulent until walk-forward is added**.

---

## 4. Missing Signals / Features (ranked by expected lift)

1. **Creator-wallet history (deployer reputation).** `dev_analyzer.py` exists but isn't wired in. For pump.fun, the single strongest predictor of rug is "this dev rugged 3 other tokens in the last 7 days." Cluster wallets by funding source (same SOL-funder = same dev), then count previous rugs. *Expected lift: massive on rug avoidance — possibly 30 %+ false-positive reduction.*
2. **Top-10-holder concentration (real, not `makers_1h`).** Currently approximated by trader-count proxy at `solana_engine.py:2906`. Helius / Solscan supplies a real top-10 holder % via `getTokenLargestAccounts`. Should be in every buy filter.
3. **Bonding-curve progress %.** Pump.fun tokens have a deterministic bonding curve (0–100 % filled, graduates to Raydium at 100 %). Buying at curve 5 % vs. 80 % is a different bet entirely. The signal is on-chain and free.
4. **Snipe-block / first-block-buyer detection.** Tokens where the deployer + 5 confederates bought in the first block are pre-coordinated rugs. Check the first 10 token-transfer instructions of the mint tx.
5. **LP burn vs. lock vs. owned.** `rug_classifier.py:50-53` lists `lp_burn_percentage` but it's never populated. Pull from the LP account directly.
6. **Twitter / social-velocity around mint address.** A simple Twitter search-API query for the mint string in the last 30 minutes; non-zero mentions correlate with sustained pumps. Free signal.
7. **Time-since-launch curve fit.** Most pump.fun tokens die in a power-law T^-1; the distribution of survival times is a known prior. Sizing should taper with age.
8. **Buy-pressure asymmetry: novel-wallet buys vs. recycler buys.** Recyclers (wallets that buy every pump.fun mint) are noise; novel wallets (first-time buyers) are alpha. Identifiable from on-chain history.
9. **SOL-funded vs. CEX-funded creator.** Wallet funded from Kraken/Binance hot wallet is much less likely to be a serial rugger than one funded from a fresh SOL deposit via Tornado-style mixer.
10. **Drift funding-rate signal.** `drift_helper.py:346` already fetches it. A simple long-funding-paid > 50 bps annualised + perp basis vs. spot > 30 bps → carry-trade signal that the entire Drift code path was meant to exploit.

---

## 5. Capital-Allocation Review

Pump.fun positions size = fixed `pumpfun_buy_amount` (`solana_engine.py:2789`), default 0.05 SOL. This is a flat micro-bet. Strengths: simple, ruin-bounded. Weaknesses:
- No Kelly: a 100 %-confident edge gets the same bet as a 51 %-confident one.
- No correlation handling: max-positions caps total exposure but does not penalise correlated bets (e.g., three near-simultaneous pump.fun snipes on similar-theme tokens behave like one bet).
- `daily_loss_limit_sol = 5.0` is a hard floor, no soft taper — a typical week with 80 losing 0.05-SOL bets ($800 loss) hits the limit fast.
- The `_get_block_duration` progressive block (`:222-230`) is well-designed for cooldowns but only triggers on consecutive losses, not on a deteriorating Sharpe / streak-z-score.

Recommendation: implement fractional-Kelly with `f* = (p*b - q)/b` capped at 5 % of bankroll, where `p` is a calibrated probability from the rug-classifier + pump-predictor ensemble and `b` is the expected upside multiple from historical tier-distribution. Currently this can't be computed because the ML stack isn't wired up.

---

## 6. ML Model Health

| Model | Status | Wired In? | Walk-forward CV? | Last retrain script? |
|-------|--------|-----------|------------------|----------------------|
| `ml/models/pump_predictor.py` (LSTM + 4 GBMs) | Built | **No** | No (`:359`, simple 80/20 split) | No `scripts/retrain_*.py` found |
| `ml/models/rug_classifier.py` (XGB+LGB+RF+GB) | Built | **No** | No (random `train_test_split`, `:277`) | No |
| `ml/models/volume_validator.py` (RF+XGB+LGB+IsoForest) | Built | **No** | No | No |
| `ml/models/ensemble_model.py` | Built | **No** | unknown (not read) | No |
| `analysis/token_scorer.py` | Built | **No** | n/a (rule-based) | n/a |

**Every Solana buy currently uses zero ML.** The scaffolding is sophisticated but the live signal is regex + thresholds. Until a model is wired in, the comparative-model exercises in the file headers are aspirational.

The model trainer signatures (`train(self, historical_data, labels)`) require a labelled dataset that nothing in the repo produces. There is no `scripts/build_pumpfun_dataset.py`, no S3 bucket of historical pump.fun launches, and no on-chain backfill job. Without that pipeline, retraining is impossible.

---

## 7. Profitability Levers (ranked, with effort estimate)

1. **Wire RugClassifier into `_open_position` (`solana_engine.py:2823`).** Replace the 20 boolean filters with `if rug_classifier.predict_proba(features)[1] > 0.4: return False`. Effort: 1 day to plumb features; 3 weeks to build the labelled dataset and retrain. *Expected: 20–40 % reduction in rug exposure, 5–10 % increase in qualifying tokens (fewer over-rejections).*
2. **Replace fixed Pump.fun size with fractional-Kelly + ruin-cap.** Use rug-classifier-derived `p` and historical tier-distribution-derived `b`. Effort: 2 days once models exist. *Expected: 1.5–2× Sharpe on the strategy.*
3. **Per-token-risk-band trailing parameters.** Compute a token-volatility-tier at entry (low-mcap meme vs. mid-mcap meme vs. graduated token) and load the corresponding tier-ladder constants from `ConfigManager`. Effort: 1 day. *Expected: 10–20 % better risk-adjusted return.*
4. **Slippage-aware reported PnL.** Replace the −50 % emergency-close assumption (`:2076`) with the actual fill SOL value. Fix `actual_entry_price` to net-of-fee. Add Jupiter actual slippage. Effort: 0.5 day. *Expected: honest metrics → all subsequent tuning becomes valid.*
5. **Creator-wallet clustering & blacklist auto-population.** `dev_analyzer.py` outputs → `scam_blacklist.add_to_blacklist`. Effort: 3 days. *Expected: catches serial ruggers pre-buy; possibly 15 %+ of pump.fun rugs share creator-cluster fingerprints.*
6. **Walk-forward CV harness for pump/rug models.** `sklearn.model_selection.TimeSeriesSplit` + embargo. Effort: 1 day. Prerequisite for any honest model.
7. **Bonding-curve-progress feature for Pump.fun.** Single `getAccountInfo` per candidate. Effort: 1 day. *Expected: removes the worst cohort (curve >85 %, almost-graduated, near-peak risk).*
8. **Drift carry-trade reactivation.** Funding-rate threshold + perp-spot basis. The infrastructure exists, the alpha is well-documented in Solana derivatives literature. Effort: 1 week. *Expected: low-Sharpe but uncorrelated PnL stream.*

---

## 8. Proposed Action Backlog

| ID | Title | Files | Effort | Priority |
|----|-------|-------|--------|----------|
| SOL-Q-01 | Fix `_save_trade_to_db` to record true fill PnL (incl. slippage + fees) | `solana_engine.py:1758, 3157` | S | P0 |
| SOL-Q-02 | Replace hard-coded −50 % emergency-close PnL with actual SOL received | `solana_engine.py:2076-2098` | S | P0 |
| SOL-Q-03 | Add `TimeSeriesSplit` + embargo to rug-classifier + pump-predictor training | `ml/models/rug_classifier.py:277`, `ml/models/pump_predictor.py:359` | S | P0 |
| SOL-Q-04 | Build `scripts/build_pumpfun_dataset.py` (mint addresses, on-chain history, T+24h outcomes) | new | L | P0 |
| SOL-Q-05 | Wire RugClassifier into `_open_position` as a probability gate | `solana_engine.py:2823`, `ml/models/rug_classifier.py` | M | P1 |
| SOL-Q-06 | Wire PumpPredictor as positive-side score; combine with rug-prob into composite score | same | M | P1 |
| SOL-Q-07 | Implement fractional-Kelly sizing in `calculate_dynamic_position_size` | `solana_engine.py:1637` | M | P1 |
| SOL-Q-08 | Honest sell-route verification: fail-closed when quote=None on non-rate-limit | `safety_engine.py:222` | S | P1 |
| SOL-Q-09 | Add top-10-holder concentration feature via Helius | `solana_engine.py:2826`, new helper | M | P1 |
| SOL-Q-10 | Add bonding-curve-progress feature for Pump.fun entries | new helper, `solana_engine.py:2826` | M | P2 |
| SOL-Q-11 | Cluster creator wallets (funding-source graph) and auto-blacklist serial ruggers | `analysis/dev_analyzer.py`, `scam_blacklist.py` | L | P2 |
| SOL-Q-12 | Surface trailing-tier constants per risk-band via `ConfigManager` | `solana_engine.py:2254-2574`, `solana_config_manager.py` | M | P2 |
| SOL-Q-13 | Replace hard-coded $100 SOL price in honeypot check with `self.sol_price_usd` | `safety_engine.py:255` | XS | P2 |
| SOL-Q-14 | Activate Drift funding-rate carry-trade logic in `_scan_drift_opportunities` | `solana_engine.py:2702`, `drift_helper.py:346` | L | P3 |
| SOL-Q-15 | Persist `dev_analyzer` known-rugger set to DB | `dev_analyzer.py:88` | M | P3 |
| SOL-Q-16 | A/B harness for individual filter constants (`min_buys_5m`, `vol/liq` bands) | new | M | P3 |
| SOL-Q-17 | Replace 5 s DexScreener price cache for active positions with WebSocket subscription | `solana_engine.py:371` | L | P3 |
| SOL-Q-18 | Add daily strategy attribution dashboard (PnL per filter rejection) | new | M | P3 |

---

## 9. Open Questions

1. Is there *any* labelled historical dataset for Pump.fun rugs vs. winners? If not, SOL-Q-04 is the gating item for the entire ML wiring path.
2. What is the production target win-rate vs. expected payoff distribution? Tier-ladder constants are only optimal once we know the empirical distribution of peak-gains for tokens that pass current filters.
3. Are Drift perpetuals strategically intended or vestigial? If kept, allocate engineering time; if not, delete the dead code.
4. Is the analyst module aware that the ML stack in `analysis/` and `ml/models/` is dead code on the Solana path? Risk-policy assumptions may depend on signals that aren't actually computed.
5. What slippage is *actually* realised on close? The 7-retry escalation up to 35 % `max_emergency_slippage_bps` (`safety_engine.py:76`) suggests realised slippage on stuck closes is far higher than the optimistic 200 bps used in some PnL paths.
6. Is the `_demo_trade_counter` periodic-DRY-RUN trade trigger (`solana_engine.py:2674-2679`) ever active in production? If yes, it pollutes the trade history with synthetic bets.

---

*End of SOLANA quant audit.*
