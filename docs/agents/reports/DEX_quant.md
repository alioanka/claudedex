# DEX_MODULE — Quant / Algorithm Audit

## Executive summary

The DEX module is a thin wrapper (`modules/dex_trading/dex_module.py:1-394`) over four sources of alpha: a momentum strategy (`trading/strategies/momentum.py`), a scalping strategy (`trading/strategies/scalping.py`), an AI/ML ensemble strategy (`trading/strategies/ai_strategy.py`), and a meta-decider (`core/decision_maker.py`). Each is supported by ~150 hand-coded features (`data/processors/feature_extractor.py`), TA-Lib indicators (`core/pattern_analyzer.py`), and an ensemble of tree + neural models (`ml/models/ensemble_model.py`, `pump_predictor.py`, `rug_classifier.py`, `volume_validator.py`). On paper the surface area is large; in practice **there is no demonstrable edge** because virtually every numeric threshold (RSI 30/70, 5% stop, 2x volume, 0.7 confidence, 0.65 weighted-score gate, etc.) is hard-coded and has never been walk-forward validated against realised P&L net of cost. The strongest near-term lever is fixing cost modeling: the strategies enter at `current_price * (1 + spread/2)` (`scalping.py:394`) and exit at `current_price * 0.95` (`momentum.py:275`) without paying gas, swap fee, MEV/sandwich premium, or post-execution slippage. With Solana-style $0.001 gas, a 1% router fee, plus 30-100 bps adverse selection, the 0.5%-2% scalping targets (`scalping.py:59-60`) are *negative-expectation* trades by construction.

Top three algorithmic gaps: (1) the AI strategy fits `StandardScaler` per single feature vector at predict time (`ai_strategy.py:295`), an order-of-magnitude leakage that makes every "scaled" feature meaningless; (2) the `MomentumStrategy.analyze` adds positions to `self.active_positions` *before* the order manager has executed (`momentum.py:100`), so backtest P&L is computed at zero-slippage fills; (3) the `DecisionMaker` injects `np.random.uniform(0.9, 1.1)` "exploration" into strategy selection (`decision_maker.py:535`) which is exploration without exploit credit — pure noise during live trading.

Top three backtest/validation concerns: (1) `BaseStrategy.backtest` uses the same lookback window as live (`base_strategy.py:765`), and `analyze` peeks at `market_data["price"]` which is the last bar's close — perfect look-ahead in the simplest case; (2) `_calculate_volatility` annualises by `* np.sqrt(252)` on minute-bar returns (`ai_strategy.py:619`) — wrong scaling factor for high-frequency data; (3) there is no model-version, training-data-snapshot, or train/test split metadata stored anywhere — `EnsemblePredictor.retrain` rebuilds all models in place (`ensemble_model.py:786`) and overwrites disk without versioning.

## Existing-edge audit

### MomentumStrategy (`trading/strategies/momentum.py`)
- **Alleged edge**: Confluence of breakout + trend + volume + smart-money signals beats a single indicator in trending crypto markets.
- **Cost-aware?** No. Stop loss at `current_price * 0.95` (`momentum.py:275`), entry at raw price; no fees, no slippage, no gas. Smart-money stops use whale entry price (`momentum.py:740`) without verifying timestamp freshness.
- **Walk-forward / OOS evidence?** None in repo. `_calculate_volume_ratio` divides current volume by an `average` field supplied by the data layer — but the data layer's `volume_history` includes the current bar (`feature_extractor.py:155-159`), so the ratio is anchored to a window that contains the test sample. Look-ahead.
- **Failure modes**: Whipsaws in ranging markets (no regime gate; the `decision_maker` does classify markets but the strategy itself does not consult the classifier). Smart-money flow uses `whale_activity.net_flow > 0` as a binary gate (`momentum.py:305`) — gameable by sybil whales. Resistance levels from local maxima (`momentum.py:458`) are computed on `highs` that include the current bar.

### ScalpingStrategy (`trading/strategies/scalping.py`)
- **Alleged edge**: Tight stops + Bollinger + RSI mean-revert capture intraday noise.
- **Cost-aware?** Half-spread is added to entry (`scalping.py:394`), which models *the quoted spread*, not realised slippage. Min profit target = 0.5% (`scalping.py:59`); stop = 0.3% (`scalping.py:61`). On Uniswap V3 with 5-bps fee tier you pay 5 bps each leg = 10 bps round-trip; on V2 0.30% tier you pay 60 bps round-trip — eating the entire 50-bps min target before MEV. **Strategy is structurally unprofitable on EVM and only viable on Solana with Jupiter aggregator pricing.**
- **Walk-forward / OOS evidence?** None. `_calculate_confidence` (`scalping.py:438`) sums hand-picked +0.1/+0.15 bumps with no empirical fitting.
- **Failure modes**: Position-size is hardcoded "100" string (`scalping.py:584`), risk-management is bypassed. Win rate calculation uses `t.get("profit", 0)` but no field named `profit` is ever written to `completed_trades` (`scalping.py:613-618`), so `win_rate` is always 0. Effectively no live performance signal feeds back.

### AIStrategy (`trading/strategies/ai_strategy.py`)
- **Alleged edge**: Ensemble of pump_predictor + rug_classifier + pattern_analyzer + technical score should outperform any single model.
- **Cost-aware?** No cost modeling. The `risk_adjusted_score = weighted_score * (1 - rug_probability)` (`ai_strategy.py:396`) is a probability product, not a P&L adjustment.
- **Walk-forward / OOS evidence?** None. `_load_models` (`ai_strategy.py:928-965`) simply globs filenames; there is no held-out validation set, no calibration, no concept-drift monitor. If models aren't loaded the strategy quietly raises thresholds — a fail-safe, but it also means "trained" status is unverified.
- **Failure modes**: `_extract_features` fits the `StandardScaler` on a *single* row at predict time (`ai_strategy.py:295`), producing zeros (mean) for every feature. Model agreement uses `1.0 - std_dev*2` (`ai_strategy.py:474`), which can be negative-clipped but trains the strategy to fire on collinear models.

### DecisionMaker (`core/decision_maker.py`)
- **Alleged edge**: Dynamic strategy-selection beats any single strategy across regimes.
- **Cost-aware?** Position-size uses Kelly with a hard-coded `b=2.5` win/loss ratio (`decision_maker.py:706`) — not estimated from history. Stop losses set per strategy (`decision_maker.py:626-666`) are flat percentages.
- **Walk-forward / OOS evidence?** None. Strategy weights (`decision_maker.py:93-100`) are constants, never updated despite `_adjust_strategy_weights` being a stub (`decision_maker.py:837-856`).
- **Failure modes**: `np.random.uniform(0.9, 1.1)` multiplier on weighted scores (`decision_maker.py:535`) injects 20% pure noise on every decision. The Bull-market boost `score *= 1.3` (`decision_maker.py:326`) is a regime tail-wind that has no symmetric anti-bear deflator — long-only bias baked in.

## Bias & leakage findings

| ID | File:Line | Bias type | Description | Fix |
|----|-----------|-----------|-------------|-----|
| B-01 | `ai_strategy.py:289-295` | look-ahead / scaler-leak | `StandardScaler` is `fit_transform`-ed on the single live feature row, then re-fit on every subsequent call. Live features are always centred to zero. | Persist a scaler fitted on the **training** set only; load via `joblib.load` in `_load_models` and call `transform()` only at predict time. |
| B-02 | `base_strategy.py:765-783` | look-ahead | Backtest passes `data` (the current bar) directly into `analyze`, which keys on `data["price"]` — the bar's close. Entry at close means signal sees its own outcome. | Pass `data[t]` as features but enter at `data[t+1]["open"]` (T+1 bar fill). |
| B-03 | `momentum.py:447-465` | look-ahead | `_find_resistance_levels` reads `candles` including the current/latest bar's `high`; local maxima detection is biased to "now". | Use `candles[:-1]` only when computing levels for entry on `candles[-1]`. |
| B-04 | `momentum.py:100-101` | snapshot mismatch | `self.active_positions[token] = best_signal` is set inside `analyze` *before* the order executes. Backtest treats every signal as filled at the quoted entry. | Move position-insertion to `_on_position_opened` after order_manager success. |
| B-05 | `ai_strategy.py:619` | wrong-frequency scaling | `np.std(returns) * np.sqrt(252)` annualises any timeframe as if daily. For 5-min bars correct factor is `sqrt(252*78)`. Volatility feed is silently mis-scaled by ~9x. | Make annualisation factor a config param tied to `self.timeframe`. |
| B-06 | `decision_maker.py:535` | p-hacking / noise injection | Per-decision random multiplier `np.random.uniform(0.9, 1.1)` flips strategy selection ~20% of the time. | Replace with epsilon-greedy and log the exploration choice; gate behind `if config['exploration_mode']`. |
| B-07 | `pattern_analyzer.py:780-785` | look-ahead VWAP | VWAP is computed over the whole window including the last bar, then used to evaluate that bar. | Compute on `closes[:-1]` for prediction at `t`. |
| B-08 | `pattern_analyzer.py:910-925` | survivorship-style breakout | Breakout uses `recent_resistance = max(highs[-20:-1])`, then triggers if `closes[-1] > recent_resistance * 1.02`. Fine for forward analysis, but the volume confirmation uses `volumes[-1]` against `mean(volumes[-20:])` *including the breakout bar* — biases ratio upward. | Compare against `mean(volumes[-20:-1])`. |
| B-09 | `decision_maker.py:706-727` | in-sample tuning | Kelly uses fixed `b=2.5` and `confidence` as `p`. Confidence is itself fitted from the same data the strategy trades on. | Estimate `b` from rolling realized `avg_win/avg_loss` (already tracked in `StrategyPerformance`). |
| B-10 | `ensemble_model.py:786-849` | training-set rebake | `retrain()` calls `self.scaler.fit_transform(X)` over *all* data including data from currently-open positions; no time-series split. | Use `TimeSeriesSplit` (already imported) and freeze the scaler before training. |
| B-11 | `volume_validator.py:112-150` | survivorship | Features include `whale_accumulation`, `smart_money_flow` from a labeled list; the labels themselves are produced post-hoc. | Require labels to be timestamped earlier than the prediction window. |
| B-12 | `ai_strategy.py:705-722` | mis-applied estimator | Roll's spread estimator requires negative serial covariance from a *single* security's tick data; here it's applied to whatever `prices` list is passed in. If prices are minute closes, the estimator is invalid. | Document timeframe requirement; gate the call behind tick-availability check. |
| B-13 | `feature_extractor.py:101-105` | future-feature leak | `price_change_{period}` uses `prices[-period]` as denominator and `prices[-1]` as numerator; fine, but is also fed to a model whose label may be derived from `prices[-1]`. | Label must look at `prices[t+k]`, not `prices[t]`. |
| B-14 | `pump_predictor.py:183-188` | label leak | `prepare_sequences` labels `y[i] = 1 if future_price > current_price * 1.10` where `future_price = price_data.iloc[i]['price']` and `current_price = price_data.iloc[i-1]['price']` — that's same-bar, not future. | Use `price_data.iloc[i+k]` for a forward horizon `k`. |
| B-15 | `ensemble_model.py:716-717` | confidence inflation | `confidence = min(model_agreement*1.2, 1.0)`. Multiplying agreement by 1.2 inflates a well-calibrated agreement score to 1.0 in most cases. | Drop the 1.2 multiplier; calibrate via isotonic regression on held-out set. |

## Missing signals / features (ranked by expected lift)

1. **5-minute realised-vol ratio (current/30-day median)** — add to `feature_extractor.py:138-203`. Best single predictor of regime shift; expected lift 15-25 bps on momentum strategy by gating entries when ratio < 0.5 (compression) or > 3 (blowoff).
2. **Order-flow imbalance (signed taker volume)** — currently approximated by alternating buy/sell volumes (`ai_strategy.py:730`), which is garbage. Replace with real swap-direction parsing from `data/collectors` and feed to feature_extractor. Lift: 10-30 bps directional.
3. **Top-of-pool tick liquidity (Uni V3) or curve depth at ±2%** — present as raw `liquidity` USD only (`base_strategy.py:127`). Convert to "slippage-for-target-size" curve. Eliminates the largest cost-model bug.
4. **Funding/borrow rate proxy (perp basis)** — momentum trades hold across funding intervals on CEX-listed tokens; basis predicts squeeze risk. Lift: avoids ~3-5% drawdowns per cycle.
5. **MEV pressure index (per-block sandwich count from mempool monitor)** — exists in `data/collectors` but not surfaced as a feature. Should gate scalping size; the `mempool_data.sandwich_attack_risk` field is referenced in `ensemble_model.py:497` but no upstream populates it.
6. **Cross-DEX price dispersion (std of mid-prices across N pools)** — predicts informed-trader flow before public quotes converge. Add to `feature_extractor.py`.
7. **Whale-wallet age-weighted PnL** — current smart-money score (`momentum.py:429-445`) treats all whale wallets equally; weighting by historical realised PnL would up-weight informed wallets. Lift: 5-15 bps on smart-money trades.
8. **Token-level holder-Herfindahl index over time** (1h, 4h, 24h slopes) — feature_extractor exposes only static concentration. The *change* in concentration is a leading indicator of distribution-phase pumps.
9. **Volume-vs-mcap z-score against same-decile peers** — instead of absolute `volume_to_mcap_ratio` (`ensemble_model.py:384`), compare to peer percentile. Removes a regime-dependent feature shift.
10. **Time-since-last-listing on each DEX** (newness decay) — predicts launchpad-era volatility. Currently `contract_age_hours` is binary-treated.

## Capital-allocation & sizing review

`BaseStrategy.calculate_position_size` (`base_strategy.py:308-348`) implements three regimes: (a) Kelly-capped (`_calculate_kelly_position`, line 350), (b) fixed-fraction (default 2% of balance), (c) strength-multiplier override (×0.5 to ×1.5). The implementation is mathematically correct in form, but:

- **Kelly inputs are mis-estimated.** `b = average_win / average_loss` uses `StrategyPerformance` running averages, which on a fresh strategy default to 1:1 → `kelly_fraction ≈ 2p-1`. With `confidence=0.65` you get `f ≈ 0.30`, scaled by safety 0.25 → 7.5% per trade. Reasonable, but on a low-trade-count strategy the estimator has huge variance.
- **Strength multiplier (×1.5 for VERY_STRONG)** stacks on top of Kelly. A Kelly-sized position then multiplied by 1.5 violates Kelly's optimality and tail risk.
- **Decision-maker also resizes by strategy** (`decision_maker.py:624-660`, e.g. `base_size * 1.5` for scalping). With both layers active, scalping can run at 3× Kelly. **This is a blow-up vector.**
- **No correlation handling across modules.** DEX module's positions don't see futures, copy-trading, sniper exposure. The portfolio_manager exists but the strategy's position-size call ignores it. Concentrated correlated exposure is the most common large-loss mode.
- **No volatility targeting.** Position size is independent of asset volatility, so a 200% IV memecoin and a 30% IV major get the same dollar size.

Recommended: (i) compute `target_dollar_vol = balance * daily_vol_target / asset_atr`; (ii) cap via portfolio-level VaR / correlation matrix; (iii) drop the strength multiplier when Kelly is on; (iv) pass live `b` rolling-mean of last 50 trades not all-time.

## ML model health

`EnsemblePredictor` (`ensemble_model.py`):
- **Training cadence**: `retrain()` exists but has no scheduler. There is no `scripts/retrain_*.py` entrypoint per agent guidelines.
- **Drift detection**: None. `update_weights` (line 1067) takes performance dict but is never called by a monitor.
- **Online-vs-offline parity**: Offline features come from `extract_features(data: Dict)`; online features come from `AIStrategy._extract_features` (`ai_strategy.py:247-301`), which uses a *different* set of fields. Parity is unverified — likely broken.
- **Model versioning**: `save_models` writes to fixed filenames (`xgboost_pump.pkl` etc.). No version tag, no train-date metadata, no git-SHA. `load_models` happily loads stale models. Fix: name files `{model}_{train_date}_{sha}.pkl` and store metadata JSON.

`PumpPredictor` (`ml/models/pump_predictor.py`):
- **Critical leak (B-14)**: label uses same-bar price.
- LSTM input shape `(seq_len, len(price_features))` but `predict_sequences` calls `fit_transform` on each call — same scaler bug.
- TF/Keras 2.x dependency adds runtime overhead and conflicts with the PyTorch models in `ensemble_model.py`. Consolidate.

`RugClassifier` (`ml/models/rug_classifier.py`):
- Features include `has_blacklist`, `hidden_owner`, etc. — sourced from on-chain detectors. Quality depends on the upstream `analysis/rug_detector.py` (not in this audit). Class imbalance not addressed: rugs are rare, so `RandomForestClassifier` with default settings predicts "not rug" most of the time → high accuracy, useless precision.

`VolumeValidatorML` (`ml/models/volume_validator.py`):
- 40+ features but no training labels are produced anywhere in repo. Effectively dead code.
- `IsolationForest(contamination=0.1)` (`ensemble_model.py:277`) assumes 10% of all tokens are wash-trading; this is way too high for liquid pairs and way too low for memecoins. Make contamination per-chain configurable.

Fixes needed:
1. `scripts/retrain_ensemble.py` with: load history → time-series split → fit → eval → version-tag → write DB row.
2. `scripts/score_drift.py` that compares last-week feature distribution to training distribution (PSI/KL); flips a config flag if drift > threshold.
3. Add `MODEL_VERSION` to every prediction dict and persist with trades for forensic linkage.

## Profitability levers (ranked by ROI)

1. **Fix the scaler leak (B-01)** — single highest-impact change. Today the AI strategy's "ML score" is essentially random; properly scaled features could unlock the ensemble's nominal lift. Expected lift on AI strategy hit-rate: +10-20 percentage points.
2. **Add cost model to every entry/exit** (`base_strategy._create_order_from_signal`) — subtract `gas_usd + 2*pool_fee + estimated_slippage` from `expected_return`; only fire signals with `expected_return_net > 0`. Expected lift: turns scalping from -EV to +EV or kills it cleanly.
3. **Remove noise injection (B-06)** — `decision_maker.py:535`. Free win, immediate.
4. **Implement vol-targeting in `calculate_position_size`** — most consistent way to flatten drawdowns. Expected: Sharpe +0.3-0.5 by smoothing concentration risk.
5. **Add 5-min realised-vol ratio as features (#1 above)** + regime gate (skip momentum when ratio < 0.5). Expected: hit-rate +5%.
6. **Walk-forward harness in `BaseStrategy.backtest`** — currently single-pass over history; needs rolling 30/7 split with reset. Without it you cannot rank strategies honestly.
7. **Sandwich-aware sizing**: when `mempool_data.sandwich_attack_risk > X`, multiply size by `(1 - risk)`. Eliminates a known tail loss source.
8. **Calibrate confidence**: replace `min(agreement*1.2, 1.0)` (B-15) with isotonic calibration vs realised outcomes. Pre-req for any Kelly to work.
9. **Per-chain model fork**: train separate ensemble for Solana memecoins vs Ethereum bluechips; the joint model is dominated by the noisier class.
10. **Wire `_adjust_strategy_weights` stub** (`decision_maker.py:837`) to actually update weights from rolling PnL — currently the multi-strategy idea has no learning loop.

## Proposed action backlog

- [ ] **QT-01** Persist trained `StandardScaler` and load via `joblib` in `AIStrategy._load_models` — touches `ai_strategy.py`, `ensemble_model.py` — expected lift: +10-20 pp hit-rate on AI signal — owner: quant.
- [ ] **QT-02** Subtract gas + fee + spread cost from `expected_return` in `decision_maker._calculate_position_parameters` and gate `should_trade` on net-positive — touches `decision_maker.py`, `base_strategy.py` — expected lift: removes -EV trades, +30-50 bps avg/trade — owner: quant.
- [ ] **QT-03** Delete `np.random.uniform(0.9,1.1)` noise in `_select_best_strategy`, replace with epsilon-greedy logged — touches `decision_maker.py:535` — expected lift: Sharpe +0.1 — owner: quant.
- [ ] **QT-04** Add `realised_vol_ratio_5m_30d` feature to `feature_extractor.extract_volume_features` and gate momentum entries on it — touches `feature_extractor.py`, `momentum.py` — expected lift: hit-rate +3-5 pp — owner: quant.
- [ ] **QT-05** Fix `pump_predictor.prepare_sequences` label to use `iloc[i+horizon]` — touches `ml/models/pump_predictor.py:183` — expected lift: model becomes predictive instead of identity — owner: quant.
- [ ] **QT-06** Implement vol-target sizing in `BaseStrategy.calculate_position_size` (replace strength multiplier) — touches `base_strategy.py:308-348` — expected lift: max-DD -20-30% — owner: quant.
- [ ] **QT-07** Add `MODEL_VERSION` field to every `predict()` return dict + DB schema — touches `ensemble_model.py`, `pump_predictor.py`, `rug_classifier.py` — expected lift: enables drift forensics — owner: quant.
- [ ] **QT-08** Create `scripts/retrain_ensemble.py` with `TimeSeriesSplit`, version tag, drift report — touches `scripts/` (new) — owner: quant.
- [ ] **QT-09** Replace `np.std * sqrt(252)` annualisation in `ai_strategy._calculate_volatility` with timeframe-aware factor — touches `ai_strategy.py:619` — owner: quant.
- [ ] **QT-10** Remove duplicate position-insertion in `MomentumStrategy.analyze` (line 100); rely on `_on_position_opened` — touches `momentum.py` — expected lift: fixes backtest fidelity — owner: quant.
- [ ] **QT-11** Compute volume-confirmation against `mean(volumes[-N:-1])` not including current bar in `pattern_analyzer._detect_breakouts` — touches `pattern_analyzer.py:910-925` — owner: quant.
- [ ] **QT-12** Add isotonic calibration step after `EnsemblePredictor.retrain` using held-out set; replace `*1.2` confidence inflator — touches `ensemble_model.py:716` — owner: quant.
- [ ] **QT-13** Compute live `b = avg_win/avg_loss` from rolling last 50 trades in `_kelly_criterion` instead of hardcoded 2.5 — touches `decision_maker.py:706` — owner: quant.
- [ ] **QT-14** Wire `_adjust_strategy_weights` to update `strategy_weights` from rolling PnL via softmax — touches `decision_maker.py:837` — owner: quant.

## Open questions

1. Where is ground-truth labelled training data stored? `retrain()` expects `pump_label`/`rug_label` columns but no labeller is in repo.
2. Are minute-bar prices actually OHLCV minute candles, or are they tick snapshots? Many indicator scalings depend on this.
3. Is `data/collectors/mempool_monitor.py` populating `mempool_data.sandwich_attack_risk` end-to-end? Referenced but never seen written.
4. Where does the smart-money / whale list come from? If it's static, momentum's whale signal is gamed within days.
5. Does the orchestrator persist model files anywhere durable, or are they re-trained every container restart? Affects entire AI strategy reliability.
