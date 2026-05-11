# AI_MODULE — Quant Audit

Auditor: `quant-algo-expert`
Date: 2026-05-11
Scope: AI / ML / sentiment / ensemble layer. Files in `modules/ai_analysis/`, `trading/strategies/ai_strategy.py`, `ml/models/*`, `ml/training/auto_trainer.py`, `ml/optimization/*`, `analysis/{token_scorer,market_analyzer,pump_predictor,rug_detector}.py`.

---

## 1. Executive verdict — RED

The AI module is a stack of **plausible-looking ML scaffolding wrapped around two genuinely dangerous bugs and one strategic confusion**:

1. **`ai_strategy.AIStrategy._extract_features` fits a `StandardScaler` on a single live row** (`trading/strategies/ai_strategy.py:50, 292-295`) every time no prior fit exists, then uses it forever. **A scaler fit on 1 row produces `mean = x, std = 0` → division by zero → `NaN` features → garbage downstream predictions**. The branch `if len(self.scaler.mean_) > 0` (`:292`) actually raises `AttributeError` until the scaler is fit at least once, so the first call falls through to the `else` branch (`:295`) which `fit_transform`s on 1 row. This is **the bug `dex-audit` flagged**, confirmed. Severity: critical.
2. **`SentimentEngine` treats LLM output as a hard trading signal**: it parses a raw float from `gpt-4o-mini` / `claude-3-5-haiku`, thresholds at 0.5, and executes a market order on Binance perp (`modules/ai_analysis/core/sentiment_engine.py:438-469, 741-790`). There is no probability calibration, no confidence interval, no agreement check between providers (the `both` mode just averages two LLMs — `:443`). LLM outputs at this scale are not stationary probabilities.
3. **`AIStrategyGenerator` (`modules/ai_analysis/core/ai_trading_engine.py:347-500`) literally asks the LLM to write strategy rules in JSON and stores them in the DB**, then `evolve_strategy` asks the same LLM to "improve" them based on performance text. There is no backtester gating the new strategy. The LLM is making allocator decisions with no statistical validation. This is product-prompt-engineering, not quant.

There are also more conventional bugs: synthetic-data class balance trained as a substitute for real walk-forward (`ml/training/auto_trainer.py:163-209`), no time-series CV in any training path, no drift detector anywhere, model files loaded by glob-latest with no atomic version pinning (`trading/strategies/ai_strategy.py:937-961`), and a fake MACD signal-line in `ensemble_model`'s technical-feature pipeline mirroring the futures bug.

I cannot recommend flipping `DRY_RUN=false` for this module under any circumstance until items 1, 2, 3 are fixed.

---

## 2. Edge audit, by strategy / model

### 2.1 `AIStrategy` (`trading/strategies/ai_strategy.py`)
**Claimed edge.** Weighted combination of pump-predictor, pattern-strength, and technical-score, gated by rug-classifier ≤ 0.2 and pump ≥ 0.6 (`:399-403`).

**Reality.**
- **Scaler bug** (see § 3.1). Until fixed, every prediction downstream is garbage.
- **Model agreement** is computed as `1 - 2*std([pump, technical, pattern])` (`:471-474`). With three predictions in [0,1], std is at most ~0.47 → agreement ∈ [0.06, 1]. That is a serviceable proxy but it's not actually a "model agreement" — pump and technical-score are scalar floats from different models with different *meanings* (technical-score = heuristic 0–1; pump = sigmoid prob). Comparing them via std-dev assumes calibrated scales, which they aren't.
- **`_retrain_models` is a stub.** Lines 902-926: it builds X, y, logs "Retrained models with N samples" — and does nothing else. Online learning is fake.
- **`_load_models` only loads pump_predictor** (`:937-948`); ensemble, rug, pattern are never disk-loaded. The strategy runs against untrained models 99% of the time.
- **`_get_ml_predictions` calls `self.ensemble_model.predict(token, chain)`** which in turn calls the network to fetch DexScreener data (`ml/models/ensemble_model.py:521-597`) — **on the inference path**. So every signal evaluation hits external HTTP, adding ~200-500ms latency and a single point of failure. Tag `AI-Q-04`.

### 2.2 `SentimentEngine` (`modules/ai_analysis/core/sentiment_engine.py`)
**Claimed edge.** LLM reads news headlines, returns a single float -1..+1; if `|score| ≥ 0.5` and `direct_trading=True`, market-buys/sells ETH at $50.

**Reality.**
- **No financial signal aggregation.** A single LLM call on ~10 headlines is the entire alpha. There is no:
  - de-duplication of headlines (same story across 5 outlets = 5x weight)
  - recency weighting
  - source credibility weighting (despite the prompt asking for it — the LLM doesn't have that data)
  - market-impact filter
- **Binary symbol choice**: hardcoded `symbol = "ETH"` (`:745`). Score → side mapping is `sign(score) → long/short`. No size scaling on confidence. No regime gate.
- **Caching is correct** for the AIProviderManager path (`ai_provider.py:79-104`), TTL 300s, keyed by SHA256(model:prompt). Good.
- **Rate limiting is correct** (`ai_provider.py:106-138`) — token bucket at 60 rpm.
- **Cost gating is correct** (`ai_provider.py:286-297`) — daily budget abort.
- **Legacy path** (`sentiment_engine._analyze_with_llm`, `_analyze_with_claude`, `:505-682`) bypasses the AIProviderManager — no rate limit, no cache, no cost gate — and is the fallback when `ai_provider_manager` fails to init. Two parallel cost models. Tag `AI-Q-05`.
- **Cooldown**: 1h per symbol after closing (`:251-252, :756-764`). OK.
- **TP/SL fixed at +5% / −3% on ETH spot at $50 trade size** (`:240-242`). No vol-adjustment; in a 4% daily ATR environment, SL is hit on noise.

### 2.3 `AIMarketAnalyzer` and `AIStrategyGenerator` (`ai_trading_engine.py`)
**Claimed edge.** LLM analyzes BTC dominance / fear-greed / news → produces a `MarketCondition` and a JSON strategy with entry/exit conditions and risk params; `evolve_strategy` improves it from perf data.

**Reality.**
- This is **agentic LLM-as-portfolio-manager**. There is no quant content. The "strategy" is a JSON of natural-language conditions (`:381-388`) — there is no executor that parses "RSI > 30 AND volume > 2x AND social sentiment > 0.5" into trades. Either (a) there's a downstream interpreter I haven't found, or (b) the generator stores strategies that nothing executes. Confirm with PM. Tag `AI-Q-06`.
- `_extract_json` (`:327-344`) silently returns `{}` on parse failures; default risk params take over (`:418-424`), so a *broken* LLM response yields a default-aggressive 5% position size.
- `analyze_market` cache TTL = 300s (`:178-187`) is too long for the volatility/Fear-Greed signal it claims to produce, but too short for cost-control. Decoupling cache TTL by output-field is needed.
- **`tokens_to_buy: ['<token1>', ...]` parsed from LLM output and stored in `MarketCondition`** (`:233`). Nothing in the rest of the code validates that these tokens *exist* on chain or have liquidity. An LLM hallucination here could route capital into a typo token. There is no `is_token_valid` check.

### 2.4 `EnsembleModel` / `EnsemblePredictor` (`ml/models/ensemble_model.py`)
**Claimed edge.** 9-model ensemble (XGB-rug, XGB-pump, LGB-rug, LGB-pump, RF, GB, LSTM, Transformer, IsolationForest) weighted to produce `pump_probability`, `rug_probability`, `expected_return`.

**Reality.**
- **Weights are static** (`:191-201`), hand-set, no learned stacking, no regime-conditional weights. The rubric asked: "stacking, boosted residuals, or per-regime weights?" — answer: **plain average with hand-set weights**. This is fine as a v0 but caps Sharpe.
- **TimeSeriesSplit is imported but never used** (`:21`). All training/validation uses `train_test_split` (`auto_trainer.py:259-263`) with `stratify=y`, which is **forbidden for time-series**. This is the canonical source of look-ahead leakage. Tag `AI-Q-LEAK-01`.
- **The `predict` method downloads token data inline** (`ensemble_model.py:560-565`). On the trading hot path. Synchronous IO inside an "async" wrapper that doesn't actually parallelize.
- **`RobustScaler` is initialized but only loaded from disk if `scaler.pkl` exists** (`:280-283`). If absent, `scaler.transform(features)` on line 616 raises `NotFittedError`. Caught by the outer try/except (line 582 → returns safe defaults). Net: when scaler is missing, every prediction is the default 0.5/0.5 — fail-safe, but silently.
- **`_predict_from_features` casts torch.FloatTensor of shape (1, F)** then `.unsqueeze(0)` → (1, 1, F), passes to LSTM expecting (batch, seq=20, features) (`:661-665`). The LSTM was trained on sequences of length 20; at inference it receives sequence of length 1. **Output is meaningless** (the LSTM will run, but the hidden state is unrolled over a single timestep — no temporal information). Tag `AI-Q-LSTM-INFER`.

### 2.5 `PumpPredictor` (`ml/models/pump_predictor.py`)
**Claimed edge.** LSTM + XGB + LGB + RF + GB ensemble regressing pump probability over a 20-bar lookback.

**Reality.**
- **Label is "future_price > current_price * 1.10"** (`:184-188`) — i.e., **the next-bar price moves +10%**. This is the **target leakage classic**: the *prepare_sequences* function builds y[i] from `price_data.iloc[i]` while X[i] uses `price_data[i-lookback:i]` — that's fine (X ends at i-1, y is at i). **But the `_calculate_technical_indicators` features include data through `data.iloc[-1]`** and are then used as the inference vector. If the feature window includes the same bar as the label, that's leakage. Need to inspect the inference call-site to confirm — looks alignment-safe as written, but the convention is fragile.
- **`fit_transform` on the price scaler inside `prepare_sequences`** (`:176`): scaler is re-fit each call. If called repeatedly, it never learns a stable scale. If called at training-time on the full dataset then re-called at inference with one bar, you get the same single-row bug pattern from § 3.1. Tag `AI-Q-SCALER-02`.
- **TensorFlow LSTM mixed with PyTorch LSTM elsewhere** (`pump_predictor.py:19` uses Keras LSTM; `ensemble_model.py:54-101` uses torch.nn.LSTM). Two ML stacks shipped in one inference path = double the surface, double the version-pinning headache. Tag `AI-Q-FRAMEWORK`.

### 2.6 `auto_trainer.AutoMLTrainer` (`ml/training/auto_trainer.py`)
**Claimed edge.** Periodic retrain on real trades; falls back to balanced synthetic.

**Reality.**
- **No time-series CV.** `train_test_split(stratify=y)` (`:261-263`). For a strategy trained on closed trades **ordered in time**, random shuffle = look-ahead in the most literal sense: the test set contains trades from *before* training trades. Tag `AI-Q-LEAK-01`.
- **Synthetic-data fallback `_generate_synthetic_data`** (`:163-209`) generates a perfectly separable 50/50 dataset where winning trades have higher pump_prob, volume, liquidity by design. Any model will hit ~95% accuracy on this. The "Training Complete! Accuracy: 0.95" in the dashboard means **nothing** when fallback was used. Tag `AI-Q-SYNTH`.
- **Scaler persisted but not versioned** (`:237-241`): `feature_scaler.joblib` is overwritten each run. **No model_version field in DB**. If a deploy mid-cycle swaps the scaler under a live ai_strategy, predictions will silently shift.
- **Models saved with predictable names but no version tag** (`:365, :420, :460`). Atomic swap is impossible — there is no `current_model_path` pointer in `config_settings`. The rubric asks: "is model file path versioned and stored in DB so retrain → swap is atomic?" Answer: **No.**
- **No drift detector.** No PSI, no KS test, no feature-distribution monitor. Tag `AI-Q-DRIFT` — P0.
- **`min_trades_for_training=100`** is fine as a guardrail but never enforced — when too few trades, the code "still trains on synthetic data for testing" (`:252-254`). Production should refuse-to-deploy in that case, not blindly train + save.
- **Triggers**: only manual / cron. No drift-triggered retrain, no walk-forward fold rotation.

### 2.7 `RugClassifier`, `VolumeValidator`, `analysis/rug_detector.py`, `analysis/pump_predictor.py`
Not deep-read in this session; tagged in backlog for follow-up audit. The `RugClassifier` interface is invoked by `AIStrategy._check_rug_probability` (`:303-337`) and gates entries at `rug_prob ≤ 0.2`. If the classifier is untrained (likely — see § 2.4), this gate is wide open (random output around 0.5 would block most entries — fail-safe). Confirm.

### 2.8 `token_scorer.py`
**Claim.** Multi-factor 0-100 composite score with category weights, risk score, opportunity score, grade A+..F.

**Reality.**
- **Weights are hand-set** (`:42-50`): liquidity 20%, volume 15%, holder-dist 15%, dev 10%, contract 15%, price 10%, sentiment 5%, market 5%, innovation 5%. No calibration against realized P&L. The rubric asks: "Weights handcrafted or learned? Calibrated against realized P&L?" — **Handcrafted, uncalibrated.**
- Grade thresholds 90/80/70/60/50 (`:79-86`) — arbitrary. There's no histogram in the codebase showing where tokens land or what win rate each grade has.
- Risk-factors dict on lines 89-95 sums to 1.0 — clean. But again, **no learned weights**.

### 2.9 Hyperparameter / RL (`ml/optimization/*`)
Not deep-read; flagged for follow-up. The presence of `reinforcement.py` (334 LOC) and `hyperparameter.py` (323 LOC) suggests an Optuna+RL stack was started but never wired into `auto_trainer.py` — the auto-trainer uses fixed hyperparameters. Tag `AI-Q-HPO-WIRE`.

---

## 3. Bias inventory

| Bias | Where | Mechanism | Severity |
|---|---|---|---|
| **Scaler-fit-on-live-row** | `ai_strategy.py:292-295` | First inference fits StandardScaler on `features.reshape(1, -1)` (1 row). `mean = x`, `std = 0`. Subsequent `transform` divides by 0 → NaN or inf. | **CRITICAL** |
| **Train/test split for time-series via `train_test_split(stratify=y)`** | `auto_trainer.py:259-263` | Random shuffle reorders trades; future trades end up in training fold and past trades in test fold. Classic look-ahead leakage. | **CRITICAL** |
| **Synthetic data fabricated to be separable** | `auto_trainer.py:163-209` | Generated wins differ from losses in feature distributions by construction. Model gets 95% acc, false sense of edge. | High |
| **Scaler not version-pinned to model** | `auto_trainer.py:237-241, ai_strategy.py:50` | Scaler can be retrained independently of the models that use it. Distribution mismatch silent. | High |
| **LSTM inference with seq_len=1 when trained on seq_len=20** | `ensemble_model.py:661-665` | Untrained inference shape. Output is noise. | High |
| **No drift detection** | global | No PSI/KS/feature monitor anywhere | High (P0 per rubric) |
| **LLM-as-signal without calibration** | `sentiment_engine.py:438-469` | LLM float → binary trade. No backtest of "LLM score X → realized 24h ETH return". | High |
| **Hand-set ensemble weights** | `ensemble_model.py:191-201`, `token_scorer.py:42-50` | No learned stacking; fragile to changes in component model quality | Medium |
| **Inference-time HTTP call inside `EnsembleModel.predict`** | `ensemble_model.py:560-565` | Adds latency + failure mode + non-determinism (different DexScreener cache → different features) | Medium |
| **MACD signal-line shortcut** (same as futures) | Likely mirrored if ensemble's tech features come from same util | Need to verify; tag follow-up | Medium |
| **Survivorship** | `auto_trainer.py:108-113` | Query is `WHERE status='closed'` — open positions are dropped, OK. But **failed-to-execute** trades (rejected, slippage-killed) are never seen by trainer. Models learn from trades that *got filled*, biasing toward easy fills. | Medium |
| **In-sample tuning of thresholds** | `ai_strategy.py:64-66` (`max_rug_probability=0.2`, `min_pump_probability=0.6`, `min_pattern_score=0.7`) | Hand-picked. No grid-search log. | Medium |
| **p-hacking via LLM evolution** | `ai_trading_engine.py:453-500` | `evolve_strategy` sees perf and asks LLM to "improve" → produces a *post-hoc* rule. Every iteration overfits. | High |

---

## 4. Feature audit

| Feature | Source | Informativeness | Issue |
|---|---|---|---|
| `pump_probability` (as a *feature* of `prepare_features`) | `auto_trainer.py:214` | **Self-prediction loop**: the trainer learns to predict `is_win` from `pump_probability` which was emitted by the *previous* version of the pump model. Predicting your own past output. | **Leakage** |
| `volume_24h`, `liquidity` | DexScreener via `EnsembleModel.predict` | OK but extracted at *inference* time | Slow |
| `volatility` | Market data | OK | OK |
| `holder_count` | Chain data | High informativeness for rug, OK | OK |
| `social_score`, `sentiment_score` | LLM-derived | Risky as *features* if LLM is also the *signal* | Double-counting |
| `pattern_strength` (from `core/pattern_analyzer.py`) | Hand-coded patterns | Low to medium | Verify in pattern_analyzer audit |
| `bid_ask_spread` from Roll estimator (`ai_strategy.py:_estimate_spread`) | Price changes | OK as a feature; needs ≥2 changes | OK |
| `hurst_exponent` (`:680-704`) | log-log of stddev over lags | OK if `len(prices) ≥ 100`; cheaply approximated | OK |
| `order_imbalance = (sum(volumes[::2]) - sum(volumes[1::2])) / total` (`:724-738`) | **Garbage** — assumes alternating buy/sell. There is no buy-vs-sell metadata used. | None | **Fix or drop** |

---

## 5. Model audit

- **Training data origin:** `trades` table (`auto_trainer.py:95-113`), 30-day lookback. Metadata column parsed for features (`:124-153`). Falls back to synthetic if empty.
- **Labels:** `is_win = profit_loss > 0` (`:142`). Binary, ignores P&L magnitude. A "1 cent profit after fees" is treated as equivalent to a 10x. Tag `AI-Q-LABEL`.
- **CV scheme:** `train_test_split(test_size=0.2, stratify=y)`. **Wrong for time-series** (see § 3).
- **Calibration:** None. `predict_proba` outputs are used directly with a 0.4 threshold (`:352, :408, :448`) — uncalibrated probabilities. No Platt scaling, no isotonic regression.
- **Drift detection:** None.
- **Version pinning:** None — models saved with constant filenames, no `model_version` in DB.

---

## 6. Profitability levers (ranked)

| Rank | Lever | Expected ROI | Effort | Tag |
|---|---|---|---|---|
| 1 | **Fix `_extract_features` scaler** — load persisted scaler at init; refuse to predict if scaler not loaded; never `fit_transform` on a single row | Eliminates NaN-poisoned signals; the upper bound on improvement is "from random to whatever the real model would produce" | XS | `AI-Q-01` |
| 2 | **Replace `train_test_split` with `TimeSeriesSplit(n_splits=5)`** in `auto_trainer.py` and require 100+ real trades before saving | Removes lookahead; metrics become real | S | `AI-Q-02` |
| 3 | **Add PSI/KS drift detector** on top-5 features daily; alert + auto-disable strategy if drift > threshold | Catches silent edge decay | M | `AI-Q-03` |
| 4 | **Version model files** (`pump_predictor_<timestamp>.pkl`) and store `current_model_version` in `config_settings`; atomic swap | Safe retrains | S | `AI-Q-04` |
| 5 | **Drop LLM-direct-trading**; use LLM output as a *feature* into the calibrated ML stack, not as a primary signal | Eliminates the cluster of LLM-related risks | M | `AI-Q-05` |
| 6 | **Calibrate model probabilities** (sklearn `CalibratedClassifierCV` with isotonic) and pick threshold via expected-utility max under realized fee schedule | Improves precision-at-threshold by ~15-25% historically | S | `AI-Q-06` |
| 7 | **Fix LSTM inference**: rolling window of last 20 bars per symbol, fed as `(1, 20, F)` not `(1, 1, F)` | Restores the LSTM's actual signal | S | `AI-Q-07` |
| 8 | **Move EnsembleModel feature fetch out of `predict()`** — accept pre-fetched features dict, fetcher is the caller's job | Cuts inference latency 200-500ms | S | `AI-Q-08` |
| 9 | **Learn `token_scorer` weights** from realized 24h PnL via `LogisticRegression` on the same categories | Replaces vibes with regression | M | `AI-Q-09` |
| 10 | **Wire `optimization/hyperparameter.py` (Optuna) into `auto_trainer`** under nested TimeSeriesSplit | One-shot edge boost; rare in crypto where market shifts faster than the search | M | `AI-Q-10` |
| 11 | **Replace `evolve_strategy` LLM-improvement with rule-by-rule walk-forward A/B** | Stops p-hacking via LLM | L | `AI-Q-11` |
| 12 | **Drop `auto_trainer._generate_synthetic_data` from production path** — fail loudly if no real trades; allow only via `--synthetic` flag for testing | Stops phantom-edge metrics | XS | `AI-Q-12` |

---

## 7. Live-trading hygiene

- **Model staleness.** No tracked `model_trained_at`. `ai_strategy._check_retrain` (`:884-900`) triggers if `≥50` outcomes accumulated and 24h elapsed — but `_retrain_models` is a no-op stub. **Models can be days/weeks old with no warning.**
- **Retrain cadence.** Manual (`python -m ml.training.auto_trainer --train`) or cron (suggested in docstring `:14-15`). Not driven by drift, not driven by perf decay.
- **Signal latency.** `SentimentEngine` runs every 15 min (`:476`). Latency of LLM call: ~1-3s. On 15m cadence this is fine.
- **Scaler-fit-on-live-row bug.** **Confirmed present in `ai_strategy.py:292-295` and re-occurring in `pump_predictor.py:176`.** Both must be fixed.
- **LLM caching.** Correct via `ResponseCache` (`ai_provider.py:79-103`) at 5-min TTL keyed by `sha256(model:prompt)`. Note: prompt includes timestamped market data, so cache hit rate will be low for sentiment but high for token-analysis prompts that repeat.
- **LLM rate limiting.** Correct (`ai_provider.py:106-138`).
- **LLM cost gate.** Correct (`ai_provider.py:286-297`). Daily budget defaults to $10 (`:160`), configurable from DB.
- **LLM outputs as features vs signals.** Currently used as **signals** (`sentiment_engine` ETH trade, `ai_trading_engine.analyze_token` direct recommendation). Must be reclassified as features into a calibrated model. **This is the highest live-trading risk in the AI module.**

---

## 8. Proposed action backlog

| ID | Action | Files | Effort |
|---|---|---|---|
| AI-Q-01 | Replace `_extract_features` scaler logic: load `feature_scaler.joblib` at `initialize()`; raise on missing; remove the `fit_transform` fallback | `trading/strategies/ai_strategy.py:50, 247-301` | XS |
| AI-Q-02 | Switch `auto_trainer` CV to `TimeSeriesSplit(n_splits=5)`; refuse to save if test-fold AUC < min threshold | `ml/training/auto_trainer.py:258-298` | S |
| AI-Q-03 | Add `ml/monitoring/drift_detector.py` — PSI on top features, daily job, dashboard alert | new file | M |
| AI-Q-04 | Versioned model paths + DB-stored `current_pump_model_path` etc.; atomic swap via single DB write | `ml/training/auto_trainer.py:300-462`, `trading/strategies/ai_strategy.py:928-965` | S |
| AI-Q-05 | Gate `SentimentEngine._execute_trade` behind `if calibrated_model_prob >= dynamic_threshold`; LLM output becomes one feature in that model | `modules/ai_analysis/core/sentiment_engine.py:438-790` | M |
| AI-Q-06 | Wrap each booster in `CalibratedClassifierCV(method='isotonic', cv='prefit')` after training | `ml/training/auto_trainer.py:312-464` | S |
| AI-Q-07 | Build per-symbol rolling 20-bar feature buffer; feed LSTM `(1, 20, F)`; cache in memory | `ml/models/ensemble_model.py:660-665, ml/models/pump_predictor.py` | S |
| AI-Q-08 | Refactor `EnsembleModel.predict(token, chain)` to `predict(features_dict)`; caller fetches data | `ml/models/ensemble_model.py:521-597` | S |
| AI-Q-09 | Fit logistic regression of `is_win` on category-score vector across last 90d trades; learn `ScoringWeights` from coefficients (renormalized to sum=1) | `analysis/token_scorer.py:42-50, 101-184` | M |
| AI-Q-10 | Wire `optimization/hyperparameter.py` Optuna into `_train_xgboost / _train_lightgbm` under nested CV | `ml/training/auto_trainer.py, ml/optimization/hyperparameter.py` | M |
| AI-Q-11 | Disable `AIStrategyGenerator.evolve_strategy` until a walk-forward A/B harness is built | `modules/ai_analysis/core/ai_trading_engine.py:453-500` | XS |
| AI-Q-12 | Refuse to save model when `len(real_trades) < min_trades_for_training`; remove silent synthetic-fallback save | `ml/training/auto_trainer.py:251-254` | XS |
| AI-Q-13 | Fix `_calculate_order_imbalance` (`ai_strategy.py:724-738`) — drop or replace with real buy/sell-volume split from data collector | `trading/strategies/ai_strategy.py` | S |
| AI-Q-14 | Remove `pump_probability` as a *feature* of `prepare_features` (`auto_trainer.py:214`) — self-prediction leakage | `ml/training/auto_trainer.py:211-243` | XS |
| AI-Q-15 | Add `is_token_valid` validation to LLM-extracted `tokens_to_buy / tokens_to_sell` before any allocator uses them | `modules/ai_analysis/core/ai_trading_engine.py:224-249` | S |
| AI-Q-16 | Label = `pnl / fees` (R-multiple) regression target instead of binary `is_win` | `ml/training/auto_trainer.py:142` | S |
| AI-Q-17 | Persist `model_trained_at` in DB; `ai_strategy.analyze` refuses to emit signals if `age > 7d` | DB schema + `ai_strategy.py` | S |
| AI-Q-18 | Audit `RugClassifier`, `analysis/rug_detector.py`, `analysis/pump_predictor.py`, `ml/optimization/{hyperparameter,reinforcement}.py` (not deep-read this session) | follow-up | M |

---

## 9. Open questions for PM / analyst / backend

1. Is `AIStrategy` (`trading/strategies/ai_strategy.py`) actually invoked in any live path, or only by the AI module's `SentimentEngine` headline loop? The two flows seem disjoint.
2. Does `AIStrategyGenerator`'s output strategy JSON get *executed* anywhere, or stored only? If executed: where is the JSON→trade interpreter?
3. Is the `trades` table referenced by `auto_trainer.fetch_training_data` actually populated with DEX/futures trades? The futures module uses `futures_trades`, not `trades`. There may be a schema mismatch silently routing the trainer to the synthetic path.
4. Confirm with backend agent: are model files stored on the filesystem persisted across container redeploys? If not, every restart wipes models and the cold-start fallbacks (`ai_strategy._load_models:952-961`) tighten thresholds — strategy effectively disables.
5. Is `pattern_analyzer.py` (`core/pattern_analyzer.py`) tested anywhere? `AIStrategy._get_ml_predictions` calls it; its output reliability drives the `min_pattern_score = 0.7` gate.
6. What is the design intent for `ai_trading_engine.AIStrategyGenerator` — is it experimental and behind a flag, or part of the live recommendation path?

---

End of report — `AI_quant.md`.
