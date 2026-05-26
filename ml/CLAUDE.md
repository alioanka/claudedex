# ml/ — Models, feature contract, and retraining (operator guide)

This module owns the offline ML training/inference for the DEX entry path.
The live (DRY_RUN) consumer is `core/engine.py::TradingBotEngine`, which calls
`EnsemblePredictor.predict_decoupled()` via `_ml_predict_opportunity()`.

## Current activation state (read this first)

The ensemble runs in **honest heuristic fallback** until trained artifacts
exist on disk. With no artifacts:

1. `EnsemblePredictor()` is built with the default `model_dir="models/"`
   (repo-root relative — NOT `ml/models/`).
2. `load_models()` finds no `scaler.pkl`, so `self.scaler` stays an unfitted
   `RobustScaler`.
3. At inference, `_predict_from_features()` calls `self.scaler.transform(...)`
   which raises `NotFittedError`. `predict_decoupled()` catches it and returns
   the degenerate `{pump=0.5, rug=0.5, confidence=0.1, error=...}`.
4. `engine._is_trustworthy_ml_result()` rejects that (error key + neutral 0.5)
   and the engine uses the heuristic fallback, labelled
   `metadata.ml_source = 'heuristic_fallback'`.

**Fail-soft is confirmed working** — a missing/unfitted artifact never crashes
the path and never fabricates a confident ML signal.

## The feature contract (critical — train must match inference)

Inference builds a **fixed-order 82-element** feature vector via
`EnsemblePredictor.extract_features(data_dict)`. The canonical column order is
the module constant `ENSEMBLE_FEATURE_NAMES` in `ml/models/ensemble_model.py`,
which mirrors `extract_features` field-for-field.

`EnsemblePredictor.retrain(training_data)` now **reindexes the training
DataFrame to `ENSEMBLE_FEATURE_NAMES` before fitting** the scaler and models,
and persists the feature list (as `models/features.json`) in that same
canonical order. This makes
train-time and inference-time feature order match *by construction*. If you
add/remove/reorder a feature in `extract_features`, update
`ENSEMBLE_FEATURE_NAMES` in lock-step and retrain (old artifacts are invalid).

If a future training source supplies columns that don't include all 82
canonical names, `retrain()` fills the gaps with `0.0` and prints which
features were missing — review that warning; large gaps mean the model is
under-fed even if it activates.

## GAP CLOSED — `scripts/train_ensemble.py` trains + activates the ensemble

The former gap (no DB→canonical-82 pipeline; `retrain_models.py` wrote 3
*separate* legacy models to `ml/models/`, never built `EnsemblePredictor`, never
wrote the `models/` artifacts the engine reads → could not flip `ml_source` off
`heuristic_fallback`) is **CLOSED** by `scripts/train_ensemble.py` (Wave-10).

`scripts/retrain_models.py` is **superseded for the ensemble** (it still trains
the legacy `PumpPredictor`/`RugClassifier`/`VolumeValidatorML` to `ml/models/` —
not deleted, out of scope — but it does NOT produce the ensemble artifacts and
does NOT activate the live DEX ML path). Use `train_ensemble.py` for the
ensemble.

### Feature parity (train == inference, by construction)

The trainer does NOT copy the feature mapping. The live inference path and the
trainer BOTH call the shared pure function
`ml.feature_builder.build_ml_feature_dict(pair, risk_score, patterns)` →
`EnsemblePredictor.extract_features` → the canonical 82-vector. `core/engine.py`
`_build_ml_feature_dict` is now a thin wrapper around that shared function
(byte-identical output — verified). The trainer wraps each stored
`trades.metadata.risk_score` *dict* in a `SimpleNamespace` so the same
`getattr`/`hasattr` branch runs. No skew possible.

### Training source + labels

Source = `trades` rows with `status='closed'` AND `metadata NOT NULL`, filtered
by `created_at >= now - --days`. Labels derived per row:

- `rug_label = 1` if `profit_loss_percentage <= -50` OR
  `metadata.close_reason == 'stop_loss_rapid'`, else 0.
- `pump_label = 1` if `profit_loss_percentage >= 20`, else 0.
- `returns = profit_loss_percentage / 100.0`.

These match `EnsemblePredictor.retrain()`'s expected columns (`retrain` drops
`['label','pump_label','rug_label','returns']` from the feature set and reads
`pump_label`/`rug_label`/`returns`). Class imbalance is handled in the CV report
via `scale_pos_weight`; `retrain()` itself fits the trees on the 82-vector after
reindexing to canonical order.

## How to activate the ensemble on the VPS

1. **Inspect first (writes nothing):**
   ```bash
   python scripts/train_ensemble.py --dry-run-no-save --days 90
   ```
   Prints rows used, class balance, and StratifiedKFold CV AUC/precision/recall
   per head. No artifacts, no DB write. Use this to sanity-check signal before
   committing artifacts.
2. **Train + persist + version:**
   ```bash
   python scripts/train_ensemble.py --days 90
   ```
   This runs `retrain → update_models → save_models`, writing to repo-root
   `models/` (the engine's default `model_dir`):
   `models/{xgboost_rug,xgboost_pump,lightgbm_rug,lightgbm_pump,random_forest,
   gradient_boosting,isolation_forest}.pkl`, `models/scaler.pkl`, and
   `models/features.json` (82 names, canonical order — what `load_models()`
   reads back). It then UPSERTs ONE row into
   `config_settings(config_type='ml_models', key='ensemble_version')` with JSON
   `{version, trained_at, git_sha, n_samples, pump_pos, rug_pos, cv_metrics,
   source}`. That version row is the script's **only** DB write — it touches no
   `trades`, no `dry_run`, no trading-behaviour config.
3. **Confirm artifacts + version:** `ls models/` shows the 9 `.pkl` files +
   `scaler.pkl` + `features.json`; query
   `SELECT value FROM config_settings WHERE config_type='ml_models' AND
   key='ensemble_version'` to see the live version.
4. **Restart the DEX module** so `load_models()` re-reads `models/`.
5. **Confirm activation:** watch the DEX log for the per-opportunity line
   `🤖 ML[ensemble] conf=... pump=... rug=...`. When it reads `ML[ensemble]`
   (not `ML[heuristic_fallback]`), the trained ensemble is live; the same
   provenance is persisted at `metadata.ml_source == 'ensemble'`.
6. **Keep DRY_RUN = true.** Activating the ensemble must NOT be paired with
   enabling LIVE — validate the signal in DRY_RUN first. The trainer itself is
   offline/analytical and never places trades or alters DRY_RUN.

### Offline self-test (no DB, no VPS)

```bash
python scripts/train_ensemble.py --mock
```
Generates ~400 synthetic metadata rows (production JSONB shape) and runs the
FULL fit+save pipeline (skipping only the DB version write), so the pipeline is
validated end-to-end. Requires the bot image's heavy deps
(xgboost/lightgbm/sklearn/torch/pandas); in a dep-less env it exits non-zero
with a clear "deps live in the bot image" message and writes nothing partial.
Mock CV numbers are near-random by design (synthetic features carry no signal —
mock validates plumbing, not predictive quality).

## Model versioning

`scripts/train_ensemble.py` records the version in
`config_settings(ml_models, ensemble_version)` (JSON) on every successful save,
alongside `models/features.json` (the schema fingerprint), so the dashboard can
surface which ensemble version is live.
