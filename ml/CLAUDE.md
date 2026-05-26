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
and persists `feature_names.json` in that same canonical order. This makes
train-time and inference-time feature order match *by construction*. If you
add/remove/reorder a feature in `extract_features`, update
`ENSEMBLE_FEATURE_NAMES` in lock-step and retrain (old artifacts are invalid).

If a future training source supplies columns that don't include all 82
canonical names, `retrain()` fills the gaps with `0.0` and prints which
features were missing — review that warning; large gaps mean the model is
under-fed even if it activates.

## KNOWN GAP — `scripts/retrain_models.py` does NOT train the ensemble

As of Wave-9, `scripts/retrain_models.py` trains three *separate* models
(`PumpPredictor`, `RugClassifier`, `VolumeValidatorML`) and writes them to
`ml/models/` with names like `latest` / `volume_validator_latest.pkl`. It
**never** constructs `EnsemblePredictor` and **never** writes the artifacts the
engine reads (`models/scaler.pkl`, `models/xgboost_rug.pkl`,
`models/feature_names.json`, ...). Two consequences:

- **Path mismatch:** retrain → `ml/models/`; ensemble reads → `models/`.
- **Model mismatch:** running today's `retrain_models.py` does NOT flip
  `ml_source` off `heuristic_fallback`.

Closing this requires a DB→canonical-82-feature pipeline (the
`ai_feature_store` rows written by `AIStrategy` are a *different* shape —
`scaler_v1`/`raw_v1` — not the ensemble's `extract_features` layout). That is a
data-pipeline task, intentionally NOT done in Wave-9 (no DB/data here). Until
it is built, the ensemble stays in fallback. Treat this as the open item that
gates ensemble activation.

## How to activate the ensemble on the VPS (when the pipeline exists)

1. Produce a labeled training DataFrame whose columns are (a superset of)
   `ENSEMBLE_FEATURE_NAMES`, plus label columns `pump_label`, `rug_label`
   (0/1), and optionally `returns`.
2. Train + persist to the path the engine reads:
   ```python
   from ml.models.ensemble_model import EnsemblePredictor
   p = EnsemblePredictor({"model_dir": "models/"})  # MUST match engine default
   new_models = await p.retrain(training_df)          # reindexes to canonical order
   await p.update_models(new_models)                  # calls save_models()
   ```
   `save_models()` writes `models/{xgboost_rug,xgboost_pump,lightgbm_rug,
   lightgbm_pump,random_forest,gradient_boosting,isolation_forest}.pkl`,
   `models/scaler.pkl`, and `models/feature_names.json`.
3. Restart the DEX module (or any process holding the `TradingBotEngine`) so
   `load_models()` re-reads `models/`.
4. **Confirm activation:** watch the DEX log for the per-opportunity line
   `🤖 ML[ensemble] conf=... pump=... rug=...`. When it reads `ML[ensemble]`
   (not `ML[heuristic_fallback]`), the trained ensemble is live. The same
   provenance is persisted on every opportunity at
   `metadata.ml_source == 'ensemble'`.
5. Keep DRY_RUN = true. Activating the ensemble must not be paired with
   enabling LIVE; validate the signal in DRY_RUN first.

## Model versioning

Per project rule, any trainer that writes ensemble artifacts must stamp a model
version in DB. `save_models()` writes `models/feature_names.json` (the schema
fingerprint). When the DB-backed ensemble retrain entrypoint is built, it must
also record a `model_version` row (e.g. in `config_settings` or a
`model_registry` table) alongside the artifact write so the dashboard can show
which version is live. This is part of the open KNOWN-GAP item above.
