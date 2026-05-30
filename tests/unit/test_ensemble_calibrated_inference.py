"""AI-Q-05 — EnsemblePredictor inference-side calibrated booster wrap.

Verifies that:
  * `ai_calibrated_predictions_enabled` is OFF by default (operator must
    flip after first calibration sample).
  * `_load_calibrated_models` accepts both the deliverable filename
    (`calibrated_<name>.pkl`) and the AI-Q-06 trainer-side filename
    (`<name>_calibrated.{pkl,joblib}`).
  * `calibrated_predict_proba` returns the calibrated probability when
    wrapper is loaded AND flag is on; falls back to raw model otherwise.
  * `fit_and_persist_calibration` writes per-model `.pkl` sidecars that
    round-trip through `joblib.load` and produce 2-column probabilities.
  * Inference path is numerically identical to the legacy code when the
    flag is off (regression guard).
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
joblib = pytest.importorskip("joblib")
pytest.importorskip("sklearn")
pytest.importorskip("torch")

from ml.models.ensemble_model import EnsemblePredictor  # noqa: E402


def _make_dataset(n: int = 400, n_features: int = 95, seed: int = 11):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    # Inject a real signal so the classifier produces non-trivial probas.
    y = ((X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, n)) > 0).astype(int)
    # Walk-forward split: train < calib < held-out test.
    a, b = int(0.5 * n), int(0.75 * n)
    return X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:]


def _fit_minimal_predictor(tmp_dir: str) -> EnsemblePredictor:
    """Build an EnsemblePredictor with two fitted tree models + scaler."""
    from sklearn.linear_model import LogisticRegression
    p = EnsemblePredictor(config={'model_dir': tmp_dir})
    X_tr, y_tr, *_ = _make_dataset()
    # We don't actually need xgboost installed for the unit test —
    # any sklearn classifier with predict_proba satisfies the contract.
    p.scaler.fit(X_tr)
    base = LogisticRegression(max_iter=200)
    base.fit(p.scaler.transform(X_tr), y_tr)
    p.models['xgboost_pump'] = base
    p.models['random_forest'] = base
    return p


def test_calibration_off_by_default():
    """Operator MUST flip the flag after first calibration sample."""
    p = EnsemblePredictor(config={})
    assert p.calibrated_predictions_enabled is False
    assert p.calibrated_models == {}


def test_fit_and_persist_writes_pkl_sidecars():
    """`fit_and_persist_calibration` writes `calibrated_<name>.pkl`."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _fit_minimal_predictor(tmp)
        _, _, X_cal, y_cal, _, _ = _make_dataset()
        saved = p.fit_and_persist_calibration(X_cal, y_cal)
        # Two base models were loaded -> two sidecars persisted.
        assert set(saved) == {'xgboost_pump', 'random_forest'}
        for name in saved:
            path = Path(tmp) / f"calibrated_{name}.pkl"
            assert path.exists()
            loaded = joblib.load(path)
            proba = loaded.predict_proba(p.scaler.transform(X_cal))
            assert proba.shape == (len(X_cal), 2)
            assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_load_accepts_both_naming_patterns():
    """Loader must read `calibrated_<name>.pkl` AND `<name>_calibrated.joblib`."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    with tempfile.TemporaryDirectory() as tmp:
        X_tr, y_tr, X_cal, y_cal, _, _ = _make_dataset()
        base = LogisticRegression(max_iter=200).fit(X_tr, y_tr)
        wrapper = CalibratedClassifierCV(base, method='sigmoid', cv='prefit')
        wrapper.fit(X_cal, y_cal)
        # AI-Q-05 deliverable naming.
        joblib.dump(wrapper, Path(tmp) / 'calibrated_xgboost_pump.pkl')
        # AI-Q-06 trainer naming (cross-compat with auto_trainer.py).
        joblib.dump(wrapper, Path(tmp) / 'random_forest_calibrated.joblib')

        p = EnsemblePredictor(config={
            'model_dir': tmp,
            'ai_calibrated_predictions_enabled': True,
        })
        p._load_calibrated_models()
        assert 'xgboost_pump' in p.calibrated_models
        assert 'random_forest' in p.calibrated_models


def test_calibrated_predict_proba_uses_wrapper_when_enabled():
    """When flag on + wrapper loaded, helper returns calibrator's value."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _fit_minimal_predictor(tmp)
        _, _, X_cal, y_cal, X_te, _ = _make_dataset()
        p.fit_and_persist_calibration(X_cal, y_cal)
        # Enable the flag.
        p.calibrated_predictions_enabled = True

        sample = p.scaler.transform(X_te[:1])
        cal_proba = p.calibrated_predict_proba('xgboost_pump', sample)
        # Reference: run wrapper directly.
        ref = float(
            p.calibrated_models['xgboost_pump'].predict_proba(sample)[0][1]
        )
        assert abs(cal_proba - ref) < 1e-9


def test_calibrated_predict_proba_falls_back_when_disabled():
    """Flag OFF -> raw model output (regression guard for legacy callers)."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _fit_minimal_predictor(tmp)
        _, _, X_cal, y_cal, X_te, _ = _make_dataset()
        p.fit_and_persist_calibration(X_cal, y_cal)
        # Flag stays off — operator hasn't flipped it.
        assert p.calibrated_predictions_enabled is False

        sample = p.scaler.transform(X_te[:1])
        helper_val = p.calibrated_predict_proba('xgboost_pump', sample)
        raw_val = float(p.models['xgboost_pump'].predict_proba(sample)[0][1])
        assert abs(helper_val - raw_val) < 1e-9


def test_calibrated_predict_proba_neutral_when_missing():
    """No model + no wrapper -> 0.5 (legacy behavior preserved)."""
    p = EnsemblePredictor(config={})
    sample = np.zeros((1, 95), dtype=np.float32)
    assert p.calibrated_predict_proba('xgboost_pump', sample) == 0.5
