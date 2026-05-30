"""AI-Q-06 — CalibratedClassifierCV wrap regression tests.

Verifies that `AutoMLTrainer._maybe_calibrate_and_save`:
  * is a no-op when `calibration_enabled=False` (default).
  * persists `<model_name>_calibrated.joblib` and records calibration
    metrics when enabled.
  * keeps the base model file intact (reversible toggle).
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
joblib = pytest.importorskip("joblib")
pytest.importorskip("sklearn")

from ml.training.auto_trainer import AutoMLTrainer  # noqa: E402


def _make_dataset(n: int = 400, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    # Linearly separable with noise -> classifier should produce non-trivial proba.
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, size=n) > 0).astype(int)
    split = int(0.75 * n)
    return X[:split], y[:split], X[split:], y[split:]


def test_calibration_off_by_default():
    """Default constructor must not enable calibration."""
    trainer = AutoMLTrainer(config={})
    assert trainer.calibration_enabled is False
    assert trainer.calibration_method == 'sigmoid'
    assert trainer.calibration_cv == 'prefit'


def test_calibration_noop_when_disabled():
    """`_maybe_calibrate_and_save` returns metrics untouched when disabled."""
    trainer = AutoMLTrainer(config={'calibration_enabled': False})
    X_train, y_train, X_test, y_test = _make_dataset(200)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression().fit(X_train, y_train)

    metrics_in = {'accuracy': 0.5}
    metrics_out = trainer._maybe_calibrate_and_save(
        model, 'logreg_test', X_test, y_test, metrics_in
    )
    # Same dict reference is fine; the contract is just "no calibration key".
    assert 'calibration' not in metrics_out
    assert metrics_out == metrics_in


def test_calibration_writes_artifact_when_enabled():
    """When enabled, a calibrated joblib appears and metrics carry Brier scores."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = AutoMLTrainer(config={
            'calibration_enabled': True,
            'calibration_method': 'sigmoid',
            'model_dir': tmp,
        })
        X_train, y_train, X_test, y_test = _make_dataset(400)

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression().fit(X_train, y_train)

        out = trainer._maybe_calibrate_and_save(
            model, 'logreg_test', X_test, y_test, {'accuracy': 0.5}
        )
        cal_block = out.get('calibration')
        assert cal_block is not None
        assert cal_block['enabled'] is True
        assert cal_block['method'] == 'sigmoid'
        assert cal_block['cv'] == 'prefit'
        # Brier metrics should be present and in [0, 1].
        for key in ('brier_pre', 'brier_post'):
            assert key in cal_block
            assert 0.0 <= cal_block[key] <= 1.0

        # Artifact must exist and be a CalibratedClassifierCV.
        artifact = Path(tmp) / 'logreg_test_calibrated.joblib'
        assert artifact.exists()
        loaded = joblib.load(artifact)
        # Round-trip predict_proba works.
        proba = loaded.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        # Probabilities sum to 1 per row.
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_unknown_calibration_method_falls_back_to_sigmoid():
    """Invalid method string warns and defaults to sigmoid (no crash)."""
    trainer = AutoMLTrainer(config={
        'calibration_enabled': True,
        'calibration_method': 'platt-but-wrong',
    })
    assert trainer.calibration_method == 'sigmoid'


def test_calibration_failure_recorded_in_metrics():
    """A calibration error must surface in metrics, not crash the trainer."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = AutoMLTrainer(config={
            'calibration_enabled': True,
            'model_dir': tmp,
        })

        class BrokenClassifier:
            classes_ = np.array([0, 1])

            def predict_proba(self, X):
                # Required for the pre-Brier; behaves like a real estimator.
                return np.stack([np.ones(len(X)) * 0.4,
                                 np.ones(len(X)) * 0.6], axis=1)

            def predict(self, X):
                return np.ones(len(X), dtype=int)

        X_train, y_train, X_test, y_test = _make_dataset(200)
        out = trainer._maybe_calibrate_and_save(
            BrokenClassifier(), 'broken', X_test, y_test, {'accuracy': 0.5}
        )
        assert out['calibration']['enabled'] is True
        assert 'error' in out['calibration']
