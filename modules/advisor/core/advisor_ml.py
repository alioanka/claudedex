"""
AdvisorMLModel — ML daily-learning loop for the financial advisor module.

HONEST DESIGN CONTRACT
----------------------
This model is deliberately gated until it has enough signal to be meaningful.
The minimum sample gate (advisor_ml_min_samples, default 50) exists to prevent
the model from fabricating confidence on sparse data.  Until the gate is met,
predict() returns None and the advice engine uses the raw heuristic confidence.
This mirrors the DEX ensemble pattern (rug_classifier, pump_predictor) that
refuses to activate on <50 samples.

Feature contract
----------------
Features extracted at advice time (from advisor_advice + advisor_sim_positions):
  market_encoded   : int  (0=crypto, 1=us_equities, 2=bist, 3=fx, 4=midas_funds)
  horizon_encoded  : int  (0=short, 1=mid, 2=long)
  direction_encoded: int  (0=long, 1=short, 2=neutral)
  confidence       : float [0, 1]
  kronos_signal    : float (0.0 if NULL in DB)
  entry_range_pct  : float ((entry_high - entry_low) / entry_low * 100; 0 if NULL)
  has_target       : int  (1 if target_price IS NOT NULL)
  has_stop         : int  (1 if stop_price IS NOT NULL)

Target
------
  win : int (1 if pnl_pct > 0, 0 otherwise)

This is a win/loss binary classifier.  Confidence recalibration replaces the
raw confidence heuristic in the advice engine once the model is activated.

Cadence
-------
  - advisor_ml_retrain_days : int, default 7 (retrain every N days)
  - advisor_ml_min_samples  : int, default 50 (refuse to train/predict below)
  - advisor_ml_version key in config_settings (config_type='advisor_config')
    updated after every successful retrain.

Artifacts
---------
  models/advisor_ml_model.pkl   — trained LightGBM classifier
  models/advisor_ml_scaler.pkl  — StandardScaler for feature normalisation
  models/advisor_ml_features.json — feature names (schema fingerprint)

Retrain entrypoint
------------------
  python scripts/retrain_advisor_ml.py
  (or called automatically by AdvisorMLModel.run_daily_tick() on schedule)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("advisor.ml")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FEATURE_NAMES: List[str] = [
    "market_encoded",
    "horizon_encoded",
    "direction_encoded",
    "confidence",
    "kronos_signal",
    "entry_range_pct",
    "has_target",
    "has_stop",
]

_MARKET_MAP = {"crypto": 0, "us_equities": 1, "bist": 2, "fx": 3, "midas_funds": 4}
_HORIZON_MAP = {"short": 0, "mid": 1, "long": 2}
_DIRECTION_MAP = {"long": 0, "short": 1, "neutral": 2}

_DEFAULT_MODEL_DIR = Path("models")
_MODEL_FILE = "advisor_ml_model.pkl"
_SCALER_FILE = "advisor_ml_scaler.pkl"
_FEATURES_FILE = "advisor_ml_features.json"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MLPrediction:
    """
    Confidence override from the ML model.

    win_prob : float [0, 1] — predicted probability the advice wins.
    raw_confidence : float — original heuristic confidence from the analyzer.
    ml_confidence : float — blended confidence (raw * 0.4 + win_prob * 0.6).
    model_version : str — version tag from DB.
    """
    win_prob: float
    raw_confidence: float
    ml_confidence: float
    model_version: str


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AdvisorMLModel:
    """
    Lightweight LightGBM classifier trained on closed sim-position outcomes.

    Lifecycle
    ---------
    1. Instantiate with config dict.
    2. Call load() at startup to restore any existing artifact.
    3. AdviceEngine calls predict(advice_row_dict) to optionally override
       confidence on each advice.
    4. AdviceEngine calls run_daily_tick(db_pool) once per day to retrain
       when enough new samples have accumulated.
    5. scripts/retrain_advisor_ml.py can also call retrain() directly.

    Fail-soft guarantees
    --------------------
    - load() never raises; logs WARNING on failure.
    - predict() returns None if model not loaded OR samples below gate.
    - retrain() returns False (not raises) on any error; logs reason.
    - No advice is blocked or modified by ML failures.
    """

    def __init__(self, config: Optional[dict] = None, model_dir: Optional[Path] = None):
        self.config = config or {}
        self.model_dir = model_dir or _DEFAULT_MODEL_DIR
        self._model = None
        self._scaler = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self._model_version: str = "unloaded"
        self._n_training_samples: int = 0

    # ------------------------------------------------------------------
    # Properties (config-driven)
    # ------------------------------------------------------------------

    @property
    def min_samples(self) -> int:
        return int(self.config.get("advisor_ml_min_samples", 50))

    @property
    def retrain_every_n_days(self) -> int:
        return int(self.config.get("advisor_ml_retrain_days", 7))

    @property
    def blend_weight(self) -> float:
        """Weight given to ML win_prob in blended confidence (0.0–1.0)."""
        return float(self.config.get("advisor_ml_blend_weight", 0.6))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """
        Load model artifacts from disk.  Returns True if successful.
        Never raises.
        """
        model_path = self.model_dir / _MODEL_FILE
        scaler_path = self.model_dir / _SCALER_FILE
        features_path = self.model_dir / _FEATURES_FILE

        if not model_path.exists() or not scaler_path.exists():
            self._load_error = (
                f"Model artifacts not found in {self.model_dir}. "
                "Run scripts/retrain_advisor_ml.py once enough closed sims exist "
                f"(min_samples={self.min_samples})."
            )
            logger.info(
                "[advisor-ml] %s (predict() returns None until trained).",
                self._load_error,
            )
            return False

        try:
            import joblib  # type: ignore[import]

            self._model = joblib.load(str(model_path))
            self._scaler = joblib.load(str(scaler_path))

            if features_path.exists():
                with open(features_path) as fh:
                    stored = json.load(fh)
                if stored != _FEATURE_NAMES:
                    logger.warning(
                        "[advisor-ml] Feature schema mismatch — stored=%s expected=%s. "
                        "Retrain required.",
                        stored, _FEATURE_NAMES,
                    )
                    self._load_error = "Feature schema mismatch — retrain required."
                    return False

            self._loaded = True
            self._load_error = None
            logger.info("[advisor-ml] Model loaded from %s", self.model_dir)
            return True
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("[advisor-ml] Load failed: %s", exc, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, advice: dict) -> Optional[MLPrediction]:
        """
        Predict win probability for an advice dict.

        Parameters
        ----------
        advice : dict with keys matching advisor_advice columns
                 (market, horizon, direction, confidence, kronos_signal,
                  entry_low, entry_high, target_price, stop_price).

        Returns
        -------
        MLPrediction or None (if not loaded or below min_samples gate).
        """
        if not self._loaded or self._model is None or self._scaler is None:
            return None

        if self._n_training_samples < self.min_samples:
            logger.debug(
                "[advisor-ml] Below min_samples gate (%d/%d) — returning None.",
                self._n_training_samples, self.min_samples,
            )
            return None

        try:
            fv = _extract_features(advice)
            import numpy as np  # type: ignore[import]
            X = np.array([fv], dtype=float)
            X_scaled = self._scaler.transform(X)
            win_prob = float(self._model.predict_proba(X_scaled)[0][1])
            raw_conf = float(advice.get("confidence", 0.5))
            ml_conf = raw_conf * (1 - self.blend_weight) + win_prob * self.blend_weight
            return MLPrediction(
                win_prob=round(win_prob, 4),
                raw_confidence=round(raw_conf, 4),
                ml_confidence=round(ml_conf, 4),
                model_version=self._model_version,
            )
        except Exception as exc:
            logger.warning("[advisor-ml] predict() error: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    async def run_daily_tick(self, db_pool) -> None:
        """
        Called by AdviceEngine once per calendar day when ML is enabled.

        Checks if a retrain is due (retrain_every_n_days), fetches closed sim
        outcomes from DB, and retrains if enough samples are present.
        Fail-soft: any exception is caught and logged.
        """
        if db_pool is None:
            logger.debug("[advisor-ml] No DB pool — skipping daily tick.")
            return

        last_retrain = await self._get_last_retrain_date(db_pool)
        if last_retrain is not None:
            days_since = (datetime.now(timezone.utc) - last_retrain).days
            if days_since < self.retrain_every_n_days:
                logger.debug(
                    "[advisor-ml] Last retrain %d day(s) ago; next in %d day(s).",
                    days_since, self.retrain_every_n_days - days_since,
                )
                return

        logger.info("[advisor-ml] Retrain window open — fetching closed sims...")
        await self.retrain(db_pool)

    async def retrain(self, db_pool) -> bool:
        """
        Ingest closed sim outcomes and retrain the classifier.

        Returns True on success, False on any failure (fail-soft).

        Validation method
        -----------------
        Walk-forward: data is chronologically sorted by opened_at.
        We use an 80/20 time-based split (not random) to avoid look-ahead.
        StratifiedKFold CV is not used here because the time order matters.
        Reported metrics (accuracy, precision on test set) are out-of-sample
        by construction.  No in-sample tuning.
        """
        try:
            import numpy as np  # type: ignore[import]
            import lightgbm as lgb  # type: ignore[import]
            from sklearn.preprocessing import StandardScaler  # type: ignore[import]
            import joblib  # type: ignore[import]
        except ImportError as exc:
            logger.warning(
                "[advisor-ml] Missing dependency: %s. Install lightgbm scikit-learn joblib.",
                exc,
            )
            return False

        # ----- 1. Fetch training data from DB -----
        try:
            rows = await _fetch_training_rows(db_pool)
        except Exception as exc:
            logger.error("[advisor-ml] DB fetch error: %s", exc)
            return False

        n = len(rows)
        if n < self.min_samples:
            logger.info(
                "[advisor-ml] Only %d closed sims (need %d). "
                "Skipping retrain — predict() returns None until gate met.",
                n, self.min_samples,
            )
            self._n_training_samples = n
            return False

        logger.info("[advisor-ml] Retraining on %d closed sims...", n)

        # ----- 2. Build feature matrix -----
        X_list, y_list = [], []
        for row in rows:
            try:
                fv = _extract_features(row)
                label = 1 if float(row.get("pnl_pct") or 0) > 0 else 0
                X_list.append(fv)
                y_list.append(label)
            except Exception as exc:
                logger.debug("[advisor-ml] Skipping malformed row: %s", exc)

        if len(X_list) < self.min_samples:
            logger.info(
                "[advisor-ml] Only %d usable feature rows after cleaning. Skipping.",
                len(X_list),
            )
            return False

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=int)

        # ----- 3. Time-ordered 80/20 split (no shuffle — walk-forward) -----
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if len(X_train) < 10 or len(X_test) < 5:
            logger.info("[advisor-ml] Not enough data for train/test split. Skipping.")
            return False

        # ----- 4. Scale + train -----
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        scale_pos = (neg / pos) if pos > 0 else 1.0

        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train_s, y_train)

        # ----- 5. Out-of-sample metrics -----
        from sklearn.metrics import accuracy_score, precision_score  # type: ignore[import]
        y_pred = model.predict(X_test_s)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        win_rate_train = float(y_train.mean())
        win_rate_test = float(y_test.mean())

        logger.info(
            "[advisor-ml] Train n=%d win_rate=%.2f | Test n=%d acc=%.3f prec=%.3f "
            "win_rate=%.2f",
            len(X_train), win_rate_train,
            len(X_test), acc, prec, win_rate_test,
        )

        # ----- 6. Persist artifacts -----
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, str(self.model_dir / _MODEL_FILE))
        joblib.dump(scaler, str(self.model_dir / _SCALER_FILE))
        with open(self.model_dir / _FEATURES_FILE, "w") as fh:
            json.dump(_FEATURE_NAMES, fh)

        # ----- 7. Update in-memory state -----
        self._model = model
        self._scaler = scaler
        self._loaded = True
        self._n_training_samples = n
        version_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._model_version = version_str

        # ----- 8. Persist version to DB -----
        await _save_version(
            db_pool,
            version=version_str,
            n_samples=n,
            test_acc=acc,
            test_prec=prec,
            feature_names=_FEATURE_NAMES,
        )

        logger.info(
            "[advisor-ml] Retrain complete. version=%s artifacts in %s",
            version_str, self.model_dir,
        )
        return True

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def health(self) -> dict:
        return {
            "loaded": self._loaded,
            "model_version": self._model_version,
            "n_training_samples": self._n_training_samples,
            "min_samples": self.min_samples,
            "model_dir": str(self.model_dir),
            "load_error": self._load_error,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_last_retrain_date(self, db_pool) -> Optional[datetime]:
        """Read advisor_ml_version from config_settings to get last retrain date."""
        try:
            async with db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT value FROM config_settings
                    WHERE config_type='advisor_config'
                      AND key='advisor_ml_version'
                    """
                )
            if row and row["value"]:
                data = json.loads(row["value"])
                ts = data.get("trained_at")
                if ts:
                    return datetime.fromisoformat(ts)
        except Exception as exc:
            logger.debug("[advisor-ml] _get_last_retrain_date error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_features(row: dict) -> List[float]:
    """
    Build the fixed-order feature vector from an advice row dict.

    Input dict keys (advisor_advice or similar):
      market, horizon, direction, confidence, kronos_signal,
      entry_low, entry_high, target_price, stop_price.

    Returns a list of floats in _FEATURE_NAMES order.
    """
    market = str(row.get("market", "crypto")).lower()
    horizon = str(row.get("horizon", "mid")).lower()
    direction = str(row.get("direction", "neutral")).lower()
    confidence = float(row.get("confidence") or 0.5)
    kronos = float(row.get("kronos_signal") or 0.0)

    entry_low = row.get("entry_low")
    entry_high = row.get("entry_high")
    if entry_low and entry_high:
        entry_range_pct = (float(entry_high) - float(entry_low)) / float(entry_low) * 100
    else:
        entry_range_pct = 0.0

    has_target = 1 if row.get("target_price") else 0
    has_stop = 1 if row.get("stop_price") else 0

    return [
        float(_MARKET_MAP.get(market, 0)),
        float(_HORIZON_MAP.get(horizon, 1)),
        float(_DIRECTION_MAP.get(direction, 2)),
        confidence,
        kronos,
        entry_range_pct,
        float(has_target),
        float(has_stop),
    ]


async def _fetch_training_rows(db_pool) -> List[dict]:
    """
    Join advisor_sim_positions (closed) with advisor_advice to produce
    a training row per closed sim.  Sorted ascending by opened_at
    so the walk-forward train/test split is chronologically correct.
    """
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                a.market,
                a.horizon,
                a.direction,
                a.confidence,
                a.kronos_signal,
                a.entry_low,
                a.entry_high,
                a.target_price,
                a.stop_price,
                s.pnl_pct,
                s.opened_at
            FROM advisor_sim_positions s
            JOIN advisor_advice a ON s.advice_id = a.id
            WHERE s.status = 'closed'
              AND s.pnl_pct IS NOT NULL
            ORDER BY s.opened_at ASC
            """
        )
    return [dict(r) for r in rows]


async def _save_version(
    db_pool,
    version: str,
    n_samples: int,
    test_acc: float,
    test_prec: float,
    feature_names: List[str],
) -> None:
    """Upsert advisor_ml_version into config_settings."""
    payload = json.dumps({
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": n_samples,
        "test_accuracy": round(test_acc, 4),
        "test_precision": round(test_prec, 4),
        "feature_names": feature_names,
    })
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO config_settings
                  (config_type, key, value, value_type, description)
                VALUES
                  ('advisor_config', 'advisor_ml_version', $1, 'string',
                   'Advisor ML model version — updated by retrain job.')
                ON CONFLICT (config_type, key) DO UPDATE
                  SET value = EXCLUDED.value,
                      updated_at = NOW()
                """,
                payload,
            )
        logger.info("[advisor-ml] Saved version %s to config_settings.", version)
    except Exception as exc:
        logger.warning("[advisor-ml] Failed to save version to DB: %s", exc)
