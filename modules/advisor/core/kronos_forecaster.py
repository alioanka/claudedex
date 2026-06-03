"""
KronosForecaster — REAL Kronos K-line foundation-model interface.

=== WHAT KRONOS ACTUALLY IS (verified 2026-06-03) ===

Source of truth: https://github.com/shiyu-coder/Kronos  (NeoQuasar / shiyu-coder),
README "Model Zoo" + examples/prediction_example.py, and the HuggingFace model
cards under org NeoQuasar.

Kronos is NOT a HuggingFace causal-LM. The earlier implementation tried
``transformers.AutoModelForCausalLM`` / ``AutoTokenizer`` which fails at load
("Couldn't instantiate the backend tokenizer ... need sentencepiece or tiktoken")
because Kronos ships its OWN custom model code: a hierarchical BSQ tokenizer
(``KronosTokenizer``) + a two-stage autoregressive transformer (``Kronos``) +
a forecasting wrapper (``KronosPredictor``). That code is vendored offline in
``modules/advisor/core/kronos_vendor/`` (MIT). Generic transformers can never
load it; sentencepiece does not help.

REAL API (verified)
    from modules.advisor.core.kronos_vendor import Kronos, KronosTokenizer, KronosPredictor
    tokenizer = KronosTokenizer.from_pretrained("<tokenizer repo/dir>")
    model     = Kronos.from_pretrained("<model repo/dir>")
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    pred_df   = predictor.predict(
        df=ohlcv_df,                 # columns ['open','high','low','close'] (+'volume','amount')
        x_timestamp=pd.Series(...),  # datetimes for the history rows
        y_timestamp=pd.Series(...),  # datetimes for the FUTURE rows to forecast
        pred_len=N, T=1.0, top_p=0.9, sample_count=1, verbose=False,
    )                                # -> DataFrame of forecasted OHLCV bars

TWO downloads are required (model repo AND a separate tokenizer repo):
    Kronos-mini  -> model NeoQuasar/Kronos-mini  + tokenizer NeoQuasar/Kronos-Tokenizer-2k   (max_context 2048)
    Kronos-small -> model NeoQuasar/Kronos-small + tokenizer NeoQuasar/Kronos-Tokenizer-base (max_context 512)
    Kronos-base  -> model NeoQuasar/Kronos-base  + tokenizer NeoQuasar/Kronos-Tokenizer-base (max_context 512)
Use ``scripts/download_kronos_weights.py`` to fetch both into /app/data/kronos/.

License: MIT. Runtime deps: torch>=2.0, numpy, pandas, einops, huggingface_hub,
safetensors, tqdm (baked into the Docker image, Stage 7b).

=== FAIL-SOFT CONTRACT (MB-19) ===
Kronos is OPTIONAL. If anything is missing/unavailable (deps, weights,
tokenizer, runtime error), ``initialize()`` returns False and ``predict()``
returns ``None`` — it NEVER raises into the advice loop. The advice cycle
continues LLM+technicals-only. No forecast is ever fabricated.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("advisor.kronos_forecaster")

# Environment variable the operator sets to point at the downloaded MODEL weights.
# Example: ADVISOR_KRONOS_WEIGHTS_PATH=/app/data/kronos/Kronos-mini
_WEIGHTS_ENV = "ADVISOR_KRONOS_WEIGHTS_PATH"
# Optional override for the TOKENIZER directory. If unset we look for a sibling
# folder next to the model dir, then fall back to the matching HF tokenizer repo.
_TOKENIZER_ENV = "ADVISOR_KRONOS_TOKENIZER_PATH"

# Default model variant name for logging/health surface.
_DEFAULT_VARIANT = "Kronos-mini"

# Verified model -> (tokenizer repo, max_context) pairing from the upstream
# README "Model Zoo". mini uses the 2k tokenizer (ctx 2048); small/base use the
# base tokenizer (ctx 512).
_VARIANT_TOKENIZER: dict[str, str] = {
    "Kronos-mini": "NeoQuasar/Kronos-Tokenizer-2k",
    "Kronos-small": "NeoQuasar/Kronos-Tokenizer-base",
    "Kronos-base": "NeoQuasar/Kronos-Tokenizer-base",
}
_VARIANT_MAX_CONTEXT: dict[str, int] = {
    "Kronos-mini": 2048,
    "Kronos-small": 512,
    "Kronos-base": 512,
}

# How many future bars to forecast before collapsing to a directional signal.
_DEFAULT_PRED_LEN = 12
# Minimum history rows for a meaningful forecast.
_MIN_HISTORY = 30
# Cap the history fed to the model so CPU latency stays bounded (and never
# exceeds the tokenizer context). Trimmed to max_context at predict time too.
_MAX_HISTORY = 512

# Logged once per process to avoid logspam on every predict() call.
_IMPORT_WARN_LOGGED = False


class KronosForecaster:
    """
    Thin interface over the REAL Kronos foundation model for K-line forecasting.

    Interface preserved for advice_engine: ``initialize()`` then ``predict(df)``
    returning a directional float (positive = bullish) or ``None``.

    Fail-soft (MB-19)
    -----------------
    If weights / tokenizer / deps are absent or inference fails, predict()
    returns None rather than raising. advice_engine treats None as "Kronos not
    available this cycle" and continues with LLM-only advice.

    Operator setup
    --------------
    1. ``python scripts/download_kronos_weights.py --variant mini`` downloads
       BOTH the model repo and its matching tokenizer repo.
    2. Set ``ADVISOR_KRONOS_WEIGHTS_PATH=/app/data/kronos/Kronos-mini`` in .env.
       (Optionally ``ADVISOR_KRONOS_TOKENIZER_PATH`` to override tokenizer dir.)
    3. Set ``advisor_kronos_enabled=true`` in advisor_config (DB-backed).
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        variant: str = _DEFAULT_VARIANT,
        device: str = "cpu",
        tokenizer_path: Optional[str] = None,
        pred_len: int = _DEFAULT_PRED_LEN,
    ):
        self._weights_path = weights_path or os.getenv(_WEIGHTS_ENV)
        self._tokenizer_path = tokenizer_path or os.getenv(_TOKENIZER_ENV)
        # Normalise variant (accept "mini"/"Kronos-mini").
        self._variant = self._normalise_variant(variant)
        self._device = device or "cpu"
        self._pred_len = max(1, int(pred_len))
        self._predictor = None       # KronosPredictor, loaded in initialize()
        self._loaded = False
        self._load_error: Optional[str] = None

    @staticmethod
    def _normalise_variant(variant: str) -> str:
        v = (variant or "").strip()
        if not v:
            return _DEFAULT_VARIANT
        if v in _VARIANT_TOKENIZER:
            return v
        short = v.lower().replace("kronos-", "")
        candidate = f"Kronos-{short}"
        return candidate if candidate in _VARIANT_TOKENIZER else _DEFAULT_VARIANT

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> bool:
        """
        Load the Kronos model + tokenizer + predictor. Returns True on success.
        Never raises — all errors are stored in self._load_error.
        """
        if not self._weights_path:
            self._load_error = (
                f"{_WEIGHTS_ENV} not set. Run scripts/download_kronos_weights.py "
                f"--variant {self._variant.replace('Kronos-', '')} then set "
                f"{_WEIGHTS_ENV}=/app/data/kronos/{self._variant}."
            )
            logger.warning(
                "[kronos] WEIGHTS NOT CONFIGURED — %s predict() returns None.",
                self._load_error,
            )
            return False

        model_dir = Path(self._weights_path)
        if not model_dir.exists():
            self._load_error = (
                f"Model weights directory not found: {model_dir}. "
                "Run scripts/download_kronos_weights.py first."
            )
            logger.warning("[kronos] %s predict() returns None.", self._load_error)
            return False

        try:
            self._predictor = _load_predictor(
                model_dir=model_dir,
                tokenizer_path=self._tokenizer_path,
                variant=self._variant,
                device=self._device,
            )
            self._loaded = True
            logger.info(
                "[kronos] %s loaded (device=%s, max_context=%d) — model=%s",
                self._variant,
                self._device,
                _VARIANT_MAX_CONTEXT.get(self._variant, 512),
                model_dir,
            )
            return True
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning(
                "[kronos] Failed to load %s: %s. predict() returns None.",
                self._variant, exc,
            )
            logger.debug("[kronos] Load failure traceback:", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def predict(self, df_klines) -> Optional[float]:
        """
        Forecast future bars and collapse them into a directional signal.

        Parameters
        ----------
        df_klines : pd.DataFrame with columns [open, high, low, close] (+volume,
                    +amount optional) and an ascending UTC DatetimeIndex.
                    Minimum length: 30 rows.

        Returns
        -------
        float or None
            Forecast return: mean(forecast close over pred_len bars) / last close
            - 1. Positive = Kronos expects price UP (bullish); negative = down.
            Roughly bounded but NOT normalised. None on any failure (fail-soft).
        """
        if not self._loaded or self._predictor is None:
            return None

        if df_klines is None or len(df_klines) < _MIN_HISTORY:
            logger.debug("[kronos] Insufficient K-line rows for prediction (<30)")
            return None

        try:
            return _run_inference(self._predictor, df_klines, self._pred_len)
        except Exception as exc:
            logger.warning("[kronos] Inference error: %s", exc)
            logger.debug("[kronos] Inference traceback:", exc_info=True)
            return None

    async def predict_batch(self, dfs: list) -> list:
        """Run inference on multiple DataFrames; returns list of float|None."""
        results = []
        for df in dfs:
            results.append(await self.predict(df))
        return results

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def health(self) -> dict:
        return {
            "loaded": self._loaded,
            "variant": self._variant,
            "device": self._device,
            "weights_path": self._weights_path,
            "tokenizer_path": self._tokenizer_path,
            "max_context": _VARIANT_MAX_CONTEXT.get(self._variant),
            "pred_len": self._pred_len,
            "load_error": self._load_error,
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_tokenizer_source(model_dir: Path, tokenizer_path: Optional[str],
                              variant: str) -> str:
    """
    Decide where to load the tokenizer from, in priority order:
      1. Explicit ADVISOR_KRONOS_TOKENIZER_PATH (dir).
      2. A sibling directory next to the model dir whose name contains
         "Tokenizer" (what download_kronos_weights.py creates).
      3. The matching HuggingFace tokenizer repo id (requires network at load).
    Returns a path string or HF repo id suitable for from_pretrained().
    """
    if tokenizer_path:
        return tokenizer_path

    parent = model_dir.parent
    if parent.exists():
        # Prefer the exact expected folder name first.
        repo_id = _VARIANT_TOKENIZER.get(variant, "NeoQuasar/Kronos-Tokenizer-base")
        expected = parent / repo_id.split("/")[-1]
        if expected.exists():
            return str(expected)
        for child in sorted(parent.iterdir()):
            if child.is_dir() and "tokenizer" in child.name.lower():
                return str(child)

    # Fall back to the HF repo id (needs egress). Logged by caller on failure.
    return _VARIANT_TOKENIZER.get(variant, "NeoQuasar/Kronos-Tokenizer-base")


def _load_predictor(model_dir: Path, tokenizer_path: Optional[str],
                    variant: str, device: str):
    """
    Load the REAL Kronos model + tokenizer + predictor from local dirs.

    Raises on any failure so initialize() records the error and predict()
    fail-softs to None.
    """
    global _IMPORT_WARN_LOGGED

    try:
        from modules.advisor.core.kronos_vendor import (  # type: ignore[import]
            Kronos,
            KronosTokenizer,
            KronosPredictor,
        )
    except Exception as exc:  # torch / einops / numpy / pandas missing, etc.
        if not _IMPORT_WARN_LOGGED:
            logger.warning(
                "[kronos] Vendored Kronos model deps unavailable "
                "(need torch>=2.0, einops, huggingface_hub, safetensors, "
                "pandas, numpy): %s",
                exc,
            )
            _IMPORT_WARN_LOGGED = True
        raise ImportError(f"Kronos deps missing: {exc}") from exc

    tok_source = _resolve_tokenizer_source(model_dir, tokenizer_path, variant)

    try:
        tokenizer = KronosTokenizer.from_pretrained(tok_source)
    except Exception as exc:
        raise RuntimeError(
            f"KronosTokenizer.from_pretrained({tok_source!r}) failed: {exc}. "
            f"Download the tokenizer repo "
            f"({_VARIANT_TOKENIZER.get(variant)}) or set {_TOKENIZER_ENV}."
        ) from exc

    try:
        model = Kronos.from_pretrained(str(model_dir))
    except Exception as exc:
        raise RuntimeError(
            f"Kronos.from_pretrained({model_dir}) failed: {exc}"
        ) from exc

    max_context = _VARIANT_MAX_CONTEXT.get(variant, 512)
    try:
        predictor = KronosPredictor(
            model, tokenizer, device=device, max_context=max_context
        )
    except Exception as exc:
        raise RuntimeError(f"KronosPredictor init failed: {exc}") from exc

    return predictor


def _prepare_ohlcv(df_klines, max_history: int):
    """
    Coerce the analyzer's klines DataFrame into the (df, x_timestamp) pair
    Kronos.predict expects: lowercase OHLC(V) columns + a datetime Series of
    the history timestamps. Trims to the most recent `max_history` rows.
    Returns (x_df, x_timestamp_series). Raises if OHLC columns are absent.
    """
    import pandas as pd  # type: ignore[import]

    df = df_klines.copy()
    # Normalise column names to lowercase so 'Open'/'OPEN' both work.
    df.columns = [str(c).lower() for c in df.columns]

    required = ["open", "high", "low", "close"]
    if not all(c in df.columns for c in required):
        raise ValueError(
            f"klines missing OHLC columns; have {list(df.columns)}"
        )

    keep = required + [c for c in ("volume", "amount") if c in df.columns]
    df = df[keep].astype("float64")

    # Build the history timestamp Series from the index (UTC DatetimeIndex).
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True, errors="coerce")
    # Drop tz so the .dt.* fields Kronos uses (minute/hour/weekday/day/month)
    # are computed consistently across all rows.
    try:
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
    except (AttributeError, TypeError):
        pass

    x_timestamp = pd.Series(idx)
    df = df.reset_index(drop=True)

    if len(df) > max_history:
        df = df.iloc[-max_history:].reset_index(drop=True)
        x_timestamp = x_timestamp.iloc[-max_history:].reset_index(drop=True)

    if df.isnull().values.any():
        df = df.ffill().bfill()
    if df.isnull().values.any():
        raise ValueError("klines contain unfillable NaNs")

    return df, x_timestamp


def _future_timestamps(x_timestamp, pred_len: int):
    """
    Build `pred_len` future timestamps continuing the bar cadence inferred from
    the median spacing of the history. Returns a pandas Series of datetimes.
    """
    import pandas as pd  # type: ignore[import]

    ts = pd.to_datetime(x_timestamp)
    if len(ts) >= 2:
        deltas = ts.diff().dropna()
        step = deltas.median()
        if pd.isna(step) or step.total_seconds() <= 0:
            step = pd.Timedelta(hours=1)
    else:
        step = pd.Timedelta(hours=1)

    last = ts.iloc[-1]
    future = [last + step * (i + 1) for i in range(pred_len)]
    return pd.Series(pd.to_datetime(future))


def _run_inference(predictor, df_klines, pred_len: int) -> Optional[float]:
    """
    Run the real KronosPredictor.predict and collapse the OHLCV forecast into a
    single directional float = mean(forecast close)/last close - 1.
    """
    x_df, x_timestamp = _prepare_ohlcv(df_klines, _MAX_HISTORY)
    y_timestamp = _future_timestamps(x_timestamp, pred_len)

    last_close = float(x_df["close"].iloc[-1])
    if last_close <= 0:
        return None

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )

    if pred_df is None or "close" not in pred_df.columns or len(pred_df) == 0:
        return None

    mean_future_close = float(pred_df["close"].astype("float64").mean())
    # Directional signal: expected forward return over the horizon.
    signal = (mean_future_close / last_close) - 1.0
    if signal != signal:  # NaN guard
        return None
    return signal
