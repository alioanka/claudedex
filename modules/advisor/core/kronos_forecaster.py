"""
KronosForecaster — interface + fail-soft stub for the Kronos foundation model.

=== KRONOS FEASIBILITY SUMMARY (Wave-20, 2026-06-02) ===

Source: https://github.com/shiyu-coder/Kronos

Architecture
    Decoder-only autoregressive Transformer (two-stage):
    Stage 1 — a hierarchical tokenizer converts OHLCV K-line (candlestick)
    data into discrete tokens.
    Stage 2 — large-scale transformer pre-training on K-lines from 45+ exchanges.

Model variants (HuggingFace org: NeoQuasar)
    Kronos-mini  :  4.1 M parameters  (CPU-feasible, low RAM)
    Kronos-small : 24.7 M parameters  (CPU or light GPU)
    Kronos-base  :102.3 M parameters  (GPU recommended; ~400 MB weights)
    Kronos-large :499.2 M parameters  (NOT open-sourced as of 2026-06-02)

License
    MIT — permissive, no usage restriction for commercial/production use.

Dependencies (inferred from README / HuggingFace ecosystem)
    python >= 3.10
    torch >= 2.0 (CPU or CUDA)
    transformers >= 4.38
    pyqlib (Microsoft Qlib — financial data preprocessing)
    pandas, numpy

GPU / CPU
    Kronos-mini and Kronos-small run on CPU with ~1-3 s per batch inference.
    Kronos-base works on CPU but 5-15 s per batch; GPU (8 GB VRAM) recommended.
    Kronos-large requires GPU and is not publicly available.
    RECOMMENDATION: start with Kronos-mini for the advisor scaffold; upgrade to
    Kronos-small if accuracy improves meaningfully in backtest.

Weights location
    HuggingFace Hub: NeoQuasar/Kronos-mini, NeoQuasar/Kronos-small,
    NeoQuasar/Kronos-base.
    Download via: huggingface_hub.snapshot_download("NeoQuasar/Kronos-mini")
    Approximate sizes: mini ~50 MB, small ~100 MB, base ~400 MB.
    NOTE: Do NOT auto-download at startup. Weights must be present in
    ADVISOR_KRONOS_WEIGHTS_PATH before the forecaster activates.
    If weights are absent, predict() returns None (fail-soft, MB-19 pattern).

Inference API shape (from README)
    predictor = KronosPredictor(model_path, device="cpu")
    # DataFrame must have columns: open, high, low, close (+ optional volume)
    # Index: datetime, sorted ascending.
    signal = predictor.predict(df_klines)   # float: positive=bullish
    signals = predictor.predict_batch(list_of_dfs)

Integration plan (specialist wiring tasks)
    1. Install torch + transformers + pyqlib in requirements.txt (quant agent).
    2. Download weights to ADVISOR_KRONOS_WEIGHTS_PATH at deploy time via
       scripts/download_kronos_weights.py (backend agent).
    3. Implement the actual HuggingFace model loading inside _load_model()
       below (quant/data agent).
    4. Wire KronosForecaster.predict() call into AdviceEngine._run_cycle()
       after the analyzer produces its AdviceResult (advice_engine.py).
    5. Store kronos_signal on the advisor_advice DB row.

=== END FEASIBILITY SUMMARY ===
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("advisor.kronos_forecaster")

# Environment variable the operator sets to point at downloaded weights.
# Example: ADVISOR_KRONOS_WEIGHTS_PATH=/data/kronos/Kronos-mini
_WEIGHTS_ENV = "ADVISOR_KRONOS_WEIGHTS_PATH"

# Default model variant name for logging/health surface.
_DEFAULT_VARIANT = "Kronos-mini"


class KronosForecaster:
    """
    Thin interface over the Kronos foundation model for K-line forecasting.

    Fail-soft (MB-19 pattern)
    -------------------------
    If weights are absent, the model cannot be loaded, or the underlying
    inference library is not installed, predict() returns None rather than
    raising. The advice_engine treats a None kronos_signal as "Kronos not
    available this cycle" and continues with LLM-only advice.

    The operator must:
    1. Set ADVISOR_KRONOS_WEIGHTS_PATH in .env to the directory containing
       downloaded HuggingFace model files.
    2. Ensure torch + transformers + pyqlib are installed
       (pip install torch transformers pyqlib).
    3. Keep ADVISOR_KRONOS_ENABLED=true in advisor_config (DB-backed).

    Usage (once wired by the quant agent)
    --------------------------------------
        forecaster = KronosForecaster()
        await forecaster.initialize()
        signal = await forecaster.predict(df_klines)
        # signal: float (positive = bullish) or None if unavailable.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        variant: str = _DEFAULT_VARIANT,
        device: str = "cpu",
    ):
        self._weights_path = weights_path or os.getenv(_WEIGHTS_ENV)
        self._variant = variant
        self._device = device
        self._model = None          # loaded lazily in initialize()
        self._tokenizer = None
        self._loaded = False
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> bool:
        """
        Attempt to load the Kronos model from weights_path.

        Returns True if model loaded successfully, False otherwise.
        Never raises — all errors stored in self._load_error.
        """
        if not self._weights_path:
            self._load_error = (
                f"{_WEIGHTS_ENV} not set. "
                f"Download weights (e.g. NeoQuasar/{self._variant} from HuggingFace) "
                f"and set {_WEIGHTS_ENV}=/path/to/weights."
            )
            logger.warning(
                f"[kronos] WEIGHTS NOT CONFIGURED — {self._load_error}. "
                "predict() will return None."
            )
            return False

        weights_dir = Path(self._weights_path)
        if not weights_dir.exists():
            self._load_error = (
                f"Weights directory not found: {weights_dir}. "
                "Run scripts/download_kronos_weights.py first."
            )
            logger.warning(f"[kronos] {self._load_error}. predict() will return None.")
            return False

        try:
            self._model, self._tokenizer = _load_model(
                weights_dir, self._variant, self._device
            )
            self._loaded = True
            logger.info(
                f"[kronos] {self._variant} loaded from {weights_dir} "
                f"on device={self._device}"
            )
            return True
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning(
                f"[kronos] Failed to load {self._variant}: {exc}. "
                "predict() will return None.",
                exc_info=True,
            )
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def predict(self, df_klines) -> Optional[float]:
        """
        Produce a directional forecast from historical K-line data.

        Parameters
        ----------
        df_klines : pd.DataFrame with columns [open, high, low, close]
                    and a datetime index (UTC, sorted ascending).
                    Optional: volume, amount columns.
                    Minimum length: 30 rows for meaningful prediction.

        Returns
        -------
        float or None
            Positive value indicates bullish bias; negative = bearish.
            Magnitude indicates confidence (not normalised — varies by model).
            None if model not loaded or inference failed (fail-soft).
        """
        if not self._loaded:
            return None

        if df_klines is None or len(df_klines) < 30:
            logger.debug("[kronos] Insufficient K-line rows for prediction (<30)")
            return None

        try:
            return _run_inference(self._model, self._tokenizer, df_klines)
        except Exception as exc:
            logger.warning(f"[kronos] Inference error: {exc}", exc_info=True)
            return None

    async def predict_batch(self, dfs: list) -> list:
        """
        Run inference on multiple DataFrames.
        Returns list of float or None, same order as input.
        """
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
            "load_error": self._load_error,
        }


# ---------------------------------------------------------------------------
# Private helpers — STUBS (quant agent fills these in Wave-21)
# ---------------------------------------------------------------------------

def _load_model(weights_dir: Path, variant: str, device: str):
    """
    STUB: Load Kronos tokenizer + model from a HuggingFace weights directory.

    Expected implementation (quant agent, Wave-21):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # or the Kronos-specific classes from the shiyu-coder/Kronos repo:
        from kronos import KronosTokenizer, KronosModel
        tokenizer = KronosTokenizer.from_pretrained(str(weights_dir))
        model = KronosModel.from_pretrained(str(weights_dir)).to(device)
        model.eval()
        return model, tokenizer

    For now, raise NotImplementedError so _loaded stays False and predict()
    returns None (fail-soft).
    """
    raise NotImplementedError(
        "Kronos model loading not yet implemented. "
        "Wave-21 quant agent task: implement _load_model() in kronos_forecaster.py."
    )


def _run_inference(model, tokenizer, df_klines) -> float:
    """
    STUB: Run a single forward pass through the loaded Kronos model.

    Expected implementation (quant agent, Wave-21):
        # Tokenize K-lines into Kronos's discrete token format.
        tokens = tokenizer.encode(df_klines)  # returns torch.Tensor
        with torch.no_grad():
            output = model.predict(tokens)
        # output is a scalar or distribution; extract directional signal.
        signal = float(output["signal"])
        return signal

    For now, raise so predict() returns None.
    """
    raise NotImplementedError(
        "Kronos inference not yet implemented. "
        "Wave-21 quant agent task: implement _run_inference() in kronos_forecaster.py."
    )
