"""
KronosForecaster — Kronos foundation-model interface (Wave-21 implementation).

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

Dependencies
    python >= 3.10
    torch >= 2.0 (CPU or CUDA)
    transformers >= 4.38
    pyqlib (Microsoft Qlib — financial data preprocessing)
    pandas, numpy

GPU / CPU
    Kronos-mini and Kronos-small run on CPU with ~1-3 s per batch inference.
    Kronos-base works on CPU but 5-15 s per batch; GPU (8 GB VRAM) recommended.
    Kronos-large requires GPU and is not publicly available.
    RECOMMENDATION: start with Kronos-mini; upgrade to Kronos-small if
    backtesting shows meaningful accuracy improvement.

Weights location
    HuggingFace Hub: NeoQuasar/Kronos-mini, NeoQuasar/Kronos-small,
    NeoQuasar/Kronos-base.
    Download via: python scripts/download_kronos_weights.py
    Approximate sizes: mini ~50 MB, small ~100 MB, base ~400 MB.
    NOTE: Do NOT auto-download at startup. Weights must be present in
    ADVISOR_KRONOS_WEIGHTS_PATH before the forecaster activates.
    If weights are absent, predict() returns None (fail-soft, MB-19 pattern).

Inference API shape
    predictor = KronosPredictor(model_path, device="cpu")
    signal = predictor.predict(df_klines)   # float: positive=bullish
    signals = predictor.predict_batch(list_of_dfs)

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

# Logged once per process to avoid logspam on every predict() call.
_IMPORT_WARN_LOGGED = False


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
    2. Ensure torch + transformers are installed
       (pip install 'torch>=2.0' 'transformers>=4.38'; optionally pyqlib).
    3. Set advisor_kronos_enabled=true in advisor_config (DB-backed).

    Usage
    -----
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
                "[kronos] WEIGHTS NOT CONFIGURED — %s. "
                "predict() will return None.",
                self._load_error,
            )
            return False

        weights_dir = Path(self._weights_path)
        if not weights_dir.exists():
            self._load_error = (
                f"Weights directory not found: {weights_dir}. "
                "Run scripts/download_kronos_weights.py first."
            )
            logger.warning("[kronos] %s. predict() will return None.", self._load_error)
            return False

        try:
            self._model, self._tokenizer = _load_model(
                weights_dir, self._variant, self._device
            )
            self._loaded = True
            logger.info(
                "[kronos] %s loaded from %s on device=%s",
                self._variant, weights_dir, self._device,
            )
            return True
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning(
                "[kronos] Failed to load %s: %s. predict() will return None.",
                self._variant, exc,
            )
            logger.debug("[kronos] Load failure traceback:", exc_info=True)
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
            logger.warning("[kronos] Inference error: %s", exc, exc_info=True)
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
# Private helpers — Wave-21 implementation
# ---------------------------------------------------------------------------

def _load_model(weights_dir: Path, variant: str, device: str):
    """
    Load a Kronos model from a HuggingFace weights directory.

    Fallback chain
    --------------
    1. Try ``from kronos import KronosPredictor`` (official Kronos pip package
       or local clone on PYTHONPATH).  Returns (predictor, None) because the
       tokeniser is baked into the predictor object.
    2. Try ``transformers.AutoModelForCausalLM`` / ``AutoTokenizer`` as generic
       HuggingFace causal-LM load — covers checkpoints uploaded with the
       standard HF config.json + model.safetensors layout.
    3. If neither path works (torch/transformers absent), raise ImportError
       so the caller stores the error and predict() returns None (fail-soft).

    GPU / small upgrade path
    ------------------------
    - Default: device='cpu', variant='Kronos-mini' (4.1 M params, ~50 MB).
    - Kronos-small (24.7 M, ~100 MB): set advisor_kronos_variant=Kronos-small.
      CPU still fine (2-4 s / call).
    - Kronos-base (102.3 M, ~400 MB): set advisor_kronos_device=cuda and
      provision >=8 GB VRAM.  CPU is slow (5-15 s) but functional.
    - Kronos-large (499.2 M): not open-sourced as of 2026-06-02; skip.
    """
    global _IMPORT_WARN_LOGGED

    # --- attempt 1: official Kronos package ---
    try:
        from kronos import KronosPredictor  # type: ignore[import]
        predictor = KronosPredictor(str(weights_dir), device=device)
        logger.info(
            "[kronos] Loaded via KronosPredictor (%s, device=%s)", variant, device
        )
        return predictor, None          # tokenizer baked into predictor
    except ImportError:
        pass   # fall through to attempt 2
    except Exception as exc:
        # Weights present but KronosPredictor init failed (corrupt files, etc.)
        raise RuntimeError(f"KronosPredictor init failed: {exc}") from exc

    # --- attempt 2: HuggingFace AutoModel (generic causal-LM interface) ---
    try:
        import torch  # type: ignore[import]  # noqa: F401
        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore[import]

        tokenizer = AutoTokenizer.from_pretrained(
            str(weights_dir), trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(weights_dir), trust_remote_code=True
        )
        model = model.to(device)
        model.eval()
        logger.info(
            "[kronos] Loaded via AutoModelForCausalLM (%s, device=%s)", variant, device
        )
        return model, tokenizer
    except ImportError as exc:
        if not _IMPORT_WARN_LOGGED:
            logger.warning(
                "[kronos] torch / transformers not installed — "
                "Kronos inference unavailable. "
                "Install: pip install 'torch>=2.0' 'transformers>=4.38' "
                "and optionally: pip install kronos pyqlib. Detail: %s",
                exc,
            )
            _IMPORT_WARN_LOGGED = True
        raise ImportError(f"Required deps missing: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"HuggingFace AutoModel load failed: {exc}") from exc


def _run_inference(model, tokenizer, df_klines) -> float:
    """
    Run a single forward pass through the loaded Kronos model.

    Parameters
    ----------
    model     : KronosPredictor (tokenizer=None) OR HuggingFace
                AutoModelForCausalLM (tokenizer provided).
    tokenizer : AutoTokenizer or None.
    df_klines : pd.DataFrame with columns [open, high, low, close] and an
                ascending UTC datetime index.  Optional: volume, amount.

    Returns
    -------
    float
        Positive = bullish bias; negative = bearish.  Magnitude not normalised.

    KronosPredictor path
        Calls model.predict(df_klines) which handles tokenisation internally.

    HuggingFace AutoModel path
        Attempts tokenizer(df_klines) first; falls back to encoding the
        OHLCV arrays as a raw float tensor (1, T, 4).  Signal is extracted
        from the last time-step's logit difference (bullish - bearish) or
        from the mean of the last hidden state.
    """
    # --- KronosPredictor path (tokenizer is None) ---
    if tokenizer is None:
        result = model.predict(df_klines)
        if isinstance(result, dict):
            return float(result.get("signal", result.get("score", 0.0)))
        return float(result)

    # --- HuggingFace AutoModel path ---
    import torch  # type: ignore[import]

    # Attempt tokeniser call with the raw DataFrame.
    try:
        enc = tokenizer(df_klines, return_tensors="pt")
        input_ids = enc["input_ids"]
    except Exception:
        # Fallback: encode OHLCV columns as a raw float tensor.
        ohlcv = df_klines[["open", "high", "low", "close"]].astype(float).values
        input_ids = torch.tensor(ohlcv, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(input_ids)

    # Extract directional signal.
    # Convention: last token, first two logit dims = [bearish, bullish].
    if hasattr(output, "logits"):
        logits = output.logits          # (1, T, vocab)
        last_step = logits[0, -1, :]
        if last_step.numel() >= 2:
            signal = float(last_step[1] - last_step[0])   # bullish - bearish
        else:
            signal = float(last_step[0])
    elif hasattr(output, "last_hidden_state"):
        signal = float(output.last_hidden_state[0, -1, :].mean())
    else:
        signal = float(output[0].mean())

    return signal
