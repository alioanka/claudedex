"""
Vendored Kronos foundation-model code (MIT License).

UPSTREAM
--------
Kronos: A Foundation Model for the Language of Financial Markets
  Repo:    https://github.com/shiyu-coder/Kronos  (NeoQuasar / shiyu-coder)
  License: MIT (c) 2025 ShiYu  — see ./LICENSE in this directory.
  Vendored files: ``kronos.py`` and ``module.py`` copied VERBATIM from the
  upstream ``model/`` package, with ONE change: the
  ``sys.path.append("../"); from model.module import *`` line in ``kronos.py``
  is replaced with a package-relative ``from .module import *`` so the code is
  importable from anywhere on PYTHONPATH and works fully offline.

WHY VENDORED (not pip-installed)
--------------------------------
Kronos is NOT a clean PyPI package — it is repo ``model/`` source. Vendoring
the minimal model module guarantees the advisor's Kronos forecaster works
without any network egress at build OR runtime; only the pretrained *weights*
(model + tokenizer repos) must be downloaded separately by the operator (see
``scripts/download_kronos_weights.py`` and ``docs/ADVISOR_SETUP_AND_USAGE.md``
§5). Kronos is OPTIONAL and FAIL-SOFT: if torch / einops / the weights are
missing, ``KronosForecaster.predict()`` returns ``None`` and the advice cycle
continues.

PUBLIC API
----------
    from modules.advisor.core.kronos_vendor import (
        Kronos, KronosTokenizer, KronosPredictor,
    )
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    pred_df   = predictor.predict(df=ohlcv_df, x_timestamp=..., y_timestamp=...,
                                  pred_len=N, T=1.0, top_p=0.9, sample_count=1)

Runtime deps for the vendored code: torch>=2.0, numpy, pandas, einops,
huggingface_hub, safetensors, tqdm.
"""

from .kronos import Kronos, KronosTokenizer, KronosPredictor  # noqa: F401

__all__ = ["Kronos", "KronosTokenizer", "KronosPredictor"]
