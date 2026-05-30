#!/usr/bin/env bash
# AI AI-Q-05 calibrated booster inference helper presence check
# (catalog: script_ai_calibrated_helper_present;
# wave-4 commits 16b7dab / a6c3a89).
#
# AI-Q-05 added `EnsemblePredictor.calibrated_predict_proba` plus the
# sidecar loader `_load_calibrated_models` so the 6 tree-based base
# classifiers route through CalibratedClassifierCV wrappers when the
# operator flips `ai_calibrated_predictions_enabled` to true. This
# script grep-asserts both the helper definition AND the predict-path
# callsite so a refactor that drops the wrap is caught immediately.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=ml/models/ensemble_model.py
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

if ! grep -nE 'def calibrated_predict_proba' "$f" >/dev/null; then
  echo "FAIL — calibrated_predict_proba definition missing from $f"
  exit 1
fi

if ! grep -nE 'def _load_calibrated_models' "$f" >/dev/null; then
  echo "FAIL — _load_calibrated_models loader missing from $f"
  exit 1
fi

# Verify the helper is actually USED in the predict path (not just
# defined). A dangling definition would silently leave the raw booster
# in place.
if ! grep -nE 'self\.calibrated_predict_proba\(' "$f" >/dev/null; then
  echo "FAIL — no callsite for self.calibrated_predict_proba in $f"
  exit 1
fi

# Flag gating must also be present so the default-OFF contract holds.
if ! grep -nE 'ai_calibrated_predictions_enabled' "$f" >/dev/null; then
  echo "FAIL — ai_calibrated_predictions_enabled flag not referenced in $f"
  exit 1
fi

echo "PASS — calibrated_predict_proba + _load_calibrated_models + callsite + flag-gate present"
