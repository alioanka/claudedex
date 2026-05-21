#!/usr/bin/env bash
# FUTURES settings page W5 risk knobs presence check
# (catalog: script_futures_settings_w5_knobs_present; W6 commit 5a857e0).
#
# Wave-6 close-out surfaced the missing W5 FUT-RM-16/17/18 knobs on
# the operator-facing settings_futures.html page so they can be
# tuned without psql. The three required inputs:
#   - futures_min_signal_confluence_count    (FUT-RM-17)
#   - futures_atr_dynamic_sl_tp_enabled      (FUT-RM-16)
#   - futures_post_loss_cooloff_minutes      (FUT-RM-18)
# A template refactor that drops any of these regresses to "edit via
# DB" only.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=dashboard/templates/settings_futures.html
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

missing=()
for sym in 'futures_min_signal_confluence_count' \
           'futures_atr_dynamic_sl_tp_enabled' \
           'futures_post_loss_cooloff_minutes'; do
  if ! grep -nF "$sym" "$f" >/dev/null; then
    missing+=("$sym")
  fi
done

if [ ${#missing[@]} -ne 0 ]; then
  echo "FAIL — settings_futures.html missing W5 risk knobs:"
  for m in "${missing[@]}"; do echo "  - $m"; done
  exit 1
fi

echo "PASS — futures W5 knobs (confluence + ATR dynamic SL/TP + post-loss cool-off) all present in settings_futures.html"
