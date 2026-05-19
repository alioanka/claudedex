#!/usr/bin/env bash
# SNIPER per-chain listener-health widget presence check (catalog:
# script_sniper_listener_widget_present; wave-2 commit 5a0a3e9).
#
# The wave-2 fix landed an operator-visible panel on
# dashboard/templates/performance_sniper.html that consumes
# /api/sniper/stats and surfaces WSS saturation, processed-hit ratio,
# rejection profile, and per-chain WSS/polling deltas. This script
# grep-asserts the canonical DOM ids and the API URL it polls, so a
# template refactor that drops them is caught immediately.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=dashboard/templates/performance_sniper.html
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

# Required widget anchors.
declare -a anchors=(
  'id="listenerHealthPanel"'
  'id="listenerHealthBody"'
  '/api/sniper/stats'
)

missing=()
for a in "${anchors[@]}"; do
  if ! grep -F "$a" "$f" >/dev/null; then
    missing+=("$a")
  fi
done

if [ ${#missing[@]} -ne 0 ]; then
  echo "FAIL — widget anchors missing from $f:"
  for a in "${missing[@]}"; do echo "  - $a"; done
  exit 1
fi

echo "PASS — listenerHealthPanel + listenerHealthBody + /api/sniper/stats present"
