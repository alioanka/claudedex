#!/usr/bin/env bash
# FUTURES funding-forecast widget presence check (catalog:
# script_futures_funding_forecast_widget_present; wave-4 commit b805626).
#
# FUT-RM-09b landed an operator-visible per-symbol 24h forward
# funding-cost forecast panel on dashboard/templates/dashboard_futures.html
# that polls GET /api/futures/funding-forecast?window_hours=24. This
# script grep-asserts the canonical DOM ids and the API URL so a
# template refactor that drops them is caught immediately.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=dashboard/templates/dashboard_futures.html
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

# Required widget anchors.
declare -a anchors=(
  'id="funding-forecast-window"'
  'id="funding-forecast-total"'
  'id="funding-forecast-table"'
  '/api/futures/funding-forecast'
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

echo "PASS — funding-forecast widget (window+total+table) + /api/futures/funding-forecast present"
