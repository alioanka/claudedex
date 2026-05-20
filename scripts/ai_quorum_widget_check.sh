#!/usr/bin/env bash
# AI quorum-metrics dashboard widget presence check
# (catalog: script_ai_quorum_widget_present; wave-4 commit 50bd9c3).
#
# Wave-4 landed an agreement-rate panel on
# dashboard/templates/dashboard_ai.html that polls
# GET /api/ai/quorum-metrics?hours=24. This script grep-asserts the URL
# is wired up so a template refactor that drops the panel is caught.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=dashboard/templates/dashboard_ai.html
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

if ! grep -F '/api/ai/quorum-metrics' "$f" >/dev/null; then
  echo "FAIL — /api/ai/quorum-metrics URL missing from $f"
  exit 1
fi

echo "PASS — /api/ai/quorum-metrics widget URL present in $f"
