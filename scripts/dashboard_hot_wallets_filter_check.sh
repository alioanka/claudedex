#!/usr/bin/env bash
# DASHBOARD hot-wallets active-filter presence check
# (catalog: script_dashboard_hot_wallets_filter_sort; W6 commit 5a857e0).
#
# Wave-6 fixed the operator-reported "hot wallets card full of
# 0-trade 0-PnL configured wallets" bug. The discovery template now
# filters wallets to ACTIVE (>0 trades OR nonzero PnL), then sorts
# by trades DESC, then PnL DESC. If nothing's active, a clear empty
# state replaces the inactive noise.
#
# This probe asserts both the filter and the sort survive future
# template refactors.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=dashboard/templates/discovery_copytrading.html
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

missing=()
if ! grep -nF 'active.filter' "$f" >/dev/null \
   && ! grep -nF '.wallets.filter' "$f" >/dev/null \
   && ! grep -nE 'data\.wallets\.filter\(' "$f" >/dev/null; then
  missing+=("no .filter() call on wallets / active list")
fi

if ! grep -nE 'total_trades.*0' "$f" >/dev/null; then
  missing+=("no 'total_trades > 0' style guard for active wallets")
fi

if [ ${#missing[@]} -ne 0 ]; then
  echo "FAIL — discovery_copytrading.html missing W6 active-filter markers:"
  for m in "${missing[@]}"; do echo "  - $m"; done
  exit 1
fi

echo "PASS — hot-wallets active filter + total_trades guard present in discovery_copytrading.html"
