#!/usr/bin/env bash
# COPY trades-page UI icons presence check
# (catalog: script_copy_trade_ui_icons_present; W6 commits 2466ec0 / f0fb2bd).
#
# Wave-6 enriched the trades_copytrading.html UI with:
#   - a copy-to-clipboard icon (copyTokenAddress) for the token mint,
#   - a Birdeye explorer link (birdeye.so/token/<mint>),
#   - a Solscan/Etherscan token link (solscan.io/token for Solana).
#
# A template refactor that drops any of these regressions the operator
# UX — they can no longer one-click jump to the token's chart or
# explorer page from the trade row.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=dashboard/templates/trades_copytrading.html
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

missing=()
for sym in 'birdeye.so/token' 'solscan.io/token' 'copyTokenAddress'; do
  if ! grep -nF "$sym" "$f" >/dev/null; then
    missing+=("$sym")
  fi
done

if [ ${#missing[@]} -ne 0 ]; then
  echo "FAIL — trades_copytrading.html missing W6 UI icons:"
  for m in "${missing[@]}"; do echo "  - $m"; done
  exit 1
fi

echo "PASS — birdeye + solscan + copyTokenAddress icons present in trades_copytrading.html"
