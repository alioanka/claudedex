#!/usr/bin/env bash
# COPY BUY/SELL detector delta-based partition check
# (catalog: script_copy_buy_sell_detector_rewrite; W6 commit 2d30f3c).
#
# Wave-6 rewrote the leader-trade direction classifier to split token
# balance deltas into base_deltas (non-stablecoin mints) vs
# quote_deltas (stablecoin mints) so a USDC->memecoin swap is
# correctly classified as a BUY of the memecoin (not a SELL of USDC).
# The pre-W6 heuristic occasionally mirrored stablecoin movements as
# the principal side of the trade.
#
# This probe grep-asserts the new identifiers exist in
# modules/copy_trading/copy_engine.py.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=modules/copy_trading/copy_engine.py
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

missing=()
for sym in 'base_deltas' 'quote_deltas'; do
  if ! grep -nF "$sym" "$f" >/dev/null; then
    missing+=("$sym")
  fi
done

# Both names must co-occur to constitute the W6 partition pattern.
if [ ${#missing[@]} -ne 0 ]; then
  echo "FAIL — copy_engine.py missing W6 BUY/SELL detector identifiers:"
  for m in "${missing[@]}"; do echo "  - $m"; done
  exit 1
fi

# Sanity: the W6 partition uses STABLECOIN_MINTS to split the two
# dicts; if STABLECOIN_MINTS is gone the partition is meaningless.
if ! grep -nF 'STABLECOIN_MINTS' "$f" >/dev/null; then
  echo "FAIL — base_deltas/quote_deltas present but STABLECOIN_MINTS gone"
  exit 1
fi

echo "PASS — delta-based BUY/SELL partition (base_deltas + quote_deltas + STABLECOIN_MINTS) present"
