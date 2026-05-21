#!/usr/bin/env bash
# COPY stablecoin-guard presence check
# (catalog: script_copy_stablecoin_guard_present; W6 commit f81144f).
#
# Wave-6 W6-COPY closes the "mirrored stablecoin transfer treated as
# tradeable token" bug. The copy_engine BUY path must:
#   1) maintain a Solana STABLECOIN_MINTS set,
#   2) maintain an EVM_STABLECOIN_ADDRESSES set,
#   3) gate SELLs on a real open position (_has_open_copy_position),
#   4) raise the 'stablecoin_not_tradeable' refusal label so the
#      operator counters surface the refusal count on the dashboard.
#
# A regression in any of these silently re-introduces ghost SELLs
# against stablecoin mints and ghost BUYs of USDC/USDT/DAI.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=modules/copy_trading/copy_engine.py
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

missing=()
for sym in 'STABLECOIN_MINTS' 'EVM_STABLECOIN_ADDRESSES' \
           '_has_open_copy_position' 'stablecoin_not_tradeable'; do
  if ! grep -nF "$sym" "$f" >/dev/null; then
    missing+=("$sym")
  fi
done

if [ ${#missing[@]} -ne 0 ]; then
  echo "FAIL — copy_engine.py missing W6 stablecoin-guard symbols:"
  for m in "${missing[@]}"; do echo "  - $m"; done
  exit 1
fi

echo "PASS — STABLECOIN_MINTS + EVM_STABLECOIN_ADDRESSES + _has_open_copy_position + stablecoin_not_tradeable present in copy_engine.py"
