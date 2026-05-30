#!/usr/bin/env bash
# DEX V3 QuoterV2 per-chain address map presence check (catalog:
# script_dex_quoter_v2_addresses_present; wave-4 commit 115cb35).
#
# wave-3 added _quote_v3 with real QuoterV2 round-trip; wave-4 routes
# the chunked price-impact probe through the same helper. Both paths
# depend on UNISWAP_V3_QUOTER_V2_ADDRESSES being populated for the
# chains the operator runs. If a chain entry is dropped (template
# refactor, key rename, etc.) the V3 quote silently falls through to
# zero impact and the new refusal gate becomes a no-op for that chain.
#
# This script source-grep asserts the canonical chain entries exist
# in trading/executors/direct_dex.py. Operator-runnable via the Test
# Runner panel without any DB credentials or RPC reachability.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=trading/executors/direct_dex.py
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

# Required: the map itself must be declared as a typed dict literal so
# overrides can layer on top via config['v3_quoter_addresses'][chain].
if ! grep -nE "^UNISWAP_V3_QUOTER_V2_ADDRESSES:[[:space:]]*Dict\[str, str\][[:space:]]*=" "$f" >/dev/null; then
  echo "FAIL — UNISWAP_V3_QUOTER_V2_ADDRESSES literal not declared in $f"
  exit 1
fi

# Required chains: every chain the DEX module runs on. Missing rows
# means the chunked V3 price-impact probe returns None for that chain
# and the refusal gate (max_price_impact_bps) silently disengages.
declare -a required=(
  "'ethereum':"
  "'polygon':"
  "'arbitrum':"
  "'base':"
  "'optimism':"
  "'bsc':"
)

missing=()
for chain in "${required[@]}"; do
  if ! grep -F "$chain" "$f" >/dev/null; then
    missing+=("$chain")
  fi
done

if [ ${#missing[@]} -gt 0 ]; then
  echo "FAIL — UNISWAP_V3_QUOTER_V2_ADDRESSES missing chain entries:"
  for m in "${missing[@]}"; do
    echo "  - $m"
  done
  exit 1
fi

echo "PASS — UNISWAP_V3_QUOTER_V2_ADDRESSES has all 6 required chain entries"
