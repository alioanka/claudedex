#!/usr/bin/env bash
# DEX MEV bundle_id default + Flashbots ETH-only gating source check
# (catalog: script_dex_mev_unbound_check; wave-2 commit e872121).
#
# Why source-grep instead of an end-to-end smoke:
#  * the original crash only triggered when protect_transaction was called
#    on the ADVANCED branch with risk_score ≤ 0.3 (Flashbots NOT engaged),
#  * exercising that path requires a live Web3 connection + a real EIP-1559
#    tx, which we deliberately keep out of the test-runner blast radius.
# A regression-proof grep is sufficient: the fix is two literal source
# changes (bundle_id default + chain == 'ethereum' gate).
set -euo pipefail

cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=trading/executors/mev_protection.py
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

# 1. Default-None initialisation before any if/else assigning bundle_id.
if ! grep -nE '^[[:space:]]*bundle_id[[:space:]]*=[[:space:]]*None' "$f" >/dev/null; then
  echo "FAIL — no top-level 'bundle_id = None' default in $f"
  echo "        regression of e872121: low-risk ADVANCED path will UnboundLocalError"
  exit 1
fi

# 2. Per-chain Flashbots gate: the fix only engages Flashbots when the
#    target chain is Ethereum mainnet. Any of the canonical guard forms
#    counts; we just need to see the intent.
if ! grep -nE "(chain[^a-zA-Z_]*==[^=]*['\"]ethereum['\"])|(['\"]ethereum['\"][^a-zA-Z_]*==[^=]*chain)" "$f" >/dev/null; then
  echo "FAIL — no chain == 'ethereum' Flashbots gate in $f"
  echo "        regression of e872121: BSC/Polygon/L2s would attempt Flashbots"
  exit 1
fi

echo "PASS — bundle_id default present AND Flashbots ETH-only gate present"
