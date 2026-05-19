#!/usr/bin/env bash
# ARB spatial-arb NameError regression check (catalog:
# script_arb_nameerror_regression; wave-2 commit 9e6a7d1).
#
# Pre-9e6a7d1 the spatial-arb opportunity-log line in
# `_check_arb_opportunity` still referenced `forward_output` /
# `final_output` from a deleted refactor. Every profitable opp raised
# NameError, the broad `except` swallowed it, and the trade was
# silently dropped — so the engine couldn't broadcast a single live
# trade. This script grep-asserts those names are gone from the hot
# path. Source-grep is sufficient: the live execution path is gated
# behind DRY_RUN=false + chain wallets + flash-loan receiver, none of
# which are appropriate for the test-runner blast radius.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=modules/arbitrage/arbitrage_engine.py
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

# Stale identifiers that must NOT reappear in the spatial-arb path.
# tokens_bought / weth_returned replaced them in 9e6a7d1. We tolerate
# the named identifier inside a comment (-> after #) — only flag a
# code-level appearance (assignment / call / reference).
for name in forward_output final_output; do
  hits=$(grep -nE "[^A-Za-z0-9_]${name}[^A-Za-z0-9_]" "$f" \
         | grep -vE '^[[:space:]]*[0-9]+:[[:space:]]*#') || true
  if [ -n "$hits" ]; then
    echo "FAIL — stale identifier '${name}' present in $f (non-comment use):"
    echo "$hits" | head -5
    exit 1
  fi
done

# Positive control: the replacement names must be present, otherwise
# the refactor was reverted.
for name in tokens_bought weth_returned; do
  if ! grep -nE "[^A-Za-z0-9_]${name}[^A-Za-z0-9_]" "$f" >/dev/null; then
    echo "FAIL — replacement identifier '${name}' missing in $f"
    echo "        (9e6a7d1 rename was reverted)"
    exit 1
  fi
done

echo "PASS — spatial-arb opportunity-log uses tokens_bought/weth_returned only"
