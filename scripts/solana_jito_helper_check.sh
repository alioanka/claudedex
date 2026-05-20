#!/usr/bin/env bash
# SOLANA Jito-bundle helper-presence check (catalog:
# script_solana_jito_helper_present; wave-4 commits d6a4a8c + a6c3a89).
#
# wave-4 wired the Jito bundle path into modules/solana_trading/core/
# solana_engine.py. The flag-gated path lives behind
# `solana_jito_bundle_enabled` (default False) so a regression that
# drops the helper would not surface in normal test runs -- the engine
# would silently fall back to vanilla Jupiter swaps and the operator
# would lose the MEV-protection lever they explicitly enabled.
#
# This script source-grep asserts:
#   1. The `_execute_swap_via_jito` async helper is still defined on
#      the engine class.
#   2. `_open_position` actually CALLS it (so the wiring isn't dead
#      code -- the wave-4 fix patches both define + call together).
#   3. The `JitoClient` import is present (so the helper can resolve
#      at runtime).
#
# Operator-runnable via the Test Runner panel; no DB / RPC needed.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=modules/solana_trading/core/solana_engine.py
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

# 1. helper definition
if ! grep -nE "async def _execute_swap_via_jito\(" "$f" >/dev/null; then
  echo "FAIL — _execute_swap_via_jito() not defined in $f"
  echo "        (wave-4 d6a4a8c/a6c3a89 helper was removed or renamed)"
  exit 1
fi

# 2. callsite in _open_position swap path
if ! grep -nE "self\._execute_swap_via_jito\(" "$f" >/dev/null; then
  echo "FAIL — _execute_swap_via_jito() defined but never called in $f"
  echo "        (Jito bundle path is dead code)"
  exit 1
fi

# 3. JitoClient import (lazy or top-level)
if ! grep -nE "JitoClient" "$f" >/dev/null; then
  echo "FAIL — JitoClient never referenced in $f"
  echo "        (helper cannot instantiate the bundle client)"
  exit 1
fi

echo "PASS — _execute_swap_via_jito defined + called + JitoClient imported"
