#!/usr/bin/env bash
# COPY close-button dual-table fallback presence check
# (catalog: script_close_button_endpoint_dual_table; W6 commit 5a857e0).
#
# Wave-6 fixed the operator-reported "close button does nothing" bug.
# The _process_close_flag_files path looked at copytrading_positions
# only; rows that lived ONLY in copytrading_trades (the legacy
# soft-delete pattern) silently ignored the flag-file close request.
# The fix adds a `from_trades_table` flag that escalates the lookup
# to copytrading_trades when the positions table comes up empty.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=modules/copy_trading/copy_engine.py
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

# Sanity: the helper must exist.
if ! grep -nE 'async def _process_close_flag_files' "$f" >/dev/null; then
  echo "FAIL — _process_close_flag_files definition missing from $f"
  exit 1
fi

# The dual-table fallback marker.
if ! grep -nF 'from_trades_table' "$f" >/dev/null; then
  echo "FAIL — from_trades_table fallback marker missing from $f"
  exit 1
fi

# Belt-and-braces: copytrading_trades must be referenced from the helper.
if ! grep -nF 'copytrading_trades' "$f" >/dev/null; then
  echo "FAIL — copytrading_trades reference missing from $f"
  exit 1
fi

echo "PASS — _process_close_flag_files + from_trades_table fallback + copytrading_trades references present"
