#!/usr/bin/env bash
# FUTURES FUT-RM-07b emergency-close Telegram notify helper presence
# check (catalog: script_futures_fut_rm_07b_notify_helper;
# wave-4 commits 34e6c95 / 6f66608).
#
# FUT-RM-07b wires a `priority="critical"` Telegram payload from the
# FUT-RM-07 verify path when a fill came back CROSS-margin instead of
# ISOLATED. The dispatcher lives in
# modules/futures_trading/core/futures_engine.py as
# `_notify_fut_rm_07_emergency_close`. This script grep-asserts the
# helper is defined AND that it's invoked from the verify-close path,
# so a refactor that silently drops the alert is caught immediately.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=modules/futures_trading/core/futures_engine.py
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

if ! grep -nE 'async def _notify_fut_rm_07_emergency_close' "$f" >/dev/null; then
  echo "FAIL — _notify_fut_rm_07_emergency_close definition missing from $f"
  exit 1
fi

# Verify the helper is actually CALLED (definition without a callsite
# would silently swallow every alert).
if ! grep -nE 'await self\._notify_fut_rm_07_emergency_close\(' "$f" >/dev/null; then
  echo "FAIL — no await callsite for _notify_fut_rm_07_emergency_close in $f"
  exit 1
fi

# Telegram flag gating must also be present.
if ! grep -nE 'telegram_emergency_close_enabled' "$f" >/dev/null; then
  echo "FAIL — telegram_emergency_close_enabled flag not referenced in $f"
  exit 1
fi

echo "PASS — _notify_fut_rm_07_emergency_close defined + called + flag-gated"
