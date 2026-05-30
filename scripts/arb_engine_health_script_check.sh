#!/usr/bin/env bash
# ARB engine-health diagnostic script presence check
# (catalog: script_arb_engine_health_script_present; W6 commit e32cf63).
#
# Wave-6 added scripts/arb_engine_health.py — a read-only operator
# diagnostic that surfaces arbitrage_runtime_stats freshness, the
# last arbitrage_trades row, kill/pause/restart flag files, and the
# tail of arbitrage_errors.log. Suggests the next operator action
# (restart via touch logs/.restart_arbitrage) when the engine is dead.
#
# This probe asserts:
#   1. The script exists on disk.
#   2. It carries an executable bit.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=scripts/arb_engine_health.py
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

if [ ! -x "$f" ]; then
  echo "FAIL — $f present but not executable (chmod +x missing)"
  exit 1
fi

# Sanity: the script must mention arbitrage_runtime_stats — its raison d'etre.
if ! grep -nF 'arbitrage_runtime_stats' "$f" >/dev/null; then
  echo "FAIL — $f exists but does not query arbitrage_runtime_stats"
  exit 1
fi

echo "PASS — scripts/arb_engine_health.py present, executable, queries arbitrage_runtime_stats"
