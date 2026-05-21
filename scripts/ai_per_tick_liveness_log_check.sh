#!/usr/bin/env bash
# AI per-tick liveness log presence check
# (catalog: script_ai_per_tick_liveness_log; W6 commit b675ab1).
#
# The Wave-6 sentiment_engine fix emits a per-cycle liveness log line
# at the top of every run() iteration so an operator can verify the
# subprocess is actually ticking (vs alive-but-stalled). The log line
# is the substring `sentiment cycle tick` followed by counters; a
# refactor that drops the line silently re-introduces the "ai looks
# alive but no trades" debugging dead end.
#
# This probe asserts the literal log marker is present in
# modules/ai_analysis/core/sentiment_engine.py.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=modules/ai_analysis/core/sentiment_engine.py
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

if ! grep -nF 'sentiment cycle tick' "$f" >/dev/null; then
  echo "FAIL — per-tick liveness log 'sentiment cycle tick' missing from $f"
  exit 1
fi

# Also confirm _last_tick_at is stamped — the subprocess_health surface
# (api_ai_diagnostics_subprocess_health) depends on it.
if ! grep -nE 'self\._last_tick_at\s*=' "$f" >/dev/null; then
  echo "FAIL — self._last_tick_at assignment missing from $f"
  exit 1
fi

echo "PASS — 'sentiment cycle tick' log + _last_tick_at heartbeat present in sentiment_engine.py"
