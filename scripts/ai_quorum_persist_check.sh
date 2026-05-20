#!/usr/bin/env bash
# AI quorum-outcome persistence helper presence check
# (catalog: script_ai_quorum_persist_present; wave-4 commit c7e4a27).
#
# The wave-4 fix landed `_record_quorum_outcome` + `_persist_quorum_outcome`
# in modules/ai_analysis/core/sentiment_engine.py so every quorum vote
# (agreement / disagreement / collapsed-to-zero) ends up as a row in
# ai_feature_store with metadata.quorum_outcome set. The
# /api/ai/quorum-metrics endpoint reads from that surface. This script
# grep-asserts both functions exist AND that the persist helper is
# invoked from the engine tick.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=modules/ai_analysis/core/sentiment_engine.py
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

if ! grep -nE 'async def _persist_quorum_outcome' "$f" >/dev/null; then
  echo "FAIL — _persist_quorum_outcome definition missing from $f"
  exit 1
fi

if ! grep -nE '_record_quorum_outcome' "$f" >/dev/null; then
  echo "FAIL — _record_quorum_outcome reference missing from $f"
  exit 1
fi

# Verify the persist helper is awaited from the tick path.
if ! grep -nE 'await self\._persist_quorum_outcome\(' "$f" >/dev/null; then
  echo "FAIL — no await callsite for _persist_quorum_outcome in $f"
  exit 1
fi

echo "PASS — _persist_quorum_outcome + _record_quorum_outcome defined + callsite present"
