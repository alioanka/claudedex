#!/usr/bin/env bash
# AI main_ai secrets-after-db-pool ordering check
# (catalog: script_ai_secrets_after_db_pool; W6 commit cca8d94).
#
# Wave-6 root cause for "0 trades despite signals": main_ai.py was
# calling secrets.get(...) BEFORE asyncpg.create_pool(...), so the
# secrets_manager stayed in bootstrap mode and never reached the
# DB-encrypted credentials. ai_provider therefore initialized with
# empty API keys, then the sentiment loop skipped every tick.
#
# This probe asserts:
#   1. modules/ai_analysis/main_ai.py exists.
#   2. asyncpg.create_pool(...) appears BEFORE the FIRST
#      `await _secrets.get_async(` or `secrets.get(` call.
#   3. _secrets.initialize(db_pool) is invoked between the two.
# A regression that re-orders these calls fails this script.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=modules/ai_analysis/main_ai.py
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

# Skip comment lines so a docstring referencing the old ordering can't
# spoof the first hit.
pool_line=$(grep -nE '^[^#]*asyncpg\.create_pool\(' "$f" | head -1 | cut -d: -f1 || true)
init_line=$(grep -nE '^[^#]*_secrets\.initialize\(db_pool\)' "$f" | head -1 | cut -d: -f1 || true)
get_line=$(grep -nE '^[^#]*(_secrets\.get_async|secrets\.get_async|_secrets\.get|secrets\.get)\(' "$f" | head -1 | cut -d: -f1 || true)

missing=()
[ -z "$pool_line" ] && missing+=("no asyncpg.create_pool(...) call")
[ -z "$init_line" ] && missing+=("no _secrets.initialize(db_pool) call")
[ -z "$get_line" ]  && missing+=("no secrets.get / get_async call")

if [ ${#missing[@]} -ne 0 ]; then
  echo "FAIL — main_ai.py missing expected calls:"
  for m in "${missing[@]}"; do echo "  - $m"; done
  exit 1
fi

if [ "$pool_line" -ge "$get_line" ] || [ "$init_line" -ge "$get_line" ] \
   || [ "$pool_line" -ge "$init_line" ]; then
  echo "FAIL — ordering wrong in $f:"
  echo "  asyncpg.create_pool      line $pool_line"
  echo "  _secrets.initialize      line $init_line"
  echo "  secrets.get / get_async  line $get_line"
  echo "Expected: create_pool < initialize(db_pool) < secrets.get_async(...)"
  exit 1
fi

echo "PASS — main_ai.py ordering: create_pool ($pool_line) < initialize ($init_line) < get_async ($get_line)"
