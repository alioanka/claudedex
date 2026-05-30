#!/usr/bin/env bash
# COPY reconcile CSRF header presence check
# (catalog: script_reconcile_csrf_header; W6 commit 5a857e0).
#
# Wave-6 fixed the operator-reported "Reconcile button returns 403"
# bug: dashboard_copytrading.html previously POSTed to
# /api/copytrading/reconcile without the X-CSRF-Token header, and
# the CSRF middleware rejected the request. The fix wires the call
# through window.withCsrfHeaders('POST') so the token is attached
# automatically.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"

f=dashboard/templates/dashboard_copytrading.html
if [ ! -f "$f" ]; then
  echo "FAIL — $f not found"
  exit 1
fi

# The withCsrfHeaders('POST') call must appear inside the
# reconcileTrades / /api/copytrading/reconcile block. We grep for the
# co-occurrence in a 25-line window centred on `reconcile`.
if ! awk '
  /reconcile/ { in_block=1; lines=0 }
  in_block { lines++; if (/withCsrfHeaders\(.?POST.?\)/) { print "found"; exit 0 } if (lines>25) in_block=0 }
' "$f" | grep -q found; then
  echo "FAIL — withCsrfHeaders('POST') not found near 'reconcile' in $f"
  exit 1
fi

echo "PASS — withCsrfHeaders('POST') wired to /api/copytrading/reconcile call in $f"
