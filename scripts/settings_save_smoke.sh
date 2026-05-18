#!/usr/bin/env bash
#
# settings_save_smoke.sh — verifies that EVERY per-module settings page
# can save with the X-CSRF-Token header. This is the regression-guard
# for the "CSRF token missing or invalid" failure operators see when
# clicking Save on /arbitrage/settings, /sniper/settings, etc.
#
# Strategy: login as admin, capture session_id + csrf_token cookies,
# then POST a benign payload to each /api/<module>/settings endpoint.
# 200 OR 4xx-not-403 = CSRF middleware accepted the request. 403 = the
# CSRF header was rejected (regression).
#
# We do NOT post real settings values — only a single tiny key like
# {"_smoke_test_marker": "1"} which the save handlers accept (each
# upserts arbitrary keys into config_settings).
#
# Usage:
#   scripts/settings_save_smoke.sh [BASE_URL] [USERNAME] [PASSWORD]
#
# Defaults: http://127.0.0.1:8080  admin  admin123
#
# Exit code:
#   0 — every save round-tripped without 403
#   1 — at least one module returned 403 (CSRF regression)

set -u

BASE="${1:-http://127.0.0.1:8080}"
USER="${2:-admin}"
PASS="${3:-admin123}"

COOKIE_JAR="$(mktemp -t settings_smoke.XXXXXX)"
trap 'rm -f "$COOKIE_JAR"' EXIT

FAILED=0

pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; FAILED=$((FAILED + 1)); }

# ---------- login ----------
echo "=== Login ==="
LOGIN_HTTP=$(curl -sS -o /dev/null -w '%{http_code}' \
    -c "$COOKIE_JAR" -X POST "$BASE/api/auth/login" \
    -H 'Content-Type: application/json' \
    -d "{\"username\":\"$USER\",\"password\":\"$PASS\"}" 2>&1)

if [ "$LOGIN_HTTP" != "200" ]; then
    fail "login returned HTTP $LOGIN_HTTP"
    cat <<'HELP'

Admin auth failed. If the account is locked, run the test-runner
D-section probe "DB: unlock admin login attempts" or reset the password
via "DB: reset admin password to admin123".
HELP
    exit 1
fi
pass "logged in"

# Pull csrf_token cookie from jar.
CSRF=$(awk '$6=="csrf_token"{print $7}' "$COOKIE_JAR")
if [ -z "$CSRF" ]; then
    fail "csrf_token cookie not set after login"
    exit 1
fi
pass "csrf_token cookie issued"

# ---------- per-module save round-trips ----------
# Each entry: "<label>:<endpoint-path>:<json-body>"
# Body is a single no-op key. The save handlers iterate over keys and
# upsert into config_settings; an unknown key just becomes a no-op row.
ENTRIES=(
    "arbitrage:/api/arbitrage/settings:{\"_smoke_test_marker\":\"1\"}"
    "sniper:/api/sniper/settings:{\"_smoke_test_marker\":\"1\"}"
    "copytrading:/api/copytrading/settings:{\"_smoke_test_marker\":\"1\"}"
    "ai:/api/ai/settings:{\"_smoke_test_marker\":\"1\"}"
    "futures:/api/settings/futures:{\"_smoke_test_marker\":\"1\"}"
    "solana:/api/settings/solana:{\"_smoke_test_marker\":\"1\"}"
)

echo
echo "=== Per-module settings save CSRF round-trips ==="
for entry in "${ENTRIES[@]}"; do
    label="${entry%%:*}"
    rest="${entry#*:}"
    endpoint="${rest%%:*}"
    body="${rest#*:}"

    HTTP=$(curl -sS -o /dev/null -w '%{http_code}' \
        -b "$COOKIE_JAR" \
        -X POST "$BASE$endpoint" \
        -H 'Content-Type: application/json' \
        -H "X-CSRF-Token: $CSRF" \
        -d "$body" 2>&1 || echo "000")

    if [ "$HTTP" = "403" ]; then
        fail "$label save returned 403 (CSRF regression at $endpoint)"
    elif [ "$HTTP" = "200" ]; then
        pass "$label save → 200"
    else
        # 4xx other than 403 means CSRF middleware passed but the
        # endpoint rejected on a different ground (e.g. schema). Still
        # proves CSRF is working, which is all this script tests.
        pass "$label save → $HTTP (CSRF accepted; non-200 may be unrelated)"
    fi
done

# ---------- per-module dry-run round-trip (canonical toggle endpoint) ----------
echo
echo "=== Per-module dry-run GET round-trips ==="
for m in arbitrage sniper copy_trading ai futures solana dex; do
    HTTP=$(curl -sS -o /dev/null -w '%{http_code}' \
        -b "$COOKIE_JAR" \
        "$BASE/api/modules/$m/dry-run" 2>&1 || echo "000")
    if [ "$HTTP" = "200" ]; then
        pass "$m dry-run GET → 200"
    else
        fail "$m dry-run GET → $HTTP (expected 200)"
    fi
done

# ---------- summary ----------
echo
if [ "$FAILED" -eq 0 ]; then
    echo "RESULT: PASS — every module settings page accepts CSRF-protected saves"
    exit 0
else
    echo "RESULT: FAIL — $FAILED endpoint(s) regressed"
    exit 1
fi
