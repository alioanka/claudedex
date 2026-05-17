#!/usr/bin/env bash
#
# dashboard_smoke.sh — authenticated end-to-end smoke test for the
# session-18 dashboard endpoints. Covers every new field shipped in
# commits b53feb7..HEAD plus the post-recovery fixes.
#
# Usage:
#   scripts/dashboard_smoke.sh [BASE_URL] [USERNAME] [PASSWORD]
#
# Defaults: http://127.0.0.1:8080  admin  admin123
#
# Exit code:
#   0 — every assertion passed
#   1 — at least one assertion failed; check stdout for FAIL lines
#
# Why this script exists: the older quick-script used
#   curl -s -c - -X POST $BASE/login -d 'username=admin&password=admin123'
# which is wrong on 4 counts (per VPS failure 5):
#   * the endpoint is /api/auth/login, not /login
#   * the cookie is named session_id, not session
#   * login expects JSON body, not form-urlencoded
#   * awk-grepping a cookie jar is fragile and breaks once csrf_token shows up
# This script does it correctly: JSON body, cookie jar via -c/-b, no awk.

set -u

BASE="${1:-http://127.0.0.1:8080}"
USER="${2:-admin}"
PASS="${3:-admin123}"

COOKIE_JAR="$(mktemp -t dashboard_smoke.XXXXXX)"
trap 'rm -f "$COOKIE_JAR"' EXIT

FAILED=0

pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; FAILED=$((FAILED + 1)); }

# ---------- 0. Login ----------
echo "=== 0. Login ==="
LOGIN_RESP="$(curl -fsS -c "$COOKIE_JAR" -X POST "$BASE/api/auth/login" \
    -H 'Content-Type: application/json' \
    -d "{\"username\":\"$USER\",\"password\":\"$PASS\"}" 2>&1)" || {
    fail "login HTTP request itself failed: $LOGIN_RESP"
    echo
    echo "Smoke test aborted: cannot authenticate. Make sure the dashboard"
    echo "is reachable at $BASE and admin credentials are correct."
    exit 1
}

if echo "$LOGIN_RESP" | jq -e '.success == true' >/dev/null 2>&1; then
    pass "login returns success:true"
else
    fail "login response shape: $LOGIN_RESP"
    exit 1
fi

# Cookie sanity check — session_id should now be in the jar.
if grep -q 'session_id' "$COOKIE_JAR"; then
    pass "session_id cookie set"
else
    fail "session_id cookie NOT set after login (cookie jar: $(cat "$COOKIE_JAR"))"
    exit 1
fi

# ---------- helper ----------
# All authenticated GETs ride through this so the cookie jar gets reused.
auth_get() { curl -fsS -b "$COOKIE_JAR" "$@"; }

# ---------- 1. /api/bot/status — MODE badge data ----------
echo
echo "=== 1. /api/bot/status (MODE badge: dry_run + mode) ==="
BOT_STATUS="$(auth_get "$BASE/api/bot/status")"
if echo "$BOT_STATUS" | jq -e '.success == true and .data.dry_run == true' >/dev/null 2>&1; then
    pass "success:true, data.dry_run:true (matches DRY_RUN=true in .env)"
else
    fail "expected success:true + data.dry_run:true. Got: $(echo "$BOT_STATUS" | jq -c '{success, dry_run: .data.dry_run, mode: .data.mode}')"
fi
if echo "$BOT_STATUS" | jq -e '.data.mode' >/dev/null 2>&1; then
    pass "data.mode field present (= $(echo "$BOT_STATUS" | jq -r '.data.mode'))"
else
    fail "data.mode field missing"
fi

# ---------- 2. /api/copytrading/stats — live + simulated split ----------
echo
echo "=== 2. /api/copytrading/stats (DASH-Q-04 live vs simulated split) ==="
CT_STATS="$(auth_get "$BASE/api/copytrading/stats")"
for k in total_trades total_pnl live_trades simulated_trades live_pnl live_win_rate; do
    if echo "$CT_STATS" | jq -e ".stats.\"$k\" != null" >/dev/null 2>&1; then
        pass "stats.$k present"
    else
        fail "stats.$k missing — got: $(echo "$CT_STATS" | jq -c '.stats | keys')"
    fi
done

# ---------- 3. /api/sniper/stats — cap fallback ----------
echo
echo "=== 3. /api/sniper/stats (cap fallback, VPS failure 3) ==="
SNIPER_STATS="$(auth_get "$BASE/api/sniper/stats")"
# active_positions_effective + max_active_positions must BOTH be present
# AND max_active_positions must be > 0 even right after restart (because
# the post-recovery patch falls back to config_settings or 500).
for k in active_positions active_positions_effective max_active_positions status; do
    if echo "$SNIPER_STATS" | jq -e ". | has(\"$k\")" >/dev/null 2>&1; then
        pass "$k present"
    else
        fail "$k missing"
    fi
done
MAX_ACTIVE="$(echo "$SNIPER_STATS" | jq -r '.max_active_positions')"
if [ "$MAX_ACTIVE" != "null" ] && [ "$MAX_ACTIVE" != "0" ] && [ -n "$MAX_ACTIVE" ]; then
    pass "max_active_positions > 0 ($MAX_ACTIVE) — fallback path working"
else
    fail "max_active_positions is 0/null — cap tile will show 'Active: N' with no /cap"
fi

# ---------- 4. /api/analytics/risk/dex — VaR + CVaR ----------
echo
echo "=== 4. /api/analytics/risk/dex (DASH-Q-12 var_99 + cvar_95) ==="
RISK="$(auth_get "$BASE/api/analytics/risk/dex")"
for k in var_95 var_99 cvar_95 annual_volatility; do
    if echo "$RISK" | jq -e ".data.\"$k\" != null" >/dev/null 2>&1; then
        pass "data.$k present"
    else
        fail "data.$k missing — got: $(echo "$RISK" | jq -c '.data | keys')"
    fi
done

# ---------- 5. /api/arbitrage/stats — honest status ----------
echo
echo "=== 5. /api/arbitrage/stats (honest status string) ==="
ARB_STATS="$(auth_get "$BASE/api/arbitrage/stats")"
if echo "$ARB_STATS" | jq -e '.status' >/dev/null 2>&1; then
    pass "status field present (= $(echo "$ARB_STATS" | jq -r '.status'))"
else
    fail "status field missing"
fi

# ---------- 6. /health canary ----------
echo
echo "=== 6. /health canary ==="
H="$(auth_get "$BASE/health")"
if echo "$H" | jq -e '.ok == true' >/dev/null 2>&1; then
    pass "ok:true"
else
    fail "ok != true. Got: $H"
fi

# ---------- 7. CSRF round-trip ----------
echo
echo "=== 7. CSRF round-trip (VPS failure 1) ==="
# After any GET, the server sets csrf_token cookie. Sending a state-changing
# request without the X-CSRF-Token header should 403. Sending it WITH the
# header (echoed from the cookie) should NOT 403. We use a benign endpoint
# (/api/auth/logout's POST — but we don't actually want to log out yet, so
# we test a no-op-ish endpoint instead: try a CSRF-required POST and assert
# the 403 path actually fires when the header is missing).
CSRF="$(awk '$6=="csrf_token"{print $7}' "$COOKIE_JAR")"
if [ -n "$CSRF" ]; then
    pass "csrf_token cookie present in jar"
else
    fail "csrf_token cookie missing — CSRF middleware never issued one"
fi

# Send a POST WITHOUT the CSRF header — must 403.
HTTP_NO_HDR="$(curl -s -o /dev/null -w '%{http_code}' -b "$COOKIE_JAR" -X POST "$BASE/api/settings/update" \
    -H 'Content-Type: application/json' -d '{"config_type":"x","updates":{}}')"
if [ "$HTTP_NO_HDR" = "403" ]; then
    pass "POST without X-CSRF-Token returns 403 (middleware enforcing)"
else
    fail "POST without X-CSRF-Token returned $HTTP_NO_HDR (expected 403)"
fi

# Send the SAME POST WITH the CSRF header — must NOT 403. (200 means the
# endpoint accepted it; 4xx other than 403 means it got past CSRF and the
# server then rejected on its own; either way CSRF is working.)
HTTP_WITH_HDR="$(curl -s -o /dev/null -w '%{http_code}' -b "$COOKIE_JAR" -X POST "$BASE/api/settings/update" \
    -H 'Content-Type: application/json' \
    -H "X-CSRF-Token: $CSRF" \
    -d '{"config_type":"smoke_test_noop","updates":{}}')"
if [ "$HTTP_WITH_HDR" != "403" ]; then
    pass "POST WITH X-CSRF-Token did not 403 (got $HTTP_WITH_HDR)"
else
    fail "POST WITH X-CSRF-Token still 403 — CSRF check rejecting valid token"
fi

# ---------- 8. /__routes__ count ----------
echo
echo "=== 8. /__routes__ ==="
RC="$(auth_get "$BASE/__routes__" | jq -r '.routes | length')"
if [ -n "$RC" ] && [ "$RC" -ge 470 ]; then
    pass "route count $RC (>= 470 baseline)"
else
    fail "route count $RC < 470 baseline"
fi

# ---------- summary ----------
echo
if [ "$FAILED" -eq 0 ]; then
    echo "RESULT: PASS — all assertions held"
    exit 0
else
    echo "RESULT: FAIL — $FAILED assertion(s) failed"
    exit 1
fi
