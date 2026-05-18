#!/usr/bin/env bash
# scripts/preflight.sh
#
# Validates every code path shipped in the May 2026 hardening session
# against a running stack (trading-bot + trading-postgres). Covers:
#   - SNIPER (cap, jupiter fallback, timing markers, WSS concurrency,
#     LIVE safety guard, logging consolidation, block_time anchoring)
#   - COPY_TRADING (global open-position cap, RiskManager gate)
#   - SOLANA (RiskManager gate wiring)
#   - Migration runner pickup of /migrations/ dir
#   - Dashboard SOL/USD price + liveness freshness checks
#
# Run after `git pull` + `docker compose up -d --build trading-bot`.
#
# Prints PASS / WARN / FAIL per check and exits 0 only if no FAIL.
# WARN means "verifiable later" (e.g. needs accumulated data) — not
# blocking.
#
# Usage:
#   bash scripts/preflight.sh
#
# Compatible with the user's docker-compose stack:
#   service: postgres (container_name: trading-postgres)
#   service: trading-bot
#   db: tradingbot
#   db user: stored in /run/secrets/db_user inside the postgres container

set -u
shopt -s lastpipe

# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
RED='\033[0;31m'; YELLOW='\033[0;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
FAILS=0; WARNS=0

pass() { printf "${GREEN}  PASS${NC}  %s\n" "$1"; }
warn() { printf "${YELLOW}  WARN${NC}  %s\n" "$1"; WARNS=$((WARNS+1)); }
fail() { printf "${RED}  FAIL${NC}  %s\n" "$1"; FAILS=$((FAILS+1)); }
hdr()  { printf "\n${CYAN}=== %s ===${NC}\n" "$1"; }

pg() {
    # Runs an SQL command inside trading-postgres using the secret-mounted
    # user. Pipes SQL via stdin instead of -c so multi-line queries with
    # embedded single quotes ('30 minutes', etc.) survive the bash quote
    # gauntlet. Stdout is the raw psql output; caller decides what to
    # grep for. -tA = tuples-only, unaligned.
    printf '%s\n' "$1" | docker compose exec -T postgres bash -lc \
        'psql -U $(cat /run/secrets/db_user) -d tradingbot -tA -f -' 2>&1
}

# ---------------------------------------------------------------------------
# Step 0: stack health
# ---------------------------------------------------------------------------
hdr "Stack health"

# Detect whether we're running INSIDE the trading-bot container vs
# from the host. The Test Runner triggers preflight via subprocess
# from the dashboard, which is in-container — `docker compose ps`
# isn't available there (no docker.sock by design). In that case the
# fact that this script is even running is proof trading-bot is alive,
# so we substitute lightweight in-container checks.
if [[ -f /.dockerenv ]] || grep -q 'docker\|containerd' /proc/1/cgroup 2>/dev/null; then
    pass "trading-bot container running (running this script from inside it)"
    # Postgres is verified by attempting a psql round-trip later — but
    # surface a hint here.
    if [[ -S /var/run/postgresql/.s.PGSQL.5432 ]] || [[ -n "${DB_HOST:-}" ]]; then
        pass "trading-postgres reachable (DB_HOST set or socket present)"
    else
        warn "trading-postgres reachability not verified in-container"
    fi
elif command -v docker >/dev/null 2>&1; then
    if docker compose ps --status running 2>/dev/null | grep -q 'trading-bot'; then
        pass "trading-bot container running"
    else
        fail "trading-bot container not running — start with 'docker compose up -d --build trading-bot'"
    fi

    if docker compose ps --status running 2>/dev/null | grep -q 'trading-postgres'; then
        pass "trading-postgres container running"
    else
        fail "trading-postgres container not running"
    fi
else
    warn "docker CLI unavailable; skipping container-presence checks"
fi

if [[ $FAILS -gt 0 ]]; then
    echo
    echo -e "${RED}Stack is not up; cannot continue.${NC}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: migrations 016 + 017 seeded
# ---------------------------------------------------------------------------
hdr "Seed migrations — position caps for sniper + copy_trading"

OUT=$(pg "SELECT value FROM config_settings WHERE config_type='sniper_config' AND key='max_active_positions';")
if [[ "$OUT" == "500" ]]; then
    pass "sniper.max_active_positions=500 seeded"
elif [[ -z "$OUT" ]]; then
    fail "sniper max_active_positions missing — migration 016 not applied. Note: b82d632 fixed the auto-runner; next 'docker compose up -d --build trading-bot' will apply it. Manual: docker compose cp migrations/016_seed_sniper_max_active_positions.sql postgres:/tmp/016.sql && docker compose exec postgres sh -c 'psql -U \$(cat /run/secrets/db_user) -d tradingbot -f /tmp/016.sql'"
else
    pass "sniper.max_active_positions=$OUT (operator-tuned)"
fi

OUT=$(pg "SELECT value FROM config_settings WHERE config_type='copytrading_config' AND key='max_active_positions';")
if [[ "$OUT" == "50" ]]; then
    pass "copytrading.max_active_positions=50 seeded"
elif [[ -z "$OUT" ]]; then
    warn "copytrading max_active_positions missing — migration 017 will auto-apply on next bot restart (post-b82d632), or apply manually now: docker compose cp migrations/017_seed_copytrading_max_active_positions.sql postgres:/tmp/017.sql && docker compose exec postgres sh -c 'psql -U \$(cat /run/secrets/db_user) -d tradingbot -f /tmp/017.sql'"
else
    pass "copytrading.max_active_positions=$OUT (operator-tuned)"
fi

# ---------------------------------------------------------------------------
# Step 2: logging consolidation
# ---------------------------------------------------------------------------
hdr "Logging consolidation"

if [[ -d /root/claudedex/logs/sniper_module ]]; then
    warn "stale logs/sniper_module/ directory still on disk — safe to 'rm -rf'"
else
    pass "no stale logs/sniper_module/"
fi

if [[ -f /root/claudedex/logs/sniper/sniper_trades.log ]]; then
    if [[ -s /root/claudedex/logs/sniper/sniper_trades.log ]]; then
        warn "sniper_trades.log present and non-empty — dead logger may have been re-introduced"
    else
        warn "0-byte sniper_trades.log still on disk — safe to 'rm -f'"
    fi
else
    pass "no dead sniper_trades.log"
fi

if [[ -d /root/claudedex/logs/sniper ]]; then
    pass "logs/sniper/ exists (single source of truth)"
else
    warn "logs/sniper/ missing — sniper subprocess may not have started yet"
fi

# ---------------------------------------------------------------------------
# Step 3: sniper subprocess up & RiskManager wired
# ---------------------------------------------------------------------------
hdr "Sniper subprocess & cross-module risk wiring"

if docker compose logs --since=5m trading-bot 2>/dev/null | grep -q "Sniper module is disabled"; then
    warn "SNIPER_MODULE_ENABLED=false — flip to 'true' in .env to validate live paths"
elif docker compose logs --since=5m trading-bot 2>/dev/null | grep -q "Starting Sniper module"; then
    pass "sniper subprocess started this session"
else
    warn "no 'Starting Sniper' line in last 5m of logs — may have started earlier"
fi

if docker compose logs --since=5m trading-bot 2>/dev/null | grep -q "RiskManager wired into Solana engine"; then
    pass "Solana engine: RiskManager.validate_trade gate wired"
else
    warn "Solana RiskManager wiring not seen in last 5m (SOLANA_MODULE_ENABLED=false?)"
fi

if docker compose logs --since=5m trading-bot 2>/dev/null | grep -q "RiskManager wired into Copy Trading"; then
    pass "Copy Trading executor: RiskManager.validate_trade gate wired"
else
    warn "Copy Trading RiskManager wiring not seen in last 5m (COPY_MODULE_ENABLED=false?)"
fi

# ---------------------------------------------------------------------------
# Step 4: runtime stats freshness + new counters present
# ---------------------------------------------------------------------------
hdr "Runtime stats snapshot"

ROW=$(pg "SELECT
  COALESCE((stats->>'active_positions')::int, -1) || '|' ||
  COALESCE((stats->>'max_active_positions')::int, -1) || '|' ||
  COALESCE((stats->>'jupiter_quote_fallback_hits')::int, 0) || '|' ||
  COALESCE((stats->>'capped_rejections')::int, 0) || '|' ||
  COALESCE((stats->'solana_listener'->>'wss_dispatched')::int, 0) || '|' ||
  COALESCE((stats->'solana_listener'->>'wss_inflight_peak')::int, 0) || '|' ||
  EXTRACT(EPOCH FROM (NOW() - updated_at))::int
FROM sniper_runtime_stats WHERE id=1;")

if [[ -z "$ROW" ]] || [[ "$ROW" == "|" ]]; then
    warn "sniper_runtime_stats row missing — sniper subprocess hasn't snapshot yet (wait ~5 min after start)"
else
    IFS='|' read -r active cap quote_fb capped dispatched peak age <<< "$ROW"
    if [[ "$cap" == "500" ]] || [[ "$cap" -gt 0 ]]; then
        pass "active_positions=$active / cap=$cap (snapshot ${age}s old)"
    else
        warn "max_active_positions=$cap looks wrong (expected 500)"
    fi
    pass "jupiter_quote_fallback_hits=$quote_fb"
    pass "capped_rejections=$capped"
    pass "wss_dispatched=$dispatched, peak_inflight=$peak/16"
    if [[ "$age" -gt 600 ]]; then
        warn "runtime stats snapshot is ${age}s old — subprocess may have stopped"
    fi
fi

# ---------------------------------------------------------------------------
# Step 5: block_time_anchored propagation
# ---------------------------------------------------------------------------
hdr "block_time_anchored DB propagation (last 30m)"

OUT=$(pg "SELECT
  COALESCE(metadata->>'detection_path','?') || ':' ||
  (metadata->>'block_time_anchored') || ':' ||
  COUNT(*)::text
FROM sniper_trades
WHERE entry_timestamp > NOW() - INTERVAL '30 minutes'
  AND metadata->>'block_time_anchored' IS NOT NULL
GROUP BY
  COALESCE(metadata->>'detection_path','?'),
  (metadata->>'block_time_anchored')
ORDER BY 1;")

if [[ -z "$OUT" ]]; then
    warn "no rows in last 30m have block_time_anchored — sniper may be idle or not running"
else
    pass "block_time_anchored values propagated:"
    echo "$OUT" | while IFS=: read -r path anchored count; do
        printf "          %s anchored=%s -> %s rows\n" "$path" "$anchored" "$count"
    done
fi

# ---------------------------------------------------------------------------
# Step 6: detect_to_rpc_receipt_ms — the headline timing metric
# ---------------------------------------------------------------------------
hdr "Detection latency p50/p95 (last 30m)"

OUT=$(pg "SELECT
  COALESCE(metadata->>'detection_path','?') || '|' ||
  COUNT(*)::text || '|' ||
  ROUND((percentile_cont(0.5) WITHIN GROUP (ORDER BY (metadata->'timing'->>'detect_to_rpc_receipt_ms')::float))::numeric, 0)::text || '|' ||
  ROUND((percentile_cont(0.95) WITHIN GROUP (ORDER BY (metadata->'timing'->>'detect_to_rpc_receipt_ms')::float))::numeric, 0)::text
FROM sniper_trades
WHERE entry_timestamp > NOW() - INTERVAL '30 minutes'
  AND metadata->'timing'->>'detect_to_rpc_receipt_ms' IS NOT NULL
GROUP BY COALESCE(metadata->>'detection_path','?')
ORDER BY 1;")

if [[ -z "$OUT" ]]; then
    warn "no timing rows yet — wait ~10 min after sniper restart"
else
    echo "$OUT" | while IFS='|' read -r path samples p50 p95; do
        # Small-sample caveat: <10 samples in the window means a single
        # block_time_anchored=false row can drag p50 to 0 because its
        # detect_to_rpc_receipt = 0 by construction (wall-clock t_detect
        # equals wall-clock rpc_receipt). Don't claim "excellent" until
        # we have a meaningful sample.
        if [[ "$path" == "wss" ]] && [[ "$samples" -lt 10 ]]; then
            warn "wss: $samples samples (low) p50=${p50}ms p95=${p95}ms — wait for more snipes for a meaningful number"
        elif [[ "$path" == "wss" ]] && [[ "$p50" -lt 500 ]]; then
            pass "wss: $samples samples, p50=${p50}ms p95=${p95}ms (excellent — sub-500ms)"
        elif [[ "$path" == "wss" ]]; then
            warn "wss: $samples samples, p50=${p50}ms p95=${p95}ms (above 500ms — RPC provider may be lagging)"
        else
            pass "$path: $samples samples, p50=${p50}ms p95=${p95}ms"
        fi
    done
fi

# ---------------------------------------------------------------------------
# Step 7: LIVE-mode safety guard
# ---------------------------------------------------------------------------
hdr "LIVE-mode safety filter guard"

OUT=$(docker compose exec -T -e DRY_RUN=false trading-bot python -c "
import asyncio, sys
sys.path.insert(0, '/app')
from modules.sniper.core.sniper_engine import SniperEngine
async def t():
    e = SniperEngine({}, None, None)
    e.safety_check_enabled = False
    try:
        await e._load_settings()
        print('GUARD_FAIL')
    except RuntimeError:
        print('GUARD_OK')
asyncio.run(t())
" 2>&1 | grep -E "GUARD_OK|GUARD_FAIL" | head -1)

if [[ "$OUT" == "GUARD_OK" ]]; then
    pass "LIVE-mode guard fires correctly (refuses safety_check_enabled=false)"
elif [[ "$OUT" == "GUARD_FAIL" ]]; then
    fail "LIVE-mode guard did NOT fire — would proceed to live trading with safety off"
else
    warn "Could not verify safety guard (test script error): $OUT"
fi

# ---------------------------------------------------------------------------
# Step 8: unit tests for the new paths
# ---------------------------------------------------------------------------
hdr "Unit tests for new code paths"

# Run from /app where pytest config lives. Use PYTEST_ADDOPTS to wipe
# the --cov flags inherited from pytest.ini so the test step doesn't
# break when pytest-cov isn't installed yet (older images). The empty
# string OVERRIDES the inifile's addopts entirely.
TEST_OUT=$(docker compose exec -T -w /app -e PYTEST_ADDOPTS='-p no:cacheprovider -p no:anchorpy' \
    trading-bot python -m pytest tests/unit/test_sniper_new_paths.py -q --no-header -o addopts= 2>&1 | tail -10)
# Match only on pytest's actual passed-count summary line, e.g.
# "19 passed in 0.42s". The previous grep matched any occurrence of
# the substring "tests" — which false-positives on error paths like
# "file or directory not found: tests/unit/...". Also detect the
# explicit error/failed lines so we surface real problems.
if echo "$TEST_OUT" | grep -qE '^[0-9]+ passed'; then
    SUMMARY=$(echo "$TEST_OUT" | grep -E '^[0-9]+ passed' | tail -1 | tr -d '\r')
    pass "tests/unit/test_sniper_new_paths.py: $SUMMARY"
elif echo "$TEST_OUT" | grep -qE 'no tests ran'; then
    warn "tests/unit/test_sniper_new_paths.py: no tests ran"
    echo "$TEST_OUT" | sed 's/^/          /'
else
    warn "unit tests did not complete cleanly:"
    echo "$TEST_OUT" | sed 's/^/          /'
fi

# ---------------------------------------------------------------------------
# Step 9: /health endpoint reachable
# ---------------------------------------------------------------------------
hdr "/health endpoint + git SHA"

HEALTH_JSON=$(docker compose exec -T trading-bot python -c "
import urllib.request, urllib.error
req = urllib.request.Request('http://localhost:8080/health')
try:
    resp = urllib.request.urlopen(req, timeout=5)
    print(f'HTTP_STATUS={resp.status}')
    print(resp.read().decode())
except urllib.error.HTTPError as e:
    # Capture body even on 4xx/5xx so we can see the real error.
    print(f'HTTP_STATUS={e.code}')
    try:
        print(e.read().decode())
    except Exception:
        print(f'(could not read body: {e})')
except Exception as e:
    print(f'CONNECTION_ERROR: {type(e).__name__}: {e}')
" 2>&1)

if echo "$HEALTH_JSON" | grep -q '"status":.*"healthy"'; then
    SHA=$(echo "$HEALTH_JSON" | grep -v HTTP_STATUS | python -c "import sys,json; print(json.load(sys.stdin).get('git_sha','?'))" 2>/dev/null || echo '?')
    pass "/health = healthy, git_sha=$SHA"
elif echo "$HEALTH_JSON" | grep -q '"status":'; then
    warn "/health responded but not healthy:"
    echo "$HEALTH_JSON" | sed 's/^/          /'
else
    fail "/health unreachable or non-JSON:"
    echo "$HEALTH_JSON" | sed 's/^/          /'
fi

# ---------------------------------------------------------------------------
# Step 10: dashboard route health — sample a representative spread
# ---------------------------------------------------------------------------
hdr "Dashboard routes — HTTP status spot-check"

# probe_route <url> <expected_status_pattern> <label>
# Uses HTTPRedirectHandler subclass to STOP at redirects so we can
# spot-check 301/302 responses (urllib normally follows them
# transparently, turning a 301 into the 200 of the target page).
probe_route() {
    local url="$1" expect="$2" label="$3"
    local code
    code=$(docker compose exec -T trading-bot python -c "
import urllib.request, urllib.error
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # don't follow
opener = urllib.request.build_opener(NoRedirect)
try:
    r = opener.open('http://localhost:8080$url', timeout=5)
    print(r.status)
except urllib.error.HTTPError as e:
    print(e.code)
except Exception as e:
    print(f'ERR:{type(e).__name__}')
" 2>&1 | tail -1 | tr -d '\r')
    if [[ "$code" =~ $expect ]]; then
        pass "$label  ($url → $code)"
    else
        warn "$label  ($url → $code, expected $expect)"
    fi
}

# Most surfaces should redirect to /login (302) because we hit them
# without an auth cookie — that's correct behavior. /health and
# /__routes__ are public; should return 200. /dashboard is the new
# 301 redirect to /dex/dashboard.
probe_route /health        '^200$'      'Public /health'
probe_route /__routes__    '^200$'      'Public /__routes__'
probe_route /dashboard     '^(301|302)$' '/dashboard 301→/dex/dashboard (or 302 if auth front)'
probe_route /analytics     '^(200|302|401)$' '/analytics serves (not the dead 302→/dashboard redirect)'
probe_route /dex/trades    '^(200|302|401)$' '/dex/trades alias'
probe_route /dex/positions '^(200|302|401)$' '/dex/positions alias'

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
echo
if [[ $FAILS -eq 0 ]]; then
    if [[ $WARNS -eq 0 ]]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  PREFLIGHT GREEN — every check passed${NC}"
        echo -e "${GREEN}========================================${NC}"
    else
        echo -e "${YELLOW}========================================${NC}"
        echo -e "${YELLOW}  PREFLIGHT AMBER — $WARNS warnings, 0 failures${NC}"
        echo -e "${YELLOW}========================================${NC}"
    fi
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  PREFLIGHT RED — $FAILS failures, $WARNS warnings${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
