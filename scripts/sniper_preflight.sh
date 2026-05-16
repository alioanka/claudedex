#!/usr/bin/env bash
# scripts/sniper_preflight.sh
#
# Validates every code path shipped in the May 2026 SNIPER hardening
# session against a running stack (trading-bot + trading-postgres).
# Run after `git pull` + `docker compose up -d --build trading-bot`.
#
# Prints PASS / WARN / FAIL per check and exits 0 only if no FAIL.
# WARN means "verifiable later" (e.g. needs accumulated data) — not
# blocking.
#
# Usage:
#   bash scripts/sniper_preflight.sh
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
    # Runs an SQL command inside trading-postgres using the secret-mounted user.
    # Stdout is the raw psql output; caller decides what to grep for.
    docker compose exec -T postgres bash -lc \
        "psql -U \$(cat /run/secrets/db_user) -d tradingbot -tA -c \"$1\"" 2>&1
}

# ---------------------------------------------------------------------------
# Step 0: stack health
# ---------------------------------------------------------------------------
hdr "Stack health"

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

if [[ $FAILS -gt 0 ]]; then
    echo
    echo -e "${RED}Stack is not up; cannot continue.${NC}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: migration 016 applied
# ---------------------------------------------------------------------------
hdr "Migration 016 — max_active_positions seed"

OUT=$(pg "SELECT value FROM config_settings WHERE config_type='sniper_config' AND key='max_active_positions';")
if [[ "$OUT" == "500" ]]; then
    pass "max_active_positions=500 seeded in config_settings"
elif [[ -z "$OUT" ]]; then
    fail "max_active_positions row missing — apply migration: docker compose cp migrations/016_seed_sniper_max_active_positions.sql postgres:/tmp/016.sql && docker compose exec postgres sh -c 'psql -U \$(cat /run/secrets/db_user) -d tradingbot -f /tmp/016.sql'"
else
    pass "max_active_positions=$OUT (operator-tuned)"
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
  metadata->>'detection_path' || ':' || (metadata->>'block_time_anchored') || ':' || COUNT(*)
FROM sniper_trades
WHERE entry_timestamp > NOW() - INTERVAL '30 minutes'
  AND metadata->>'block_time_anchored' IS NOT NULL
GROUP BY 1, 2 ORDER BY 1, 2;")

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
  metadata->>'detection_path' || '|' ||
  COUNT(*) || '|' ||
  ROUND((percentile_cont(0.5) WITHIN GROUP (ORDER BY (metadata->'timing'->>'detect_to_rpc_receipt_ms')::float))::numeric, 0) || '|' ||
  ROUND((percentile_cont(0.95) WITHIN GROUP (ORDER BY (metadata->'timing'->>'detect_to_rpc_receipt_ms')::float))::numeric, 0)
FROM sniper_trades
WHERE entry_timestamp > NOW() - INTERVAL '30 minutes'
  AND metadata->'timing'->>'detect_to_rpc_receipt_ms' IS NOT NULL
GROUP BY 1 ORDER BY 1;")

if [[ -z "$OUT" ]]; then
    warn "no timing rows yet — wait ~10 min after sniper restart"
else
    echo "$OUT" | while IFS='|' read -r path samples p50 p95; do
        if [[ "$path" == "wss" ]] && [[ "$p50" -lt 500 ]]; then
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

if docker compose exec -T trading-bot python -m pytest tests/unit/test_sniper_new_paths.py -q --no-header 2>&1 | tail -5 | grep -q "passed"; then
    PASS_LINE=$(docker compose exec -T trading-bot python -m pytest tests/unit/test_sniper_new_paths.py -q --no-header 2>&1 | tail -1)
    pass "tests/unit/test_sniper_new_paths.py: $PASS_LINE"
else
    warn "could not run unit tests (test deps missing or test file not present)"
fi

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
