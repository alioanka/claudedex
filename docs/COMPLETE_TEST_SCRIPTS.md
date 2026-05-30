# Complete Test Scripts — All Session Commits

> ⚠️ **SUPERSEDED.** Every script + SQL probe + API call below has been
> ported into the **/test-runner** dashboard page (commit `d5eb661`).
> Open `http://<host>:8080/test-runner`, click each Run button, then
> **Copy All Results** at the top right. This is the single source of
> truth for testing the session-18 fixes.
>
> Kept for historical context. Scenarios live in
> `monitoring/test_runner_routes.py` `TEST_CATALOG`.

Generated for the May 2026 hardening session. Covers all 90+ commits
across SNIPER hardening, cross-module RiskManager gates, dashboard
audit fixes, infrastructure hardening, and operator tooling.

All commands paste cleanly into a `bash` shell on the VPS at
`/root/claudedex`. Container service names: `postgres`, `trading-bot`,
`redis`. Database: `tradingbot`. Credentials live in Docker secrets
at `/run/secrets/db_user` and `/run/secrets/db_password`.

---

## 1. Bootstrap — pull + rebuild + restart

```bash
cd /root/claudedex
git fetch origin claude/create-expert-agents-JFSF5
git checkout claude/create-expert-agents-JFSF5
git pull origin claude/create-expert-agents-JFSF5

# Stale dirs from before logging consolidation
rm -rf logs/sniper_module logs/arbitrage_module logs/copy_trading_module
rm -f  logs/sniper/sniper_trades.log

# Rebuild — Dockerfile changes (pytest-asyncio, pytest-cov, tests
# included via .dockerignore) require a fresh build, but pip layers
# are cached so this is fast.
docker compose up -d --build trading-bot
sleep 10

# Optional: clean up stale-open trade rows accumulated from prior runs.
docker compose exec -T trading-bot python scripts/cleanup_stale_positions.py
docker compose exec -T trading-bot python scripts/cleanup_stale_positions.py --apply
```

---

## 2. Single-command preflight (covers ~80% of validation)

```bash
bash scripts/preflight.sh
```

Exit code 0 if no FAIL. Output shows PASS / WARN / FAIL per check
across:
- Stack health (containers running)
- Migrations 016 + 017 seeded
- Logging consolidation (no stale dirs, no dead trades log)
- Sniper subprocess up
- RiskManager wiring observed for enabled modules
- Runtime stats snapshot fresh
- block_time_anchored propagation
- detection latency p50/p95
- LIVE-mode safety guard fires
- 19 unit tests pass inside container
- /health endpoint healthy
- /dashboard routes spot-check (status codes)

Expected verdict: **PREFLIGHT AMBER** (3 warnings if SOLANA / COPY are
disabled in `.env`, otherwise GREEN).

---

## 3. SQL validation queries

Set up the helper once per shell session:

```bash
PG_RUN() { docker compose exec -T postgres bash -lc "psql -U \$(cat /run/secrets/db_user) -d tradingbot -tA -c \"$1\""; }
```

### 3a. Migration seeds present

```bash
PG_RUN "SELECT config_type, key, value FROM config_settings WHERE key='max_active_positions' ORDER BY config_type;"
```
Expected: `copytrading_config|max_active_positions|50` + `sniper_config|max_active_positions|500`.

### 3b. Runtime stats snapshot — all new SNIPER counters

```bash
PG_RUN "SELECT
  (stats->>'active_positions')::int                AS sniper_in_mem,
  (stats->>'active_positions_effective')::int      AS sniper_effective,
  (stats->>'max_active_positions')::int            AS sniper_cap,
  (stats->>'jupiter_quote_fallback_hits')::int     AS quote_fb_hits,
  (stats->>'capped_rejections')::int               AS capped,
  (stats->>'positions_synthetic_closed')::int      AS synth_closes,
  (stats->'solana_listener'->>'wss_dispatched')::int AS wss_dispatched,
  (stats->'solana_listener'->>'wss_inflight_peak')::int AS wss_peak,
  (stats->'solana_listener'->>'block_time_anchored')::int AS sol_anchored,
  (stats->'evm_listener'->>'block_time_anchored')::int   AS evm_anchored,
  EXTRACT(EPOCH FROM (NOW() - updated_at))::int AS age_s
FROM sniper_runtime_stats WHERE id=1;"
```

Health bands: cap=500, effective ≥ in_mem (DB count never below memory),
wss_peak < 16 normally, age_s < 600.

### 3c. block_time_anchored propagation

```bash
PG_RUN "SELECT
  COALESCE(metadata->>'detection_path','?') || ':' ||
  (metadata->>'block_time_anchored') || ':' ||
  COUNT(*)::text
FROM sniper_trades
WHERE entry_timestamp > NOW() - INTERVAL '30 minutes'
  AND metadata->>'block_time_anchored' IS NOT NULL
GROUP BY COALESCE(metadata->>'detection_path','?'), (metadata->>'block_time_anchored')
ORDER BY 1;"
```

Expected: `polling:true:N`, `wss:true:N`, occasional `wss:false:M`
(rare RPC drops). EVM polling rows also `:true` once the block_ts
cache populates.

### 3d. Detection latency p50/p95 (the headline timing metric)

```bash
PG_RUN "SELECT
  COALESCE(metadata->>'detection_path','?') || '|' ||
  COUNT(*)::text || '|' ||
  ROUND((percentile_cont(0.5) WITHIN GROUP (ORDER BY (metadata->'timing'->>'detect_to_rpc_receipt_ms')::float))::numeric, 0)::text || '|' ||
  ROUND((percentile_cont(0.95) WITHIN GROUP (ORDER BY (metadata->'timing'->>'detect_to_rpc_receipt_ms')::float))::numeric, 0)::text
FROM sniper_trades
WHERE entry_timestamp > NOW() - INTERVAL '30 minutes'
  AND metadata->'timing'->>'detect_to_rpc_receipt_ms' IS NOT NULL
GROUP BY COALESCE(metadata->>'detection_path','?')
ORDER BY 1;"
```

Last measured:
- wss   p50 ≈ 1500-1900 ms (RPC provider's logsSubscribe push latency)
- poll  p50 ≈ 1900-2400 ms

### 3e. LIVE-mode safety guard fires

```bash
docker compose exec -T -e DRY_RUN=false trading-bot python -c "
import asyncio, sys
sys.path.insert(0, '/app')
from modules.sniper.core.sniper_engine import SniperEngine
async def t():
    e = SniperEngine({}, None, None)
    e.safety_check_enabled = False
    try:
        await e._load_settings()
        print('UNEXPECTED: guard did not fire')
    except RuntimeError as err:
        print(f'GUARD FIRED OK: {err}')
asyncio.run(t())
"
```

Expected: `GUARD FIRED OK: Sniper refused to start: safety_check_enabled=false in LIVE mode`.

### 3f. Unit tests inside container (19 tests)

```bash
docker compose exec -T -w /app -e PYTEST_ADDOPTS='-p no:cacheprovider -p no:anchorpy' \
    trading-bot python -m pytest tests/unit/test_sniper_new_paths.py -q --no-header -o addopts=
```

Expected: `19 passed`.

---

## 4. Dashboard endpoint validation

### 4a. /health (no auth required)

```bash
curl -s http://38.242.251.156:8080/health | python3 -m json.tool 2>/dev/null || \
  docker compose exec -T trading-bot python -c "
import urllib.request, json
r = urllib.request.urlopen('http://localhost:8080/health', timeout=5)
print(json.dumps(json.loads(r.read()), indent=2))
"
```

Expected JSON: `{"status":"healthy","service":"claudedex-dashboard","time":"...","git_sha":"","db":"reachable"}`.

### 4b. /__routes__ (no auth, diagnostic)

```bash
docker compose exec -T trading-bot python -c "
import urllib.request, json
r = urllib.request.urlopen('http://localhost:8080/__routes__', timeout=5)
j = json.loads(r.read())
print(f'Routes registered: {j[\"count\"]}')
# Confirm key new endpoints exist
for path in ['/health', '/dashboard', '/dex/trades', '/dex/positions',
             '/analytics', '/api/copytrading/wallets/remove',
             '/api/copytrading/wallets/add',
             '/api/backtest/results/{test_id}/export']:
    found = any(x.get('path') == path for x in j['routes'])
    print(f'  {\"✓\" if found else \"✗\"} {path}')
"
```

### 4c. /api/sniper/timing cache (response should include 'cached' key on 2nd call)

```bash
docker compose exec -T trading-bot python -c "
import urllib.request, json
# Two back-to-back requests; second should be cached
for i in (1, 2):
    r = urllib.request.urlopen('http://localhost:8080/api/sniper/timing?days=7', timeout=10)
    body = json.loads(r.read())
    print(f'call {i}: cached={body.get(\"cached\", False)}, paths={list((body.get(\"data\",{}) or {}).get(\"paths\",{}).keys())}')
"
```

Note: the /api/sniper/timing endpoint requires auth — if you see a 302
you need to test from a logged-in browser session. The cache behavior
is internal so this is more of a sanity check than a hard test.

---

## 5. Browser UI manual checks

| Page | What to verify |
|---|---|
| `/` | All 7 module cards rendered (DEX, Futures, Solana, Sniper, Arbitrage, Copy Trading, AI) — previously only first 3 |
| `/` | Disabled modules show "DISABLED" not "RUNNING" |
| `/full-dashboard` | "Last updated HH:MM:SS" + ⚠ marker if fetch failed |
| `/full-dashboard` | Charts with no data render "No data yet" overlay (not blank axes) |
| `/dashboard` | 301 redirects to `/dex/dashboard` (check Network tab → first request status 301) |
| `/dex/trades` `/dex/positions` `/dex/performance` | Same content as root `/trades` etc. (new alias) |
| `/analytics` | Real Advanced Analytics page (portfolio summary, per-module performance/risk/equity) |
| `/logs` | Module dropdown + Level dropdown, table shows Module column, real entries from logs/sniper/, logs/arbitrage/, etc. |
| `/sniper/dashboard` | Win Rate displays e.g. `0.05%` (2-decimal format), Trade Amount unit follows target chain |
| `/sniper/dashboard` | "WSS task peak (queued+running): N · M dispatched (concurrent RPC capped at 16)" note in Pool Detection card |
| `/sniper/positions` | Pagination footer "Prev / Next / Showing X-Y of Z (page A/B)" |
| `/sniper/performance` | "Top 25 of N tokens" in Performance by Token header |
| `/sniper/performance` | Yellow p50 detect→rpc card under each path's Detection Latency |
| `/sniper/performance` | Equity Curve / Daily P&L / Win-Loss charts populated if any closed trades; "No realized P&L yet" if all DRY_RUN |
| `/futures/performance` `/arbitrage/performance` | Same top-25 + empty-state pattern as sniper |
| `/arbitrage/dashboard` | Triangular tile shows "Disabled" badge + "scope cut" caption (not "0 trades, $0.00") |
| `/arbitrage/dashboard` | DAI rows display "DAI" symbol (not shortened address) |
| `/arbitrage/dashboard` | Token Size column produces sensible values (not off by ~3x) |
| `/arbitrage/positions` | Renders explanation page (not silent 302→/arbitrage/trades) |
| `/arbitrage/trades` | Yellow ⚠ "Showing N (cap)" warning if 2000 rows fetched |
| `/arbitrage/settings` | Triangular section shows yellow "Experimental — currently gated" banner |
| `/copytrading/dashboard` | No Win/Loss + Chain Distribution pies (link to /copytrading/performance instead) |
| `/copytrading/discovery` | "Click to load top traders" button (not auto-firing paid API) |
| `/copytrading/wallets` | Removing a wallet only drops that one (uses atomic endpoint) |
| `/copytrading/trades` `/copytrading/performance` | Yellow ⚠ "Showing N (cap)" when capped |
| `/ai/dashboard` | Sentiment shows compact summary + "View full gauge →" link (not duplicate gauge) |
| `/ai/dashboard` | "20% / $50 / 5% / 3%" hardcoded defaults replaced with "--" until settings load |
| `/ai/performance` | "Avg Hold Time" shows real duration (was permanently `--`) |
| `/ai/logs` | Cost estimate uses per-model pricing (gpt-4o-mini, claude haiku rates) |
| `/simulator` | "Wired (per-chain bps)" / "Wall-clock only" honest reporting (no fake "98.5%") |
| `/module-control` | Module status badges: green RUNNING / yellow Stale / blue Starting / gray DISABLED |
| `/pro-controls` | Sidebar highlights "Pro Controls" (not "DEX Dashboard") |
| `/global-settings` | Number inputs reject non-numeric, JSON fields render as textarea, real error messages on save |
| `/settings` | Shows "Module Control moved" card linking to `/module-control` (no duplicate cards) |
| `/positions` (DEX) | "Emergency Exit (All Modules)" button (not "Close All") with expanded confirmation |
| `/trades` (DEX) | "Showing 100 of N" footer when more than 100 |
| Bot Start / Stop / Restart buttons | Affect all enabled modules, not just DEX |

---

## 6. Test new endpoints directly

### Atomic wallet add (requires auth — easiest via browser dev tools console after login):

```javascript
fetch('/api/copytrading/wallets/add', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({wallet: '0xabc...'})
}).then(r => r.json()).then(console.log)
```

### Backtest CSV export (after running a backtest):

```bash
# Replace <test_id> with the id from a backtest run
curl -s "http://38.242.251.156:8080/api/backtest/results/<test_id>/export" \
     -o /tmp/backtest.csv
head /tmp/backtest.csv
```

---

## 7. Pre-LIVE flip checklist

ONLY run when ready to actually go live — not for tonight's validation.

```bash
# 1. Re-enable the sniper safety filter in DB (was disabled for Phase 2)
docker compose exec -T postgres bash -lc \
  "psql -U \$(cat /run/secrets/db_user) -d tradingbot -c \"
   UPDATE config_settings SET value='true'
   WHERE config_type='sniper_config' AND key='safety_check_enabled';\""

# 2. Confirm it stuck
PG_RUN "SELECT value FROM config_settings WHERE config_type='sniper_config' AND key='safety_check_enabled';"

# 3. Flip global DRY_RUN
sed -i 's/^DRY_RUN=.*/DRY_RUN=false/' .env

# 4. Restart
docker compose restart trading-bot

# 5. Watch for the safety guard's REFUSE line
docker compose logs --tail=100 trading-bot | grep -iE "REFUS|safety|guard"
```

If safety_check_enabled was NOT flipped to true, the engine will refuse
to start with `🛑 REFUSING TO RUN: safety_check_enabled=false while
DRY_RUN=false`. Fix the DB and restart.

---

## 8. Rollback

```bash
git checkout 2faa707  # last commit before this session
docker compose up -d --build trading-bot
```
