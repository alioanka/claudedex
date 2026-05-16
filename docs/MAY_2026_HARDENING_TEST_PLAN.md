# May 2026 Hardening — Consolidated Test Plan

Run this top-to-bottom in your evening session. Each phase has a copy-pasteable command block + expected output. If anything diverges, paste the actual output and we'll triage.

The session pushed **23 commits** to `claude/create-expert-agents-JFSF5` covering: SNIPER (active-positions cap, Jupiter quote fallback, LIVE safety guard, block-time anchoring, t_rpc_receipt timing, WSS+polling concurrency, logging consolidation), SOLANA + COPY_TRADING (cross-module RiskManager.validate_trade gates, COPY position cap), dashboard (Win Rate format, top-25 table limit, empty-state chart messages, real SOL/USD price, freshness-based liveness check), and infrastructure (migration runner path fix, docker-compose version cleanup, unit tests, preflight script).

---

## Step 0 — Pull the branch + cold restart

```bash
cd /root/claudedex
git fetch origin claude/create-expert-agents-JFSF5
git checkout claude/create-expert-agents-JFSF5
git pull origin claude/create-expert-agents-JFSF5

# Stale dirs from before logging consolidation
rm -rf logs/sniper_module logs/arbitrage_module logs/copy_trading_module
rm -f  logs/sniper/sniper_trades.log

# Rebuild + restart only the bot (postgres+redis untouched)
docker compose up -d --build trading-bot
sleep 8
```

Watch the boot sequence:

```bash
docker compose logs --tail=200 trading-bot | grep -iE "🚀|🔫|RiskManager|migration"
```

Expected lines (subset):
- `🚀 Starting Sniper module...`
- `Logs: logs/sniper/stdout.log, logs/sniper/stderr.log`
- `✅ RiskManager wired into Solana engine`
- `✅ RiskManager wired into Copy Trading executor`
- Migration runner scans `migrations/` (not `data/storage/migrations/`)
- `017_seed_copytrading_max_active_positions` either runs or shows `[ALREADY APPLIED]`

---

## Step 1 — Define the psql helper

```bash
PG_RUN() { docker compose exec -T postgres bash -lc "psql -U \$(cat /run/secrets/db_user) -d tradingbot -tA -c \"$1\""; }
```

---

## Step 2 — Run the preflight script (replaces 90% of manual checks)

```bash
bash scripts/preflight.sh
```

Exit code 0 + a green banner ⇒ all critical paths validated. Amber banner ⇒ warnings only, safe to continue. Red banner ⇒ paste output back.

If you want to do the manual checks individually, continue with steps 3-9 below.

---

## Step 3 — Migration seeds present

```bash
PG_RUN "SELECT key, value FROM config_settings WHERE config_type IN ('sniper_config','copytrading_config') AND key='max_active_positions' ORDER BY config_type;"
```

Expected:
```
max_active_positions|50
max_active_positions|500
```

---

## Step 4 — Runtime stats with new counters

```bash
PG_RUN "SELECT
  (stats->>'active_positions')::int           AS sniper_active,
  (stats->>'max_active_positions')::int       AS sniper_cap,
  (stats->>'jupiter_quote_fallback_hits')::int AS quote_fb,
  (stats->>'capped_rejections')::int          AS capped,
  (stats->'solana_listener'->>'wss_dispatched')::int AS wss_dispatched,
  (stats->'solana_listener'->>'wss_inflight_peak')::int AS peak_inflight,
  (stats->'solana_listener'->>'block_time_anchored')::int AS sol_anchored,
  (stats->'evm_listener'->>'block_time_anchored')::int   AS evm_anchored,
  EXTRACT(EPOCH FROM (NOW() - updated_at))::int AS age_s
FROM sniper_runtime_stats WHERE id=1;"
```

Healthy ranges: `sniper_cap=500`, `peak_inflight < 16`, `age_s < 600`. Both `sol_anchored` and `evm_anchored` should grow as new pools are detected.

---

## Step 5 — block_time_anchored propagation (last 30 min)

```bash
PG_RUN "SELECT
  metadata->>'detection_path' AS path,
  (metadata->>'block_time_anchored')::bool AS anchored,
  COUNT(*)
FROM sniper_trades
WHERE entry_timestamp > NOW() - INTERVAL '30 minutes'
  AND metadata->>'block_time_anchored' IS NOT NULL
GROUP BY 1, 2 ORDER BY 1, 2;"
```

After this branch is running you should see `t` (true) dominate. EVM polling rows will pick up real `block.timestamp` via the new `_block_ts_cache`; EVM WSS reuses that cache so common bursts also show `t`.

---

## Step 6 — Detection latency p50/p95 (the headline timing metric)

```bash
PG_RUN "SELECT
  metadata->>'detection_path' AS path,
  COUNT(*) AS samples,
  ROUND((percentile_cont(0.5) WITHIN GROUP (
    ORDER BY (metadata->'timing'->>'detect_to_rpc_receipt_ms')::float))::numeric, 0) AS p50_ms,
  ROUND((percentile_cont(0.95) WITHIN GROUP (
    ORDER BY (metadata->'timing'->>'detect_to_rpc_receipt_ms')::float))::numeric, 0) AS p95_ms
FROM sniper_trades
WHERE entry_timestamp > NOW() - INTERVAL '30 minutes'
  AND metadata->'timing'->>'detect_to_rpc_receipt_ms' IS NOT NULL
GROUP BY 1 ORDER BY 1;"
```

Last validation showed WSS p50 ~1.9s and polling p50 ~1.9s. After the WSS+polling concurrency rewrite (fb4767e + 76c1924) the WSS p95 tightened from 4.1s → 2.8s. The p50 floor is your RPC provider's `logsSubscribe` push latency (subscription is already at `commitment='processed'` — sub-second on a premium RPC, multi-second on free tier).

---

## Step 7 — LIVE-mode safety filter guard

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

Expected: `GUARD FIRED OK: Sniper refused to start: safety_check_enabled=false in LIVE mode`

---

## Step 8 — Unit tests for the new code paths

```bash
docker compose exec -T trading-bot python -m pytest tests/unit/test_sniper_new_paths.py -v
```

Expected: 19 passed.

---

## Step 9 — Browser UI smoke checks

| Page | Check |
|---|---|
| `/sniper/dashboard` | Win Rate renders as `0.05%` (not 17 decimals); Trade Amount unit is correct (`SOL`/`ETH`/`(multi-chain)`) |
| `/sniper/dashboard` | Status reads `Online` (or `Online (no snapshot yet)` during startup grace, never stuck on stale env-only `Online`) |
| `/sniper/performance` | "Performance by Token/Chain" header shows `Top 25 of N tokens`; table scrolls inside its 520px box |
| `/sniper/performance` | Each path card under "Detection Latency" shows yellow `p50 detect→rpc (N samples)` |
| `/sniper/performance` | Equity Curve / Daily P&L / Win-Loss charts show *"No realized P&L yet — DRY_RUN trades close at entry price"* (not a misleading flat $10k line) |
| `/futures/performance` | Same top-25 + empty-state pattern |
| `/arbitrage/performance` | Same top-25 + empty-state pattern |

---

## Going LIVE checklist (when ready, separate session)

DO NOT do this tonight unless you've validated everything above and explicitly want to flip:

```sql
-- Inside trading-postgres
UPDATE config_settings SET value='true'
WHERE config_type='sniper_config' AND key='safety_check_enabled';
```

Then in `.env`:
```
DRY_RUN=false
```

Then `docker compose restart trading-bot`. The LIVE-mode safety guard from Step 7 will refuse to start if safety_check_enabled is still false, so you can't forget the SQL flip without the engine telling you.

---

## Quick smoke checks for the late-additions

### /health endpoint
```bash
curl -s http://38.242.251.156:8080/health | python -m json.tool
```
Expected: `{"status": "healthy", "service": "claudedex-dashboard", "git_sha": "...", "db": "reachable"}`

### Migration runner picks up /migrations/ on restart
```bash
docker compose logs --since=2m trading-bot | grep -E "017_seed_copytrading|016_seed_sniper|migration"
```
Expected: either `[ALREADY APPLIED]` or `applied successfully` lines for 016 and 017.

### Subprocess auto-restart cooldown
```bash
# After bot has been up >1 hour, kill the dex subprocess to test restart counter reset
pgrep -f main_dex.py | head -1 | xargs -r kill
# In ~10s logs should show "stable for ... reset restart budget" if a previous crash had bumped the counter
```

### Restart budget log
```bash
docker compose logs --since=24h trading-bot | grep "restart budget"
```

### COPY_TRADING bounded sets
After leader monitoring runs for a while:
```bash
docker compose exec -T trading-bot python -c "
import gc, sys
sys.path.insert(0, '/app')
# Pull a live engine instance if reachable, else just sanity-check the helper code
from modules.copy_trading.copy_engine import CopyTradingEngine
e = CopyTradingEngine.__new__(CopyTradingEngine)
e._known_tx_hashes = set(); e._known_solana_sigs = set()
e._known_tx_order = []; e._known_sig_order = []; e._known_max = 5
for i in range(8):
    e._remember_tx_hash(f'tx{i}')
print(f'Bounded at: {len(e._known_tx_hashes)} (expected 5)')
print(f'Oldest evicted: {\"tx0\" not in e._known_tx_hashes}')
"
```

---

## Dashboard audit pass — additional UI/UX validation

After 3 parallel review agents inventoried 61 findings across the
dashboard surface, the following extra checks are worth running in
the evening session alongside the SNIPER preflight:

| Check | URL | Expected |
|---|---|---|
| Index has all 7 module cards | `/` | DEX, Futures, Solana, Sniper, Arbitrage, Copy Trading, AI all present |
| Status reflects actual liveness | `/`, `/full-dashboard` | Disabled modules say DISABLED, not RUNNING |
| Analytics page actually loads | `/analytics` | No redirect to /dashboard; real Advanced Analytics page renders |
| Logs show real per-module entries | `/logs` | Sniper/arbitrage/dashboard log lines visible (was always empty) |
| Bot Start/Stop affects all modules | side-nav buttons | All enabled subprocesses restart, not just DEX |
| /health returns healthy JSON | `curl http://localhost:8080/health` | 200 + `{status: healthy, db: reachable}` |
| Sniper performance Top-25 table | `/sniper/performance` | "Top 25 of N tokens" header |
| Sniper / Futures / Arbitrage charts honest empty state | each `/performance` | "No realized P&L yet" placeholder when DRY_RUN |
| Sniper positions paginated | `/sniper/positions` | Prev/Next + "Showing X-Y of Z" |
| Trades cap warning | `/arbitrage/trades`, `/copytrading/trades`, `/copytrading/performance` | Yellow ⚠ when fetch hits cap, blank otherwise |
| Arbitrage Triangular tile honest | `/arbitrage/dashboard` | "Disabled / scope cut" badge |
| AI cost estimate per-model | `/ai/logs` | Reasonable dollar figure (~$0.001-0.01) |
| AI Avg Hold Time populated | `/ai/performance` | Real duration, not `--` |
| Copy discovery doesn't auto-fire | `/copytrading/discovery` | "Click to load top traders" button, no auto-call |
| Wallet remove atomic | `/copytrading/wallets` → Remove | Only that wallet drops; full list intact |
| DEX trades capped at 100 rows | `/trades` | "Showing 100 of N" footer |
| Last updated honest | `/full-dashboard` | Timestamp + ⚠ if fetch failed |
| Pro Controls sidebar | `/pro-controls` | "Pro Controls" highlighted, not "DEX Dashboard" |

If any of these regress, paste the relevant page output and we'll
triage.

---

## Commit log this session

```
9407da0 [backend] dashboard: tab-hidden polling backoff for /ai/logs + /copytrading/positions
0bf7784 [backend] /arbitrage/positions: explain the empty state instead of silent redirect to /trades
4e3acec [backend] dashboard: overflow warnings on capped trade fetches (audit #15-17)
e33b41f [backend] /full-dashboard: Last updated timestamp shows attempt time + ⚠ on failure
96f3135 [backend] /sniper/positions: paginate (50/page) — was rendering 1000+ rows
1e9ae76 [backend] /trades (DEX): cap visible rows to 100 + 'Showing N of M' footer
20d1676 [backend] /api/arbitrage/trading/status: prefer arbitrage_config.dry_run over global env
ca5c3df [backend] /api/backtest/results/{id}/export: CSV download endpoint (was 404)
b45a979 [backend] /copytrading/dashboard: raise trades fetch limit 100→2000 + guard future timestamps
a52e010 [backend] copytrading: atomic /api/copytrading/wallets/{add,remove} — stop risking full list loss
cca4231 [backend] index.html: render module cards for Sniper, Arbitrage, Copy Trading, AI
584fadb [backend] dashboard: drop misleading "ETH" suffix on mixed-chain arbitrage stats + de-dup AI avgSentiment
44df8da [backend] /ai/performance: compute + render Avg Hold Time (was permanently --)
4f15cce [backend] /ai/logs: per-model cost table instead of single GPT-3.5 $0.002/1K rate
c8bd9a3 [backend] /arbitrage/dashboard: mark Triangular tile as Disabled (scope cut), not 0 trades
65688eb [backend] /copytrading/discovery: make Hot Wallets opt-in to stop burning paid API quota
5e4ea12 [backend] dashboard: clearer DEX page titles + honest emergency-exit warning
9151757 [backend] dashboard: delete unused index_new.html + modules_old_backup.html templates
2e01b43 [backend] dashboard.html: define missing loadInsights() — silent ReferenceError
49672a7 [backend] /api/solana/stats: real SOL/USD via cached helper instead of 200 sentinel
5443c0b [backend] dashboard: pro_controls page-context + drop fake [1,0] win-rate data
54864aa [backend] dashboard: let module_routes own Start/Stop/Restart, not engine-only
1476955 [backend] /analytics: remove dead redirect stub so AnalyticsRoutes wins
9eabc3e [backend] /api/logs: walk per-module logs dirs instead of dead /app/logs path
9a49c7b [backend] full_dashboard: stop treating env=true as fallback for RUNNING
88e404b [backend] dashboard: fix Arbitrage Token Size math + DAI address typo
7a4ebf3 [backend] dashboard: AI + COPY_TRADING status no longer spoofed by historical data
9bd4cec [backend] cache /api/sniper/timing 30s — percentile_cont over 130k rows was hot
3edbcac [backend] dashboard: status no longer reports RUNNING for disabled modules with stale trades
6198edc [backend] dockerignore: ship tests/ into image + tighten preflight grep
34621c6 [backend] preflight: install pytest-cov + override pytest.ini's --cov addopts
7390d47 [backend] /health verified working; harden /__routes__ debug endpoint
ced8af0 [backend] diagnose /health 500: multi-channel logging + /__routes__ debug
938300a [backend] disable anchorpy pytest plugin + diagnostic log on /health hits
dbba9fa [backend] fix preflight round-2: bulletproof /health, SQL GROUP BY shape, pytest-asyncio
713da9f [backend] fix preflight findings: /health 500, cap counts wrong, preflight SQL escape, test runner
21e3d4c [backend] preflight: add /health endpoint reachability + git SHA report
a586796 [backend] cleanup_stale_positions: extend to copytrading_trades
9452577 [market] COPY_TRADING: only gate BUYs through RiskManager, never SELLs
761d5c1 [backend] dashboard: module_control honors Stale / Starting status states
bf3508d [backend] scripts: cleanup_stale_positions.py operator utility
2ae04b2 [docs] extend test plan with the latest 8 commits (health endpoint, restart budget, bounded sets, etc.)
08039d7 [backend] scripts: rewrite health_check.py to call /health, drop stale creds
22534bc [backend] dashboard: real /health endpoint + Dockerfile healthcheck that uses it
6f7cb85 [backend] dashboard: surface WSS concurrency observability on sniper page
0fa8a2b [backend] main: reset module restart budget after sustained uptime
6fbd137 [market] COPY_TRADING: bound _known_tx_hashes / _known_solana_sigs against memory growth
35a41e8 [backend] dashboard: clarify "Passed Safety" label when safety_check_enabled=false
d773f40 [backend] dashboard: rename misleading "Tokens Detected/Sniped" labels
458c0c7 [docs] consolidated test plan for the evening validation session
5d7f94c [backend] docs: replace brittle line-number refs in module CLAUDE.md files
ab4a845 [backend] dashboard: empty-state messages for futures + arbitrage charts
6a6b1ab [backend] dashboard: apply top-25+scroll pattern to futures & arbitrage perf tables
c4e4c3d [backend] preflight: extend to migration 017 + COPY cap, rename script
b82d632 [backend] migrate_database: pick up migrations from the canonical /migrations dir
99839b2 [backend] tests: extend coverage to EVM cache, COPY cap, dashboard SOL price
ee7fe62 [backend] dashboard: real SOL/USD price for SOL-denominated PnL displays
de6662a [backend] dashboard: snipe status reflects snapshot freshness, not just env
ae5ba8a [market] SNIPER: close stale slippage_tolerance TODOs in trade_executor
a28dd22 [market] COPY_TRADING: global open-position cap to bound multi-leader fanout
b88d336 [backend] preflight: consolidated SNIPER validation script
66cd935 [backend] tests: cover SNIPER hardening paths from May 2026 session
74396ed [backend] dashboard: empty-state messages for charts with no realized P&L
76c1924 [market] SNIPER: parallelize Solana polling under the same WSS semaphore
dd8bb51 [market] SNIPER: EVM block-time anchoring via cached eth.get_block
3f26532 [market] SOLANA + COPY: wire cross-module RiskManager.validate_trade gate
fb4767e [market] SNIPER: parallelize WSS candidate processing under bounded semaphore
f898a18 [market] SNIPER: t_rpc_receipt marker isolates detection latency from RPC wait
12ec9cc [market] SNIPER: block_time_anchored DB propagation + LIVE safety-filter guard
c9a7bb8 [market] SNIPER: Jupiter /quote price fallback for fresh Solana mints
c59a32c [market] SNIPER: active-positions cap to brake runaway accumulation
67dae9e [backend] SNIPER: consolidate logging + dashboard UI polish
55cecb7 [market] SNIPER: bump verdict to GREEN candidate after Phase 2 volume validation
```
