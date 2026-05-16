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

## Commit log this session

```
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
