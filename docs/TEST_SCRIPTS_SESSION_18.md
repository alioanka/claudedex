# Test Scripts — 18-commit autonomous session

Covers commits `b53feb7` → `19b5a7b` on branch
`claude/create-expert-agents-JFSF5`. Paste sections in order. All
commands assume:

- Working dir: `/root/claudedex`
- Services: `postgres` (container `trading-postgres`), `trading-bot`, `redis`
- DB: `tradingbot`; credentials in `/run/secrets/db_user`, `/run/secrets/db_password`
- Dashboard: `http://localhost:8080` (browser) or via `127.0.0.1:8080`

Each section reports PASS/FAIL. Run section 0 first; everything else
is independent.

---

## 0. Bootstrap — pull, rebuild, restart

```bash
cd /root/claudedex

# Pull the new commits
git fetch origin
git checkout claude/create-expert-agents-JFSF5
git pull origin claude/create-expert-agents-JFSF5

# Verify HEAD
git log --oneline -1
# Expect: 19b5a7b [backend] sniper: dedupe _evaluate_target on token_address (SNIPE-RM-18)

# Rebuild the bot image (dashboard is part of trading-bot in this stack)
docker compose build trading-bot

# Restart the trading-bot service (postgres + redis keep running)
docker compose up -d trading-bot

# Wait for healthy
sleep 8
docker ps --format '{{.Names}}\t{{.Status}}' | grep -E 'trading-bot|trading-postgres|trading-redis'
# Expect all three "(healthy)" or "Up"

# Tail logs for 10s to confirm clean boot
timeout 10 docker logs -f trading-bot 2>&1 | head -50
# Expect no Python tracebacks; expect:
#   "📋 Sniper settings loaded:" line
#   no "REFUSING TO RUN" lines (DRY_RUN=true keeps both LIVE guards quiet)
```

PASS if HEAD matches `19b5a7b`, all 3 containers up, no tracebacks in
the first 10s of logs.

---

## 1. Preflight (existing matrix — confirms no regressions)

```bash
cd /root/claudedex
bash scripts/preflight.sh
```

Expected: **AMBER with 0 FAIL** (3 by-design warnings are OK). Compare
against the AMBER result from the prior session — line count and pass
matrix must be identical.

PASS if `exit 0` and no `FAIL:` lines.

---

## 2. Browser UI tests (open dashboard, follow the table)

Open `http://localhost:8080` in a browser, login as `admin / admin123`.

| # | Page | Action | Expected |
|---|------|--------|----------|
| 2.1 | any page | look at top-right corner | Pill reads `🔵 DRY-RUN` (blue). Was `UNKNOWN` (gray) before commit `5c7777b`. |
| 2.2 | `/pro-controls` | click `Panic Sell` → confirm | Alert shows either "Emergency exit triggered!" with HTTP 200, or "Emergency exit FAILED: <reason>" — **NOT** the old "Emergency exit triggered!" while nothing happened. |
| 2.3 | `/sniper/dashboard` | look at "Tokens Sniped" tile | Detail line reads `Active: N/M` where M = `max_active_positions` from DB. If N ≥ 0.8·M shows `⚠ near cap`; if ≥ 0.95·M shows `⛔ CAPPED — new snipes blocked`. |
| 2.4 | `/analytics` | scroll to Risk panel | See 3 tiles: `VaR (95%)`, `VaR (99%)`, `CVaR (95%)`. All three show non-empty `$X.XX`. Hover on 99% — tooltip explains "daily loss exceeded only 1% of the time". |
| 2.5 | `/global-settings` | expand any category | `max_drawdown_pct`, `max_position_size`, `max_slippage`, `min_liquidity`, `daily_loss_limit`, `circuit_breaker_threshold` (whichever exist) render with **red border + "⚠ RISK" label suffix**. Non-risk keys render normally. |
| 2.6 | `/global-settings` | edit a risk-critical numeric value (e.g. drawdown from 15 → 14), click Save | Confirm dialog shows `You are changing RISK-CRITICAL values:\n  max_drawdown_pct: 15 → 14\n\nThese controls how much capital can be lost. Confirm?` — Cancel and verify it does NOT save. Then redo and Confirm — it saves. |
| 2.7 | `/global-settings` | edit a non-risk key | No confirm dialog — saves directly. |
| 2.8 | `/copytrading/dashboard` | look at P&L hero tile | If `is_simulated=true` rows exist in DB, tile reads e.g. `+$0.00 live (+$340.00 incl. DRY_RUN)`. If only live or only simulated, tile shows just the single number. |
| 2.9 | `/futures/dashboard` | If no FUTURES API keys configured, look at network indicator | Shows `--` (not `BINANCE TESTNET`). Configure Bybit keys → shows `BYBIT MAINNET`/`TESTNET`. |
| 2.10 | `/settings/futures` | open while futures module disabled | Module status row shows `--` for mode/network (not fabricated `BINANCE TESTNET`). |
| 2.11 | sidebar nav | click DEX → Positions, Trades, Performance, Backtest, Reports, Analysis | All 6 routes load; the sidebar item for the active page highlights. URLs use `/dex/*` aliases. |
| 2.12 | `/positions` | click `Close All` confirmation flow | Hits `/api/bot/emergency-exit` (kebab, canonical) — see network tab. |

PASS if all 12 rows behave as described.

---

## 3. API sanity (one paste, no UI)

> **CORRECTED LOGIN PATTERN (VPS failure 5):** the older quick-script used
> `curl -X POST $BASE/login -d 'username=admin&password=admin123'` and then
> `awk` to extract a `session` cookie. That is wrong on 4 counts: the
> endpoint is `/api/auth/login`, the cookie is `session_id`, the body is
> JSON (not form-urlencoded), and grepping the cookie jar with `awk` breaks
> as soon as `csrf_token` also lands in the jar. Use the pattern below or
> the canned script `scripts/dashboard_smoke.sh`.

```bash
BASE=http://127.0.0.1:8080

# Login + grab session_id cookie (and csrf_token cookie, set by the
# CSRF middleware on the same response).
COOKIE_JAR=$(mktemp)
curl -s -c "$COOKIE_JAR" -X POST "$BASE/api/auth/login" \
    -H 'Content-Type: application/json' \
    -d '{"username":"admin","password":"admin123"}' | jq '.success'
# Expect: true

curl_auth() { curl -s -b "$COOKIE_JAR" "$@"; }

echo "=== 3.1 /api/bot/status (MODE badge data shape) ==="
curl_auth "$BASE/api/bot/status" | jq '{success, dry_run: .data.dry_run, mode: .data.mode, running: .data.running}'
# Expect: success:true, dry_run:true, mode:"DRY_RUN", running:true|false

echo "=== 3.2 /api/copytrading/stats (DASH-Q-04 live siblings) ==="
curl_auth "$BASE/api/copytrading/stats" | \
    jq '.stats | {total_trades, total_pnl, live_trades, simulated_trades, live_pnl, live_win_rate}'
# Expect: all six keys present (live_* may be 0 if no real trades — but key must exist)

echo "=== 3.3 /api/sniper/stats (effective count) ==="
curl_auth "$BASE/api/sniper/stats" | \
    jq '{active_positions, active_positions_live, active_positions_effective, max_active_positions, status}'
# Expect: all four numeric keys present; status is "Online"/"Offline"/"Idle"/...

echo "=== 3.4 /api/analytics/risk/dex (var_99 + cvar_95) ==="
curl_auth "$BASE/api/analytics/risk/dex" | jq '.data | {var_95, var_99, cvar_95, annual_volatility}'
# Expect: all four numeric

echo "=== 3.5 /api/arbitrage/stats (honest liveness) ==="
curl_auth "$BASE/api/arbitrage/stats" | jq '. | {status, total_trades}'
# Expect: status reflects env+recent-activity, NOT just "Online if total_trades>0"

echo "=== 3.6 /__routes__ — every new endpoint registered ==="
curl_auth "$BASE/__routes__" | jq '.routes | length'
# Expect: ≥ 470 (existing baseline)

echo "=== 3.7 /health (canary) ==="
curl_auth "$BASE/health" | jq .
# Expect: {"ok": true, ...}

rm -f "$COOKIE_JAR"
```

PASS if all 7 sections return the expected shape with no `null` /
`undefined` for the new fields.

---

## 4. Safety-guard tests (these MUST refuse to start)

These verify the new sniper LIVE-mode startup guards. They are
**destructive to runtime state** — they will restart `trading-bot`.

### 4.1 SNIPE-RM-19 — test_mode + LIVE refuses

```bash
# Enable test_mode in DB
docker exec trading-postgres psql -U $(docker exec trading-postgres cat /run/secrets/db_user) tradingbot -c "
  UPDATE config_settings SET value='true'
  WHERE config_type='sniper_config' AND key='test_mode';
" 2>&1

# Flip DRY_RUN to false in .env (REMEMBER TO REVERT)
sed -i.bak 's/^DRY_RUN=true$/DRY_RUN=false/' .env
grep "^DRY_RUN=" .env  # Confirm: DRY_RUN=false

# Restart and check logs
docker compose restart trading-bot
sleep 5
docker logs trading-bot --tail 80 2>&1 | grep -E "REFUSING TO RUN|test_mode=true"

# Expect a line like:
#   🛑 REFUSING TO RUN: test_mode=true while DRY_RUN=false.
# And a RuntimeError stack trace.

# REVERT:
sed -i.bak 's/^DRY_RUN=false$/DRY_RUN=true/' .env
docker exec trading-postgres psql -U $(docker exec trading-postgres cat /run/secrets/db_user) tradingbot -c "
  UPDATE config_settings SET value='false'
  WHERE config_type='sniper_config' AND key='test_mode';
"
docker compose restart trading-bot
sleep 5
docker logs trading-bot --tail 30 2>&1 | grep -E "Sniper settings loaded|REFUSING"
# Expect "Sniper settings loaded" and NO "REFUSING" line.
```

PASS if step 4.1 prints "REFUSING TO RUN: test_mode=true" then, after
revert, prints "Sniper settings loaded" cleanly.

### 4.2 Existing safety_check_enabled guard (regression check)

```bash
# Same shape — flip safety_check_enabled off + DRY_RUN=false → must refuse
docker exec trading-postgres psql -U $(docker exec trading-postgres cat /run/secrets/db_user) tradingbot -c "
  UPDATE config_settings SET value='false'
  WHERE config_type='sniper_config' AND key='safety_check_enabled';
"
sed -i.bak 's/^DRY_RUN=true$/DRY_RUN=false/' .env
docker compose restart trading-bot
sleep 5
docker logs trading-bot --tail 80 2>&1 | grep -E "REFUSING TO RUN|safety_check"

# Expect:
#   🛑 REFUSING TO RUN: safety_check_enabled=false while DRY_RUN=false.

# REVERT (BOTH must be reverted before any live use!):
sed -i.bak 's/^DRY_RUN=false$/DRY_RUN=true/' .env
# NB: when going LIVE later, set safety_check_enabled=true FIRST:
# docker exec trading-postgres psql ... -c "UPDATE config_settings SET value='true' WHERE config_type='sniper_config' AND key='safety_check_enabled';"
docker compose restart trading-bot
sleep 5
```

PASS if both guards fire correctly and the engine restarts cleanly
after revert.

---

## 5. Behavioral tests (let logs accumulate 30-120 min)

These need running time to surface. Run them after sections 1-4 pass.

### 5.1 SNIPE-RM-18 — _evaluate_target dedup

```bash
# Same address should never appear twice in TARGET ACQUIRED within 60s
docker logs trading-bot --since 1h 2>&1 \
  | grep "SNIPER TARGET ACQUIRED" \
  | awk '{print $NF}' \
  | sort | uniq -c | sort -rn | head -10

# Each address should appear EXACTLY ONCE. Before the fix you'd see
# repeats from polling+WSS overlap.
```

PASS if no address appears > 1 in the last hour.

### 5.2 SNIPE-RM-12 — max_hold_minutes time-stop (opt-in)

```bash
# Enable a short 15-min hold cap
docker exec trading-postgres psql -U $(docker exec trading-postgres cat /run/secrets/db_user) tradingbot -c "
  INSERT INTO config_settings(config_type,key,value,value_type)
  VALUES ('sniper_config','max_hold_minutes','15','int')
  ON CONFLICT(config_type,key) DO UPDATE SET value='15';
"
docker compose restart trading-bot

# Wait 20 min — any position older than 15 min should retire with TIME_STOP
sleep 1200
docker logs trading-bot --since 25m 2>&1 | grep "TIME STOP"
# Expect at least one "⏰ TIME STOP triggered for ... (held Nm, cap 15m, P&L ...)"

# REVERT to disable (or set higher cap):
docker exec trading-postgres psql -U $(docker exec trading-postgres cat /run/secrets/db_user) tradingbot -c "
  UPDATE config_settings SET value='0'
  WHERE config_type='sniper_config' AND key='max_hold_minutes';
"
docker compose restart trading-bot
```

PASS if at least one TIME_STOP appears in the 20-min window. (Requires
the engine to have ≥1 stale position; if you have zero open snipes,
this test trivially passes with no output — re-run after sniper activity.)

### 5.3 AI-BE-09 — RateLimiter no longer stalls at low RPM

```bash
# Watch for the symptom: long delays between AI provider calls
docker logs trading-bot --since 1h 2>&1 \
  | grep -E "RateLimit|wait_for_token|429" \
  | tail -20
# Should NOT see "waited 3+ seconds" patterns at startup. Quiet log is normal.

# Cross-check: did the AI module make calls at all?
docker logs trading-bot --since 1h 2>&1 | grep -c "ai_provider\|sentiment_engine"
# Expect > 0 if AI_MODULE_ENABLED=true
```

PASS if no `waited >3s` warnings AND module made calls.

### 5.4 FUT-RM-04 — newClientOrderId logged

Only if futures has been actively trading (DRY_RUN or live):

```bash
docker logs trading-bot --since 24h 2>&1 | grep "clientOrderId=cd-" | head -5
# Expect each open_long/open_short/close_position to log clientOrderId=cd-l-... / cd-s-... / cd-c-...
```

PASS if any matches appear (or n/a if no futures trades).

### 5.5 FUT-RM-14 — fee-adjusted breakeven

After any TP1 hit on a real or paper trade:

```bash
docker logs trading-bot --since 24h 2>&1 | grep "fee-adjusted breakeven"
# Expect: "🔒 BTCUSDT: Stop moved to fee-adjusted breakeven $X.XX (entry $Y.YY, buffer 0.090%)"
# buffer should be ~2*taker_fee + 1bp ≈ 0.09% Binance, 0.13% Bybit
```

PASS if the line appears with buffer > 0 (or n/a if no TP1 hits yet).

### 5.6 SOL-RM-14 — jupiter_helper killswitch defense

```bash
# Create the killswitch file
touch /root/claudedex/logs/.killswitch

# Trigger any solana swap via dashboard or wait for engine's next attempt
sleep 30
docker logs trading-bot --since 1m 2>&1 | grep "JupiterHelper DRY-RUN/PAUSED gate"
# Expect at least one such line if any jupiter_helper.execute_swap was attempted

# REVERT:
rm /root/claudedex/logs/.killswitch
```

PASS if killswitch file presence blocks jupiter_helper from broadcasting
(or n/a if no jupiter swap was attempted in the window).

### 5.7 AR-10 — Arbitrage Solana slippage

```bash
docker exec trading-postgres psql -U $(docker exec trading-postgres cat /run/secrets/db_user) tradingbot \
    -c "SELECT value FROM config_settings WHERE config_type='arbitrage_config' AND key='sol_arb_slippage_bps';"
# If row exists, value is whatever was configured.
# If no row, engine uses new default = 75 (was 150).

docker logs trading-bot --since 1h 2>&1 | grep -E "arb_slippage_bps|slippage_bps=" | head -5
# Look for slippage value in arb quote logs — should be ≤ 75 by default.
```

PASS if default is observed as 75 in logs (or as configured if row exists).

---

## 6. Quick regression smoke (every page reachable)

```bash
COOKIE_JAR=$(mktemp)
curl -s -c "$COOKIE_JAR" -X POST "http://127.0.0.1:8080/login" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d 'username=admin&password=admin123' >/dev/null

for path in / /dashboard /modules /positions /trades /performance /analytics \
            /backtest /reports /analysis /logs /global-settings /settings/credentials \
            /settings/rpc-api /settings/copytrading /settings/sniper /settings/arbitrage \
            /settings/ai /settings/solana /settings/futures \
            /sniper/dashboard /sniper/timing /sniper/performance \
            /futures/dashboard /solana/dashboard /copytrading/dashboard \
            /arbitrage/dashboard /pro-controls /simulator \
            /dex/positions /dex/trades /dex/performance /dex/backtest /dex/reports /dex/analysis; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -b "$COOKIE_JAR" "http://127.0.0.1:8080$path")
    printf "%-3s  %s\n" "$code" "$path"
done | grep -vE '^200|^301'
# Expect: no output (every page returns 200 or 301). Any 404/500 is a fail.

rm -f "$COOKIE_JAR"
```

PASS if the loop prints nothing (all 200/301).

---

## 7. Cleanup / final state verification

```bash
# Confirm DRY_RUN restored
grep "^DRY_RUN=" .env
# Expect: DRY_RUN=true

# Confirm test_mode off
docker exec trading-postgres psql -U $(docker exec trading-postgres cat /run/secrets/db_user) tradingbot -c "
  SELECT key, value FROM config_settings
  WHERE config_type='sniper_config'
    AND key IN ('test_mode','safety_check_enabled','max_hold_minutes');
"
# Expected (or your post-test setting):
#   test_mode             | false
#   safety_check_enabled  | false  ← still false from Phase 2 measurement
#   max_hold_minutes      | 0 (or your chosen value)

# Confirm services healthy
docker ps --format '{{.Names}}\t{{.Status}}' | grep -E 'trading-bot|trading-postgres|trading-redis'

# Confirm no Python tracebacks since last bootstrap
docker logs trading-bot --since 30m 2>&1 | grep -E "Traceback|CRITICAL|FATAL" | head
# Expect: empty (or only the deliberate "REFUSING TO RUN" criticals from section 4)
```

PASS if `.env` has DRY_RUN=true, all three containers up, no
unexpected tracebacks.

---

## Going-live checklist (DO NOT DO THIS TONIGHT)

Before flipping `DRY_RUN=false` for real:

1. `safety_check_enabled=true` in DB (currently `false` from Phase 2 measurement):
   ```sql
   UPDATE config_settings SET value='true'
   WHERE config_type='sniper_config' AND key='safety_check_enabled';
   ```
2. `test_mode=false` in DB (verify with section 7).
3. Flip `DRY_RUN=false` in `.env`.
4. `docker compose restart trading-bot`.
5. Watch `docker logs -f trading-bot` for 10 min — verify no `REFUSING TO RUN`,
   verify first snipe completes through the **non-simulated** path.
6. The two new LIVE guards (`test_mode + LIVE` and `safety_check_enabled=false +
   LIVE`) will refuse to start if either misconfiguration is left. That is
   the intended safety net — fix the DB row, do not bypass the guard.

---

## Per-commit traceability

| Commit | Test section |
|--------|--------------|
| `5c7777b` MODE badge unwrap | 2.1, 3.1 |
| `e858930` pro_controls + BINANCE fallback | 2.2, 2.9, 2.10 |
| `cc107f0` main.js emergency-exit URL | 2.12 |
| `08bdcc1` positions.html emergency-exit URL | 2.12 |
| `b53feb7` sniper effective count tile | 2.3, 3.3 |
| `20bc0c7` sniper near-cap warning UI | 2.3 |
| `f2d3fe5` global-settings risk-critical | 2.5, 2.6, 2.7 |
| `5aa7a08` /analytics VaR(99)+CVaR(95) | 2.4, 3.4 |
| `7ad4bd0` /api/copytrading/stats live_* | 3.2 |
| `433cd5f` /copytrading/dashboard P&L hint | 2.8 |
| `5bd120c` ai_provider RateLimiter float math | 5.3 |
| `c463c62` jupiter_helper defense gate | 5.6 |
| `0a09e35` binance_futures clientOrderId | 5.4 |
| `8b71526` futures fee-adjusted breakeven | 5.5 |
| `9d6e07e` solana scam_blacklist regex | (passive — affects what gets sniped) |
| `654ee31` arbitrage sol_arb_slippage_bps default | 5.7 |
| `832cfb9` sniper max_hold_minutes | 5.2 |
| `bb0af24` sniper test_mode+LIVE refuse | 4.1 |
| `19b5a7b` sniper _evaluate_target dedup | 5.1 |

---

## Agent 1 Re-test (post-recovery)

Sprint commits `dddb1a5..HEAD` fix five VPS failures the operator reported
on top of the 18-commit session. Each block below is paste-friendly: copy
the whole fenced block into your shell, no edits needed (the cookie+CSRF
plumbing is identical across all five).

### Required: the CORRECT cookie + CSRF pattern

Everything below assumes you ran this first to populate the cookie jar.

```bash
BASE=http://127.0.0.1:8080
COOKIE_JAR=$(mktemp)
curl -s -c "$COOKIE_JAR" -X POST "$BASE/api/auth/login" \
    -H 'Content-Type: application/json' \
    -d '{"username":"admin","password":"admin123"}' | jq '.success'
# Expect: true

# Helpers
auth_get() { curl -s -b "$COOKIE_JAR" "$@"; }
csrf() { awk '$6=="csrf_token"{print $7}' "$COOKIE_JAR"; }
auth_post() {
    local url="$1"; shift
    curl -s -b "$COOKIE_JAR" -X POST "$url" \
        -H 'Content-Type: application/json' \
        -H "X-CSRF-Token: $(csrf)" "$@"
}
```

The single-command alternative is `scripts/dashboard_smoke.sh` which runs
every assertion in this section and exits non-zero if anything fails.

### VPS DB note

Postgres credentials live at `/run/secrets/db_password` and
`/run/secrets/db_user` *inside* the `trading-postgres` container. The
host shell does NOT have access to those paths (`cat: No such file or
directory`). Always run psql FROM INSIDE the container:

```bash
docker exec -it trading-postgres bash -c \
  'psql -U "$(cat /run/secrets/db_user)" -d trading_bot'

# Or for a one-shot query without an interactive shell:
docker exec -i trading-postgres bash -c \
  'PGPASSWORD="$(cat /run/secrets/db_password)" \
   psql -U "$(cat /run/secrets/db_user)" -d trading_bot -At -c "SELECT 1;"'
```

### Re-test 1 — CSRF on emergency-exit + global-settings save

Fixes: `dddb1a5` pro_controls panicSell sends `X-CSRF-Token`;
`bd0a0a6` global_settings saveSettings sends `X-CSRF-Token`.

```bash
# 1a. CSRF middleware must 403 a POST that lacks the header.
echo -n "no-header POST status: "
curl -s -o /dev/null -w '%{http_code}\n' -b "$COOKIE_JAR" -X POST \
    "$BASE/api/bot/emergency-exit"
# Expect: 403

# 1b. Same POST WITH the X-CSRF-Token header must NOT 403.
echo -n "with-header POST status: "
curl -s -o /dev/null -w '%{http_code}\n' -b "$COOKIE_JAR" -X POST \
    "$BASE/api/bot/emergency-exit" \
    -H "X-CSRF-Token: $(csrf)"
# Expect: 200 (DRY_RUN — no actual sells) — definitely NOT 403.

# 1c. settings save echoes the token correctly.
echo -n "settings save status: "
auth_post "$BASE/api/settings/update" \
    -d '{"config_type":"smoke_noop","updates":{}}' \
    -o /dev/null -w '%{http_code}\n'
# Expect: 200 (the backend may reject the unknown config_type but it
# must NOT be a 403)
```

UI re-test: open `/pro-controls` → click "Panic Sell" (DRY_RUN — safe).
Should alert "Emergency exit triggered!" not "Emergency exit FAILED:
CSRF token missing or invalid". Open `/global-settings` → expand any
category → click Save → should alert "Saved <category> settings." not
"Save failed: CSRF token missing or invalid".

### Re-test 2 — MODE badge no longer UNKNOWN

Fix: `f8375db` adds `?v={git-sha}` cache buster to all `<script>` tags
and `Accept: application/json` + content-type sanity check in
`refreshBotModeBadge`.

```bash
# 2a. Confirm payload shape.
auth_get "$BASE/api/bot/status" | jq '{success, dry_run: .data.dry_run, mode: .data.mode}'
# Expect: {"success": true, "dry_run": true, "mode": "DRY_RUN"}

# 2b. Confirm the dashboard's HTML now appends asset_version.
auth_get "$BASE/" | grep -oE '/static/js/main\.js\?v=[^"]+'
# Expect: /static/js/main.js?v=<sha-or-timestamp> (NOT just main.js)
```

UI re-test: hard-reload any dashboard page (Ctrl-Shift-R). Top-right
badge should read `🔵 DRY-RUN` within 5 seconds. If still UNKNOWN, open
DevTools console — look for `[bot-mode]` warnings explaining the
payload shape or non-JSON response.

### Re-test 3 — Sniper cap tile shows X/Y

Fix: `118f2ae` (rolled into Agent 2's commit) — `api_get_sniper_stats`
falls back to `config_settings.max_active_positions` (then 500) when
the runtime snapshot is missing or 0.

```bash
# 3a. Even right after a restart, max_active_positions must be > 0.
auth_get "$BASE/api/sniper/stats" | jq '{
    active_positions,
    active_positions_effective,
    max_active_positions,
    status
}'
# Expect: max_active_positions >= 1 (the config_settings value, or 500).
# active_positions_effective >= active_positions (uses max() of the two).
```

UI re-test: open `/sniper/dashboard`. The "Tokens Sniped" tile detail
line should read `Active: N/M` (with the `/M`), with the `⚠ near cap`
warning chip appearing when `N/M >= 0.8`.

### Re-test 4 — Risk-critical CSS readable on dark theme

Fix: `bd0a0a6` — `.risk-critical` defaults to dark-red bg + pink text
(legible on the default dark theme); `.theme-light .risk-critical`
restores the original light pink + dark red.

```bash
# 4a. Confirm the new CSS shipped.
auth_get "$BASE/global-settings" | \
    grep -E '\.risk-critical\s*\{|\.theme-light\s+\.risk-critical' | head -4
# Expect: two rules — the default (dark theme) and the .theme-light override.
```

UI re-test: open `/global-settings` on the dark theme. Risk-critical
rows (e.g. `max_drawdown_pct`) should have a dark-red background with
clearly readable pinkish text. Toggle to light theme via the sun/moon
icon → the same rows render with the original light-pink background +
dark-red text.

### Re-test 5 — Smoke test script authenticates correctly

Fix: `scripts/dashboard_smoke.sh` — one-shot script that runs all of
the above plus VaR/CVaR + copytrading-stats + arbitrage-stats checks.

```bash
# Run it from the host (NOT from inside the container — the script
# only needs HTTP access to port 8080, no DB):
bash scripts/dashboard_smoke.sh http://127.0.0.1:8080 admin admin123
# Expect: every line prefixed "  PASS  ...", final line:
#   "RESULT: PASS — all assertions held"
# Exit code: 0 on full pass, 1 if any FAIL appeared.
```

If any assertion FAILs, the line tells you which field/endpoint
returned the wrong shape — paste the FAIL block to the next agent.
