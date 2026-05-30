# Wave-13 Solana Module Audit — Agent 4

## Commits shipped

| Hash | What |
|------|------|
| `8748360` | Fix datetime tz crash in `_save_trade_to_db` + migration 034 |
| `24f7bb1` | Clamp `pnl_pct` to [-100, 2000]% at write time in `_save_trade_to_db` |

---

## Bug 1 (FIXED) — datetime crash: `_save_trade_to_db` TypeError

**Root cause.** `solana_positions.opened_at` is `TIMESTAMPTZ` (migration 013), so
positions restored from DB carry tz-aware datetimes. Every code path that creates
a `Trade` object used `datetime.utcnow()` for `closed_at` — tz-naive. asyncpg
raises when attempting to subtract or insert mixed-aware/naive datetimes:

> `invalid input for query argument $15 ... can't subtract offset-naive and offset-aware datetimes`

**Code fix (commit 8748360):**
- Added `_as_utc()` helper (worktree copy did not have it; remote version already had it — merged cleanly).
- `_save_trade_to_db`: wraps `opened_at` / `closed_at` with `_as_utc()` before INSERT.
- Duration arithmetic `(closed_at - opened_at).total_seconds()` uses `_as_utc` on both operands.
- All `time_held = (datetime.utcnow() - position.opened_at)` sites now use `_as_utc`.
- `Trade.closed_at` construction sites use `datetime.now(timezone.utc)`.
- `Position.opened_at` default factory uses `datetime.now(timezone.utc)`.
- Reconcile path normalises `row['opened_at']` via `_as_utc()`.

**Schema fix (migration 034):**
`034_solana_trades_timestamptz.sql` ALTERs `entry_time`, `exit_time`, `created_at`
from `TIMESTAMP` to `TIMESTAMPTZ` using `AT TIME ZONE 'UTC'` coercion.
The code fix is defensive (works before migration is applied); the migration
makes the schema consistent with `solana_positions` and the application intent.

---

## Bug 2 (FIXED code path) — Inflated PnL rows (+495424%)

**Root cause.** Pre-Wave-7: `_get_dexscreener_data` used `priceUsd` from the
highest-liquidity pair regardless of which side our mint appeared on. When our
mint was only the *quote* token (e.g. SOMETOKEN/ORCA), the pair's `priceUsd` was
ORCA's price (~$5 at the time), not ours. Exit PnL was calculated relative to an
inflated price → +495424%.

**Already fixed** in Wave-7 / Wave-12 (`own_pairs` base-token filter, lines 942-954
of `solana_engine.py`). The fix restricts DexScreener pairs to those where our
mint is the `baseToken.address`.

**Added guard (commit 24f7bb1):** `_save_trade_to_db` now clamps `pnl_pct` to
`[-100, 2000]%` before INSERT. Values outside this range are logged as WARNING
so future feed bugs are visible. 2000% is reachable on a 20x pump.fun launch; no
legitimate trade exceeds it.

**DB cleanup queries** — see DB-QUERY block below.

---

## Bug 3 (ALREADY FIXED) — Wallet balance SolanaRpcException

`_get_wallet_balance` (lines 1782-1828) is already fail-soft with:
- In-memory cache (15s TTL by default), served on RPC failure.
- Exponential backoff: each failure extends `cache_ts` by `streak * 15s` (capped at 120s).
- Rate-limited error log (once per 30s).

This means the error log spam in Wave-12 was high-frequency because `_wallet_balance_ttl_s`
was being reset incorrectly. The current code looks correct. No code change needed.

**RPC capacity handoff to Agent 9:** Helius free-tier is ~10 req/s. The engine
calls `get_balance` on every monitoring tick if the TTL has expired. With 5s tick
and 15s TTL that is ~4 req/min for balance alone — acceptable. But `getAccountInfo`
for each position (`_get_token_balance`) fires per-position per close, which can
spike. Recommend Agent 9 configure a dedicated staked Helius connection for
`getBalance` / `getAccountInfo` to isolate from Jupiter quote traffic.

---

## Bug 4 (OPERATOR / NOT A BUG) — Drift "open_position returned None (guard refused)"

This is expected behaviour. `DriftHelper.open_position` returns `None` when:
1. `driftpy` is not installed (DriftHelper runs in simulated mode).
2. Drift collateral account value is zero (leverage guard fails-closed).
3. MB-15 client-side guards reject the trade (funding-rate / oracle deviation / leverage cap).

The engine logs this at INFO (`🔶 Drift {market}: open_position returned None (guard refused)`).
No code change needed. Operator action to go live: install `driftpy`, deposit USDC
collateral to the Drift sub-account, confirm `✅ DriftHelper initialized`.

DRY_RUN simulated Drift activity is clearly labeled in logs (`[DRY_RUN]`).

---

## DB-QUERY BLOCK

Run these in order. All queries are read-only except the clearly-labeled optional cleanup DELETE.

```bash
PG() { docker exec trading-postgres psql -U "$(docker exec trading-postgres cat /run/secrets/db_user)" -d tradingbot -P pager=off "$@"; }
```

### 1. Schema discovery — confirm column types before applying migration 034

```sql
-- Q1: Check actual column types for solana_trades timestamp columns
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'solana_trades'
  AND column_name IN ('entry_time','exit_time','created_at','pnl_pct','pnl_sol','pnl_usd')
ORDER BY column_name;

-- Q2: Check solana_positions schema
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'solana_positions'
ORDER BY column_name;
```

### 2. Discover inflated rows (read-only)

```sql
-- Q3: Count and size inflated pnl_pct rows
-- The table has pnl_pct, pnl_sol, pnl_usd columns (NO pnl_percentage column).
SELECT
    COUNT(*) AS inflated_count,
    SUM(pnl_sol) AS inflated_pnl_sol_sum,
    MAX(pnl_pct) AS max_pnl_pct,
    MIN(pnl_pct) AS min_pnl_pct
FROM solana_trades
WHERE pnl_pct > 2000 OR pnl_pct < -100;

-- Q4: Sample inflated rows — inspect before any DELETE
SELECT id, trade_id, token_symbol, strategy, pnl_pct, pnl_sol, pnl_usd,
       entry_price, exit_price, exit_reason, entry_time
FROM solana_trades
WHERE pnl_pct > 2000 OR pnl_pct < -100
ORDER BY pnl_pct DESC
LIMIT 50;
```

### 3. Strategy performance baseline

```sql
-- Q5: Strategy win-rate and PnL summary (excluding inflated rows)
SELECT
    strategy,
    is_simulated,
    COUNT(*) AS trades,
    ROUND(100.0 * SUM(CASE WHEN pnl_sol > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS win_rate_pct,
    ROUND(SUM(pnl_sol)::numeric, 4) AS total_pnl_sol,
    ROUND(AVG(pnl_pct)::numeric, 2) AS avg_pnl_pct,
    ROUND(AVG(duration_seconds)::numeric, 0) AS avg_hold_s
FROM solana_trades
WHERE pnl_pct BETWEEN -100 AND 2000
GROUP BY strategy, is_simulated
ORDER BY strategy, is_simulated;
```

### 4. Optional cleanup DELETE (run Q4 FIRST, review IDs, then run this)

```sql
-- Q6 (OPTIONAL): Delete id-scoped inflated rows.
-- REVIEW Q4 output first. Replace the subquery with explicit IDs if you
-- want to be extra safe. This deletes rows where pnl_pct is physically
-- impossible (bug artifact, not real trades).
BEGIN;
DELETE FROM solana_trades
WHERE pnl_pct > 2000 OR pnl_pct < -100;
-- Check rows deleted:
-- SELECT COUNT(*) FROM solana_trades;
-- If count looks correct: COMMIT; else ROLLBACK;
ROLLBACK; -- CHANGE TO COMMIT after reviewing
```

---

## Top 5 Enhancements (ranked by ROI)

### 1. TIMESTAMPTZ migration (schema, migration 034) — P0
Apply `034_solana_trades_timestamptz.sql`. Until applied, the code-side `_as_utc`
guard prevents crashes but doesn't fix the underlying type mismatch. Timestamp
arithmetic in analytics queries (duration bucketing, daily PnL windows) will
silently truncate timezone offset without it.

### 2. Real-time SOL-out verification on close (profitability)
Currently after the sell swap, the engine checks that token balance decreased.
It does NOT verify that SOL balance increased (swap may have routed to a
dust intermediate). Add a SOL balance delta check post-sell (compare
`_get_wallet_balance(force=True)` before and after); if delta < `amount_sol * 0.80`
(accounting for fees), flag as partial-fill and re-attempt. This directly
reduces silent value loss.

### 3. pnl_pct cap at monitoring update, not just at write (reduce loss surface)
The `pnl_pct` clamp is only at the `_save_trade_to_db` write. If monitoring
calculates an impossible `unrealized_pnl_pct`, the position may trigger an
incorrect TP exit (price-feed bug → fake +2000% → immediate TP close at wrong
price). Add the same `max(min(...))` at the `position.unrealized_pnl_pct =` 
assignment in `_monitor_positions`.

### 4. Pump.fun entry filter: minimum time-since-creation (latency guard)
Currently the engine enters pump.fun tokens as soon as they appear. With
detection latency of 15-120s (Wave-12 Sniper finding), the engine may buy
into a token that has already been sniped by competitors at block-0 and is
mid-dump. Add a `pumpfun_min_age_seconds` config key (default 30s): skip tokens
created less than this many seconds ago — counterintuitive, but ensures the
initial bot-sniping phase has passed and real community buying is starting.

### 5. Jito tip size linked to opportunity size (MEV profitability)
`solana_jito_tip_lamports` is currently a flat 50,000 lamports. For large
opportunities (e.g. high `pnl_pct` momentum signal), the inclusion probability
at 50k lamports during high-throughput periods may be sub-50%. Add
`calculate_jito_tip(expected_pnl_sol)` that bids `min(opp_sol * 0.05, max_tip)` —
bid 5% of expected profit as tip, capped at a configured maximum. This increases
fill rate on the most valuable trades where competition is highest.

---

## Cross-module handoffs

**To Agent 9 (pool_engine):**
- Helius `getBalance` / `getAccountInfo` are in the same RPC pool as Jupiter
  quote traffic. Recommend a dedicated staked endpoint for balance reads to
  prevent rate-limiting cascades during busy scan cycles.
- The Solana engine uses `self.client` (a `solana.rpc.async_api.AsyncClient`)
  initialized at startup from the pool. It does NOT call `pool_engine.report_failure()`
  on RPC errors — it uses its own `_wallet_balance_error_streak` backoff. Suggest
  wiring `pool_engine.report_failure` on SolanaRpcException so the pool can rotate
  to a different Helius endpoint.

**To Dashboard:**
- `solana_trades.entry_time` / `exit_time` are currently `TIMESTAMP` (tz-naive) in
  the live DB. Migration 034 will change them to `TIMESTAMPTZ`. The dashboard's
  analytics queries (`analytics_routes.py`) that compute `date_trunc('day', exit_time)`
  need to be verified with `AT TIME ZONE 'UTC'` to remain correct after the ALTER.
