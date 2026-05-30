# Wave-13 Sniper Module — Agent 6 Report

**Date:** 2026-05-30  
**Agent:** 6 (Solana/EVM Sniper + MEV expert)  
**Branch:** claude/friendly-ramanujan-nMWNv  
**Scope:** `modules/sniper/` only

---

## Commits Delivered

| Hash | Description |
|------|-------------|
| `6a30d47` | `[sniper] wave-13: fix absurd PnL bug — USD/native price unit mismatch` |
| `8581b90` | `[sniper] wave-13: WSS 429 detection + RPC endpoint rotation on rate-limit` |

---

## Bug 1 Fixed: Absurd PnL ("+594089%", "+1613720%")

**Root cause (end-to-end):**

- `entry_price` is stored as native-per-token: `amount_in_sol / amount_out_tokens`  
  With `_simulate_buy`'s 1e6 multiplier, this is `0.1_SOL / 100_000_tokens` = `1e-6 SOL/token`.
- `current_price` from Jupiter Price v2 / Birdeye / Pyth is USD-per-token:  
  e.g. `$0.0002/token` for a fresh pump.fun mint.
- The monitor computed `pnl_pct = (0.0002 - 1e-6) / 1e-6 * 100` ≈ `+19900%` at minimum.  
  At SOL=$200, the mismatch is ~200x, producing the observed +500k–+1.6M% values.

**Fix applied in `_monitor_active_snipes` (sniper_engine.py):**

1. Derive `entry_price_usd = entry_usd / amount_held` — USD per token from stored trade data. Both legs are now in USD.
2. Add `phantom-price guard`: if `|pnl_pct| > 200%` AFTER the unit fix, the price source returned a stale/wrong value. DRY_RUN: route to `_close_position_synthetic` with reason `phantom_price_dry_run` (uses Wave-7 honest modeled exit). LIVE: log WARNING and skip — do not sell at a phantom price.
3. Log lines for TAKE PROFIT / STOP LOSS no longer include emoji (unambiguous grepping).

**DB row integrity:** Wave-12 already clamped `_simulate_sell` output to `move ∈ [0.01, 3.0]`, ensuring `profit_loss_pct ∈ [-99%, +200%]` in the DB. The engine-side fix now prevents the phantom trigger from firing at all, making the DB cap a true belt-and-suspenders.

---

## Bug 2 Fixed: WSS 429 Rate-Limit Disconnects

**Root cause:** When Helius rate-limits the WSS connection it closes with HTTP 429. Previously the generic exception handler slept `backoff` seconds and reconnected to the **same** endpoint, guaranteeing another 429 immediately.

**Fix applied in `_wss_listen_loop` (solana_listener.py):**

1. Detect 429/503/`Too Many Requests` in the exception message.
2. Call `RPCProvider.report_rate_limit('SOLANA_RPC', url, 300)` to mark the endpoint throttled.
3. Fetch next available endpoint via `RPCProvider.get_rpc('SOLANA_RPC')`.
4. Re-derive the WSS URL from the new HTTP endpoint and reconnect there.
5. Add `wss_rate_limits` counter to `_stats` (visible in `sniper_runtime_stats.solana_listener`).
6. Falls through to existing exponential backoff (1s → 2s → ... → 60s) regardless.

**Handoff to Agent 9 (pool_engine):** pool_engine needs a dedicated `SOLANA_WSS` provider slot separate from `SOLANA_RPC` so WSS endpoints can be round-robined independently. Currently the code infers the WSS URL from the HTTP RPC URL — this works for Helius/QuickNode (same hostname), but a multi-provider setup may want separate WSS-specific keys.

---

## Bug 3: Active-Positions Cap — Already Enforced

Reviewed `_effective_active_count()` and both gate sites (`_evaluate_target` and `_execute_snipe`). Cap is enforced as `max(in-memory, DB open count)` so orphan rows from prior restarts count. `max_active_positions` is DB-configurable via `config_settings` (migration 016, default 500). No change needed.

---

## DB-QUERY BLOCK

PostgreSQL was not reachable in this worktree environment (no Docker socket, no running postgres on localhost:5432). Schema discovery was done via migration files instead.

**Schema source:** `migrations/010_add_sniper_ai_tables.sql`

Key findings from schema review:
- `profit_loss_pct` is `NUMERIC` (unbounded in DDL) — the Wave-12 `_simulate_sell` ±200% cap is the only guard in the application layer. Consider adding a `CHECK (profit_loss_pct BETWEEN -100 AND 300)` constraint in a future migration to make it DB-enforced.
- No index on `(chain, status)` compound — every `_effective_active_count` query does a full scan of `sniper_trades WHERE status='open'`. Add `CREATE INDEX idx_sniper_trades_chain_status ON sniper_trades(chain, status)` in migration 034 if the table grows past ~100k rows.
- `sniper_positions` table exists but the engine never writes to it (writes only to `sniper_trades`). Dashboard `api_get_sniper_positions` should query `sniper_trades WHERE status='open'` not `sniper_positions`.

---

## Top 5 Profitability Enhancements (ranked by ROI, implementation risk)

### 1. Market-cap gate at entry (HIGH ROI, low risk, data-dependent)
Fresh pump.fun mints with < $10k liquidity and < $5k initial market cap have 90%+ rug rate. Adding `min_market_cap_usd` and `min_pool_liquidity_usd` as separate DB config keys (distinct from the existing `min_liquidity`) and deriving market cap from Jupiter quote (already available) would reduce honeypot exposure without needing GoPlus. **Currently blocked on live data to calibrate thresholds.**

### 2. Dynamic take-profit based on holder velocity (MEDIUM ROI, medium risk)
If holder count grows >50 new holders/minute in the first 60s, increase TP threshold by 2x. If it stalls, tighten SL. Requires tracking holder count time-series (Helius `getTokenLargestAccounts` + cache). **Out of scope this wave; requires new data collection path.**

### 3. Jito bundle for snipe tx (HIGH ROI for LIVE, infrastructure gap)
Currently on Solana, `_execute_solana_buy` uses standard Jupiter swap. In a competitive new-mint environment, MEV bots front-run via Jito bundles. Wrapping the buy in a Jito bundle with a configurable tip (e.g. 0.01 SOL) would guarantee ordering ahead of other snipers. **Requires Jito client wiring not yet present in `trade_executor.py`.** Ticket for Agent 9: add `JITO_BLOCK_ENGINE_URL` to pool_engine and expose a `submit_jito_bundle(txs, tip_lamports)` helper.

### 4. `max_hold_minutes` as DB config key (LOW risk, HIGH fill-rate impact)
Already implemented in code (`self.max_hold_minutes`) and read from DB, but **not seeded in any migration**. Positions that never hit TP or SL hold indefinitely, tying up cap slots. Seeding `max_hold_minutes = 60` as default would turn over the position pool 24x/day. Candidate for migration 034.

### 5. Per-chain TP/SL differentiation (MEDIUM ROI, low risk)
Pump.fun launches are high-volatility (50% loss or 5x pump within minutes). Raydium V4 V2 launches are calmer. Using chain+source-specific TP/SL multipliers (e.g. `pump_fun_take_profit_multiplier = 3.0`) would tighten exits on rugs and widen them on genuine pumps. Requires two new DB config keys + a lookup in the monitor loop.

---

## Cross-Module / RPC / Dashboard Handoffs

- **Agent 9 (pool_engine):** Add `SOLANA_WSS` provider slot separate from `SOLANA_RPC`. WSS reconnect rotation currently re-derives WSS URL from HTTP endpoint — works for single-provider Helius but breaks multi-provider setups. Also: `JITO_BLOCK_ENGINE_URL` for future Jito bundling.
- **Dashboard agent:** `api_get_sniper_positions` queries `sniper_positions` table; engine writes to `sniper_trades`. Fix the query. Also: `api_get_sniper_close` / `close-all` handlers UPDATE `trades` table, not `sniper_trades` — silent no-ops (documented in modules/sniper/CLAUDE.md).
- **RiskManager integration:** Sniper calls `TokenSafetyChecker` locally but does NOT call `core/risk_manager.py RiskManager.validate_trade`. This is a P1 gap for LIVE trading — position sizing should be bounded by portfolio-level risk manager. Add as next-wave ticket.
- **Migration needed:** Seed `max_hold_minutes` and add compound index `(chain, status)` on `sniper_trades`. Number: 034 (if not taken by sibling agents).
