# ARBITRAGE — Wave-3 Audit + Enhancement

**Branch:** `claude/create-expert-agents-JFSF5`
**Owner:** A2 — smartcontract-web3-expert
**Window:** 2026-05-19
**Status:** Two of three carry-over items closed. Triangular stays gated (operator approval pending).

## Wave-3 scope (per PM brief)

Carry-over from `PM_FINAL.md` / `ARBITRAGE_CAMPAIGN.md` "Items deferred":

1. Per-DEX realized-slippage learning to replace static `default_slippage_pct` (needs ~7d of honest PnL rows). [DONE]
2. Live `_gas_spend_usd_hour` tile in monitoring dashboard. [DONE]
3. Triangular atomic-receiver contract deploy. [SKIPPED — operator approval]

## Shipped this wave

### W3-1 — Per-(chain, dex_pair, pair_symbol) realized-slippage learning
Commit: `f2e95d3`

- New migration `025_add_arb_realized_slippage.sql` (one row per
  `(chain, dex_pair, pair_symbol)`; unique index; updated_at desc index).
- `EVMArbitrageEngine.get_realized_slippage(buy_dex, sell_dex, pair_symbol, *, use_p90=False)` — looks up median (default) or p90 from the in-memory cache; returns `None` if cold-start (no key) or below `_realized_slip_min_samples=5`. Caller falls back to `chain_config['default_slippage_pct']`.
- `EVMArbitrageEngine._refresh_realized_slippage()` — hourly TTL-gated; scans `arbitrage_trades` over trailing 7d on this chain; buckets by `(buy_dex, sell_dex, pair_symbol)`; computes median + p90 sorted by realized-slippage value; upserts `arb_realized_slippage`; rebuilds in-memory cache.
- Per-trade attribution: `realized = max(0, gross - net - flash_fee_pct - gas_cost/entry_usd)`. Negative deltas (failed-tx logging artifacts) are dropped before aggregation so they don't skew the median.
- `_log_arb_trade` now persists `pair_symbol`, `slippage_pct`, `slippage_source` (`realized_median` or `static_default`) in `arbitrage_trades.metadata` so the refresh is reproducible and the dashboard can later distinguish per-row slippage attribution.
- **Pre-execute gate uses p90** (bias toward skipping marginal trades during fat-tail regimes); **`_log_arb_trade` uses median** (PnL honesty).
- Bug fix: latent `GAS_COST_USD` `NameError` at lines 2357/2388 (metadata write + Telegram alert) — leftover from `8cf0143` — now use the live `gas_cost_usd` variable already in scope. Without this, every successful arb log row would have crashed in the `except` block (silently logged-only-as-error, no DB write).
- Refresh runs inside the existing 5-min `_log_stats_if_needed` cadence; TTL gate inside the method makes the cadence safe (no-op while fresh).

**ROI:** Reduces false-positive opportunity execution on pools with chronic high slippage (e.g. shallow L2 V2 routes vs deep Curve pools on the same pair) while still firing on pairs where realized slippage is below the chain-level default. Loss-surface reduction + profitability lever.

### W3-2 — Live `_gas_spend_usd_hour` dashboard tile
Commit: `329b784`

- New migration `028_add_arbitrage_runtime_stats.sql` — JSONB snapshot keyed by chain (one row per `EVMArbitrageEngine` subprocess).
- `EVMArbitrageEngine._persist_runtime_stats()` — snapshots `gas_spend_usd_hour`, `gas_budget_usd_per_hour`, `gas_budget_ratio`, `gas_window_age_s`, `gas_spike_multiplier`, `min_profit_threshold_effective`, `realized_slip_keys` + trusted count, scans / opps. Called from `_log_stats_if_needed` on the 5-min cadence; fail-soft.
- New HTTP route `GET /api/arbitrage/gas-spend` in `enhanced_dashboard.py` — reads all chain rows, returns per-chain breakdown + roll-up (`total_spend_usd`, `total_budget_usd`, `overall_ratio`), `stale=True` flag when `max_age_s > 600` (10-min slack on top of 5-min snapshot cadence).
- Widget on `/arbitrage/dashboard` — new stat-card tile in the existing stats grid. Colour-coded value (green <50%, amber 50-85%, red >=85%, grey if stale). Tooltip lists per-chain spend/cap detail. Reuses the existing 15s auto-refresh poll (no new timer).
- Read-only endpoint; no CSRF header needed.

**ROI:** Pure observability. Closes the explicit Wave-2 carry-over item. Without this, operators only saw budget enforcement after the engine logged `Gas budget gate: hourly gas budget $X.XX would be exceeded` — now the cap-approach is visible before the gate trips.

### W3-3 — Triangular atomic-receiver
**SKIPPED.** Per PM hard rule. Triangular stays gated by the MB-05 atomic-receiver guard (`triangular_engine.py:906-911` returns `None` before any swap build). Contract deploy requires operator approval.

## Tests

`tests/unit/test_arb_realized_slippage.py` — 12 unit tests:

- `_slip_key` key format pinned (`buy_dex->sell_dex|TOKEN/QUOTE`)
- `get_realized_slippage` returns `None` on cold start (NOT `0.0`)
- `get_realized_slippage` returns `None` below min_samples (caller fallback)
- median / p90 lookup path
- p90 explicitly via `use_p90=True` flag
- unknown key => `None`
- `_refresh_realized_slippage` aggregation: median + p90 + sample count over 10 mocked rows (sorted-realized 0.1%..1.0% step 0.1% => median 0.6%, p90 1.0%)
- negative-clamp: failed-tx artifact rows (gross < costs) dropped before aggregation
- bucketing: distinct (buy_dex, sell_dex) and pair_symbol land in distinct keys
- TTL gate: in-window refresh is a no-op (DB not hit)
- no-db path: `db_pool=None` returns cleanly
- fallback-path pin: helper returns `None` (NOT `0.0`) so caller's `if x is None: x = static` works

All 12 pass.

## Migration ledger

| File | Origin | Purpose | Status |
|---|---|---|---|
| `migrations/025_add_arb_realized_slippage.sql` | `f2e95d3` | New `arb_realized_slippage(chain, dex_pair, pair_symbol)` keyed table for the 7d rolling median+p90 estimator | Applied at boot |
| `migrations/028_add_arbitrage_runtime_stats.sql` | `329b784` | New `arbitrage_runtime_stats(chain, stats JSONB)` snapshot table for the gas-spend tile | Applied at boot |

Migration 026 / 027 / 029 were claimed by other concurrent agents in this wave (COPY slippage_tracker, FUTURES leverage overrides, FUTURES funding payments).

## HTTP surface

| Method | Route | Origin | Notes |
|---|---|---|---|
| GET | `/api/arbitrage/gas-spend` | `329b784` | Per-chain + rolled-up hourly gas spend / budget / ratio; `stale=true` if no snapshot in last 10 min |

## Deferred for Wave-4+

- Per-leg slippage attribution (split the realized estimate between buy_dex and sell_dex). Today we attribute jointly to the (buy_dex, sell_dex) tuple because the on-chain receipt doesn't expose per-leg fill price separately from the bundled flash-loan PnL. Would need either (a) two RPC simulate calls before send (expensive) or (b) emitting per-leg `Swap` event parsing post-receipt.
- Cross-engine roll-up of the realized-slippage cache (today each chain subprocess maintains its own cache; the dashboard reads the DB so cross-chain comparison already works for ops, but a single in-process planner would benefit from sharing the cache).
- Triangular atomic-receiver contract deploy (operator approval).
- Test Runner catalog entries for `/api/arbitrage/gas-spend` and the `arb_realized_slippage` schema/freshness (T1/T2 territory after Wave-3 close).

## Final state

Two of three Wave-3 carry-over items closed. Spatial arbitrage remains
GREEN candidate (ready for chain-by-chain LIVE flip). Triangular stays
explicitly gated. No new P0/P1 introduced; one latent `GAS_COST_USD`
NameError closed as part of W3-1.
