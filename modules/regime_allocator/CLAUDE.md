# REGIME_ALLOCATOR Module

## What it does
Volatility-regime-aware capital allocator. **ADVISORY ONLY.** On a slow
cadence (default hourly) it:
1. Fetches BTCUSDT + ETHUSDT closes from a FREE public kline endpoint
   (Binance spot REST, Bybit v5 fallback — no API key, no LLM, 2 requests
   per tick). If both sources fail the tick is skipped (fail-soft).
2. Classifies the market regime with pure, self-tested math
   (`core/regime_classifier.py`): realized-vol expansion vs compression
   crossed with trend vs range (Kaufman efficiency ratio).
3. Writes one audit row to `regime_snapshots` and a batch of per-module
   capital-weight PROPOSALS to `regime_allocation_proposals` for **operator
   approval**. Prior pending rows are superseded each tick (no stacking).

It **never trades, never flips a live/DRY_RUN flag, never writes
`logs/.killswitch` or any `logs/.pause_*` flag.** The only optional side
write is mirroring its proposal rows into `portfolio_allocations`
(`proposed_by='regime'`), gated behind a DB knob that defaults to false.

## Entry point
`modules/regime_allocator/main_regime_allocator.py` — launched by `main.py`
when `REGIME_ALLOCATOR_MODULE_ENABLED=true` (default **false**). Health
server on port **8091** (`REGIME_ALLOCATOR_HEALTH_PORT`) — `/health` and
`/status` (last tick summary). Decision math: `core/regime_classifier.py`
(pure, deterministic, self-tested:
`python -m modules.regime_allocator.core.regime_classifier`).
Fetch/loop/persistence: `core/regime_engine.py`.

## Regime taxonomy + decision math (operator-derivable, no black box)
Per asset (BTC weight 0.6, ETH 0.4 by default):
```
vol_ratio = stdev(log-returns, last short_vol_bars) / stdev(log-returns, last long_vol_bars)
  >= vol_expand_ratio (1.15)  -> expansion
  <= vol_compress_ratio (0.85) -> compression
ER = |net move| / sum(|bar moves|) over trend_bars   (Kaufman efficiency ratio)
  >= er_trend (0.35) -> trend;  <= er_range (0.20) -> range
```
Weighted votes across BTC/ETH map to one of five regimes:
`trend_expansion` (momentum modules up), `chop_expansion` (de-risk +
extra reserve; vol without direction is where momentum bleeds),
`trend_compression` (grind: futures/copy carry up), `range_compression`
(arbitrage / mean-capture up), `neutral` (no tilt).
`confidence = 0.5*cross-asset agreement + 0.5*threshold exceedance`.

Weights:
```
eff_tilt_i = 1 + confidence * (tilt_i(regime) - 1)     # low conviction -> ~1.0
weight_i   = base_i * eff_tilt_i, normalized to (100 - reserve_pct)
```
Below `min_confidence` the tilt is fully suppressed (equal/base weights).
The tilt matrix is in `regime_classifier.DEFAULT_TILTS`, overridable via the
`regime_tilts` JSON knob (partial overrides merge over defaults).

## Key config (DB-backed, config_type='regime_allocator'; migration 119)
| Key | Default | What it does |
|---|---|---|
| `tick_interval_seconds` | 3600 | Cadence (regimes move slowly; do not go below ~900) |
| `kline_interval` | '4h' | Candle size: '1h', '4h' or '1d' |
| `kline_limit` | 180 | Bars fetched (180 x 4h = 30 days) |
| `short_vol_bars` | 24 | Recent realized-vol window |
| `long_vol_bars` | 96 | Baseline realized-vol window |
| `trend_bars` | 42 | Efficiency-ratio window |
| `vol_expand_ratio` | 1.15 | short/long RV at/above -> expansion |
| `vol_compress_ratio` | 0.85 | at/below -> compression |
| `er_trend` / `er_range` | 0.35 / 0.20 | ER trend/range thresholds (gap = hysteresis) |
| `btc_weight` | 0.6 | BTC vote weight (ETH = 1 - this) |
| `reserve_pct` | 10.0 | Always-unallocated cushion |
| `chop_extra_reserve_pct` | 10.0 | Added reserve in chop_expansion (confidence-scaled) |
| `min_confidence` | 0.25 | Below this, tilts are suppressed entirely |
| `regime_tilts` | '' | Optional JSON override of the tilt matrix |
| `base_weights` | '' | Optional JSON per-module base weights (default equal) |
| `mirror_to_portfolio_allocations` | false | Also write rows to portfolio_allocations (proposed_by='regime') |

## Kill switch
- Global: `logs/.killswitch` — tick skipped (also polled via
  `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_regime_allocator` — tick skipped.

## Logs
`logs/regime_allocator/` — `regime_allocator.log`, `regime_allocator_errors.log`.

## DB tables (migration 118)
- `regime_snapshots` — one row per tick: regime, confidence, reason, full
  component breakdown, price source.
- `regime_allocation_proposals` — one row per module per tick; pending =
  `approved_at IS NULL AND superseded_at IS NULL`. Operator approves by
  setting `approved_at`/`approved_by` (dashboard follow-up).

## Authority hierarchy (matches the wave-15 split)
1. ORCHESTRATOR `budget_usd` — LIVE enforcement (AllocationGuard).
2. PORTFOLIO_ALLOCATOR — advisory, performance-driven (7d Sharpe Kelly).
3. REGIME_ALLOCATOR (this module) — advisory, **regime-conditioned**; answers
   "given the tape, which STYLE of module should run heavier?" while the
   portfolio allocator answers "who has earned it lately?". The portfolio
   allocator annotates its proposals with the latest regime snapshot
   (fail-soft read) so the operator sees both lenses side-by-side.

## Dashboard surface (owned by the dashboard agent — follow-up)
Read-only + fail-soft: latest `regime_snapshots` row (regime + confidence +
reason) and the pending `regime_allocation_proposals` batch with
approve/reject buttons (approve = set `approved_at`/`approved_by`). Hide the
panel when the tables are absent.

## Isolation / safety
Reads: free public klines, `config_settings`, module-enabled env flags.
Writes ONLY: `regime_snapshots`, `regime_allocation_proposals`, and (knob
default-off) `portfolio_allocations` rows tagged `proposed_by='regime'`.
Never imports a trading executor, never signs anything, no paid APIs.
