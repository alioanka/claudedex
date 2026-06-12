# MARKET_DATA_WAREHOUSE Module

## What it does
Unified historical market-data store. A slow ingest loop periodically captures
normalized OHLCV candles + realized perp funding-rate history for the symbols
the bot trades from FREE public sources (Binance/Bybit public REST — no API
keys) into compact, deduplicated Postgres tables, so consumer modules
(backtest_replay, regime_allocator, param_tuner, options_vol, advisor/ML
retraining, TCA post-mortems) read ONE consistent history instead of each
re-fetching. **PURE DATA — never trades, never signs, holds no secrets.**
The failure mode is wasted disk, not lost money.

## Entry point
`modules/market_data_warehouse/main_market_data_warehouse.py` — launched by
`main.py` when `MARKET_DATA_WAREHOUSE_MODULE_ENABLED=true` (default **false**).
Health server on port 8095 (`MARKET_DATA_WAREHOUSE_HEALTH_PORT`) with:
- `GET /health` — liveness
- `GET /status` — last ingest-tick summary (candles/series counts, errors)
- `GET /query` — read-only operator peek:
  - `/query` → stored (symbol, timeframe, source) keys
  - `/query?symbol=BTC/USDT&timeframe=1h&limit=100` → recent candles
  - `/query?symbol=BTC/USDT&metric=funding_rate&limit=50` → funding series
  - `/query?symbol=BTC/USDT&timeframe=1m&coverage=1` → history extent per source

## Read-accessor API (how other modules consume the warehouse)
Import `modules.market_data_warehouse.reader` — never query the tables
directly, so the schema can evolve behind one surface. Every function takes an
asyncpg pool/connection first, is read-only + fail-soft (errors → empty
result), accepts any symbol spelling (`BTCUSDT`, `btc/usdt`, ...), and returns
oldest-first dicts with epoch-second `ts` (replay-friendly):
```python
from modules.market_data_warehouse import reader

candles = await reader.get_candles(pool, 'BTC/USDT', '1h', start=..., end=..., limit=500)
funding = await reader.get_funding(pool, 'BTC/USDT', limit=100)          # metric sugar
series  = await reader.get_series(pool, 'BTC/USDT', 'funding_rate')      # generic
px      = await reader.get_latest_close(pool, 'SOL/USDT', '1m')          # stored, NOT live
cov     = await reader.get_coverage(pool, 'ETH/USDT', '1m')              # rows + first/last ts per source
keys    = await reader.list_symbols(pool)
```
Contract notes: with no `source=` pinned the densest stored source is used and
never mixed within one result; `get_latest_close` staleness is bounded by the
ingest interval; **consumers MUST check `get_coverage` before trusting a
backtest window** — gap-free history is not guaranteed.

## Internals
- `core/normalizer.py` — PURE (stdlib-only, no I/O): canonical symbol
  (`BASE/QUOTE`) + timeframe forms, candle/series validation (NaN/inf,
  high<low, OHLC bounds, ts flooring), dedup on the storage key. Self-tested:
  `python -m modules.market_data_warehouse.core.normalizer`.
- `core/sources.py` — fail-soft public-REST fetchers (Binance spot klines,
  Binance USD-M funding, Bybit v5 linear klines + funding). Return `[]` on any
  error, never raise. Adding a source = one fetcher + one registry entry.
- `core/warehouse_engine.py` — ingest tick: per-key high-water mark from the
  DB → fetch only what is new (minus one bar so the previously in-progress
  candle is corrected) → idempotent upsert; per-tick request budget +
  inter-request spacing; retention purge.
- `reader.py` — the consumer API above.

## Key config (DB-backed, config_type='market_data_warehouse'; migration 123)
| Key | Default | What it does |
|---|---|---|
| `ingest_interval_seconds` | 300 | Seconds between ingest ticks |
| `symbols` | BTC/USDT,ETH/USDT,SOL/USDT | CSV of pairs to archive (only add symbols with a committed consumer) |
| `candle_timeframes` | 1m,1h | CSV of timeframes (1m,5m,15m,30m,1h,4h,1d) |
| `candle_sources` | binance,bybit | Enabled candle sources |
| `funding_enabled` | true | Archive funding-rate history |
| `funding_sources` | binance,bybit | Enabled funding sources |
| `candles_per_request` | 200 | Candles per API call (venue cap 1000) |
| `max_requests_per_tick` | 40 | Hard per-tick request budget (politeness) |
| `request_spacing_seconds` | 0.35 | Client-side spacing between requests |
| `retention_days_1m` | 30 | Purge 1m candles past this (coarser TFs are the long-lived downsample) |
| `retention_days_default` | 365 | Retention for non-1m candles |
| `retention_days_series` | 365 | Retention for series rows |
| `purge_enabled` | true | Master retention switch — keep on; the warehouse must stay disk-bounded |

## Kill switch
- Global: `logs/.killswitch` — ingest tick skipped (polled via
  `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_market_data_warehouse` — tick skipped.

## Logs
`logs/market_data_warehouse/` — `market_data_warehouse.log`,
`market_data_warehouse_errors.log` (both rotating).

## DB tables (migration 123)
- `market_candles` — UNIQUE `(source, symbol, timeframe, ts)` = the
  idempotent-upsert/dedup key; index `(symbol, timeframe, ts DESC)`.
- `market_series` — generic scalar series (today: `metric='funding_rate'`;
  mark price / OI / realized vol later with NO schema change); UNIQUE
  `(source, symbol, metric, ts)`.

## Isolation / safety
Reuses read-only: `core/dry_run.py` (killswitch poll), `config_settings`.
Writes only `market_candles` + `market_series`. No executor imports, no keys,
no LLM spend, no order paths. Rate-limited + fail-soft on every source: a dead
endpoint is skipped and the tick continues.

## Orchestrator wiring still needed (NOT done by this module — report items)
- `main.py`: launch `modules/market_data_warehouse/main_market_data_warehouse.py`
  when `MARKET_DATA_WAREHOUSE_MODULE_ENABLED=true` (mirror the meta_controller
  block).
- `.env.example`: `MARKET_DATA_WAREHOUSE_MODULE_ENABLED=false` and
  `MARKET_DATA_WAREHOUSE_HEALTH_PORT=8095`.
- Root `CLAUDE.md` module table + health-port map: 8095 = market_data_warehouse.
- Optional dashboard: coverage panel reading `/status` + `reader.list_symbols`.
