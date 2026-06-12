-- Migration 123: market_data_warehouse — config seeds + candle/series tables
--
-- Unified historical market-data store (modules/market_data_warehouse/).
-- PURE DATA: the module periodically ingests normalized OHLCV candles and
-- funding-rate series for the symbols the bot trades from free public
-- sources (Binance/Bybit public REST, no keys) so consumer modules
-- (backtest_replay, regime_allocator, param_tuner, options_vol, advisor/ML
-- retraining) read one consistent history via
-- modules.market_data_warehouse.reader instead of each re-fetching.
-- Applying this migration changes NO behavior: the module only runs when
-- MARKET_DATA_WAREHOUSE_MODULE_ENABLED=true (default false) and it never
-- places a trade, signs nothing, and holds no secrets.
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('market_data_warehouse', 'ingest_interval_seconds', '300', 'int',
     'Seconds between ingest cycles (default 5 min). The warehouse is a slow '
     'archival loop, not a live ticker.', NOW(), NOW()),

    ('market_data_warehouse', 'symbols', 'BTC/USDT,ETH/USDT,SOL/USDT', 'str',
     'CSV of canonical BASE/QUOTE pairs to archive. Keep this list to symbols '
     'with a committed consumer — storage without readers is pure cost.',
     NOW(), NOW()),

    ('market_data_warehouse', 'candle_timeframes', '1m,1h', 'str',
     'CSV of canonical timeframes to archive per symbol (1m,5m,15m,30m,1h,4h,1d). '
     '1m is the fine-grained window; 1h is the long-lived history.',
     NOW(), NOW()),

    ('market_data_warehouse', 'candle_sources', 'binance,bybit', 'str',
     'CSV of enabled candle sources (free public REST, no keys). '
     'Supported: binance, bybit.', NOW(), NOW()),

    ('market_data_warehouse', 'funding_enabled', 'true', 'bool',
     'Archive realized perp funding-rate history into market_series '
     '(metric=''funding_rate''). Read-only public data.', NOW(), NOW()),

    ('market_data_warehouse', 'funding_sources', 'binance,bybit', 'str',
     'CSV of enabled funding-rate sources. Supported: binance, bybit.',
     NOW(), NOW()),

    ('market_data_warehouse', 'candles_per_request', '200', 'int',
     'Max candles requested per API call (capped at 1000 by the venues).',
     NOW(), NOW()),

    ('market_data_warehouse', 'max_requests_per_tick', '40', 'int',
     'Hard request budget per ingest tick — keeps the module polite on free '
     'public endpoints. When exhausted, the remainder waits for the next tick.',
     NOW(), NOW()),

    ('market_data_warehouse', 'request_spacing_seconds', '0.35', 'float',
     'Sleep between consecutive public-API requests (client-side rate limit).',
     NOW(), NOW()),

    ('market_data_warehouse', 'retention_days_1m', '30', 'int',
     'Disk discipline: 1m candles older than this are purged. The coarser '
     'timeframes ARE the long-lived downsampled history.', NOW(), NOW()),

    ('market_data_warehouse', 'retention_days_default', '365', 'int',
     'Retention window for all non-1m candle timeframes.', NOW(), NOW()),

    ('market_data_warehouse', 'retention_days_series', '365', 'int',
     'Retention window for scalar series rows (funding etc).', NOW(), NOW()),

    ('market_data_warehouse', 'purge_enabled', 'true', 'bool',
     'Master switch for the retention purge. Disable only temporarily (e.g. '
     'before a manual export) — the warehouse must stay disk-bounded.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- Normalized OHLCV candles. One row per (source, symbol, timeframe, bar
-- open-time); the unique constraint is the idempotent-upsert key — re-ingest
-- refreshes the previously in-progress bar instead of duplicating it.
CREATE TABLE IF NOT EXISTS market_candles (
    id           BIGSERIAL PRIMARY KEY,
    source       TEXT NOT NULL,             -- 'binance' | 'bybit' | ...
    symbol       TEXT NOT NULL,             -- canonical 'BASE/QUOTE'
    timeframe    TEXT NOT NULL,             -- canonical '1m','1h',...
    ts           TIMESTAMPTZ NOT NULL,      -- bar open time (UTC, floored)
    open         DOUBLE PRECISION NOT NULL,
    high         DOUBLE PRECISION NOT NULL,
    low          DOUBLE PRECISION NOT NULL,
    close        DOUBLE PRECISION NOT NULL,
    volume       DOUBLE PRECISION NOT NULL DEFAULT 0,
    quote_volume DOUBLE PRECISION,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_market_candles_key UNIQUE (source, symbol, timeframe, ts)
);
CREATE INDEX IF NOT EXISTS idx_market_candles_sym_tf_ts
    ON market_candles (symbol, timeframe, ts DESC);
CREATE INDEX IF NOT EXISTS idx_market_candles_ts
    ON market_candles (ts);

-- Generic scalar time series (funding rates today; mark price / open
-- interest / realized vol later — adding a metric needs NO schema change).
CREATE TABLE IF NOT EXISTS market_series (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    symbol      TEXT NOT NULL,              -- canonical 'BASE/QUOTE'
    metric      TEXT NOT NULL,              -- 'funding_rate' | ...
    ts          TIMESTAMPTZ NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_market_series_key UNIQUE (source, symbol, metric, ts)
);
CREATE INDEX IF NOT EXISTS idx_market_series_sym_metric_ts
    ON market_series (symbol, metric, ts DESC);
CREATE INDEX IF NOT EXISTS idx_market_series_ts
    ON market_series (ts);

COMMIT;
