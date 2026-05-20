-- Migration 029: futures funding-cost ledger (predicted vs realized).
--
-- Wave 3 deliverable #1. Operator wants visibility into the actual funding
-- bill the futures book is paying vs what the gate predicted, hour by hour.
--
-- Predicted = (cached funding_rate at gate time) × notional × side_sign
-- Realized  = the funding payment the exchange actually debited at the
--             funding interval (8h on Binance/Bybit USDT perps; we record
--             a per-hour rollup so the chart granularity is consistent).
--
-- The engine writes one row per (hour_bucket, symbol, source) with the
-- contributions accumulated within that hour. The dashboard widget reads
-- the trailing 24h aggregated by hour_bucket.
--
-- side_sign: +1 = LONG paying (positive funding) or SHORT receiving;
-- the predicted/realized columns are signed in USD (positive = cost to
-- the book, negative = revenue to the book). This keeps the chart simple
-- (one signed bar per bucket).

CREATE TABLE IF NOT EXISTS futures_funding_payments (
    id              BIGSERIAL PRIMARY KEY,
    hour_bucket     TIMESTAMPTZ NOT NULL,    -- date_trunc('hour', ts UTC)
    symbol          VARCHAR(32) NOT NULL,
    side            VARCHAR(8)  NOT NULL,    -- 'LONG' or 'SHORT' at time of charge
    notional_usd    NUMERIC NOT NULL DEFAULT 0,
    predicted_usd   NUMERIC NOT NULL DEFAULT 0,   -- signed; + = cost to book
    realized_usd    NUMERIC NOT NULL DEFAULT 0,   -- signed; + = cost to book
    exchange        VARCHAR(16) NOT NULL DEFAULT 'binance',
    network         VARCHAR(16) NOT NULL DEFAULT 'mainnet',
    source          VARCHAR(16) NOT NULL DEFAULT 'engine',  -- 'engine' | 'income' | 'exit'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (hour_bucket, symbol, side, exchange, network, source)
);

CREATE INDEX IF NOT EXISTS idx_futures_funding_hour
    ON futures_funding_payments (hour_bucket DESC);
CREATE INDEX IF NOT EXISTS idx_futures_funding_symbol_hour
    ON futures_funding_payments (symbol, hour_bucket DESC);
