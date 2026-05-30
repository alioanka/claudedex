-- Migration: arbitrage_runtime_stats — JSONB snapshot keyed by chain so the
-- standalone dashboard can read in-process counters (gas spend per hour, gas
-- budget, etc.) that otherwise only exist inside the per-chain arbitrage
-- subprocess.
--
-- Single row per chain (one ARBITRAGE subprocess per chain when multiple
-- EVMArbitrageEngine instances are spun up). Updated periodically from
-- EVMArbitrageEngine._persist_runtime_stats. Read by
-- /api/arbitrage/gas-spend and the /arbitrage/dashboard widget.
--
-- Date: 2026-05-19

CREATE TABLE IF NOT EXISTS arbitrage_runtime_stats (
    chain        TEXT        PRIMARY KEY,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stats        JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_arb_runtime_stats_updated
    ON arbitrage_runtime_stats (updated_at DESC);
