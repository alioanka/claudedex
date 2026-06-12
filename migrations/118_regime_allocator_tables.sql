-- Migration 118: regime_allocator — audit + proposal tables
--
-- The volatility-regime-aware capital allocator (modules/regime_allocator/).
-- ADVISORY ONLY: it classifies the market regime from free BTC/ETH price data
-- and writes per-module capital-weight PROPOSALS for operator approval. It
-- never trades, never flips a live flag, never touches the killswitch.
-- Applying this migration changes NO behavior; the module only runs when
-- REGIME_ALLOCATOR_MODULE_ENABLED=true (default false).
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

-- One row per tick: the classified regime + full component breakdown so the
-- operator can recompute every number by hand.
CREATE TABLE IF NOT EXISTS regime_snapshots (
    id            BIGSERIAL PRIMARY KEY,
    regime        TEXT NOT NULL,            -- trend_expansion | chop_expansion |
                                            -- trend_compression | range_compression | neutral
    confidence    DOUBLE PRECISION,         -- 0..1
    reason        TEXT,                     -- operator-readable, contains the numbers
    components    JSONB,                    -- per-asset vol_ratio / ER / votes / params
    price_source  TEXT,                     -- e.g. 'binance/binance:4h'
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_regime_snapshots_created
    ON regime_snapshots(created_at DESC);

-- One row per module per tick. The pending batch is
-- approved_at IS NULL AND superseded_at IS NULL; each tick supersedes the
-- prior pending batch so proposals never stack. Operator approval = setting
-- approved_at / approved_by (dashboard follow-up). Approved rows are never
-- touched by the engine.
CREATE TABLE IF NOT EXISTS regime_allocation_proposals (
    id             BIGSERIAL PRIMARY KEY,
    snapshot_id    BIGINT REFERENCES regime_snapshots(id),
    regime         TEXT NOT NULL,
    confidence     DOUBLE PRECISION,
    module         TEXT NOT NULL,            -- short engine key (sniper, dex, ...)
    weight_pct     NUMERIC(5,2) NOT NULL,    -- proposed % of total book
    reason         TEXT,
    components     JSONB,                    -- base weight, tilt, reserve, etc.
    approved_at    TIMESTAMPTZ,
    approved_by    TEXT,
    superseded_at  TIMESTAMPTZ,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_regime_proposals_module_created
    ON regime_allocation_proposals(module, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_regime_proposals_pending
    ON regime_allocation_proposals(created_at DESC)
    WHERE approved_at IS NULL AND superseded_at IS NULL;

COMMIT;
