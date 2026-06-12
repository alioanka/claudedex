-- Migration 120: execution_quality — TCA config seeds + metrics tables
--
-- Transaction Cost Analysis module (modules/execution_quality/). READ-ONLY
-- observer: it reads every trading module''s closed-trade tables, decomposes
-- quoted-vs-realized execution cost (slippage, fees, gas, flash-loan/tip
-- extras, suspected MEV), and persists per-trade rows + per-module scorecards.
-- It NEVER trades, NEVER writes a pause/killswitch flag, and has zero market
-- risk. Applying this migration changes NO behavior; the module only runs when
-- EXECUTION_QUALITY_MODULE_ENABLED=true (default false).
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('execution_quality', 'tick_interval_seconds', '1800', 'int',
     'Seconds between TCA scoring cycles (default 30 min — this is a slow, '
     'read-only analytics loop).', NOW(), NOW()),

    ('execution_quality', 'lookback_hours', '24', 'int',
     'Rolling window of closed trades scored each tick. Per-trade rows are '
     'idempotent (unique module+trade_ref), so overlapping windows never '
     'double-count.', NOW(), NOW()),

    ('execution_quality', 'max_rows_per_module', '500', 'int',
     'Per-tick cap on rows fetched from each module''s trade table (most '
     'recent first) so a busy module cannot stall the loop.', NOW(), NOW()),

    ('execution_quality', 'min_trades_for_scorecard', '3', 'int',
     'Below this many scored trades in the window the scorecard row is still '
     'written (for coverage visibility) but breach thresholds are NOT '
     'evaluated — thin data never raises a flag.', NOW(), NOW()),

    ('execution_quality', 'include_simulated', 'true', 'bool',
     'When true (default) DRY_RUN/simulated fills are scored too — useful '
     'because modeled costs should be audited before a live flip. LIVE and '
     'simulated rows are always distinguishable via is_simulated.', NOW(), NOW()),

    ('execution_quality', 'sandwich_suspect_bps', '150', 'float',
     'Entry slippage (realized vs quoted, bps) at or above which an on-chain '
     'fill is flagged mev_suspect=true. A flag, not an accusation: it marks '
     'fills worth a manual receipt review.', NOW(), NOW()),

    ('execution_quality', 'fee_warn_bps', '30', 'float',
     'Scorecard breach threshold: average fee cost per trade above this many '
     'bps of notional is recorded in the breaches JSON.', NOW(), NOW()),

    ('execution_quality', 'gas_warn_bps', '50', 'float',
     'Scorecard breach threshold: average gas cost per trade above this many '
     'bps of notional is recorded in the breaches JSON.', NOW(), NOW()),

    ('execution_quality', 'slippage_warn_bps', '50', 'float',
     'Scorecard breach threshold: average realized entry slippage above this '
     'many bps is recorded in the breaches JSON.', NOW(), NOW()),

    ('execution_quality', 'total_cost_warn_bps', '100', 'float',
     'Scorecard breach threshold: average total measured cost per trade above '
     'this many bps of notional is recorded in the breaches JSON.', NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- One row per scored trade. Idempotent across ticks via the unique
-- (module, trade_ref) index; re-scoring an already-scored trade is a no-op.
CREATE TABLE IF NOT EXISTS tca_trade_costs (
    id                  BIGSERIAL PRIMARY KEY,
    module              TEXT NOT NULL,            -- dex | solana | futures | sniper | arbitrage | copy_trading | ai
    trade_ref           TEXT NOT NULL,            -- source table trade_id (or id::text)
    venue               TEXT,                     -- chain or exchange
    side                TEXT,
    is_simulated        BOOLEAN NOT NULL DEFAULT TRUE,
    notional_usd        DOUBLE PRECISION,
    fee_bps             DOUBLE PRECISION,         -- fees / notional * 10000
    gas_bps             DOUBLE PRECISION,         -- gas / notional * 10000
    extra_bps           DOUBLE PRECISION,         -- flash-loan fee, tips, modeled slippage already deducted upstream
    entry_slippage_bps  DOUBLE PRECISION,         -- realized vs quoted (NULL when no quote captured)
    exit_slippage_bps   DOUBLE PRECISION,
    total_cost_bps      DOUBLE PRECISION,         -- sum of the KNOWN components only
    total_cost_usd      DOUBLE PRECISION,
    gross_pnl_usd       DOUBLE PRECISION,
    net_pnl_usd         DOUBLE PRECISION,
    cost_to_gross_pct   DOUBLE PRECISION,         -- total cost as % of |gross pnl|
    quote_covered       BOOLEAN NOT NULL DEFAULT FALSE,  -- true iff a captured quote made slippage measurable
    mev_suspect         BOOLEAN NOT NULL DEFAULT FALSE,
    components          JSONB,                    -- raw inputs + unit assumptions, operator-recomputable
    trade_time          TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_tca_trade_costs_module_ref
    ON tca_trade_costs(module, trade_ref);
CREATE INDEX IF NOT EXISTS idx_tca_trade_costs_module_time
    ON tca_trade_costs(module, trade_time DESC);
CREATE INDEX IF NOT EXISTS idx_tca_trade_costs_created
    ON tca_trade_costs(created_at DESC);

-- One row per module per tick (full audit history; dashboard reads the latest
-- per module). breaches JSON lists threshold violations; empty list = clean.
CREATE TABLE IF NOT EXISTS tca_scorecards (
    id                   BIGSERIAL PRIMARY KEY,
    module               TEXT NOT NULL,
    window_hours         INTEGER,
    trades_scored        INTEGER NOT NULL DEFAULT 0,
    quote_coverage_pct   DOUBLE PRECISION,        -- % of scored trades with a captured quote
    avg_fee_bps          DOUBLE PRECISION,
    avg_gas_bps          DOUBLE PRECISION,
    avg_extra_bps        DOUBLE PRECISION,
    avg_entry_slippage_bps    DOUBLE PRECISION,
    median_entry_slippage_bps DOUBLE PRECISION,
    avg_total_cost_bps   DOUBLE PRECISION,
    median_total_cost_bps DOUBLE PRECISION,
    total_cost_usd       DOUBLE PRECISION,
    gross_pnl_usd        DOUBLE PRECISION,
    net_pnl_usd          DOUBLE PRECISION,
    cost_to_gross_pct    DOUBLE PRECISION,
    mev_suspect_count    INTEGER NOT NULL DEFAULT 0,
    breaches             JSONB,
    components           JSONB,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tca_scorecards_module_created
    ON tca_scorecards(module, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tca_scorecards_created
    ON tca_scorecards(created_at DESC);

COMMIT;
