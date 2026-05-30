-- Migration 038: Add entry_usd column to solana_positions
--
-- Context (Agent 7 / Copy Trading cross-module handoff):
--   The cross-module exposure aggregator (modules/copy_trading/exposure_aggregator.py)
--   queries each trading module's open-position table and sums the USD basis of
--   every open position in a given token. This total is compared against the
--   copy_cross_module_exposure_cap_usd threshold before a new BUY is broadcast.
--
--   The solana_positions table (migration 013) stores open Solana positions but
--   has no USD-denominated entry column -- only:
--     entry_price  DOUBLE PRECISION  (USD per token)
--     amount       DOUBLE PRECISION  (token units, human-readable)
--     value_sol    DOUBLE PRECISION  (SOL-denominated position size)
--
--   exposure_aggregator.py._sum_solana_positions_usd (line 115) already has a
--   best-effort two-path fallback:
--     1. If entry_usd column exists: SELECT COALESCE(SUM(entry_usd), 0)
--     2. If not: derives USD from entry_price * amount (requires both non-null)
--   The column check at runtime (information_schema query) adds ~1ms latency per
--   call and makes the aggregator path schema-version-dependent.
--
--   This migration adds entry_usd NUMERIC (nullable) so that:
--     a. The aggregator uses the direct fast path.
--     b. solana_engine.py can write entry_usd at INSERT time alongside value_sol.
--     c. The copy module's Solana exposure accounting is no longer a no-op.
--
--   The column is NULLABLE because:
--     - Existing rows may not have an accurate USD basis on hand at write time.
--     - The aggregator's fallback (entry_price * amount) remains as belt-and-
--       suspenders for rows where entry_usd IS NULL.
--
-- Idempotent: ALTER TABLE ... ADD COLUMN IF NOT EXISTS (PostgreSQL 9.6+).
--
-- Date: 2026-05-30

ALTER TABLE solana_positions
    ADD COLUMN IF NOT EXISTS entry_usd NUMERIC;

COMMENT ON COLUMN solana_positions.entry_usd IS
    'USD value of position at entry time. Populated by solana_engine._open_position '
    'from wave-13 onward. Consumed by copy_trading exposure_aggregator to sum '
    'cross-module USD exposure per token. NULL on rows written before this migration.';

-- down:
-- ALTER TABLE solana_positions DROP COLUMN IF EXISTS entry_usd;
