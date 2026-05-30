-- Migration 034: Upgrade solana_trades timestamp columns to TIMESTAMPTZ
-- Root cause: migration 008 created entry_time/exit_time/created_at as tz-naive
-- TIMESTAMP. solana_positions.opened_at (migration 013) is TIMESTAMPTZ.
-- When a position is reconciled from DB its opened_at is tz-aware; the engine
-- writes it as Trade.opened_at into entry_time. asyncpg refuses to insert a
-- tz-aware datetime into a tz-naive column and raises:
--   "can't subtract offset-naive and offset-aware datetimes"
-- Fix: ALTER to TIMESTAMPTZ. AT TIME ZONE 'UTC' coerces existing values safely.
-- This migration is idempotent: IF the column is already TIMESTAMPTZ the cast
-- is a no-op.
--
-- Discovery-first: run
--   SELECT column_name, data_type FROM information_schema.columns
--   WHERE table_name = 'solana_trades'
--     AND column_name IN ('entry_time','exit_time','created_at');
-- before applying to confirm column types on the live instance.

ALTER TABLE solana_trades
    ALTER COLUMN entry_time  TYPE TIMESTAMPTZ
        USING entry_time AT TIME ZONE 'UTC',
    ALTER COLUMN exit_time   TYPE TIMESTAMPTZ
        USING exit_time  AT TIME ZONE 'UTC',
    ALTER COLUMN created_at  TYPE TIMESTAMPTZ
        USING created_at AT TIME ZONE 'UTC';
