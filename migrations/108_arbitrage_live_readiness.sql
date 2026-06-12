-- Migration 108: arbitrage LIVE-readiness — receipt-confirmation knob
--
-- Companion to the arbitrage_engine live-honesty fix: a broadcast flash-loan
-- tx is now confirmed via its receipt (bounded wait) BEFORE being booked to
-- arbitrage_trades; a status=0 revert is recorded as a 'tx_reverted'
-- near-miss (gas burnt, no fill) instead of a profitable closed trade.
--
-- Seeded to match the code default exactly (90 seconds), so applying this
-- migration changes NO behavior on its own. Setting 0 restores the old
-- fire-and-forget booking (book at broadcast, never check the receipt).
--
-- The shadow_mode / live_execution_enabled keys this wave also enforces in
-- the EVM execute gate and the Solana engine were already seeded by
-- migration 097 (same config_type, same defaults) — no re-seed needed.
--
-- Idempotent: ON CONFLICT (config_type, key) DO NOTHING.
-- NOTE: all SQL string literals are single-quoted; '' escapes an apostrophe.

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
(
    'arbitrage_config',
    'receipt_confirm_timeout_s',
    '90',
    'integer',
    'LIVE fills only: seconds to wait for the flash-loan tx receipt before booking the trade. Reverted txs (status 0) are recorded as ''tx_reverted'' near-misses — burnt gas, not profit. 0 = book at broadcast without checking the receipt (pre-108 behavior).',
    NOW(), NOW()
)
ON CONFLICT (config_type, key) DO NOTHING;
