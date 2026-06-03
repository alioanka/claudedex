-- Migration 075: Copy-trading Helius outbound rate (req/s) knob
-- Context: the copy module monitors ~33 Solana leader wallets. On each poll
-- cycle it fired all ~33 wallet RPC calls near-simultaneously, so the Helius
-- free tier (~10 req/s) returned 429 instantly (rate-limit count 1 -> 17+ in
-- under 1 second). The endpoints were HEALTHY; the problem was the STAMPEDE.
-- copy_engine now feeds a shared pool_engine TokenBucket (provider HELIUS_API)
-- and awaits acquire_rate_limit() before each outbound Helius call, spacing the
-- fan-out to copy_helius_rps req/s. All 33 wallets are still polled every
-- cycle, just spread over time instead of bursting.
-- Idempotent: ON CONFLICT DO NOTHING. Safe to re-run; no destructive changes.
--
-- DOWN (manual rollback):
--   DELETE FROM config_settings
--    WHERE config_type = 'copytrading_config'
--      AND key = 'copy_helius_rps';

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('copytrading_config', 'copy_helius_rps',
     '8', 'float',
     'Outbound requests/second ceiling for the copy-trading Solana/Helius '
     'wallet fan-out. Feeds a shared pool_engine token bucket so 33 wallet '
     'polls are SPACED instead of stampeding the free-tier 429 limit. Default '
     '8 (just under Helius free ~10 req/s). Clamped 1..50 in copy_engine. '
     'Raise on a paid Helius plan; lower if you still see 429 bursts.')
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;
