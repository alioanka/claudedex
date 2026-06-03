-- Migration 068: Copy-trading Solana RPC cadence knobs — Wave-19
-- Context: the copy module's Solana fallback poll was hammering rate-limited
-- PUBLIC Solana endpoints (rpc.ankr.com/solana, free.rpcpool.com,
-- api.mainnet-beta) under a 33-wallet config, producing constant
-- "Solana RPC rate limited in fallback poll - backing off" and 0 copies.
-- The engine now (a) resolves the fallback endpoint fresh per cycle via
-- pool_engine, PREFERRING the operator's healthy Helius JSON-RPC endpoint,
-- and (b) honours the two cadence knobs seeded below.
-- Idempotent: ON CONFLICT DO NOTHING. Safe to re-run; no destructive changes.
--
-- DOWN (manual rollback):
--   DELETE FROM config_settings
--    WHERE config_type = 'copytrading_config'
--      AND key IN ('copy_poll_interval_s', 'copy_request_spacing_s');

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('copytrading_config', 'copy_poll_interval_s',
     '15', 'float',
     'Full monitor-cycle cadence in seconds for the copy-trading Solana/EVM '
     'poll. Larger = fewer getSignaturesForAddress bursts against the RPC. '
     'Clamped 5..300 in copy_engine. Default 15.'),
    ('copytrading_config', 'copy_request_spacing_s',
     '0', 'float',
     'Minimum gap in seconds between consecutive outbound Solana RPC calls, '
     'enforced globally across the wallet fan-out so N wallets do not fire in '
     'one synchronized burst. 0 disables. Clamped 0..5 in copy_engine. '
     'Raise toward 0.1-0.3 if even Helius 429s under a large watchlist.')
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;
