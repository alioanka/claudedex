-- Migration 100: remove dead ('solana_pumpfun', 'enabled') config row
--
-- Migration 005 seeded ('solana_pumpfun', 'enabled', 'true') but no code
-- ever reads that key: both config managers
-- (modules/solana_trading/config/solana_config_manager.py and the
-- modules/solana_strategies duplicate) map the pump.fun toggle to the key
-- 'pumpfun_enabled' (CONFIG_KEY_MAPPING), and the dashboard prefix rule
-- (pumpfun_* -> solana_pumpfun) reads/writes 'pumpfun_enabled' too. The
-- 'enabled' row is dead and misleading on raw config views.
--
-- Step 1 preserves operator intent: every dashboard edit stamps
-- updated_by, while the mig-005 seed leaves it NULL. Only an
-- operator-edited value is copied into the canonical key, and only when
-- the canonical row is absent (an existing 'pumpfun_enabled' row stays
-- authoritative via ON CONFLICT DO NOTHING). An untouched seed is NOT
-- copied: it still holds 'true', the code default for 'pumpfun_enabled'
-- is False, and silently enabling pump.fun from a stale seed would be a
-- live behavior change. After deletion the absent canonical key keeps
-- today's effective behavior (code default False).
--
-- Idempotent: re-running finds no 'enabled' row; both statements no-op.

INSERT INTO config_settings (config_type, key, value, value_type, description)
SELECT 'solana_pumpfun', 'pumpfun_enabled', dead.value, 'bool',
       'Enable Pump.fun sniping (operator value migrated from the legacy ''enabled'' row by migration 100)'
FROM config_settings AS dead
WHERE dead.config_type = 'solana_pumpfun'
  AND dead.key = 'enabled'
  AND dead.updated_by IS NOT NULL
ON CONFLICT (config_type, key) DO NOTHING;

DELETE FROM config_settings
WHERE config_type = 'solana_pumpfun'
  AND key = 'enabled';
