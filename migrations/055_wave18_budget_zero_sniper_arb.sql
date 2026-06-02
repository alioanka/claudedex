-- Migration 055: wave-18 — neutralise sniper + arbitrage via budget=0
--
-- Operator decision (2026-06-02): sniper and arbitrage are structurally
-- unprofitable on current infrastructure. Both modules are being kept in
-- the codebase but entries must be suppressed immediately.
--
-- Strategy: force allocation_guard_config.budget_usd_sniper and
-- budget_usd_arbitrage to '0'. The sniper engine now reads this value
-- at startup (sniper_budget_usd) and explicitly blocks _execute_snipe
-- when it is 0 (see modules/sniper/core/sniper_engine.py). The
-- allocation guard itself treats budget_usd==0 as "unlimited" (skip
-- the per-module check), so this explicit startup check is necessary.
--
-- IMPORTANT: Migration 046 seeded these rows with ON CONFLICT DO NOTHING,
-- so they already exist in any live database at their original values
-- (150.0 and 200.0 respectively). This migration uses
-- ON CONFLICT ... DO UPDATE SET value=excluded.value to FORCE the
-- override — a DO NOTHING here would silently leave them at their
-- original values and have zero effect.
--
-- Idempotent: safe to re-run. On fresh databases the rows are inserted
-- at '0'; on existing databases the existing rows are updated to '0'.
--
-- To re-enable either module, set the budget to a positive value:
--   UPDATE config_settings SET value='150' WHERE config_type='allocation_guard_config' AND key='budget_usd_sniper';
--   UPDATE config_settings SET value='200' WHERE config_type='allocation_guard_config' AND key='budget_usd_arbitrage';

-- UP
INSERT INTO config_settings (config_type, key, value, description)
VALUES
    ('allocation_guard_config', 'budget_usd_sniper', '0',
     'USD budget for sniper module. 0 = entries suppressed (wave-18 operator decision: structurally unprofitable). '
     'Set to a positive value to re-enable (e.g. 150).'),
    ('allocation_guard_config', 'budget_usd_arbitrage', '0',
     'USD budget for arbitrage module. 0 = entries suppressed (wave-18 operator decision: structurally unprofitable). '
     'Set to a positive value to re-enable (e.g. 200).')
ON CONFLICT (config_type, key) DO UPDATE
    SET value = excluded.value,
        description = excluded.description;

-- DOWN (restore pre-wave-18 values from migration 046)
-- UPDATE config_settings SET value='150.0' WHERE config_type='allocation_guard_config' AND key='budget_usd_sniper';
-- UPDATE config_settings SET value='200.0' WHERE config_type='allocation_guard_config' AND key='budget_usd_arbitrage';
