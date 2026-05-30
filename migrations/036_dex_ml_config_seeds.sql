-- Migration 036: DEX config seeds
-- Seeds two groups of DB-backed config keys introduced by Wave-13 (Agent 2):
--
--   GROUP 1 — trading section
--     min_vol_liq_ratio (float, default 0.05)
--       BUG-FIX P0: the old hard-coded threshold was 2.0, which required 24-hr
--       volume to exceed 2x pool liquidity before a pair was considered — a bar
--       that real active DEX pairs almost never clear. This resulted in zero
--       opportunities being discovered across all 6 chains for entire scan cycles.
--       0.05 is a "ghost-pool guard only" floor; active pairs with any meaningful
--       turnover pass easily. Tunable via the dashboard Settings → Trading section.
--
--   GROUP 2 — ml_models section (four keys)
--     ml_retrain_enabled (bool, default false)
--       Guards the auto-retrain subprocess loop. Set to true only after 50+
--       closed trades exist; the engine checks ml_retrain_min_trades before
--       spawning scripts/train_ensemble.py.
--     ml_retrain_interval_hours (float, default 24)
--       How many hours the engine waits between retrain runs.
--     ml_retrain_days (int, default 90)
--       Look-back window (days) of closed trades fed to train_ensemble.py.
--     ml_retrain_min_trades (int, default 50)
--       Minimum closed-trade count required before the engine will attempt
--       a retrain. Below this threshold the model artefacts would be trained
--       on too few samples to be meaningful.
--
-- Idempotent: ON CONFLICT (config_type, key) DO NOTHING — safe to re-apply on
-- a cluster that already has any of these rows (operator overrides are preserved).
--
-- Date: 2026-05-30

-- ============================================================================
-- GROUP 1: trading config
-- ============================================================================

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('trading', 'min_vol_liq_ratio', '0.05', 'float',
     'Minimum 24h-volume / liquidity ratio for a DEX pair to be considered. '
     '0.05 = ghost-pool guard (active pairs clear this easily). '
     'Range 0.01-1.0; raise to 0.1-0.3 for higher-turnover bias.')
ON CONFLICT (config_type, key) DO NOTHING;

-- ============================================================================
-- GROUP 2: ml_models config (DEX ensemble auto-retrain loop)
-- ============================================================================

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('ml_models', 'ml_retrain_enabled', 'false', 'bool',
     'Enable the DEX ensemble auto-retrain loop. Set true only after '
     'ml_retrain_min_trades closed trades exist in the trades table.'),

    ('ml_models', 'ml_retrain_interval_hours', '24', 'float',
     'Hours between automatic DEX ensemble retrain runs '
     '(only checked when ml_retrain_enabled=true).'),

    ('ml_models', 'ml_retrain_days', '90', 'int',
     'Look-back window (days) of closed trades fed to train_ensemble.py. '
     'Longer windows improve generalisation; shorter windows track regime drift.'),

    ('ml_models', 'ml_retrain_min_trades', '50', 'int',
     'Minimum closed-trade count required before auto-retrain will fire. '
     'Below this, model artefacts would be trained on too few samples.')

ON CONFLICT (config_type, key) DO NOTHING;

-- down:
-- DELETE FROM config_settings
-- WHERE (config_type = 'trading'   AND key = 'min_vol_liq_ratio')
--    OR (config_type = 'ml_models' AND key IN (
--        'ml_retrain_enabled', 'ml_retrain_interval_hours',
--        'ml_retrain_days', 'ml_retrain_min_trades'));
