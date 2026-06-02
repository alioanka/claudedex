-- Migration 060: Advisor ML + Sim enhancements — Wave-21
-- Adds:
--   1. horizon_end_date column to advisor_sim_positions (for auto-close).
--   2. config_settings seeds for ML loop + sim horizon config.
-- Idempotent: ADD COLUMN IF NOT EXISTS + ON CONFLICT DO NOTHING.
-- Safe to re-run; no destructive changes.

BEGIN;

-- =========================================================================
-- 1. advisor_sim_positions: add horizon_end_date
--    Used by auto_close_expired() to close sims at horizon deadline.
-- =========================================================================
ALTER TABLE advisor_sim_positions
    ADD COLUMN IF NOT EXISTS horizon_end_date TIMESTAMPTZ;

-- =========================================================================
-- 2. config_settings: ML loop keys
-- =========================================================================

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_ml_min_samples',
     '50', 'integer',
     'Minimum number of closed sim positions required before the ML '
     'confidence model will train or predict. Gate prevents fabricated '
     'confidence on sparse data. Default 50.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_ml_retrain_days',
     '7', 'integer',
     'How often (in days) to automatically retrain the advisor ML model '
     'when advisor_ml_enabled=true. Default 7.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_ml_blend_weight',
     '0.6', 'float',
     'Weight [0.0-1.0] given to the ML win-probability when blending with '
     'the raw heuristic confidence. '
     'blended = raw*(1-w) + ml_win_prob*w. Default 0.6.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_ml_version',
     '', 'string',
     'JSON blob written by retrain job: {version, trained_at, n_samples, '
     'test_accuracy, test_precision, feature_names}. Empty = not yet trained.')
ON CONFLICT (config_type, key) DO NOTHING;

-- =========================================================================
-- 3. config_settings: sim horizon config keys
--    Operator-overridable max-holding days per horizon.
-- =========================================================================

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'sim_horizon_days_short',
     '7', 'integer',
     'Max holding days for SHORT-horizon sim positions. Default 7.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'sim_horizon_days_mid',
     '90', 'integer',
     'Max holding days for MID-horizon sim positions. Default 90.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'sim_horizon_days_long',
     '365', 'integer',
     'Max holding days for LONG-horizon sim positions. Default 365.')
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;

-- DOWN (reversible rollback):
-- BEGIN;
-- ALTER TABLE advisor_sim_positions DROP COLUMN IF EXISTS horizon_end_date;
-- DELETE FROM config_settings
--   WHERE config_type='advisor_config'
--     AND key IN (
--       'advisor_ml_min_samples','advisor_ml_retrain_days',
--       'advisor_ml_blend_weight','advisor_ml_version',
--       'sim_horizon_days_short','sim_horizon_days_mid','sim_horizon_days_long'
--     );
-- COMMIT;
