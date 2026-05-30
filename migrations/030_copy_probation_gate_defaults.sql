-- Wave-4 COPY_TRADING enhancement (CT-Q-09 follow-up):
-- align probation-gate defaults with the operator-approved spec in
-- docs/agents/reports/PM_FINAL_WAVE3.md (carry-over section). The
-- Wave-3 migration 026 staged the probation columns + seeded a
-- conservative 15% loss threshold; the operator approved the Wave-4
-- default of -25% (i.e. a mirrored trade has to lose more than 25%
-- before a leader is benched).
--
-- This migration is idempotent (UPDATE ... WHERE value IS DISTINCT
-- FROM new value) so re-running on a DB where the operator already
-- tuned the threshold higher leaves their override untouched.
--
-- Also seeds the engine-consumed alias keys (copy_* prefix) so the
-- Settings page surfaced to the operator can use a consistent
-- `copy_<name>` namespace. The engine reads BOTH the prefixed and
-- unprefixed forms (prefixed takes priority); see
-- copy_engine._load_settings.

-- 1. Update the default loss threshold from 15 -> 25 ONLY when the
--    current value is still the migration-026 default (15). Operator
--    overrides survive.
UPDATE config_settings
   SET value = '25'
 WHERE config_type = 'copytrading_config'
   AND key = 'probation_loss_pct_threshold'
   AND value = '15';

-- 2. Seed `copy_` prefixed aliases used by the Settings page (dashboard
--    agent owns the UI; engine reads either form). INSERT ... ON
--    CONFLICT DO NOTHING so existing values stay.
INSERT INTO config_settings (config_type, key, value, value_type)
VALUES
    ('copytrading_config', 'copy_probation_gate_enabled',           'true', 'boolean'),
    ('copytrading_config', 'copy_probation_score_threshold',        '30',   'number'),
    ('copytrading_config', 'copy_probation_loss_pct_threshold',     '25',   'number'),
    ('copytrading_config', 'copy_probation_days',                   '7',    'number'),
    ('copytrading_config', 'copy_cross_module_exposure_check_enabled', 'true', 'boolean'),
    ('copytrading_config', 'copy_cross_module_exposure_cap_usd',    '5000', 'number')
ON CONFLICT (config_type, key) DO NOTHING;
