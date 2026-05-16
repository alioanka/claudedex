-- Seed the COPY_TRADING global open-position cap. Per-leader cooldown
-- bounds per-leader rate; this cap bounds GLOBAL exposure so a config
-- with many leaders can't simultaneously open unbounded positions.
-- Tune via config_settings.copytrading_config.max_active_positions.
INSERT INTO config_settings (config_type, key, value, value_type)
VALUES ('copytrading_config', 'max_active_positions', '50', 'integer')
ON CONFLICT (config_type, key) DO NOTHING;
