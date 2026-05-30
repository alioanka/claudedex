-- Seed the active-positions cap so the engine and dashboard both surface a
-- bounded value before the operator tunes it. 500 matches the in-code
-- default and serves as an emergency brake against the runaway position
-- accumulation observed in Phase 2 DRY_RUN (10k+ positions in 22h).
INSERT INTO config_settings (config_type, key, value, value_type)
VALUES ('sniper_config', 'max_active_positions', '500', 'integer')
ON CONFLICT (config_type, key) DO NOTHING;
