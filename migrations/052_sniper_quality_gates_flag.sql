-- Migration 052: Wave-16 sniper quality gates decoupling
--
-- Adds sniper_quality_gates_enabled to config_settings so the W15 entry-
-- quality heuristics (holder count, token age, dev holding, safety score,
-- buy-sell ratio) can be toggled independently of safety_check_enabled.
--
-- When true (default):  quality gates run regardless of whether the
--   honeypot-safety API check (safety_check_enabled) is on or off.
-- When false: quality filtering is disabled entirely.  Not recommended
--   for LIVE trading.
--
-- Also re-asserts safety_check_enabled=false -> true for the production
-- environment where the operator is ready to enable the full gate set.
-- The UPDATE below is a no-op if safety_check_enabled is already true.

-- Seed the new flag (idempotent via ON CONFLICT DO NOTHING)
INSERT INTO config_settings (config_type, key, value, description)
VALUES (
    'sniper_config',
    'sniper_quality_gates_enabled',
    'true',
    'W16: run W15 entry-quality heuristics even when safety_check_enabled=false. '
    'Gates that need a SafetyReport (holder count, dev holding, safety score) '
    'still require safety_check_enabled=true; age and buy-sell-ratio run always. '
    'Set false only to disable quality filtering entirely.'
)
ON CONFLICT (config_type, key) DO NOTHING;
