-- Migration 067: Advisor advice-quality tunables — horizon-aware levels + confidence
--
-- Seeds config_settings (config_type='advisor_config') for the shared
-- horizon-aware trade-level model and the real (varying) confidence formula
-- introduced in modules/advisor/core/analyzers/levels.py.
--
-- ROOT CAUSE these keys address:
--   Previously every analyzer used FIXED percentage bands (entry +/-0.5%,
--   target +/-5%, stop +/-3%) independent of horizon, so short/mid/long showed
--   IDENTICAL entry/target/stop for a symbol. And per-analyzer confidence was
--   abs(vote)/3 (+0.1 vol) -> a frozen discrete ladder that looked stuck at 43%.
--
--   levels.horizon_levels() now scales entry/target/stop by:
--       distance_frac = vol_unit * horizon_multiplier
--   where vol_unit = clamp(ATR%-of-close OR Bollinger-width proxy OR NAV-return
--   stdev, vol_floor, vol_ceiling). So short < mid < long distances differ, and
--   a more volatile asset gets wider bands.
--
--   levels.signal_confidence() is a continuous function of vote agreement +
--   RSI/BB magnitude + volume confirmation + a horizon factor - a dual-advice
--   DISAGREE penalty, so confidence varies per symbol/horizon.
--
-- HONESTY NOTE: these are HEURISTIC levels and a heuristic confidence, NOT a
-- prediction or guarantee. The advisor is ADVICE-ONLY. Multipliers are a
-- starting calibration; the operator should tune them per their risk appetite.
--
-- Idempotent: INSERT ... ON CONFLICT (config_type, key) DO NOTHING.
-- Reversible: DOWN block at bottom.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    -- Entry-band half-width multipliers (x vol_unit) per horizon.
    ('advisor_config', 'levels_entry_mult_short', '0.25', 'float',
     'Entry-band half-width = vol_unit * this, for SHORT horizon (1d-1w). '
     'Tighter entry band for short horizons. Default 0.25.', NOW(), NOW()),
    ('advisor_config', 'levels_entry_mult_mid', '0.50', 'float',
     'Entry-band half-width multiplier for MID horizon (1w-3m). Default 0.50.',
     NOW(), NOW()),
    ('advisor_config', 'levels_entry_mult_long', '1.00', 'float',
     'Entry-band half-width multiplier for LONG horizon (3m-2y). Default 1.00.',
     NOW(), NOW()),

    -- Target-distance multipliers (x vol_unit) per horizon.
    ('advisor_config', 'levels_target_mult_short', '3.0', 'float',
     'Target distance = vol_unit * this * levels_target_rr, for SHORT horizon. '
     'Short < mid < long target distance. Default 3.0 (~few days of vol).',
     NOW(), NOW()),
    ('advisor_config', 'levels_target_mult_mid', '8.0', 'float',
     'Target distance multiplier for MID horizon. Default 8.0 (~few weeks).',
     NOW(), NOW()),
    ('advisor_config', 'levels_target_mult_long', '20.0', 'float',
     'Target distance multiplier for LONG horizon. Default 20.0 (~a quarter+). '
     'NOTE: aggressive for high-vol assets; lower if targets look unrealistic.',
     NOW(), NOW()),

    -- Stop-distance multipliers (x vol_unit) per horizon.
    ('advisor_config', 'levels_stop_mult_short', '1.5', 'float',
     'Stop distance = vol_unit * this, for SHORT horizon. Default 1.5.',
     NOW(), NOW()),
    ('advisor_config', 'levels_stop_mult_mid', '4.0', 'float',
     'Stop distance multiplier for MID horizon. Default 4.0.', NOW(), NOW()),
    ('advisor_config', 'levels_stop_mult_long', '10.0', 'float',
     'Stop distance multiplier for LONG horizon. Default 10.0.', NOW(), NOW()),

    -- Global reward:risk shaping applied to target distance only.
    ('advisor_config', 'levels_target_rr', '1.0', 'float',
     'Reward:risk shaping factor multiplied into target distance only. '
     '>1.0 widens targets relative to stops. Default 1.0.', NOW(), NOW()),

    -- Volatility-unit clamps (fractional per-bar vol).
    ('advisor_config', 'levels_vol_floor', '0.005', 'float',
     'Minimum per-bar volatility unit (fraction of price). Floors degenerate '
     'zero-width bands on ultra-quiet series. Default 0.005 (0.5%).',
     NOW(), NOW()),
    ('advisor_config', 'levels_vol_ceiling', '0.15', 'float',
     'Maximum per-bar volatility unit. Caps absurd bands on a vol spike. '
     'Default 0.15 (15%).', NOW(), NOW()),

    -- Confidence shaping.
    ('advisor_config', 'levels_conf_floor', '0.05', 'float',
     'Minimum published confidence (never 0 — always some uncertainty). '
     'Default 0.05.', NOW(), NOW()),
    ('advisor_config', 'levels_conf_ceiling', '0.95', 'float',
     'Maximum published confidence (never 1.0 — heuristic, not a guarantee). '
     'Default 0.95.', NOW(), NOW()),
    ('advisor_config', 'levels_conf_disagree_penalty', '0.20', 'float',
     'Confidence subtracted when dual-advice (anthropic vs openai) DISAGREE '
     'on direction. A disagreement should LOWER confidence. Default 0.20.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;


-- =========================================================================
-- DOWN (reversible) -- run manually to undo
-- =========================================================================
-- BEGIN;
-- DELETE FROM config_settings
--   WHERE config_type='advisor_config'
--     AND key IN (
--       'levels_entry_mult_short','levels_entry_mult_mid','levels_entry_mult_long',
--       'levels_target_mult_short','levels_target_mult_mid','levels_target_mult_long',
--       'levels_stop_mult_short','levels_stop_mult_mid','levels_stop_mult_long',
--       'levels_target_rr','levels_vol_floor','levels_vol_ceiling',
--       'levels_conf_floor','levels_conf_ceiling','levels_conf_disagree_penalty'
--     );
-- COMMIT;
