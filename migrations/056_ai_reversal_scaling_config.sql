-- Migration 056: AI wave-18 reversal + scaling config keys
-- ai_reversal_min_score: minimum |score| for a counter-directional signal to
--   trigger position reversal (close existing + open opposite). Default 0.5.
-- ai_max_scale_ins: how many times a same-direction signal may add to an
--   already-open position. Default 0 (scaling OFF) — operator opts in via DB.
INSERT INTO config_settings (config_type, key, value, description)
VALUES
    ('ai_config', 'ai_reversal_min_score', '0.5',
     'Minimum |score| required to trigger a position reversal (close existing + open opposite direction). 0 disables reversals.'),
    ('ai_config', 'ai_max_scale_ins', '0',
     'Maximum number of scale-in additions to an already-open position on a strong same-direction signal. 0 = scaling disabled (default).')
ON CONFLICT (config_type, key) DO NOTHING;
