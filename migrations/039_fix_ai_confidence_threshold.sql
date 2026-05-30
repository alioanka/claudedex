-- Migration 039: Fix AI confidence_threshold stored value + seed model keys
--
-- Context (Wave-13, discovered from live DB after Agent 8's code fix):
--   config_settings(ai_config, confidence_threshold) was stored as '50' — a
--   percentage-era value. The sentiment engine compares a sentiment score in
--   the range [-1.0, +1.0] against this threshold, so a value of 50 can NEVER
--   be exceeded → the AI module skips every cycle ([ai-skip]
--   reason=confidence_below_threshold) and has made 0 trades for months.
--   Live scores cluster at abs(0.30..0.40); Agent 8 set the CODE default to
--   0.35, but the DB value overrides the code default, so the DB row must be
--   corrected too.
--
--   Also seeds claude_model / openai_model so they are visible/editable in the
--   dashboard AI settings page (Agent 8 + Agent 10 handoff). If absent the code
--   falls back to these same defaults, so seeding is cosmetic-but-helpful.
--
-- Idempotent:
--   * The UPDATE only rewrites the threshold when it is still a legacy
--     out-of-range value (> 1.0), so re-running is a no-op and it will not
--     stomp a value an operator later tunes by hand within [0,1].
--   * Model-key seeds use ON CONFLICT (config_type, key) DO NOTHING.

-- 1) Correct the out-of-range threshold (50 -> 0.35). Guarded so a hand-tuned
--    in-range value (e.g. 0.50) is preserved on re-run.
UPDATE config_settings
   SET value = '0.35'
 WHERE config_type = 'ai_config'
   AND key = 'confidence_threshold'
   AND value ~ '^[0-9.]+$'
   AND value::numeric > 1.0;

-- 2) Seed model id keys (cosmetic — code defaults match these).
INSERT INTO config_settings (config_type, key, value)
VALUES ('ai_config', 'claude_model', 'claude-3-5-sonnet-20241022')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value)
VALUES ('ai_config', 'openai_model', 'gpt-4o-mini')
ON CONFLICT (config_type, key) DO NOTHING;
