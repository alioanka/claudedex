-- Migration 070: advisor LLM rationale cache config — issue #12
--
-- The advisor's numeric levels + confidence are computed locally
-- (modules/advisor/core/analyzers/levels.py). The LLM is used ONLY for the
-- prose rationale, so it should NOT be re-called every advice cycle when the
-- underlying signal/direction has not materially changed. rationale_helper.py
-- now caches the generated rationale keyed on
--   market | symbol | horizon | direction | rounded(close, sma, rsi, bb, vol)
-- and reuses it until the key changes or the entry exceeds a max age.
--
-- These two keys (config_type='advisor_config') tune that cache. Surface both
-- in the Advisor Settings page (dashboard agent).
--
--   advisor_llm_rationale_cache_enabled  bool  default 'true'
--       When true, reuse a cached rationale while the rounded signal + direction
--       are unchanged (single-provider mode only; dual-advice modes always run
--       fresh so providers_disagree stays current). Set 'false' to force an LLM
--       call every cycle (original behaviour).
--   advisor_llm_rationale_max_age_minutes int default '360' (6h)
--       Hard refresh interval: even if the signal is static, regenerate the
--       rationale after this many minutes so language does not go stale.
--
-- Idempotent: INSERT ... ON CONFLICT (config_type, key) DO NOTHING.
-- ADVICE-ONLY: this only affects rationale TEXT generation cadence; numeric
-- levels/confidence are always recomputed locally every cycle.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('advisor_config', 'advisor_llm_rationale_cache_enabled', 'true', 'bool',
     'When true, the advisor reuses a cached LLM rationale while the rounded '
     'technical signal and direction are unchanged for a symbol/horizon, instead '
     'of calling the paid LLM API every advice cycle. Numeric levels and '
     'confidence are still recomputed locally every cycle. Single-provider mode '
     'only; dual-advice modes always run fresh. Default true.',
     NOW(), NOW()),
    ('advisor_config', 'advisor_llm_rationale_max_age_minutes', '360', 'int',
     'Maximum age (minutes) of a cached LLM rationale before it is regenerated '
     'even if the signal has not changed. Prevents stale prose. Default 360 (6h). '
     'Only applies when advisor_llm_rationale_cache_enabled=true.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;

-- ---------------------------------------------------------------------------
-- DOWN (manual rollback)
-- ---------------------------------------------------------------------------
-- BEGIN;
-- DELETE FROM config_settings WHERE config_type='advisor_config'
--   AND key IN ('advisor_llm_rationale_cache_enabled',
--               'advisor_llm_rationale_max_age_minutes');
-- COMMIT;
