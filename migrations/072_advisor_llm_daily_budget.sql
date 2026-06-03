-- Migration 072: advisor HARD daily paid-LLM call budget
--
-- A runaway KAP re-classification loop (the 60s classifier worker re-LLM'd the
-- same un-stored disclosures every cycle) burned a full day of paid Anthropic
-- credit in ~3 hours. The rationale cache (070) reduces advice spam but is NOT
-- a hard ceiling and does nothing for the KAP path.
--
-- modules/advisor/core/llm_budget.py now enforces a single global daily cap on
-- ALL paid LLM calls, shared by BOTH the KAP classifier and the advice
-- rationale generator. When the day's budget is spent, every caller fails soft
-- to rule-based output and makes NO API call until the UTC day rolls over. The
-- counter is persisted to logs/advisor/.llm_budget.json so a restart/crash-loop
-- cannot reset it and re-spend within the same day.
--
--   advisor_llm_daily_max_calls  int  default '50'
--       Maximum paid Anthropic/OpenAI calls per UTC day across KAP + advice.
--       <= 0 disables the cap (unlimited). Lower it (e.g. 10) to be stricter.
--
-- Surface in the Advisor Settings page (dashboard agent).
-- Idempotent: INSERT ... ON CONFLICT (config_type, key) DO NOTHING.
-- ADVICE-ONLY.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('advisor_config', 'advisor_llm_daily_max_calls', '50', 'int',
     'HARD global cap on paid LLM (Anthropic/OpenAI) calls per UTC day, shared '
     'across KAP classification AND advice rationale. When reached, the advisor '
     'falls back to rule-based output and makes NO further paid API calls until '
     'UTC rollover (counter persisted across restarts). <=0 disables the cap. '
     'Default 50 — lower it to be stricter on cost.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;

-- ---------------------------------------------------------------------------
-- DOWN (manual rollback)
-- ---------------------------------------------------------------------------
-- BEGIN;
-- DELETE FROM config_settings WHERE config_type='advisor_config'
--   AND key = 'advisor_llm_daily_max_calls';
-- COMMIT;
