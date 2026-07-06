-- Migration 138: Wave-F5 advisor repairs (docs/agents/wave-f5/06_advisor_bist.md)
--
-- 1. sim_horizon_days_long 365 -> 90: LONG-horizon sims squatted a channel
--    slot for a YEAR, saturating the per-channel caps (75/75 observed) and —
--    before the Wave-F5 code fix — silencing all advice. CONDITIONAL update:
--    only rows still at the mig-060 default '365' are touched; any operator
--    override is preserved.
--
-- Single-quoted SQL literals only ('' escapes apostrophes). Idempotent.

UPDATE config_settings
SET value = '90', updated_at = NOW()
WHERE config_type = 'advisor_config'
  AND key = 'sim_horizon_days_long'
  AND value = '365';
