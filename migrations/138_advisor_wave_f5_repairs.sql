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

-- =========================================================================
-- 2. One-off NaN cleanup (Wave-F5 fix 3). Legacy rows persisted before the
--    non-finite guards (levels.horizon_levels / portfolio_engine) hold
--    'NaN'::numeric values; json.dumps then emitted bare NaN tokens and the
--    advice-history + simulations pages died on res.json(). The endpoints are
--    now NaN-safe; this cleans the stored rows. Idempotent (Postgres treats
--    NaN = NaN as TRUE for numeric comparisons).
-- =========================================================================

UPDATE advisor_advice SET entry_low    = NULL WHERE entry_low    = 'NaN'::numeric;
UPDATE advisor_advice SET entry_high   = NULL WHERE entry_high   = 'NaN'::numeric;
UPDATE advisor_advice SET target_price = NULL WHERE target_price = 'NaN'::numeric;
UPDATE advisor_advice SET stop_price   = NULL WHERE stop_price   = 'NaN'::numeric;

UPDATE advisor_sim_positions SET current_price = NULL WHERE current_price = 'NaN'::numeric;
UPDATE advisor_sim_positions SET target_price  = NULL WHERE target_price  = 'NaN'::numeric;
UPDATE advisor_sim_positions SET stop_price    = NULL WHERE stop_price    = 'NaN'::numeric;
UPDATE advisor_sim_positions SET exit_price    = NULL WHERE exit_price    = 'NaN'::numeric;
UPDATE advisor_sim_positions SET pnl_pct       = NULL WHERE pnl_pct       = 'NaN'::numeric;
UPDATE advisor_sim_positions SET pnl_usd       = NULL WHERE pnl_usd       = 'NaN'::numeric;

-- entry_price is NOT NULL, so it cannot be nulled: a sim with a NaN entry can
-- never produce a valid PnL — close it out explicitly (bookkeeping only; the
-- advisor is ADVICE-ONLY, no orders exist). Idempotent via the status guard.
UPDATE advisor_sim_positions
SET status = 'closed', close_reason = 'nan_entry_cleanup',
    pnl_pct = NULL, pnl_usd = NULL,
    closed_at = NOW(), updated_at = NOW()
WHERE status = 'open'
  AND entry_price = 'NaN'::numeric;
