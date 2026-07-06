-- Migration 143: Wave-F5 Intelligence & Ops fixes — conditional tuning seeds
--
-- Part 1: intent_solver poll cadence. The module is a PARKED scaffold (doc
-- verdict: PARK IT) whose only source (CoW auction API) was 403-blocked for
-- 20 days while the 60s default hammered it 28,589 times. The code fix adds
-- a browser-class User-Agent + hourly dead-source backoff; this seed slows
-- the scaffold itself from 60s to hourly polls — CONDITIONAL: only applied
-- while the row still holds the mig-130 default '60', so an operator-chosen
-- cadence is never clobbered. No live flag is touched (the module has no
-- live path at all).
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

UPDATE config_settings
SET value = '3600',
    updated_at = NOW(),
    description = 'Seconds between intent-polling / shadow-evaluation cycles. '
                  'Raised 60 -> 3600 by mig 143: the scaffold is PARKED and '
                  'its only free source 403-blocks aggressive pollers.'
WHERE config_type = 'intent_solver'
  AND key = 'poll_interval_s'
  AND value = '60';

-- Part 2: param_tuner reward starvation. The three mig-129 tunables target
-- dex/ai/sniper — modules with ~0 closed trades per 6h reward window — so
-- the bandit recorded rewards once in 20 days (arms at pulls: 0, proposed=0
-- on every tick since Jun 16). Extend the registry to the two modules that
-- DO close trades: futures (atr_tp_rr_ratio, the TP R:R multiple — a target
-- knob, NOT a stop/leverage knob; those stay hard-excluded in code) and
-- solana (jupiter_auto_exit time-based exit, the solana analogue of the
-- whitelisted sniper max_hold_minutes). SHADOW-ONLY posture is unchanged:
-- auto_apply_enabled stays false; these rows only let the bandit LEARN and
-- write proposal rows.
--
-- 2a. Seed the target config rows at their exact code defaults (the tuner
-- skips a tunable whose config_settings row is absent). Values mirror
-- FuturesRiskConfig.atr_tp_rr_ratio = 2.0 and solana jupiter_auto_exit = 0,
-- so applying this changes NO runtime behavior. Idempotent.
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('futures_risk', 'atr_tp_rr_ratio', '2.0', 'float',
     'TP1 = ratio x SL distance when atr_dynamic_sl_tp_enabled (FUT-RM-16). '
     'Seeded by mig 143 at the code default so param_tuner can read/score '
     'it; registered as a bandit tunable with bounds [0.8, 2.0].',
     NOW(), NOW()),

    ('solana_jupiter', 'jupiter_auto_exit', '0', 'int',
     'Time-based Jupiter position exit in seconds (0 = disabled; positions '
     'held 3x this are force-exited). Seeded by mig 143 at the code default '
     'so param_tuner can read/score it; bandit bounds [0, 14400].',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- 2b. Append the two entries to tunable_registry — CONDITIONAL: only while
-- the registry still holds exactly the three mig-129 entries and neither
-- new key is present (an operator-extended or operator-pruned registry is
-- never clobbered; operator-tweaked BOUNDS on the original three survive
-- because this appends instead of replacing). Guarded so a malformed JSON
-- value skips the update instead of halting the deploy.
DO $mig143$
BEGIN
    UPDATE config_settings
    SET value = (value::jsonb || '[
  {"module": "futures", "config_type": "futures_risk", "key": "atr_tp_rr_ratio",
   "min": 0.8, "max": 2.0, "steps": 7, "value_type": "float"},
  {"module": "solana", "config_type": "solana_jupiter", "key": "jupiter_auto_exit",
   "min": 0, "max": 14400, "steps": 6, "value_type": "int"}
]'::jsonb)::text,
        updated_at = NOW()
    WHERE config_type = 'param_tuner'
      AND key = 'tunable_registry'
      AND value NOT LIKE '%atr_tp_rr_ratio%'
      AND value NOT LIKE '%jupiter_auto_exit%'
      AND value LIKE '%min_vol_liq_ratio%'
      AND value LIKE '%confidence_threshold%'
      AND value LIKE '%max_hold_minutes%'
      AND jsonb_typeof(value::jsonb) = 'array'
      AND jsonb_array_length(value::jsonb) = 3;
EXCEPTION WHEN others THEN
    RAISE NOTICE 'mig 143: tunable_registry extension skipped (%). Operator '
                 'can add the futures/solana entries via the param_tuner '
                 'settings page instead.', SQLERRM;
END
$mig143$;
