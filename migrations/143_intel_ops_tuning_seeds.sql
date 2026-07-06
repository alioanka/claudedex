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
