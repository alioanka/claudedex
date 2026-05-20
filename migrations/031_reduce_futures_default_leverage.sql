-- Migration 031: FUT-RM-18 — reduce futures default_leverage from 10x to 5x.
--
-- Wave-5 risk policy. After the operator's 4-trade loss streak (25% win rate,
-- 3 SL hits at -20% each on 10x), 10x default is too aggressive for the
-- current strategy strength: at 10x, a 4% adverse price move wipes out 40%
-- of the position margin and instantly trips the static 2% SL after leverage.
-- Dropping default to 5x doubles the room before SL hits without changing
-- max_leverage (which stays at 20x for opt-in per-trade aggression). The
-- per-symbol leverage override table (FUT-RM-08, migration 027) still wins
-- when set, so an operator can pin BTC/USDT at 10x and let alts default to 5x.
--
-- Idempotent: only updates if the current value is still the legacy default
-- of 10. Operator-customised values (anything ≠ 10) are NOT touched so an
-- existing 3x / 7x / 15x setting survives migration replay.

UPDATE config_settings
SET value = '5',
    description = 'Default leverage for new positions (Wave-5: lowered from 10x)'
WHERE config_type = 'futures_leverage'
  AND key = 'default_leverage'
  AND value = '10';
