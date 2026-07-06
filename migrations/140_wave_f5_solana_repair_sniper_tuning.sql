-- Migration 140: Wave-F5 Solana fake-PnL data repair + SNIPER hard tuning
--
-- Two independent, fully idempotent parts. Single-quoted literals only.
--
-- PART A — one-off Solana data repair.
--   The +2000%-pinned fake rows (and their mirror -99.98% stop rows on the
--   same poisoned mints) are tagged metadata.excluded=true so the dashboard
--   aggregates (which now filter on that tag) stop counting fabricated wins.
--   Raw rows are KEPT for audit — only the tag is added; no data is deleted.
--   Re-runnable: the WHERE NOT COALESCE(excluded) guards make it a no-op the
--   second time.
--
-- PART B — SNIPER hard tuning (config_type='sniper_config').
--   For each key: INSERT the proposed value if the row is ABSENT (a missing
--   row means no operator has ever set it), then conditionally UPDATE a row
--   that still sits at the OLD default to the proposed value. An operator
--   override (row present with a non-default value) is never clobbered:
--   the INSERT hits ON CONFLICT DO NOTHING and the UPDATE's value guard
--   does not match. Exits are unaffected by the two NEW entry-only keys.

-- =====================================================================
-- PART A: Solana poisoned-trade data repair
-- =====================================================================

-- A1. Tag the +2000%-pinned fakes (the report signature: pnl_pct >= 2000).
UPDATE solana_trades
SET metadata = jsonb_set(
        jsonb_set(COALESCE(metadata, '{}'::jsonb), '{excluded}', 'true'::jsonb, true),
        '{exclusion_reason}', '"wave_f5_historical_pinned_2000pct"'::jsonb, true)
WHERE pnl_pct >= 2000
  AND NOT COALESCE((metadata->>'excluded')::boolean, false);

-- A2. Tag the mirror -99.98% stop_loss rows that share a poisoned mint with
--     an A1 row (poisoned entry -> real price confirmed -> fake -99.98% stop).
UPDATE solana_trades s
SET metadata = jsonb_set(
        jsonb_set(COALESCE(s.metadata, '{}'::jsonb), '{excluded}', 'true'::jsonb, true),
        '{exclusion_reason}', '"wave_f5_historical_mirror_stop"'::jsonb, true)
WHERE s.pnl_pct <= -99.9
  AND NOT COALESCE((s.metadata->>'excluded')::boolean, false)
  AND s.token_mint IN (
        SELECT DISTINCT token_mint FROM solana_trades WHERE pnl_pct >= 2000
  );

-- =====================================================================
-- PART B: SNIPER hard tuning
-- =====================================================================

-- B0. NEW entry-only risk knobs (implemented in sniper_engine.py). Seed only
--     if absent — never clobber an operator value.
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('sniper_config', 'sniper_max_daily_loss_usd', '50', 'float',
     'Halt NEW sniper entries for the rest of the UTC day once realized '
     'losses breach this USD figure. Exits are unaffected. 0 disables.'),
    ('sniper_config', 'sniper_entry_cooldown_seconds', '60', 'int',
     'Minimum seconds between two sniper ENTRIES (paces snipes, bounds '
     'per-rug re-entry churn). Exits are unaffected. 0 disables.')
ON CONFLICT (config_type, key) DO NOTHING;

-- B1. Insert-if-missing at the proposed value (absent row = no operator set).
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('sniper_config', 'test_mode', 'false', 'bool',
     'Wave-F5: relaxed-safety test mode OFF (was the loss-population driver).'),
    ('sniper_config', 'min_liquidity', '25000', 'float',
     'Wave-F5: minimum pool liquidity gate (native/USD).'),
    ('sniper_config', 'max_buy_tax', '5', 'float', 'Wave-F5: max buy tax %.'),
    ('sniper_config', 'max_sell_tax', '5', 'float', 'Wave-F5: max sell tax %.'),
    ('sniper_config', 'sniper_min_holder_count', '50', 'int',
     'Wave-F5: minimum holder count.'),
    ('sniper_config', 'sniper_min_token_age_seconds', '180', 'int',
     'Wave-F5: minimum token age before entry.'),
    ('sniper_config', 'sniper_min_safety_score', '70', 'int',
     'Wave-F5: minimum safety score.'),
    ('sniper_config', 'sniper_min_buy_sell_ratio', '2.0', 'float',
     'Wave-F5: minimum buy/sell ratio (demand confirmation).'),
    ('sniper_config', 'sniper_max_dev_holding_pct', '15', 'float',
     'Wave-F5: max dev holding %.'),
    ('sniper_config', 'trade_amount', '0.05', 'float',
     'Wave-F5: per-snipe entry size (SOL).'),
    ('sniper_config', 'max_active_positions', '25', 'int',
     'Wave-F5: concurrent-position trading cap.'),
    ('sniper_config', 'max_hold_minutes', '240', 'int',
     'Wave-F5: time-stop for zombie positions.'),
    ('sniper_config', 'take_profit_pct', '100', 'float', 'Wave-F5: TP %.'),
    ('sniper_config', 'stop_loss_pct', '15', 'float', 'Wave-F5: SL %.'),
    ('sniper_config', 'sniper_partial_take_pct', '30', 'float',
     'Wave-F5: partial-take trigger %.'),
    ('sniper_config', 'chain', 'solana', 'string',
     'Wave-F5: route only Solana (EVM listener disabled).')
ON CONFLICT (config_type, key) DO NOTHING;

-- B2. Conditional UPDATE: only rows still at the OLD default move to proposed.
UPDATE config_settings SET value = 'false', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'test_mode'
   AND lower(value) IN ('true', '1', 'yes');

UPDATE config_settings SET value = '25000', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'min_liquidity'
   AND value IN ('1000', '1000.0');

UPDATE config_settings SET value = '5', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'max_buy_tax'
   AND value IN ('10', '10.0');

UPDATE config_settings SET value = '5', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'max_sell_tax'
   AND value IN ('10', '10.0');

UPDATE config_settings SET value = '50', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'sniper_min_holder_count'
   AND value IN ('10', '10.0');

UPDATE config_settings SET value = '180', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'sniper_min_token_age_seconds'
   AND value IN ('30', '30.0');

UPDATE config_settings SET value = '70', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'sniper_min_safety_score'
   AND value IN ('40', '40.0');

UPDATE config_settings SET value = '2.0', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'sniper_min_buy_sell_ratio'
   AND value IN ('1.5', '1.50');

UPDATE config_settings SET value = '15', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'sniper_max_dev_holding_pct'
   AND value IN ('30', '30.0');

UPDATE config_settings SET value = '0.05', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'trade_amount'
   AND value IN ('0.1', '0.10');

UPDATE config_settings SET value = '25', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'max_active_positions'
   AND value IN ('500', '500.0');

UPDATE config_settings SET value = '240', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'max_hold_minutes'
   AND value IN ('0', '0.0');

UPDATE config_settings SET value = '100', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'take_profit_pct'
   AND value IN ('50', '50.0');

UPDATE config_settings SET value = '15', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'stop_loss_pct'
   AND value IN ('20', '20.0');

UPDATE config_settings SET value = '30', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'sniper_partial_take_pct'
   AND value IN ('20', '20.0');

UPDATE config_settings SET value = 'solana', updated_at = NOW()
 WHERE config_type = 'sniper_config' AND key = 'chain'
   AND lower(value) = 'all';
