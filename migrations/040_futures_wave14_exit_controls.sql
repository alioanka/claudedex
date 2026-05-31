-- Migration 040: Futures Wave-14 exit control config seeds
--
-- Seeds two new keys in config_settings for the futures_risk section:
--
--   futures_risk.max_hold_minutes (FUT-RM-23)
--     Intraday max-hold cap. Positions that drift past this limit without
--     hitting TP1 are time-exited to prevent multi-hour fee bleed.
--     Default 240 min (4h) = 16 × the 15m signal bar.
--     Set to 0 to disable.
--
--   futures_risk.signal_reversal_threshold (FUT-RM-24)
--     Reversal score (sum of all 5 indicator signals) required before the
--     engine will exit a position early on an opposing signal. Pre-Wave-14
--     this was hardcoded to 6 (practically impossible). Default 4 = two
--     strong reversals, above the entry bar of 3.
--
-- Both keys use INSERT ... ON CONFLICT DO NOTHING so existing operator
-- customisations are preserved on re-run.

INSERT INTO config_settings (config_type, key, value, description, updated_at)
VALUES
    (
        'futures_risk',
        'max_hold_minutes',
        '240',
        'FUT-RM-23: intraday max-hold cap in minutes. Positions still within SL/TP bounds but held past this limit are time-exited. 0 = disabled. Default 240 (4h).',
        NOW()
    ),
    (
        'futures_risk',
        'signal_reversal_threshold',
        '4',
        'FUT-RM-24: reversal score magnitude needed for signal-reversal early exit (was hardcoded 6, effectively disabled). Minimum 3 to avoid whipsaws. Default 4.',
        NOW()
    )
ON CONFLICT (config_type, key) DO NOTHING;
