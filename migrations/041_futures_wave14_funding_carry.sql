-- Migration 041: Futures Wave-14 funding-carry strategy config seeds
--
-- Seeds FUT-RM-25 funding-carry strategy keys in futures_funding config.
-- All keys are OFF by default. Operator must review the dashboard funding-
-- forecast widget for ≥48h before enabling, then set:
--   funding_carry_enabled = 'true'
--   carry_max_positions = '1'  (start small)
--
-- carry_min_funding_bps: 8 bps/interval ≈ 87%/yr APR at 1x.
--   At 5x leverage the carry yield is ~435%/yr before fees;
--   round-trip Bybit taker (0.12%) + slippage (0.05%) = 0.17% total cost.
--   8 bps = 0.08% per interval; net = 0.08% - 0.17% = -0.09% per interval.
--   Carry becomes profitable only if held ≥ 3 intervals (24h).
--   Operator should raise to 12+ bps if they want single-interval breakeven.
--
-- carry_exit_funding_bps: exit when funding drops below 3 bps to protect
--   against sudden de-funding without waiting for TP/SL.
--
-- All inserts use ON CONFLICT DO NOTHING so existing operator config survives.

INSERT INTO config_settings (config_type, key, value, description, updated_at)
VALUES
    (
        'futures_funding',
        'funding_carry_enabled',
        'false',
        'FUT-RM-25: enable funding-rate carry strategy. Enter SHORT when funding exceeds carry_min_funding_bps to collect funding payments. Disabled by default.',
        NOW()
    ),
    (
        'futures_funding',
        'carry_min_funding_bps',
        '8.0',
        'FUT-RM-25: entry threshold (bps per interval). 8 bps ≈ 87%/yr APR. Raise to 12+ for single-interval breakeven vs Bybit taker fees.',
        NOW()
    ),
    (
        'futures_funding',
        'carry_exit_funding_bps',
        '3.0',
        'FUT-RM-25: exit carry position when funding drops below this threshold (bps per interval). Prevents holding dead carry positions.',
        NOW()
    ),
    (
        'futures_funding',
        'carry_max_positions',
        '2',
        'FUT-RM-25: max simultaneous carry positions (independent of momentum max_positions cap). Start at 1 during initial observation.',
        NOW()
    ),
    (
        'futures_funding',
        'carry_max_hold_minutes',
        '960',
        'FUT-RM-25: max hold for carry positions (minutes). Default 960 = 2 × 8h funding intervals. Overrides global max_hold_minutes for carry trades.',
        NOW()
    )
ON CONFLICT (config_type, key) DO NOTHING;
