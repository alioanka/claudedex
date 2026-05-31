-- Migration 044: Sniper Wave-15 entry quality + exit rework config seeds
--
-- Context: last-1000-trade honest data shows 21.8% WR / -$1,523 PnL.
-- Root cause: indiscriminate sniping of every launch with no minimum
-- holder count, no token-age window, no buy/sell pressure gate, no
-- dev-holding cap, and a single 50% TP that memecoins rarely reach.
--
-- NEW entry-filter keys:
--   sniper_min_holder_count (int, default 10)
--     Reject tokens with fewer than N holders at safety-check time.
--     Fresh launches with < 10 holders are overwhelmingly dev-wallets
--     that dump immediately. Start conservative; raise after data.
--
--   sniper_min_token_age_seconds (int, default 30)
--     Ignore events for pools/mints younger than N seconds relative to
--     the pool's block_time. The first 5-30 seconds are dominated by
--     competing MEV bots that front-run entry and immediately dump.
--     Conservative default 30 s; tune up to 60 s if WR stays low.
--
--   sniper_max_dev_holding_pct (numeric, default 30)
--     Reject if top-holder (proxy for dev wallet) holds > N % of supply.
--     GoPlus lp_holders[0].percent already lands in the safety report;
--     this gate uses the same field. 30% is aggressive dumping risk.
--
--   sniper_min_buy_sell_ratio (numeric, default 1.5)
--     Reject if recent buy-volume / sell-volume < N over the detection
--     window. Proxy for momentum; < 1.0 means more selling than buying.
--     Requires Birdeye trade-stats endpoint (fail-open if unavailable).
--     Default 1.5 = at least 50% more buys than sells in last window.
--     Set 0.0 to disable (fail-open on all tokens).
--
--   sniper_min_safety_score (int, default 40)
--     Hard floor on the subtractive-penalty safety score (0-100).
--     Current gate only rejects DANGER (score ≈ 0-15) + HONEYPOT.
--     Adding a floor at 40 also rejects low-CAUTION tokens.
--     0 = disabled (existing behavior).
--
-- NEW exit-control keys:
--   sniper_partial_take_pct (numeric, default 20.0)
--     % gain at which 50% of the position is sold (partial take).
--     Replaces the current all-or-nothing 50% TP. On a memecoin that
--     pumps +20% then dumps, the old strategy realizes 0% (SL later);
--     partial-take locks in +20% on half and lets the rest ride.
--     Set 0.0 to disable partial-take (revert to single TP).
--
--   sniper_partial_take_size_pct (numeric, default 50.0)
--     Fraction of position sold at sniper_partial_take_pct (%).
--     50 = sell half. 100 = revert to full-exit (same as single TP).
--
--   sniper_trail_after_partial (bool, default true)
--     After partial take, use trailing stop instead of fixed SL on the
--     remaining position. Trail distance = stop_loss_pct from the high
--     watermark reached after partial-take. false = keep fixed SL.
--
-- Idempotent: ON CONFLICT DO NOTHING on all inserts.
-- Date: 2026-05-31

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    -- Entry quality gates
    ('sniper_config', 'sniper_min_holder_count', '10', 'int',
     'Minimum holder count at safety-check time. Tokens with fewer holders are '
     'almost exclusively dev-wallets pre-dump. 0 = disabled. Default 10.'),

    ('sniper_config', 'sniper_min_token_age_seconds', '30', 'int',
     'Ignore pool events younger than N seconds (vs pool block_time). '
     'First 5-30s dominated by MEV bots that front-run then dump. '
     '0 = disabled. Default 30. Tune up to 60 after data window.'),

    ('sniper_config', 'sniper_max_dev_holding_pct', '30.0', 'float',
     'Reject if top-holder % of supply > N (proxy for dev wallet). '
     'GoPlus holders[0].percent * 100. 30% = aggressive dump risk. '
     '100 = disabled. Default 30.'),

    ('sniper_config', 'sniper_min_buy_sell_ratio', '1.5', 'float',
     'Reject if buys / sells in last window < N (momentum gate). '
     '< 1.0 = more selling than buying. Fail-open if data unavailable. '
     '0.0 = disabled. Default 1.5.'),

    ('sniper_config', 'sniper_min_safety_score', '40', 'int',
     'Hard safety score floor (0-100 subtractive-penalty scale). '
     'Supplements DANGER rejection: also rejects low-CAUTION tokens. '
     '0 = disabled (existing DANGER-only gate). Default 40.'),

    -- Exit rework
    ('sniper_config', 'sniper_partial_take_pct', '20.0', 'float',
     'P&L % at which partial-take fires. 0.0 = disabled (single TP). '
     'On a +20% spike that then dumps, partial-take locks in gains '
     'before the move reverses. Default 20.'),

    ('sniper_config', 'sniper_partial_take_size_pct', '50.0', 'float',
     'Fraction of position (%) sold at sniper_partial_take_pct. '
     '50 = sell half, let rest ride. 100 = full exit (= single TP). '
     'Default 50.'),

    ('sniper_config', 'sniper_trail_after_partial', 'true', 'bool',
     'After partial take, trail-stop the remainder from high-watermark '
     'using stop_loss_pct as the trail distance. false = fixed SL. '
     'Default true.')

ON CONFLICT (config_type, key) DO NOTHING;

-- down:
-- DELETE FROM config_settings
-- WHERE config_type = 'sniper_config'
--   AND key IN (
--     'sniper_min_holder_count', 'sniper_min_token_age_seconds',
--     'sniper_max_dev_holding_pct', 'sniper_min_buy_sell_ratio',
--     'sniper_min_safety_score', 'sniper_partial_take_pct',
--     'sniper_partial_take_size_pct', 'sniper_trail_after_partial'
--   );
