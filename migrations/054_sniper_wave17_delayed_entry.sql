-- Migration 054: Wave-17 sniper delayed-entry strategy
--
-- Root cause of -$14k / 38% WR: the sniper bought every pump.fun
-- launch at t=0 when NO vetting data exists.  At t=0 seconds:
--   - Birdeye has no trade history -> buy/sell ratio returns None
--     -> gate fails OPEN (bug: should reject when data absent at entry)
--   - block_time missing/unparseable -> age gate fails OPEN
--   - holder/dev/safety gated behind safety_check_enabled=OFF
-- Net: 100% pass-rate, immediate-rug exposure.
--
-- Fix: mandatory age floor BEFORE evaluation.  Pool is added to an
-- in-memory watchlist on detection.  Entry (safety checks + buy)
-- only fires once the token is >= sniper_min_entry_age_seconds old.
-- By then Birdeye/holder/liquidity data exist and the gates are
-- meaningful.  Missing data at that age is itself a red flag ->
-- fail-CLOSED (opposite of the t=0 fail-open behavior).
--
-- New keys seeded here (idempotent, ON CONFLICT DO NOTHING):
--   sniper_min_entry_age_seconds  : age floor before ANY evaluation
--   sniper_watchlist_max_size     : cap on in-memory watchlist
--   sniper_watchlist_recheck_secs : how often to re-check watchlist
--   sniper_fail_closed_missing_bsr: reject on missing buy/sell data
--   sniper_phantom_price_threshold: |pnl_pct| ceiling before skip

INSERT INTO config_settings (config_type, key, value, description)
VALUES
    (
        'sniper_config',
        'sniper_min_entry_age_seconds',
        '180',
        'Wave-17: minimum token age (seconds) before evaluation begins. '
        'Detection adds to watchlist; entry fires only when age >= this floor. '
        'At 3 min Birdeye/holder data exist and gates are meaningful. '
        '0 = disabled (reverts to t=0 behavior, not recommended). '
        'Dashboard note: surface as "Min entry age" knob.'
    ),
    (
        'sniper_config',
        'sniper_watchlist_max_size',
        '500',
        'Wave-17: max in-memory watchlist slots (too-young pools awaiting '
        'the age floor). When full, new detections are dropped silently. '
        'Prevents unbounded memory growth during pump.fun launch bursts.'
    ),
    (
        'sniper_config',
        'sniper_watchlist_recheck_secs',
        '15',
        'Wave-17: how often (seconds) the watchlist poller re-checks '
        'too-young tokens. Lower = enter sooner after age floor is crossed; '
        'higher = fewer Birdeye/safety API calls. Default 15s is a '
        'reasonable balance for 3-min floor (re-checks 12x per window).'
    ),
    (
        'sniper_config',
        'sniper_fail_closed_missing_bsr',
        'true',
        'Wave-17: when true, a token with no Birdeye buy/sell data at '
        'entry time (age >= min_entry_age_seconds) is REJECTED (fail-closed). '
        'At t=0 absence of data was expected; at 3min+ it signals '
        'low liquidity or a dead pool. Set false to revert to fail-open '
        '(legacy behavior, not recommended). Dashboard: surface as toggle.'
    ),
    (
        'sniper_config',
        'sniper_phantom_price_threshold',
        '300',
        'Wave-17: |pnl_pct| ceiling in the monitor loop. If the computed '
        'P&L exceeds +/-this threshold the price read is treated as a '
        'stale/wrong-unit artifact and skipped (DRY_RUN: synthetic close; '
        'LIVE: skip with warning). Replaces the hardcoded 200 in wave-16. '
        'Default 300 gives headroom for real 3x moves without acting on '
        'the +8000% phantom-price artifacts.'
    )
ON CONFLICT (config_type, key) DO NOTHING;
