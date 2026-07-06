-- Migration 137: Polymarket hardening — signal-quality knobs, live-safety
-- caps, and forward-outcome tracking (wave-F5 fix wave on top of mig 101).
--
-- Applying this migration changes NO live behavior: every seed preserves the
-- shadow-first defaults (shadow_mode=true, live_execution_enabled=false are
-- NOT touched); all new knobs default to the values the code already falls
-- back to. Idempotent (ON CONFLICT DO NOTHING / IF NOT EXISTS only).
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy).

BEGIN;

-- ── signal-quality knobs (wave-F5 BUG-1/4/5/7) ──
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('polymarket_config', 'arb_min_liquidity_usd', '1000', 'int',
     'Arb detector book-honesty gate: markets below this Gamma liquidity '
     '(USD) are skipped — a book nobody quotes cannot host an executable '
     'YES+NO<1 arb (the only arb firing in 20 shadow days was a dead book).',
     NOW(), NOW()),

    ('polymarket_config', 'arb_max_edge_bps', '500', 'int',
     'Arb detector stale-book cap: a GROSS YES+NO edge above this many bps '
     'is rejected as too-good-to-be-true (settled/dead market with drifting '
     'mids, not free money). 500 = 5%.', NOW(), NOW()),

    ('polymarket_config', 'momentum_min_volume_24h_usd', '5000', 'int',
     'Momentum signal only considers markets with at least this 24h volume '
     '(USD). Was consumed by code but never seeded (wave-F5 BUG-7).',
     NOW(), NOW()),

    ('polymarket_config', 'momentum_exclude_categories', '', 'string',
     'Comma-separated category DENYlist for momentum/new_market signals '
     '(e.g. ''Sports'' to drop in-play game noise). Empty = none excluded.',
     NOW(), NOW()),

    ('polymarket_config', 'momentum_flip_cooldown_minutes', '30', 'int',
     'Per-market direction-flip cooldown: after a recorded momentum signal, '
     'an OPPOSITE-direction signal on the same market is suppressed for this '
     'many minutes (in-play books flip YES/NO within minutes — noise).',
     NOW(), NOW()),

    ('polymarket_config', 'new_market_max_age_hours', '24', 'int',
     'new_market signals require a Gamma created_at younger than this many '
     'hours — an old market re-entering the top-volume window is NOT new '
     '(wave-F5 BUG-5).', NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;
