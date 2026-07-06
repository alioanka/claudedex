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

-- ── live-path safety knobs (wave-F5 BUG-2/3; only consulted AFTER the
--    shadow_mode/live_execution_enabled/should_skip_live chain passes) ──
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('polymarket_config', 'max_market_exposure_usd', '100', 'float',
     'Polymarket risk gate: max live USD notional per market (across both '
     'legs of an arb pair). Replaces the DEX-token RiskManager call, which '
     'ran honeypot analysis on CLOB token ids (wave-F5 BUG-3).', NOW(), NOW()),

    ('polymarket_config', 'max_total_exposure_usd', '500', 'float',
     'Polymarket risk gate: max total live USD notional across all open '
     'markets. Exceeding it records simulated with '
     'skip_reason=risk:total_exposure_cap.', NOW(), NOW()),

    ('polymarket_config', 'max_open_markets', '10', 'int',
     'Polymarket risk gate: max distinct markets with live exposure at once.',
     NOW(), NOW()),

    ('polymarket_config', 'order_fill_timeout_s', '30', 'int',
     'Live order lifecycle: seconds to poll a submitted CLOB order for a '
     'fill before cancelling it (never leave an unknown resting order).',
     NOW(), NOW()),

    ('polymarket_config', 'signature_type', '', 'string',
     'py-clob-client signature type: empty/0 = EOA-L1 wallet; 1/2 = '
     'Polymarket proxy wallet (requires funder_address).', NOW(), NOW()),

    ('polymarket_config', 'funder_address', '', 'string',
     'Proxy (funder) wallet address for signature_type 1/2; empty for EOA.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- ── forward-outcome tracking (edge proof; wave-F5 gap 8) ──
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('polymarket_config', 'snapshot_top_n_markets', '50', 'int',
     'Per-cycle price snapshots are written for the top-N watched markets '
     'by 24h volume (polymarket_price_snapshots — feeds the dashboard '
     'charts and the LATE outcome marks).', NOW(), NOW()),

    ('polymarket_config', 'snapshot_retention_days', '14', 'int',
     'Disk discipline: price snapshots older than this are pruned hourly.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- Per-cycle price/liquidity snapshots for watched markets. Feeds the
-- /polymarket dashboard charts and the outcome marker below.
CREATE TABLE IF NOT EXISTS polymarket_price_snapshots (
    id          BIGSERIAL PRIMARY KEY,
    market_id   TEXT NOT NULL,
    question    TEXT,
    category    TEXT,
    yes_price   DOUBLE PRECISION,
    no_price    DOUBLE PRECISION,
    best_bid    DOUBLE PRECISION,
    best_ask    DOUBLE PRECISION,
    volume_24h  DOUBLE PRECISION,
    liquidity   DOUBLE PRECISION,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_pm_snapshots_market_ts
    ON polymarket_price_snapshots (market_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_pm_snapshots_ts
    ON polymarket_price_snapshots (ts);

-- Forward outcomes per signal, marked LATE (mirrors smart_money): each
-- horizon is written only after it has FULLY elapsed, using the yes_price
-- observed at mark time (current cycle or latest snapshot) — never an early
-- estimate. fwd_return_* is in probability points x100 (yes moved 0.05 =
-- 5.0), SIGNED by signal direction (positive = the signal was right).
CREATE TABLE IF NOT EXISTS polymarket_signal_outcomes (
    signal_id            BIGINT PRIMARY KEY
                         REFERENCES polymarket_signals(id) ON DELETE CASCADE,
    market_id            TEXT NOT NULL,
    signal_type          TEXT NOT NULL,
    direction            TEXT,
    yes_price_at_signal  DOUBLE PRECISION,
    yes_price_1h         DOUBLE PRECISION,
    yes_price_6h         DOUBLE PRECISION,
    yes_price_24h        DOUBLE PRECISION,
    fwd_return_1h        DOUBLE PRECISION,
    fwd_return_6h        DOUBLE PRECISION,
    fwd_return_24h       DOUBLE PRECISION,
    fully_marked         BOOLEAN NOT NULL DEFAULT FALSE,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_pm_outcomes_type
    ON polymarket_signal_outcomes (signal_type, fully_marked);
CREATE INDEX IF NOT EXISTS idx_pm_outcomes_market
    ON polymarket_signal_outcomes (market_id);

COMMIT;
