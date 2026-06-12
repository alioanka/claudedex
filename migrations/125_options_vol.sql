-- Migration 125: options_vol — config seeds + suggestions/surface tables
--
-- New SHADOW-FIRST module (modules/options_vol/): Deribit crypto-options
-- HEDGING advisor. Reads the free public Deribit chain, recomputes IV from
-- marks, compares to trailing realized vol, and suggests protective-put /
-- collar hedges sized against the fleet''s net long delta — recorded
-- SIMULATED (is_simulated=true). Applying this migration changes NO
-- behavior: the module only runs when OPTIONS_VOL_MODULE_ENABLED=true
-- (default false), shadow_mode defaults TRUE, live_execution_enabled
-- defaults FALSE, and the live path is additionally gated by
-- should_skip_live + RiskManager + a hard monthly premium budget. SELL legs
-- can never go live (record-only).
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('options_vol', 'shadow_mode', 'true', 'bool',
     'Record-only mode: every suggestion is simulated and the live order '
     'path is never reached. Flip to false ONLY after >= 4 weeks of clean '
     'shadow advisories.', NOW(), NOW()),

    ('options_vol', 'live_execution_enabled', 'false', 'bool',
     'Explicit LIVE opt-in (fail-safe off). Even when true, orders also '
     'require shadow_mode=false, should_skip_live to pass, RiskManager '
     'approval and remaining monthly premium budget.', NOW(), NOW()),

    ('options_vol', 'premium_selling_enabled', 'false', 'bool',
     'Named gate reserved for a future, separately-reviewed premium-selling '
     'leg. The current code records SELL legs simulated regardless of this '
     'flag — short-vol live execution is deliberately not implemented.',
     NOW(), NOW()),

    ('options_vol', 'poll_interval_s', '300', 'int',
     'Seconds between engine cycles (vol surfaces move slowly; keep the '
     'free public API unstressed).', NOW(), NOW()),

    ('options_vol', 'max_requests_per_minute', '20', 'int',
     'Client-side cap on outbound Deribit public API requests.', NOW(), NOW()),

    ('options_vol', 'currencies', 'BTC,ETH', 'string',
     'Comma-separated currencies to build surfaces for. BTC/ETH ONLY — alt '
     'option books on Deribit are too thin for a bot this size.', NOW(), NOW()),

    ('options_vol', 'hedge_currency', 'BTC', 'string',
     'The single currency hedge advisories are issued in (deepest book; the '
     'fleet''s correlated-drawdown scenario is a beta event).', NOW(), NOW()),

    ('options_vol', 'fleet_beta', '1.0', 'float',
     'Beta mapping fleet spot exposure (mostly alts/SOL) onto the hedge '
     'currency. 1.0 = assume the book falls 1:1 with BTC in the tail.',
     NOW(), NOW()),

    ('options_vol', 'rv_window_hours', '720', 'int',
     'Trailing window (hours of hourly perp closes) for realized vol; 720 = '
     '30 days.', NOW(), NOW()),

    ('options_vol', 'hedge_delta_threshold_usd', '1000', 'float',
     'Fleet net long delta (USD) above which a hedge is suggested; only the '
     'EXCESS above this threshold is hedged.', NOW(), NOW()),

    ('options_vol', 'hedge_coverage_ratio', '0.5', 'float',
     'Fraction of the excess delta the hedge aims to offset (best-effort; '
     'the premium cap below is the hard constraint).', NOW(), NOW()),

    ('options_vol', 'put_target_delta', '-0.25', 'float',
     'Target forward delta of the protective put (25-delta OTM).', NOW(), NOW()),

    ('options_vol', 'call_target_delta', '0.25', 'float',
     'Target forward delta of the collar''s short call leg.', NOW(), NOW()),

    ('options_vol', 'tenor_min_days', '5', 'int',
     'Minimum option tenor considered (avoid pin/expiry games).', NOW(), NOW()),

    ('options_vol', 'tenor_max_days', '21', 'int',
     'Maximum option tenor considered (short-dated tail protection only).',
     NOW(), NOW()),

    ('options_vol', 'collar_min_ivrv', '1.15', 'float',
     'When ATM IV / realized vol exceeds this, vol is rich: suggest a collar '
     '(sell an OTM call to finance the put) instead of a plain put.',
     NOW(), NOW()),

    ('options_vol', 'min_open_interest', '10', 'float',
     'Skip strikes with less open interest than this (contracts).', NOW(), NOW()),

    ('options_vol', 'max_quote_spread_frac', '0.25', 'float',
     'Skip strikes whose (ask-bid)/mark exceeds this fraction.', NOW(), NOW()),

    ('options_vol', 'max_premium_per_hedge_usd', '25', 'float',
     'HARD cap on premium per suggested hedge; sizing is scaled down to fit. '
     'Systematic put-buying bleeds — keep this small.', NOW(), NOW()),

    ('options_vol', 'monthly_premium_budget_usd', '50', 'float',
     'HARD monthly cap on LIVE premium spend (counted from live rows in '
     'options_vol_suggestions; the gate fails CLOSED on query error).',
     NOW(), NOW()),

    ('options_vol', 'shadow_record_interval_s', '3600', 'int',
     'Per-currency throttle between recorded hedge advisories.', NOW(), NOW()),

    ('options_vol', 'surface_snapshot_interval_s', '3600', 'int',
     'Per-currency throttle between vol-surface snapshots.', NOW(), NOW()),

    ('options_vol', 'deribit_base_url', 'https://www.deribit.com', 'string',
     'Deribit API base URL (point at https://test.deribit.com to rehearse '
     'a future live flip against testnet).', NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- Hedge suggestion / (future) position ledger. One row per LEG; multi-leg
-- structures (collars) share a structure_id. is_simulated defaults TRUE and
-- only the gated executor ever writes FALSE. premium_usd is signed:
-- positive = premium paid (BUY), negative = premium received (SELL leg,
-- always simulated).
CREATE TABLE IF NOT EXISTS options_vol_suggestions (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    structure_id        TEXT,
    suggestion_type     TEXT NOT NULL,        -- 'protective_put' | 'collar_put_leg' | 'collar_call_leg'
    currency            TEXT NOT NULL,        -- 'BTC' | 'ETH'
    instrument_name     TEXT,                 -- e.g. 'BTC-26JUN26-90000-P'
    side                TEXT NOT NULL,        -- 'BUY' | 'SELL'
    option_type         TEXT,                 -- 'put' | 'call'
    strike              DOUBLE PRECISION,
    expiry              TIMESTAMPTZ,
    contracts           DOUBLE PRECISION,
    index_price         DOUBLE PRECISION,
    mark_iv             DOUBLE PRECISION,     -- our own bisection IV from mark
    rv                  DOUBLE PRECISION,     -- trailing realized vol
    ivrv_ratio          DOUBLE PRECISION,
    delta               DOUBLE PRECISION,
    premium_usd         DOUBLE PRECISION,
    fleet_net_delta_usd DOUBLE PRECISION,
    hedged_delta_usd    DOUBLE PRECISION,
    status              TEXT NOT NULL DEFAULT 'simulated',  -- 'simulated' | 'live_submitted' | 'live_failed'
    order_id            TEXT,
    skip_reason         TEXT,
    is_simulated        BOOLEAN NOT NULL DEFAULT TRUE,
    details             JSONB
);
CREATE INDEX IF NOT EXISTS idx_optvol_sugg_created ON options_vol_suggestions(created_at);
CREATE INDEX IF NOT EXISTS idx_optvol_sugg_structure ON options_vol_suggestions(structure_id);
CREATE INDEX IF NOT EXISTS idx_optvol_sugg_live ON options_vol_suggestions(is_simulated, created_at);

-- Transparent per-expiry vol-surface history (ATM IV vs trailing RV), so the
-- IV-vs-RV signal is auditable after the fact.
CREATE TABLE IF NOT EXISTS options_vol_surface (
    id           BIGSERIAL PRIMARY KEY,
    snapshot_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    currency     TEXT NOT NULL,
    expiry       TIMESTAMPTZ NOT NULL,
    tenor_days   DOUBLE PRECISION,
    atm_strike   DOUBLE PRECISION,
    atm_iv       DOUBLE PRECISION,
    rv           DOUBLE PRECISION,
    ivrv_ratio   DOUBLE PRECISION,
    index_price  DOUBLE PRECISION,
    n_strikes    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_optvol_surface_cur_time ON options_vol_surface(currency, snapshot_at);

COMMIT;
