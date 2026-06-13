-- Migration 135: COPY_TRADING v3 — profitable-wallet discovery + shadow-copy
-- simulator tables + config seeds.
--
-- Applying this migration changes NO behavior:
--   * copy_auto_discovery_enabled (mig 092) stays the master discovery gate
--     and defaults FALSE; every new v3 flag below also defaults SAFE
--     (auto-promote OFF, shadow sim OFF, score sizing OFF, decay demotion OFF).
--   * A discovered wallet is NEVER traded directly. Promotion to an active
--     leader requires operator approval of the copy_leader_candidates row
--     (or the explicit double-gated auto-promote: flag=true AND
--     copy_v3_auto_promote_max_leaders > 0, default 0).
--   * Shadow-sim rows are paper-only and carry is_simulated=TRUE.
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers). Idempotent: IF NOT EXISTS / ON CONFLICT DO NOTHING.

BEGIN;

-- ---------------------------------------------------------------------
-- 1. Discovered-wallet leaderboard (the v3 discovery output).
--    Complements copy_leader_candidates (mig 092): this table is the FULL
--    ranked universe with score breakdown + provenance; candidates above
--    the proposal thresholds are ALSO upserted into copy_leader_candidates
--    (status=pending) for the existing operator-approval flow.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS copy_discovered_wallets (
    id                  BIGSERIAL PRIMARY KEY,
    chain               TEXT NOT NULL,
    wallet_address      TEXT NOT NULL,
    sources             JSONB NOT NULL DEFAULT '[]',   -- provenance list, e.g. ['smart_money','onchain']
    score               DOUBLE PRECISION,               -- 0..100 composite (wallet_profitability)
    score_breakdown     JSONB,                          -- per-component values + penalties
    realized_pnl_usd    DOUBLE PRECISION,               -- trailing-window REALIZED only
    win_rate            DOUBLE PRECISION,               -- raw closed-round-trip ratio 0..1
    profit_factor       DOUBLE PRECISION,               -- gross wins / gross losses (capped)
    trade_count         INTEGER NOT NULL DEFAULT 0,     -- closed round-trips in window
    max_drawdown_pct    DOUBLE PRECISION,               -- 0..1 on cumulative realized PnL
    avg_hold_seconds    DOUBLE PRECISION,
    consistency         DOUBLE PRECISION,               -- 0..1 daily-PnL regularity
    diversification     DOUBLE PRECISION,               -- 0..1 (1 - notional HHI, token-count scaled)
    wash_penalty        DOUBLE PRECISION,               -- 0..1 fast-roundtrip / self-trade share
    lucky_penalty       DOUBLE PRECISION,               -- 0..1 single-trade PnL concentration
    pnl_basis           TEXT NOT NULL DEFAULT 'usd',    -- 'usd' | 'sol_numeraire' (honesty flag)
    window_days         INTEGER NOT NULL DEFAULT 30,
    status              TEXT NOT NULL DEFAULT 'discovered',  -- discovered|proposed|promoted|rejected
    first_seen_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_scored_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_copy_discovered_wallet UNIQUE (chain, wallet_address)
);
CREATE INDEX IF NOT EXISTS idx_copy_discovered_score
    ON copy_discovered_wallets (score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_copy_discovered_status
    ON copy_discovered_wallets (status, score DESC NULLS LAST);

COMMENT ON TABLE copy_discovered_wallets IS
    'COPY v3 wallet-discovery leaderboard. Read-only universe of candidate leader wallets scored on trailing REALIZED performance (no forward returns). Never traded directly; promotion flows through copy_leader_candidates approval.';

-- ---------------------------------------------------------------------
-- 2. Shadow-copy simulator: paper fills + open positions + equity curve.
--    Everything here is simulated; is_simulated defaults TRUE and the
--    simulator never writes FALSE.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS copy_shadow_fills (
    id                  BIGSERIAL PRIMARY KEY,
    chain               TEXT NOT NULL,
    wallet_address      TEXT NOT NULL,                  -- the LEADER wallet being shadow-copied
    token               TEXT NOT NULL,
    token_symbol        TEXT,
    side                TEXT NOT NULL,                  -- 'buy' | 'sell'
    leader_price_usd    DOUBLE PRECISION,               -- observed at-event price
    fill_price_usd      DOUBLE PRECISION,               -- slippage+fee adjusted paper fill
    qty                 DOUBLE PRECISION NOT NULL,
    notional_usd        DOUBLE PRECISION NOT NULL,
    fee_usd             DOUBLE PRECISION NOT NULL DEFAULT 0,
    realized_pnl_usd    DOUBLE PRECISION,               -- populated on sells only
    source_ref          TEXT NOT NULL,                  -- tx hash / event id provenance
    event_source        TEXT NOT NULL DEFAULT 'smart_money',  -- smart_money | helius | engine
    event_time          TIMESTAMPTZ NOT NULL,
    is_simulated        BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_copy_shadow_fill UNIQUE (chain, wallet_address, token, side, source_ref)
);
CREATE INDEX IF NOT EXISTS idx_copy_shadow_fills_wallet
    ON copy_shadow_fills (chain, wallet_address, event_time DESC);

CREATE TABLE IF NOT EXISTS copy_shadow_positions (
    id                  BIGSERIAL PRIMARY KEY,
    chain               TEXT NOT NULL,
    wallet_address      TEXT NOT NULL,                  -- leader
    token               TEXT NOT NULL,
    token_symbol        TEXT,
    qty                 DOUBLE PRECISION NOT NULL DEFAULT 0,   -- OUR paper qty
    cost_usd            DOUBLE PRECISION NOT NULL DEFAULT 0,   -- OUR remaining cost basis
    leader_qty          DOUBLE PRECISION NOT NULL DEFAULT 0,   -- leader open qty we tracked
    last_price_usd      DOUBLE PRECISION,
    adds_count          INTEGER NOT NULL DEFAULT 0,
    opened_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_simulated        BOOLEAN NOT NULL DEFAULT TRUE,
    CONSTRAINT uq_copy_shadow_position UNIQUE (chain, wallet_address, token)
);

CREATE TABLE IF NOT EXISTS copy_shadow_equity (
    id                  BIGSERIAL PRIMARY KEY,
    chain               TEXT NOT NULL,
    wallet_address      TEXT NOT NULL,                  -- leader
    realized_pnl_usd    DOUBLE PRECISION NOT NULL DEFAULT 0,   -- cumulative
    unrealized_pnl_usd  DOUBLE PRECISION NOT NULL DEFAULT 0,   -- mark at last seen price
    equity_usd          DOUBLE PRECISION NOT NULL DEFAULT 0,   -- realized + unrealized
    open_positions      INTEGER NOT NULL DEFAULT 0,
    fills_count         INTEGER NOT NULL DEFAULT 0,
    is_simulated        BOOLEAN NOT NULL DEFAULT TRUE,
    snapshot_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_copy_shadow_equity_wallet
    ON copy_shadow_equity (chain, wallet_address, snapshot_at DESC);

-- Per-wallet ingest cursor so a tick never reprocesses (or skips) events.
CREATE TABLE IF NOT EXISTS copy_shadow_cursors (
    chain               TEXT NOT NULL,
    wallet_address      TEXT NOT NULL,
    last_event_time     TIMESTAMPTZ,
    last_event_id       BIGINT NOT NULL DEFAULT 0,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (chain, wallet_address)
);

-- ---------------------------------------------------------------------
-- 3. Engine v3 columns: score-decay demotion needs the historical peak.
-- ---------------------------------------------------------------------
ALTER TABLE copy_leader_scores
    ADD COLUMN IF NOT EXISTS peak_score NUMERIC(6, 2);
COMMENT ON COLUMN copy_leader_scores.peak_score IS
    'Highest composite score ever observed for this leader; used by the (default-OFF) score-decay demotion to bench leaders whose edge decayed.';

ALTER TABLE copy_leader_candidates
    ADD COLUMN IF NOT EXISTS score_breakdown JSONB,
    ADD COLUMN IF NOT EXISTS provenance JSONB;

-- ---------------------------------------------------------------------
-- 4. Config seeds (copytrading_config). ON CONFLICT DO NOTHING so
--    operator overrides survive. value_type vocabulary follows mig 092.
-- ---------------------------------------------------------------------
INSERT INTO config_settings (config_type, key, value, value_type)
VALUES
    -- Discovery v3 (master gate stays copy_auto_discovery_enabled, mig 092,
    -- default false; interval copy_auto_discovery_interval_hours reused).
    ('copytrading_config', 'copy_v3_sources',                     'smart_money,onchain,leader_scores,dexscreener', 'string'),
    ('copytrading_config', 'copy_v3_window_days',                 '30',    'number'),
    ('copytrading_config', 'copy_v3_min_score',                   '60',    'number'),
    ('copytrading_config', 'copy_v3_min_trades',                  '10',    'number'),
    ('copytrading_config', 'copy_v3_min_realized_pnl_usd',        '500',   'number'),
    ('copytrading_config', 'copy_v3_max_lucky_share',             '0.6',   'number'),
    ('copytrading_config', 'copy_v3_max_wash_penalty',            '0.4',   'number'),
    ('copytrading_config', 'copy_v3_max_candidates_per_sweep',    '25',    'number'),
    ('copytrading_config', 'copy_v3_max_rpc_enrich_wallets',      '5',     'number'),
    -- Auto-promote: DOUBLE-gated. Even with the flag true, max_leaders=0
    -- promotes nothing. Default behaviour: operator approval only.
    ('copytrading_config', 'copy_v3_auto_promote_enabled',        'false', 'boolean'),
    ('copytrading_config', 'copy_v3_auto_promote_max_leaders',    '0',     'number'),
    ('copytrading_config', 'copy_v3_auto_promote_min_shadow_fills', '10',  'number'),
    -- Shadow-copy simulator (paper-only; writes is_simulated=true rows).
    ('copytrading_config', 'copy_shadow_sim_enabled',             'false', 'boolean'),
    ('copytrading_config', 'copy_shadow_sim_interval_s',          '300',   'number'),
    ('copytrading_config', 'copy_shadow_sim_notional_usd',        '100',   'number'),
    ('copytrading_config', 'copy_shadow_fee_bps',                 '30',    'number'),
    ('copytrading_config', 'copy_shadow_slippage_bps',            '50',    'number'),
    ('copytrading_config', 'copy_shadow_max_adds_per_position',   '3',     'number'),
    ('copytrading_config', 'copy_shadow_max_wallets',             '40',    'number'),
    -- Engine v3: confidence-weighted sizing (can only SHRINK below the
    -- existing caps; multiplier is clamped to [floor, 1.0]).
    ('copytrading_config', 'copy_score_sizing_enabled',           'false', 'boolean'),
    ('copytrading_config', 'copy_score_sizing_floor',             '0.25',  'number'),
    -- Engine v3: score-decay demotion (uses the EXISTING probation bench).
    ('copytrading_config', 'copy_score_decay_demotion_enabled',   'false', 'boolean'),
    ('copytrading_config', 'copy_score_decay_pct',                '40',    'number'),
    ('copytrading_config', 'copy_score_decay_min_trades',         '10',    'number')
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;
