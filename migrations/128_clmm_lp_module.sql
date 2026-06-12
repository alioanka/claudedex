-- Migration 128: CLMM LP module — config seeds + shadow-positions table
--
-- New SHADOW/ADVISORY-FIRST concentrated-liquidity module (modules/clmm_lp/).
-- Proposes Uniswap v3 / Orca range positions from free public pool data and
-- records them SIMULATED (is_simulated=true) with a transparent net-of-IL
-- expected APR: net = M*fee_apr*decay - M*vol^2/8 (LVR) - rebalance cost.
-- IL is marked against the HODL benchmark every cycle, never accrued silently.
-- LIVE mint/rebalance/burn is gated (shadow_mode=false AND
-- live_execution_enabled=true AND not should_skip_live AND RiskManager) and is
-- additionally NOT IMPLEMENTED in v1 — all-green gates still record simulated.
-- Applying this migration changes NO behavior on its own — the module only
-- runs when CLMM_LP_MODULE_ENABLED=true (default false).
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

-- ── config seeds (defaults mirror the in-code config.get(...) fallbacks) ──
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('clmm_lp', 'shadow_mode', 'true', 'bool',
     'Shadow mode (default ON): record every proposed CLMM position as a '
     'simulated clmm_shadow_positions row; never reach a live mint path. '
     'LP-ing below fair value is a known way to lose money — keep ON until '
     'the shadow track proves the net-of-IL model out-of-sample.', NOW(), NOW()),

    ('clmm_lp', 'live_execution_enabled', 'false', 'bool',
     'Explicit LIVE opt-in: even with shadow_mode off and module dry_run '
     'false, any live action also requires this true. Fail-safe default '
     'false. NOTE: v1 has no live transaction path regardless.', NOW(), NOW()),

    ('clmm_lp', 'poll_interval_s', '300', 'int',
     'Seconds between pool-snapshot / proposal / IL-mark cycles.', NOW(), NOW()),

    ('clmm_lp', 'candidate_pools',
     '[{"chain":"ethereum","pool_address":"0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640","fee_rate_bps":5,"label":"UNIv3 USDC/WETH 0.05%"},{"chain":"solana","pool_address":"Czfq3xZZDmsdGdUyrNLtRhGc47cXcZtLG4crryfu44zE","fee_rate_bps":30,"label":"Orca SOL/USDC whirlpool"}]',
     'string',
     'JSON list of candidate CLMM pools: chain (DexScreener slug), '
     'pool_address, fee_rate_bps, label. Quote token must be a USD stable. '
     'EVM fee tiers are verified on-chain via pool_engine (mismatch excludes '
     'the pool); Solana fee tiers are operator-verifiable — confirm before '
     'trusting the fee leg. Max 2 pools recommended in v1.', NOW(), NOW()),

    ('clmm_lp', 'range_width_pct', '10', 'int',
     'Half-width of the proposed symmetric range around spot, in percent '
     '(10 = range [0.90*P, 1.10*P]). Narrower boosts fees AND IL (both '
     'scale with the concentration factor) plus rebalance frequency.',
     NOW(), NOW()),

    ('clmm_lp', 'min_net_apr_pct', '10', 'float',
     'Proposal floor: only shadow-record a position when expected net-of-IL '
     'APR (fees minus LVR minus rebalance cost) clears this percent.',
     NOW(), NOW()),

    ('clmm_lp', 'fee_decay_factor', '0.7', 'float',
     'Haircut on the current 24h fee run-rate (volume decay + in-range LP '
     'competition). 1.0 = trust the spot fee APR; lower = more conservative.',
     NOW(), NOW()),

    ('clmm_lp', 'default_annual_vol', '0.8', 'float',
     'Annualized volatility assumption used until enough realized samples '
     'exist, and as a FLOOR while samples are sparse — underestimating vol '
     'overstates net APR (the measurement trap).', NOW(), NOW()),

    ('clmm_lp', 'min_vol_samples', '12', 'int',
     'Minimum price-return samples before realized vol replaces the default.',
     NOW(), NOW()),

    ('clmm_lp', 'rebalance_cost_bps', '30', 'int',
     'Cost per simulated re-range as bps of position size (gas + crossing '
     'the spread on the rebalancing swap). Charged on every close.',
     NOW(), NOW()),

    ('clmm_lp', 'max_position_size_usd', '100', 'float',
     'Per-position notional cap (USD). Applies to simulated sizing and any '
     'future live sizing.', NOW(), NOW()),

    ('clmm_lp', 'max_open_positions', '2', 'int',
     'Max simultaneously open shadow positions (doc guidance: max 2 pools '
     'in v1).', NOW(), NOW()),

    ('clmm_lp', 'shadow_record_interval_s', '3600', 'int',
     'Per-pool proposal throttle: at most one recorded proposal per this '
     'many seconds (keeps the shadow track honest, avoids row spam).',
     NOW(), NOW()),

    ('clmm_lp', 'max_position_age_hours', '168', 'int',
     'Simulated positions older than this are closed (close_reason='
     'max_age) so fee decay cannot be hidden by indefinite holding.',
     NOW(), NOW()),

    ('clmm_lp', 'out_of_range_exit_buffer_pct', '2', 'float',
     'Close a shadow position once price exits the range by more than this '
     'percent (simulating the re-range an LP would be forced into).',
     NOW(), NOW()),

    ('clmm_lp', 'dexscreener_base_url', 'https://api.dexscreener.com', 'string',
     'Free public pool-data API base (price / 24h volume / TVL; no key).',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- ── shadow position ledger (always simulated in v1) ──
CREATE TABLE IF NOT EXISTS clmm_shadow_positions (
    id                      BIGSERIAL PRIMARY KEY,
    chain                   TEXT NOT NULL,
    pool_address            TEXT NOT NULL,
    pool_label              TEXT,
    fee_rate_bps            INTEGER NOT NULL,
    entry_price             DOUBLE PRECISION NOT NULL,
    price_lower             DOUBLE PRECISION NOT NULL,
    price_upper             DOUBLE PRECISION NOT NULL,
    size_usd                DOUBLE PRECISION NOT NULL,
    amount0                 DOUBLE PRECISION,
    amount1                 DOUBLE PRECISION,
    liquidity               DOUBLE PRECISION,
    expected_fee_apr        DOUBLE PRECISION,
    expected_il_apr         DOUBLE PRECISION,
    expected_rebalance_apr  DOUBLE PRECISION,
    expected_net_apr        DOUBLE PRECISION,
    annual_vol_used         DOUBLE PRECISION,
    vol_source              TEXT,
    status                  TEXT NOT NULL DEFAULT 'open',
    is_simulated            BOOLEAN NOT NULL DEFAULT TRUE,
    skip_reason             TEXT,
    current_price           DOUBLE PRECISION,
    fees_usd                DOUBLE PRECISION NOT NULL DEFAULT 0,
    il_usd                  DOUBLE PRECISION NOT NULL DEFAULT 0,
    net_usd                 DOUBLE PRECISION NOT NULL DEFAULT 0,
    last_marked_at          TIMESTAMPTZ,
    closed_at               TIMESTAMPTZ,
    close_reason            TEXT,
    details                 JSONB,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_clmm_shadow_positions_status
    ON clmm_shadow_positions(status);
CREATE INDEX IF NOT EXISTS idx_clmm_shadow_positions_pool
    ON clmm_shadow_positions(chain, pool_address);
CREATE INDEX IF NOT EXISTS idx_clmm_shadow_positions_sim_created
    ON clmm_shadow_positions(is_simulated, created_at DESC);

COMMIT;
