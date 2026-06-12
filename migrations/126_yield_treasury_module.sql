-- Migration 126: yield_treasury — config seeds + yield_treasury_advice table
--
-- ADVISORY idle-capital carry observer (modules/yield_treasury/). Tracks
-- blue-chip yields (Aave v3 USDC supply via free RPC eth_call; jitoSOL and
-- stETH via free public APIs) and advises how much of the bot''s idle float
-- could earn if parked, net of roundtrip gas/fees, with a hard withdrawal-
-- latency cap so advice can never point at a venue that would lock capital
-- a trading module needs. It NEVER deposits, NEVER signs, NEVER touches the
-- killswitch in this build: the live gate chain terminates in
-- live_deposit_path_not_built even when fully open (Phase-2 build, gated
-- behind treasury Phase 2 per docs/agents/NEW_MODULE_IDEAS.md idea 7).
-- Applying this migration changes NO behavior: the module only runs when
-- YIELD_TREASURY_MODULE_ENABLED=true (default false). All defaults are
-- observe-safe: shadow_mode=true, live_execution_enabled=false.
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('yield_treasury', 'poll_interval_seconds', '900', 'int',
     'Seconds between advisory ticks (slow loop; default 15 min).',
     NOW(), NOW()),

    ('yield_treasury', 'shadow_mode', 'true', 'bool',
     'Master shadow flag. true = advice rows only. Must be false AND '
     'live_execution_enabled=true AND should_skip_live pass AND RiskManager '
     'approve before any live path is even considered — and in this build '
     'the path still terminates in live_deposit_path_not_built.',
     NOW(), NOW()),

    ('yield_treasury', 'live_execution_enabled', 'false', 'bool',
     'Second live gate (fail-safe default false). No deposit code ships in '
     'this build regardless; flipping this alone does nothing.',
     NOW(), NOW()),

    ('yield_treasury', 'min_idle_usd', '250', 'float',
     'Minimum deployable stablecoin (USD) below which advice is HOLD '
     '(idle_below_min) — sub-scale carry never clears gas.', NOW(), NOW()),

    ('yield_treasury', 'min_idle_sol', '1.0', 'float',
     'Minimum deployable SOL for the jitoSOL venue.', NOW(), NOW()),

    ('yield_treasury', 'min_idle_eth', '0.05', 'float',
     'Minimum deployable ETH for the stETH venue.', NOW(), NOW()),

    ('yield_treasury', 'max_deploy_frac', '0.5', 'float',
     'At most this fraction of touchable idle (idle minus undeployed floor) '
     'may ever be advised into yield, fleet-wide.', NOW(), NOW()),

    ('yield_treasury', 'venue_cap_frac', '0.25', 'float',
     'At most this fraction of total idle at any single venue (smart-'
     'contract concentration cap).', NOW(), NOW()),

    ('yield_treasury', 'undeployed_floor_usd', '100', 'float',
     'Stablecoin floor (USD) per chain that yield advice can never touch — '
     'trading float + buffer stays liquid.', NOW(), NOW()),

    ('yield_treasury', 'undeployed_floor_sol', '0.2', 'float',
     'SOL floor never advised into jitoSOL (fees + Jito tips + ATA rent).',
     NOW(), NOW()),

    ('yield_treasury', 'undeployed_floor_eth', '0.02', 'float',
     'ETH floor never advised into stETH (mainnet exit gas).', NOW(), NOW()),

    ('yield_treasury', 'horizon_days', '30', 'float',
     'Assumed parking horizon for the net-benefit calc; net_horizon = '
     'gross_daily * horizon_days - roundtrip_cost.', NOW(), NOW()),

    ('yield_treasury', 'max_breakeven_days', '10', 'float',
     'HOLD when roundtrip costs take longer than this to earn back '
     '(breakeven_too_slow).', NOW(), NOW()),

    ('yield_treasury', 'max_withdrawal_latency_s', '86400', 'float',
     'Hard cap: venues whose recall latency exceeds this are always HOLD '
     '(withdrawal_latency_exceeds_cap) — parked capital must be recallable '
     'within a day for the trading fleet.', NOW(), NOW()),

    ('yield_treasury', 'roundtrip_cost_usd_ethereum', '16.0', 'float',
     'Assumed deposit+withdraw gas (USD) for Aave v3 on Ethereum mainnet.',
     NOW(), NOW()),

    ('yield_treasury', 'roundtrip_cost_usd_arbitrum', '0.30', 'float',
     'Assumed deposit+withdraw gas (USD) for Aave v3 on Arbitrum.',
     NOW(), NOW()),

    ('yield_treasury', 'roundtrip_cost_usd_base', '0.20', 'float',
     'Assumed deposit+withdraw gas (USD) for Aave v3 on Base.', NOW(), NOW()),

    ('yield_treasury', 'lst_roundtrip_bps', '10', 'float',
     'Proportional roundtrip cost (bps) for LST entry+exit via swap '
     '(jitoSOL/SOL and stETH/ETH are tight pairs).', NOW(), NOW()),

    ('yield_treasury', 'sol_tx_cost_sol', '0.001', 'float',
     'Fixed SOL tx cost assumed for the jitoSOL roundtrip.', NOW(), NOW()),

    ('yield_treasury', 'eth_lst_roundtrip_cost_eth', '0.004', 'float',
     'Fixed ETH gas assumed for the stETH roundtrip.', NOW(), NOW()),

    ('yield_treasury', 'withdrawal_latency_s_aave', '120', 'float',
     'Recall latency for Aave v3 supply: one withdraw tx + confirmation.',
     NOW(), NOW()),

    ('yield_treasury', 'withdrawal_latency_s_jitosol', '120', 'float',
     'Recall latency for jitoSOL via Jupiter swap-out.', NOW(), NOW()),

    ('yield_treasury', 'withdrawal_latency_s_steth', '259200', 'float',
     'Recall latency for stETH, conservatively the withdrawal queue (~3 '
     'days) — under the default 1-day cap this venue is always HOLD, by '
     'design, until a swap-out path is reviewed.', NOW(), NOW()),

    ('yield_treasury', 'lido_apr_url',
     'https://eth-api.lido.fi/v1/protocol/steth/apr/last', 'string',
     'Free Lido endpoint for the stETH APR (percent; parsed shape-tolerant).',
     NOW(), NOW()),

    ('yield_treasury', 'jitosol_apy_url',
     'https://extra-api.sanctum.so/v1/apy/latest?lst=jitoSOL', 'string',
     'Free Sanctum endpoint for the jitoSOL APY (fraction; parsed '
     'shape-tolerant). APR feed failure = HOLD, never a stale deploy.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- One advice row per venue per tick (HOLD rows included — the record that
-- carry does NOT pay on a small float is the honest product here).
CREATE TABLE IF NOT EXISTS yield_treasury_advice (
    id                    BIGSERIAL PRIMARY KEY,
    venue                 TEXT NOT NULL,            -- aave_v3 | jito_jitosol | lido_steth
    chain                 TEXT NOT NULL,
    asset                 TEXT NOT NULL,            -- USDC | SOL | ETH
    unit                  TEXT NOT NULL,            -- USD | SOL | ETH (all amounts below)
    apr                   DOUBLE PRECISION,         -- observed; NULL = feed unavailable
    idle_amount           DOUBLE PRECISION NOT NULL DEFAULT 0,
    deployable_amount     DOUBLE PRECISION NOT NULL DEFAULT 0,
    gross_daily           DOUBLE PRECISION NOT NULL DEFAULT 0,
    roundtrip_cost        DOUBLE PRECISION NOT NULL DEFAULT 0,
    breakeven_days        DOUBLE PRECISION,         -- NULL = never (no yield)
    horizon_days          DOUBLE PRECISION NOT NULL DEFAULT 0,
    net_horizon           DOUBLE PRECISION NOT NULL DEFAULT 0,
    withdrawal_latency_s  INTEGER NOT NULL DEFAULT 0,
    latency_note          TEXT,                     -- the caveat every row carries
    recommendation        TEXT NOT NULL,            -- HOLD | DEPLOY_CANDIDATE (advice only)
    reason                TEXT,
    shadow                BOOLEAN NOT NULL DEFAULT TRUE,
    details               JSONB,                    -- {live_skip_reason} for candidates
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_yield_advice_venue_created
    ON yield_treasury_advice(venue, chain, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_yield_advice_created
    ON yield_treasury_advice(created_at DESC);

COMMIT;
