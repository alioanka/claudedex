-- Migration 141: COPY_TRADING Wave-F5 — wallet-discovery revival.
--
-- Root cause addressed (docs/agents/wave-f5/04_copy_aitrader.md): every
-- external candidate source returned 0 wallets, discovery degraded to a local
-- fallback built from already-tracked wallets, and the v3 discovery engine
-- (mig 135) had never run because copy_auto_discovery_enabled was false.
--
-- This migration is DISCOVERY/SHADOW-ONLY. It touches NO live-trading path:
--   * The tables below are read/written only by the discovery sweep and the
--     shadow simulator (paper). Neither can place an order.
--   * The conditional seeds at the bottom flip copy_auto_discovery_enabled and
--     copy_shadow_sim_enabled from 'false' to 'true' ONLY WHERE the stored
--     value is still 'false' (operator overrides are preserved). No
--     live-execution flag is created or changed; a discovered wallet is still
--     NEVER traded without operator approval (copy_leader_candidates flow) or
--     the pre-existing double-gated auto-promote (default OFF, max_leaders=0).
--
-- Single-quoted SQL literals only. Idempotent: IF NOT EXISTS / ON CONFLICT.

BEGIN;

-- ---------------------------------------------------------------------
-- 1. Cross-sweep fee-payer accumulation.
--    Replaces the broken "wallet must appear >=5 times inside ONE 100-tx
--    snapshot" sampling heuristic: swap counts accumulate PERSISTENTLY across
--    sweeps until a wallet clears discovery_min_swaps, at which point it
--    becomes a discovery candidate. Populated by the helius_tokens source.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS copy_discovery_feepayers (
    chain            TEXT NOT NULL,
    wallet_address   TEXT NOT NULL,
    source           TEXT NOT NULL DEFAULT 'helius_tokens',  -- helius_tokens | rpc_solana
    cumulative_swaps INTEGER NOT NULL DEFAULT 0,
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (chain, wallet_address, source)
);
CREATE INDEX IF NOT EXISTS idx_copy_feepayers_swaps
    ON copy_discovery_feepayers (cumulative_swaps DESC);

COMMENT ON TABLE copy_discovery_feepayers IS
    'COPY v3 cross-sweep fee-payer accumulation. Persistent per-wallet swap counts so low-frequency wallets clear the min-swaps floor over days instead of needing >=N swaps inside one Helius snapshot. Read-only screening input; never traded directly.';

-- ---------------------------------------------------------------------
-- 2. Helius daily call budget (shared discipline).
--    The same Helius key serves the copy monitor's per-wallet poll AND
--    discovery; discovery must be a good tenant. One counter row per UTC day.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS copy_helius_budget (
    day    DATE PRIMARY KEY,
    calls  INTEGER NOT NULL DEFAULT 0
);

COMMENT ON TABLE copy_helius_budget IS
    'COPY v3 Helius daily call-budget counter (UTC day). Discovery checks helius_daily_call_budget before each request so it cannot exhaust the shared key.';

-- ---------------------------------------------------------------------
-- 3. Config seeds (copytrading_config). ON CONFLICT DO NOTHING so operator
--    overrides survive. Discovery-only knobs.
-- ---------------------------------------------------------------------
INSERT INTO config_settings (config_type, key, value, value_type)
VALUES
    -- Helius quota discipline (item 4).
    ('copytrading_config', 'helius_daily_call_budget', '500', 'number'),
    ('copytrading_config', 'helius_tx_sample',         '100', 'number'),
    ('copytrading_config', 'discovery_min_swaps',      '3',   'number'),
    -- smart_money_scores source floor (item 6): smart_money_wallet_scores
    -- score threshold to propose an EVM wallet into copy_leader_candidates.
    ('copytrading_config', 'copy_sm_min_score',        '0.6', 'number'),
    -- Leader holdings-snapshot reconciliation cadence, minutes (item 8c).
    -- Advisory only: logs drift between a leader's on-chain holdings and our
    -- mirrored positions. Never trades / auto-closes.
    ('copytrading_config', 'copy_reconcile_minutes',   '360', 'number')
ON CONFLICT (config_type, key) DO NOTHING;

-- ---------------------------------------------------------------------
-- 4. Conditional activation of DISCOVERY + SHADOW SIMULATION only.
--    Root cause: the v3 discovery engine (mig 135) had NEVER executed because
--    copy_auto_discovery_enabled (mig 092) was still 'false'. Turn it on — and
--    the paper shadow simulator — ONLY WHERE the stored value is still the
--    seeded 'false'. An operator who already changed either value is NOT
--    touched (never clobber operator overrides).
--
--    SAFETY: these two flags gate READ-ONLY discovery writes (candidates for
--    operator approval) and SIMULATED shadow fills (is_simulated=true). NEITHER
--    is a live-execution flag. No order path is enabled: a discovered wallet is
--    still only traded after operator approval (copy_leader_candidates) or the
--    pre-existing double-gated auto-promote (default OFF, max_leaders=0), and
--    every live gate (should_skip_live, RiskManager, DRY_RUN) is unchanged.
UPDATE config_settings SET value = 'true', updated_at = NOW()
 WHERE config_type = 'copytrading_config'
   AND key = 'copy_auto_discovery_enabled'
   AND value = 'false';

UPDATE config_settings SET value = 'true', updated_at = NOW()
 WHERE config_type = 'copytrading_config'
   AND key = 'copy_shadow_sim_enabled'
   AND value = 'false';

COMMIT;
