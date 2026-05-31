-- Migration 046: Central allocation guard — module budgets + orchestrator budget output
--
-- Wave-15 capital allocation enforcement.
--
-- Problem:
--   Multiple modules share the SAME on-chain wallets (EVM 0x802f...1D4D and
--   Solana BZ7dWN...qedR). The portfolio_allocations table only proposes
--   USD budgets; nothing actually prevents two modules from spending the same
--   lamports. This migration seeds:
--     1. Per-module capital budgets in config_settings (the AllocationGuard
--        reads these as its static defaults).
--     2. A `budget_usd` column on orchestrator_recommendations so the
--        orchestrator can emit a recommended budget per module each tick,
--        which the AllocationGuard reads as its dynamic override.
--     3. A wallet-group registry in config_settings so the guard can sum
--        total committed capital per SHARED wallet (EVM group vs Solana group).
--
-- Wallet groups (operator configurable):
--   evm_wallet   : dex, arbitrage, ai, futures   (share 0x802f...1D4D)
--   solana_wallet: solana, sniper, copy_trading   (share BZ7dWN...qedR)
--   Note: "dex" module trades on BOTH wallets depending on chain; for guard
--   purposes we assign it to its primary group (evm_wallet) and the Solana
--   portion is captured by the solana_positions table entry_usd column.
--
-- Idempotent: ON CONFLICT DO NOTHING on all inserts, ADD COLUMN IF NOT EXISTS.
-- Date: 2026-05-31

-- -----------------------------------------------------------------------
-- A. Add budget_usd column to orchestrator_recommendations
--    The orchestrator engine writes its recommended USD budget per module
--    alongside the to_live/hold/etc verdict. The AllocationGuard reads the
--    most recent non-superseded row's budget_usd per module; NULL means
--    "no orchestrator budget override — fall back to static config".
-- -----------------------------------------------------------------------
ALTER TABLE orchestrator_recommendations
    ADD COLUMN IF NOT EXISTS budget_usd NUMERIC(20, 2);

COMMENT ON COLUMN orchestrator_recommendations.budget_usd IS
    'Recommended capital budget in USD for this module, emitted by the '
    'orchestrator scorer alongside the to_live/hold verdict. NULL = no '
    'recommendation (guard falls back to allocation_guard_config static default). '
    'Added in migration 046 (wave-15).';

-- -----------------------------------------------------------------------
-- B. Per-module static budget defaults (allocation_guard_config)
--    Operator-tunable via ConfigManager. The AllocationGuard reads these
--    as the fallback when no orchestrator budget is available.
--    Defaults are conservative: total book ~$1000 split across 7 modules.
-- -----------------------------------------------------------------------
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    -- Global caps (apply across ALL modules regardless of module budgets)
    ('allocation_guard_config', 'global_evm_wallet_cap_usd', '700.0', 'float',
     'Maximum total USD committed across ALL modules sharing the EVM wallet '
     '(dex, arbitrage, ai, futures). Veto any new entry that would push the '
     'sum above this cap. 0 = disabled. Default 700.'),

    ('allocation_guard_config', 'global_solana_wallet_cap_usd', '700.0', 'float',
     'Maximum total USD committed across ALL modules sharing the Solana wallet '
     '(solana, sniper, copy_trading). 0 = disabled. Default 700.'),

    ('allocation_guard_config', 'global_total_cap_usd', '1000.0', 'float',
     'Hard cap on total USD committed across ALL modules and ALL wallets combined. '
     'Safety net above the per-wallet caps. 0 = disabled. Default 1000.'),

    -- Per-module budget defaults
    ('allocation_guard_config', 'budget_usd_sniper', '150.0', 'float',
     'Default USD budget for the sniper module. Overridden by orchestrator '
     'budget_usd if a recent non-superseded recommendation exists. 0 = unlimited.'),

    ('allocation_guard_config', 'budget_usd_arbitrage', '200.0', 'float',
     'Default USD budget for the arbitrage module. 0 = unlimited.'),

    ('allocation_guard_config', 'budget_usd_copy_trading', '150.0', 'float',
     'Default USD budget for the copy_trading module. 0 = unlimited.'),

    ('allocation_guard_config', 'budget_usd_futures', '200.0', 'float',
     'Default USD budget for the futures module. 0 = unlimited.'),

    ('allocation_guard_config', 'budget_usd_solana', '150.0', 'float',
     'Default USD budget for the solana module. 0 = unlimited.'),

    ('allocation_guard_config', 'budget_usd_dex', '100.0', 'float',
     'Default USD budget for the DEX module. 0 = unlimited.'),

    ('allocation_guard_config', 'budget_usd_ai', '100.0', 'float',
     'Default USD budget for the AI analysis module. 0 = unlimited.'),

    -- Guard behavior flags
    ('allocation_guard_config', 'enabled', 'true', 'bool',
     'Master enable/disable for the AllocationGuard. false = advisory logging '
     'only (veto decisions are logged but never block entries). Default true.'),

    ('allocation_guard_config', 'use_orchestrator_budget', 'true', 'bool',
     'When true, the guard reads the orchestrator''s latest budget_usd '
     'recommendation (if present and non-superseded) as the module budget, '
     'overriding the static budget_usd_<module> default. When false, always '
     'uses the static default. Default true.'),

    ('allocation_guard_config', 'orchestrator_budget_ttl_minutes', '120', 'int',
     'Maximum age of an orchestrator budget recommendation before it is '
     'treated as stale and the static default is used instead. Default 120 min.'),

    -- Wallet-group membership (JSON list per group, read by guard to know
    -- which modules share which wallet for aggregate-cap enforcement)
    ('allocation_guard_config', 'evm_wallet_modules',
     '["dex","arbitrage","ai","futures"]', 'json',
     'List of module names that share the primary EVM execution wallet. '
     'Used for global_evm_wallet_cap_usd enforcement.'),

    ('allocation_guard_config', 'solana_wallet_modules',
     '["solana","sniper","copy_trading"]', 'json',
     'List of module names that share the Solana execution wallet. '
     'Used for global_solana_wallet_cap_usd enforcement.')

ON CONFLICT (config_type, key) DO NOTHING;

-- -----------------------------------------------------------------------
-- C. Index on orchestrator_recommendations.budget_usd for fast guard reads
-- -----------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_orch_recs_budget_module
    ON orchestrator_recommendations(module, created_at DESC)
    WHERE budget_usd IS NOT NULL
      AND approved IS NULL
      AND superseded_at IS NULL;

-- down:
-- ALTER TABLE orchestrator_recommendations DROP COLUMN IF EXISTS budget_usd;
-- DELETE FROM config_settings WHERE config_type = 'allocation_guard_config';
-- DROP INDEX IF EXISTS idx_orch_recs_budget_module;
