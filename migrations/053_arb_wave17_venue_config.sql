-- Migration 053: wave-17 arbitrage venue config (CORRECTED)
--
-- NOTE (Wave-17 fix): the original 053 inserted into a non-existent table
-- `bot_config`, which does not exist in this schema. The canonical config
-- table is `config_settings` (config_type, key, value, ...). The original
-- crashed the migrator with "relation bot_config does not exist", which the
-- entrypoint treats as a critical error -> the bot failed to start and
-- migration 054 never ran. This version uses config_settings.
--
-- IMPORTANT: arbitrage_engine.py reads these knobs from the hardcoded
-- CHAIN_CONFIGS dict via `self.chain_config.get(key, default)` -- the
-- code-level defaults (upper_spread_cap_bps=500, min_pool_tvl_usd=50000,
-- v3_quoter_enabled per-chain) are ALWAYS the effective values today. These
-- rows are seeded for visibility/consistency with config_settings + the
-- dashboard, and as the anchor for a future DB-override wiring. Seeding them
-- changes NO behavior on its own.
--
-- Idempotent: ON CONFLICT (config_type, key) DO NOTHING.

INSERT INTO config_settings (config_type, key, value, description)
VALUES
    ('arbitrage_config', 'upper_spread_cap_bps_ethereum', '500', 'Max believable BUY-side cross-DEX spread (bps) on Ethereum before classifying as thin_pool_artifact'),
    ('arbitrage_config', 'upper_spread_cap_bps_arbitrum', '500', 'Max believable BUY-side cross-DEX spread (bps) on Arbitrum before classifying as thin_pool_artifact'),
    ('arbitrage_config', 'upper_spread_cap_bps_base',     '500', 'Max believable BUY-side cross-DEX spread (bps) on Base before classifying as thin_pool_artifact'),
    ('arbitrage_config', 'min_pool_tvl_usd_ethereum', '50000', 'Minimum WETH-leg pool TVL (USD) to quote a DEX on Ethereum'),
    ('arbitrage_config', 'min_pool_tvl_usd_arbitrum', '50000', 'Minimum WETH-leg pool TVL (USD) to quote a DEX on Arbitrum'),
    ('arbitrage_config', 'min_pool_tvl_usd_base',     '50000', 'Minimum WETH-leg pool TVL (USD) to quote a DEX on Base'),
    ('arbitrage_config', 'arb_v3_quoter_enabled_ethereum', 'true',  'Enable Uniswap V3 QuoterV2 price discovery on Ethereum (proof-of-concept)'),
    ('arbitrage_config', 'arb_v3_quoter_enabled_arbitrum', 'false', 'Enable Uniswap V3 QuoterV2 on Arbitrum (pending wave-18 execution leg)'),
    ('arbitrage_config', 'arb_v3_quoter_enabled_base',     'false', 'Enable Uniswap V3 QuoterV2 on Base (pending wave-18 execution leg)')
ON CONFLICT (config_type, key) DO NOTHING;
