-- Migration 053: wave-17 arbitrage venue config
-- Seeds three new per-chain DB-configurable keys introduced in wave-17:
--   upper_spread_cap_bps  — BUY-side spread sanity ceiling (default 500 bps)
--   min_pool_tvl_usd      — per-DEX pool reserve floor before quoting (default $50k)
--   arb_v3_quoter_enabled — Uniswap V3 QuoterV2 flag (True for ETH, False for L2s)
--
-- These complement the existing min_price_spread_bps (wave-16) and are read via
-- chain_config.get(key, default) in arbitrage_engine.py so the code-level default
-- is always the safe fallback even if this migration is not yet applied.

-- Upper-spread sanity cap: any cross-DEX BUY divergence above 500 bps is a
-- thin-pool / wrong-ABI artifact.  Operators can raise this per chain if needed.
INSERT INTO bot_config (module, key, value, description)
VALUES
    ('arbitrage', 'upper_spread_cap_bps_ethereum', '500', 'Max believable BUY-side cross-DEX spread (bps) on Ethereum before classifying as thin_pool_artifact'),
    ('arbitrage', 'upper_spread_cap_bps_arbitrum', '500', 'Max believable BUY-side cross-DEX spread (bps) on Arbitrum before classifying as thin_pool_artifact'),
    ('arbitrage', 'upper_spread_cap_bps_base',     '500', 'Max believable BUY-side cross-DEX spread (bps) on Base before classifying as thin_pool_artifact')
ON CONFLICT (module, key) DO NOTHING;

-- Pool-reserve floor: skip a DEX/pair combination when the on-chain WETH-leg TVL
-- is below this USD threshold.  Lower values allow more small pools (more noise);
-- higher values miss real opportunities on thin-but-real pools.  $50k is
-- conservative for mainnet; may need lowering on Base if BaseSwap TVL drops.
INSERT INTO bot_config (module, key, value, description)
VALUES
    ('arbitrage', 'min_pool_tvl_usd_ethereum', '50000', 'Minimum WETH-leg pool TVL (USD) to quote a DEX on Ethereum'),
    ('arbitrage', 'min_pool_tvl_usd_arbitrum', '50000', 'Minimum WETH-leg pool TVL (USD) to quote a DEX on Arbitrum'),
    ('arbitrage', 'min_pool_tvl_usd_base',     '50000', 'Minimum WETH-leg pool TVL (USD) to quote a DEX on Base')
ON CONFLICT (module, key) DO NOTHING;

-- V3 Quoter flag: enabled for Ethereum (dominant liquidity venue); disabled for
-- Arbitrum/Base until the V3 execution leg is added in wave-18.
INSERT INTO bot_config (module, key, value, description)
VALUES
    ('arbitrage', 'arb_v3_quoter_enabled_ethereum', 'true',  'Enable Uniswap V3 QuoterV2 price discovery on Ethereum (proof-of-concept)'),
    ('arbitrage', 'arb_v3_quoter_enabled_arbitrum', 'false', 'Enable Uniswap V3 QuoterV2 on Arbitrum (pending wave-18 execution leg)'),
    ('arbitrage', 'arb_v3_quoter_enabled_base',     'false', 'Enable Uniswap V3 QuoterV2 on Base (pending wave-18 execution leg)')
ON CONFLICT (module, key) DO NOTHING;
