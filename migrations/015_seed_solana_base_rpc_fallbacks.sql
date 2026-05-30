-- Migration: 015_seed_solana_base_rpc_fallbacks
-- Description: Seed free/public Solana RPC endpoints as a fallback tier and
--              add a canonical public BASE RPC endpoint.
--
-- Context (Wave-13 known issues):
--   1. SOLANA_RPC pool only contains Helius paid endpoints. When all three are
--      rate-limited simultaneously the pool starves callers. Adding public
--      endpoints at priority=200 (lower quality but always reachable) ensures
--      at least one endpoint remains usable during Helius outage or rate-limit
--      storms. Priority 200 keeps them behind Helius (priority ≈100-150) so
--      they are never selected when Helius is healthy.
--   2. BASE_RPC and FANTOM_RPC have no seeded rows, causing
--      "No available endpoints for: BASE_RPC" at startup.
--
-- All inserts are guarded by ON CONFLICT (provider_type, url) DO NOTHING so
-- this migration is safe to apply on a cluster that already has some of these
-- rows (e.g. from manual seeding or a previous .env seed-on-startup).
--
-- Rollback / DOWN:
--   DELETE FROM rpc_api_pool
--   WHERE provider_type IN ('SOLANA_RPC', 'BASE_RPC', 'FANTOM_RPC')
--     AND name LIKE 'Public%';
--
-- Date: 2026-05-30

-- ============================================================================
-- Solana fallback tier  (priority=200, weight=30 — used only when all
-- priority≤150 Helius endpoints are unavailable)
-- ============================================================================

INSERT INTO rpc_api_pool
    (endpoint_type, provider_type, name, url, status, is_enabled,
     priority, weight, chain, notes)
VALUES
    -- Solana Foundation public mainnet-beta (anonymous, no key, hard rate-limited
    -- ~100 RPS but always available as last resort)
    ('rpc', 'SOLANA_RPC', 'Public Solana Mainnet-Beta',
     'https://api.mainnet-beta.solana.com',
     'active', TRUE, 200, 30, 'solana',
     'Free public endpoint — fallback only; expect 429s under load'),

    -- GenesysGo / RPC Pool public node (permissionless)
    ('rpc', 'SOLANA_RPC', 'Public RPCPool',
     'https://rpc.ankr.com/solana',
     'active', TRUE, 200, 30, 'solana',
     'Ankr public Solana endpoint — fallback tier'),

    -- Triton One public trial endpoint (higher limits than foundation)
    ('rpc', 'SOLANA_RPC', 'Public Triton One',
     'https://free.rpcpool.com',
     'active', TRUE, 210, 20, 'solana',
     'Triton public trial — low priority fallback')

ON CONFLICT (provider_type, url) DO NOTHING;

-- ============================================================================
-- BASE RPC fallbacks (Coinbase maintains public endpoints; Ankr free tier)
-- ============================================================================

INSERT INTO rpc_api_pool
    (endpoint_type, provider_type, name, url, status, is_enabled,
     priority, weight, chain, notes)
VALUES
    -- Coinbase / Base official public endpoint
    ('rpc', 'BASE_RPC', 'Public Base Mainnet (Coinbase)',
     'https://mainnet.base.org',
     'active', TRUE, 100, 100, 'base',
     'Official Coinbase-operated Base RPC — primary public endpoint'),

    -- Ankr public Base endpoint
    ('rpc', 'BASE_RPC', 'Public Base (Ankr)',
     'https://rpc.ankr.com/base',
     'active', TRUE, 120, 60, 'base',
     'Ankr public Base endpoint — secondary'),

    -- Blast API public Base endpoint
    ('rpc', 'BASE_RPC', 'Public Base (Blast)',
     'https://base-mainnet.public.blastapi.io',
     'active', TRUE, 130, 40, 'base',
     'Blast API public Base endpoint — tertiary')

ON CONFLICT (provider_type, url) DO NOTHING;

-- ============================================================================
-- FANTOM RPC fallbacks (honeypot_checker was failing to connect)
-- ============================================================================

INSERT INTO rpc_api_pool
    (endpoint_type, provider_type, name, url, status, is_enabled,
     priority, weight, chain, notes)
VALUES
    -- Fantom Foundation public RPC
    ('rpc', 'FANTOM_RPC', 'Public Fantom Opera (Foundation)',
     'https://rpc.ftm.tools',
     'active', TRUE, 100, 100, 'fantom',
     'Fantom Foundation public RPC'),

    -- Ankr public Fantom endpoint
    ('rpc', 'FANTOM_RPC', 'Public Fantom (Ankr)',
     'https://rpc.ankr.com/fantom',
     'active', TRUE, 110, 80, 'fantom',
     'Ankr public Fantom endpoint')

ON CONFLICT (provider_type, url) DO NOTHING;

-- ============================================================================
-- Ensure provider_types catalogue contains BASE_RPC and FANTOM_RPC
-- (migration 011 already defines them, but guard in case schema diverged)
-- ============================================================================

INSERT INTO rpc_api_provider_types
    (provider_type, endpoint_type, chain, description, default_priority, is_required)
VALUES
    ('BASE_RPC',   'rpc', 'base',   'Base mainnet RPC endpoints',  100, FALSE),
    ('FANTOM_RPC', 'rpc', 'fantom', 'Fantom Opera RPC endpoints',  100, FALSE)
ON CONFLICT (provider_type) DO NOTHING;
