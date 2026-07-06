-- Migration 144: BIRDEYE_API provider type (Wave-F5 multi-key rotation)
--
-- pool_engine now supports numbered BIRDEYE_API_KEY[_2..9] env/secret keys,
-- each registered as its own rotating endpoint under provider type
-- BIRDEYE_API. Endpoints work without this row (rpc_api_pool has no FK),
-- but the dashboard /settings/rpc-api provider dropdown is populated from
-- rpc_api_provider_types (monitoring/rpc_pool_routes.py get_provider_types),
-- so the operator cannot add Birdeye keys via the UI without it.
--
-- No other schema change is needed for multi-key rotation: rpc_api_pool
-- already supports multiple rows per provider_type, and slot>=2 keys that
-- share a base URL are stored with a ?key_slot=N tag to satisfy the
-- existing UNIQUE(provider_type, url) constraint.
--
-- Idempotent: ON CONFLICT DO NOTHING.

INSERT INTO rpc_api_provider_types
    (provider_type, endpoint_type, chain, description, default_priority, is_required)
VALUES
    ('BIRDEYE_API', 'api', 'solana',
     'Birdeye market-data API (copy-trading wallet discovery; Solana-only on free tier)',
     100, FALSE)
ON CONFLICT (provider_type) DO NOTHING;

-- down:
-- Reversible: remove the reference row and any endpoints registered under
-- it (usage-history rows cascade via the existing FK ON DELETE CASCADE).
-- DELETE FROM rpc_api_pool WHERE provider_type = 'BIRDEYE_API';
-- DELETE FROM rpc_api_provider_types WHERE provider_type = 'BIRDEYE_API';
