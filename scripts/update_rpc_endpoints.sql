-- Script to update RPC endpoints in the rpc_api_pool table with free public endpoints
-- Run this with: docker exec -i claudedex-postgres-1 psql -U claudedex -d claudedex < scripts/update_rpc_endpoints.sql

-- First, disable all existing Ankr/Alchemy endpoints that require API keys
UPDATE rpc_api_pool
SET is_enabled = FALSE
WHERE url LIKE '%ankr.com%'
   OR url LIKE '%alchemy.com%'
   OR url LIKE '%infura.io%';

-- Delete old Ethereum endpoints and insert new ones
DELETE FROM rpc_api_pool WHERE provider_type = 'ETHEREUM_RPC';

INSERT INTO rpc_api_pool (endpoint_type, provider_type, name, url, status, is_enabled, priority, weight, chain) VALUES
('rpc', 'ETHEREUM_RPC', 'LlamaRPC ETH', 'https://eth.llamarpc.com', 'active', true, 10, 100, 'ethereum'),
('rpc', 'ETHEREUM_RPC', 'PublicNode ETH', 'https://ethereum.publicnode.com', 'active', true, 20, 100, 'ethereum'),
('rpc', 'ETHEREUM_RPC', '1RPC ETH', 'https://1rpc.io/eth', 'active', true, 30, 100, 'ethereum'),
('rpc', 'ETHEREUM_RPC', 'MEV Blocker', 'https://rpc.mevblocker.io', 'active', true, 40, 100, 'ethereum'),
('rpc', 'ETHEREUM_RPC', 'DRPC ETH', 'https://eth.drpc.org', 'active', true, 50, 100, 'ethereum');

-- Delete old Polygon endpoints and insert new ones
DELETE FROM rpc_api_pool WHERE provider_type = 'POLYGON_RPC';

INSERT INTO rpc_api_pool (endpoint_type, provider_type, name, url, status, is_enabled, priority, weight, chain) VALUES
('rpc', 'POLYGON_RPC', 'Polygon RPC', 'https://polygon-rpc.com', 'active', true, 10, 100, 'polygon'),
('rpc', 'POLYGON_RPC', 'LlamaRPC Polygon', 'https://polygon.llamarpc.com', 'active', true, 20, 100, 'polygon'),
('rpc', 'POLYGON_RPC', 'PublicNode Polygon', 'https://polygon.publicnode.com', 'active', true, 30, 100, 'polygon'),
('rpc', 'POLYGON_RPC', '1RPC Polygon', 'https://1rpc.io/matic', 'active', true, 40, 100, 'polygon'),
('rpc', 'POLYGON_RPC', 'DRPC Polygon', 'https://polygon.drpc.org', 'active', true, 50, 100, 'polygon');

-- Delete old Arbitrum endpoints and insert new ones
DELETE FROM rpc_api_pool WHERE provider_type = 'ARBITRUM_RPC';

INSERT INTO rpc_api_pool (endpoint_type, provider_type, name, url, status, is_enabled, priority, weight, chain) VALUES
('rpc', 'ARBITRUM_RPC', 'Arbitrum Official', 'https://arb1.arbitrum.io/rpc', 'active', true, 10, 100, 'arbitrum'),
('rpc', 'ARBITRUM_RPC', 'PublicNode Arbitrum', 'https://arbitrum-one.publicnode.com', 'active', true, 20, 100, 'arbitrum'),
('rpc', 'ARBITRUM_RPC', 'LlamaRPC Arbitrum', 'https://arbitrum.llamarpc.com', 'active', true, 30, 100, 'arbitrum'),
('rpc', 'ARBITRUM_RPC', '1RPC Arbitrum', 'https://1rpc.io/arb', 'active', true, 40, 100, 'arbitrum'),
('rpc', 'ARBITRUM_RPC', 'DRPC Arbitrum', 'https://arbitrum.drpc.org', 'active', true, 50, 100, 'arbitrum');

-- Delete old Base endpoints and insert new ones
DELETE FROM rpc_api_pool WHERE provider_type = 'BASE_RPC';

INSERT INTO rpc_api_pool (endpoint_type, provider_type, name, url, status, is_enabled, priority, weight, chain) VALUES
('rpc', 'BASE_RPC', 'Base Official', 'https://mainnet.base.org', 'active', true, 10, 100, 'base'),
('rpc', 'BASE_RPC', 'PublicNode Base', 'https://base.publicnode.com', 'active', true, 20, 100, 'base'),
('rpc', 'BASE_RPC', 'LlamaRPC Base', 'https://base.llamarpc.com', 'active', true, 30, 100, 'base'),
('rpc', 'BASE_RPC', '1RPC Base', 'https://1rpc.io/base', 'active', true, 40, 100, 'base'),
('rpc', 'BASE_RPC', 'DRPC Base', 'https://base.drpc.org', 'active', true, 50, 100, 'base');

-- Update BSC to use more reliable endpoints (keep existing Binance endpoints, add backups)
DELETE FROM rpc_api_pool WHERE provider_type = 'BSC_RPC' AND url LIKE '%ankr%';

INSERT INTO rpc_api_pool (endpoint_type, provider_type, name, url, status, is_enabled, priority, weight, chain)
SELECT 'rpc', 'BSC_RPC', 'BSC Dataseed 1', 'https://bsc-dataseed1.binance.org', 'active', true, 10, 100, 'bsc'
WHERE NOT EXISTS (SELECT 1 FROM rpc_api_pool WHERE url = 'https://bsc-dataseed1.binance.org');

INSERT INTO rpc_api_pool (endpoint_type, provider_type, name, url, status, is_enabled, priority, weight, chain)
SELECT 'rpc', 'BSC_RPC', 'BSC Dataseed 2', 'https://bsc-dataseed2.binance.org', 'active', true, 20, 100, 'bsc'
WHERE NOT EXISTS (SELECT 1 FROM rpc_api_pool WHERE url = 'https://bsc-dataseed2.binance.org');

INSERT INTO rpc_api_pool (endpoint_type, provider_type, name, url, status, is_enabled, priority, weight, chain)
SELECT 'rpc', 'BSC_RPC', 'PublicNode BSC', 'https://bsc.publicnode.com', 'active', true, 30, 100, 'bsc'
WHERE NOT EXISTS (SELECT 1 FROM rpc_api_pool WHERE url = 'https://bsc.publicnode.com');

INSERT INTO rpc_api_pool (endpoint_type, provider_type, name, url, status, is_enabled, priority, weight, chain)
SELECT 'rpc', 'BSC_RPC', 'DRPC BSC', 'https://bsc.drpc.org', 'active', true, 40, 100, 'bsc'
WHERE NOT EXISTS (SELECT 1 FROM rpc_api_pool WHERE url = 'https://bsc.drpc.org');

-- Update Avalanche endpoints
DELETE FROM rpc_api_pool WHERE provider_type = 'AVALANCHE_RPC';

INSERT INTO rpc_api_pool (endpoint_type, provider_type, name, url, status, is_enabled, priority, weight, chain) VALUES
('rpc', 'AVALANCHE_RPC', 'Avalanche Official', 'https://api.avax.network/ext/bc/C/rpc', 'active', true, 10, 100, 'avalanche'),
('rpc', 'AVALANCHE_RPC', 'PublicNode Avalanche', 'https://avalanche.publicnode.com', 'active', true, 20, 100, 'avalanche'),
('rpc', 'AVALANCHE_RPC', 'DRPC Avalanche', 'https://avalanche.drpc.org', 'active', true, 30, 100, 'avalanche'),
('rpc', 'AVALANCHE_RPC', '1RPC Avalanche', 'https://1rpc.io/avax/c', 'active', true, 40, 100, 'avalanche');

-- Update Monad endpoints (public RPCs if available)
DELETE FROM rpc_api_pool WHERE provider_type = 'MONAD_RPC';

INSERT INTO rpc_api_pool (endpoint_type, provider_type, name, url, status, is_enabled, priority, weight, chain) VALUES
('rpc', 'MONAD_RPC', 'PublicNode Monad', 'https://monad.publicnode.com', 'active', true, 10, 100, 'monad'),
('rpc', 'MONAD_RPC', 'Monad RPC', 'https://rpc.monad.xyz', 'active', true, 20, 100, 'monad');

-- Show summary
SELECT provider_type, COUNT(*) as endpoint_count, string_agg(name, ', ') as endpoints
FROM rpc_api_pool
WHERE is_enabled = TRUE
GROUP BY provider_type
ORDER BY provider_type;
