-- Migration: Add RPC/API Pool Management Tables
-- Description: Create tables for managing RPC and API endpoints with health tracking
-- Created: 2025-12-20

-- ============================================================================
-- RPC/API Pool Table
-- Stores all RPC and API endpoints with health metrics and rate limit tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS rpc_api_pool (
    id SERIAL PRIMARY KEY,

    -- Endpoint identification
    endpoint_type VARCHAR(20) NOT NULL,  -- 'rpc' or 'api'
    provider_type VARCHAR(50) NOT NULL,  -- e.g., 'ETHEREUM_RPC', 'SOLANA_RPC', 'HELIUS_API', 'JUPITER_API'
    name VARCHAR(100) NOT NULL,          -- Human-readable name (e.g., 'Alchemy ETH Primary')
    url TEXT NOT NULL,                    -- The actual URL/endpoint
    api_key TEXT,                         -- Optional API key (encrypted)

    -- Health metrics
    status VARCHAR(20) NOT NULL DEFAULT 'active',  -- 'active', 'rate_limited', 'unhealthy', 'disabled'
    is_enabled BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 100,         -- Lower = higher priority (used for ordering)
    weight INTEGER DEFAULT 100,           -- For weighted load balancing (1-100)

    -- Rate limit tracking
    rate_limit_until TIMESTAMP WITH TIME ZONE,  -- When rate limit expires
    rate_limit_count INTEGER DEFAULT 0,   -- Number of times rate limited
    last_rate_limit_at TIMESTAMP WITH TIME ZONE,

    -- Performance metrics
    last_success_at TIMESTAMP WITH TIME ZONE,
    last_failure_at TIMESTAMP WITH TIME ZONE,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    avg_latency_ms FLOAT DEFAULT 0,       -- Rolling average latency

    -- Health check info
    last_health_check_at TIMESTAMP WITH TIME ZONE,
    health_score FLOAT DEFAULT 100,       -- 0-100 score based on success rate, latency
    consecutive_failures INTEGER DEFAULT 0,

    -- Metadata
    chain VARCHAR(50),                    -- For RPC: ethereum, bsc, solana, etc.
    supports_ws BOOLEAN DEFAULT FALSE,    -- Whether it supports WebSocket
    ws_url TEXT,                          -- WebSocket URL if different
    notes TEXT,
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    UNIQUE(provider_type, url)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_rpc_api_pool_provider_type ON rpc_api_pool(provider_type);
CREATE INDEX IF NOT EXISTS idx_rpc_api_pool_status ON rpc_api_pool(status);
CREATE INDEX IF NOT EXISTS idx_rpc_api_pool_enabled ON rpc_api_pool(is_enabled);
CREATE INDEX IF NOT EXISTS idx_rpc_api_pool_priority ON rpc_api_pool(priority);
CREATE INDEX IF NOT EXISTS idx_rpc_api_pool_chain ON rpc_api_pool(chain);
CREATE INDEX IF NOT EXISTS idx_rpc_api_pool_health_score ON rpc_api_pool(health_score DESC);
CREATE INDEX IF NOT EXISTS idx_rpc_api_pool_rate_limit ON rpc_api_pool(rate_limit_until) WHERE rate_limit_until IS NOT NULL;

-- ============================================================================
-- RPC/API Usage History
-- Tracks usage patterns for analytics and optimization
-- ============================================================================

CREATE TABLE IF NOT EXISTS rpc_api_usage_history (
    id SERIAL PRIMARY KEY,
    endpoint_id INTEGER REFERENCES rpc_api_pool(id) ON DELETE CASCADE,
    module_name VARCHAR(50),              -- Which module used this endpoint
    request_type VARCHAR(50),             -- Type of request made

    -- Request details
    success BOOLEAN NOT NULL,
    latency_ms INTEGER,
    error_type VARCHAR(100),              -- e.g., 'rate_limit', 'timeout', 'network_error'
    error_message TEXT,

    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for usage history
CREATE INDEX IF NOT EXISTS idx_rpc_usage_endpoint ON rpc_api_usage_history(endpoint_id);
CREATE INDEX IF NOT EXISTS idx_rpc_usage_module ON rpc_api_usage_history(module_name);
CREATE INDEX IF NOT EXISTS idx_rpc_usage_time ON rpc_api_usage_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rpc_usage_success ON rpc_api_usage_history(success);

-- Partition usage history by time for efficient queries (optional, for high-volume)
-- Convert to hypertable if TimescaleDB is available
DO $$
BEGIN
    -- Try to create hypertable, ignore if TimescaleDB not available
    PERFORM create_hypertable('rpc_api_usage_history', 'created_at',
                              chunk_time_interval => INTERVAL '1 day',
                              if_not_exists => TRUE);
EXCEPTION WHEN others THEN
    -- TimescaleDB extension not available, continue without hypertable
    NULL;
END $$;

-- ============================================================================
-- Provider Type Definitions
-- Reference table for valid provider types
-- ============================================================================

CREATE TABLE IF NOT EXISTS rpc_api_provider_types (
    id SERIAL PRIMARY KEY,
    provider_type VARCHAR(50) NOT NULL UNIQUE,
    endpoint_type VARCHAR(20) NOT NULL,   -- 'rpc', 'ws', or 'api'
    chain VARCHAR(50),                    -- Associated chain if applicable
    description TEXT,
    default_priority INTEGER DEFAULT 100,
    is_required BOOLEAN DEFAULT FALSE,    -- Whether at least one endpoint is required
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert provider type definitions
INSERT INTO rpc_api_provider_types (provider_type, endpoint_type, chain, description, default_priority, is_required) VALUES
-- EVM RPC Endpoints
('ETHEREUM_RPC', 'rpc', 'ethereum', 'Ethereum mainnet RPC endpoints', 100, TRUE),
('BSC_RPC', 'rpc', 'bsc', 'BNB Smart Chain RPC endpoints', 100, TRUE),
('POLYGON_RPC', 'rpc', 'polygon', 'Polygon mainnet RPC endpoints', 100, FALSE),
('ARBITRUM_RPC', 'rpc', 'arbitrum', 'Arbitrum One RPC endpoints', 100, FALSE),
('BASE_RPC', 'rpc', 'base', 'Base mainnet RPC endpoints', 100, FALSE),
('MONAD_RPC', 'rpc', 'monad', 'Monad mainnet RPC endpoints', 100, FALSE),
('PULSECHAIN_RPC', 'rpc', 'pulsechain', 'PulseChain RPC endpoints', 100, FALSE),
('FANTOM_RPC', 'rpc', 'fantom', 'Fantom Opera RPC endpoints', 100, FALSE),
('CRONOS_RPC', 'rpc', 'cronos', 'Cronos mainnet RPC endpoints', 100, FALSE),
('AVALANCHE_RPC', 'rpc', 'avalanche', 'Avalanche C-Chain RPC endpoints', 100, FALSE),
-- Solana RPC
('SOLANA_RPC', 'rpc', 'solana', 'Solana mainnet RPC endpoints', 100, TRUE),
-- WebSocket Endpoints
('ETHEREUM_WS', 'ws', 'ethereum', 'Ethereum mainnet WebSocket endpoints', 100, FALSE),
('BSC_WS', 'ws', 'bsc', 'BSC WebSocket endpoints', 100, FALSE),
('ARBITRUM_WS', 'ws', 'arbitrum', 'Arbitrum WebSocket endpoints', 100, FALSE),
('SOLANA_WS', 'ws', 'solana', 'Solana WebSocket endpoints', 100, FALSE),
-- API Endpoints
('GOPLUS_API', 'api', NULL, 'GoPlus Security API for token analysis', 100, FALSE),
('1INCH_API', 'api', NULL, '1inch DEX Aggregator API', 100, FALSE),
('HELIUS_API', 'api', 'solana', 'Helius Solana enhanced API', 100, FALSE),
('ETHERSCAN_API', 'api', 'ethereum', 'Etherscan blockchain explorer API', 100, FALSE),
('JUPITER_API', 'api', 'solana', 'Jupiter Aggregator API for Solana', 100, TRUE)
ON CONFLICT (provider_type) DO NOTHING;

-- ============================================================================
-- Triggers
-- ============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_rpc_api_pool_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_rpc_api_pool_updated_at ON rpc_api_pool;
CREATE TRIGGER update_rpc_api_pool_updated_at
    BEFORE UPDATE ON rpc_api_pool
    FOR EACH ROW
    EXECUTE FUNCTION update_rpc_api_pool_timestamp();

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE rpc_api_pool IS 'Stores RPC and API endpoints with health metrics and rate limit tracking';
COMMENT ON TABLE rpc_api_usage_history IS 'Tracks usage patterns for each endpoint for analytics';
COMMENT ON TABLE rpc_api_provider_types IS 'Reference table defining valid provider types';

COMMENT ON COLUMN rpc_api_pool.status IS 'Current status: active, rate_limited, unhealthy, disabled';
COMMENT ON COLUMN rpc_api_pool.priority IS 'Lower number = higher priority for selection';
COMMENT ON COLUMN rpc_api_pool.weight IS 'For weighted load balancing (1-100)';
COMMENT ON COLUMN rpc_api_pool.health_score IS 'Calculated score (0-100) based on success rate and latency';
COMMENT ON COLUMN rpc_api_pool.rate_limit_until IS 'Timestamp when rate limit expires, NULL if not limited';
