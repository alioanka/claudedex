-- Migration 002: Add Module Support
-- Phase 1: Modular Architecture System
-- This migration adds tables for module tracking, metrics, and positions
-- Safe for both fresh installations and existing databases

-- ============================================================================
-- MODULE TRACKING TABLE
-- ============================================================================
-- Stores configuration and state for each trading module
CREATE TABLE IF NOT EXISTS modules (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    module_type VARCHAR(50) NOT NULL,
    enabled BOOLEAN DEFAULT true,
    status VARCHAR(20) DEFAULT 'stopped',
    capital_allocation DECIMAL(18,8) DEFAULT 0,
    capital_used DECIMAL(18,8) DEFAULT 0,
    config JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_health_check TIMESTAMP,

    CONSTRAINT valid_module_type CHECK (module_type IN (
        'dex_trading',
        'futures_trading',
        'arbitrage',
        'liquidity_provision',
        'solana_strategies',
        'custom'
    )),

    CONSTRAINT valid_status CHECK (status IN (
        'disabled',
        'initializing',
        'running',
        'paused',
        'error',
        'stopping',
        'stopped'
    ))
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_modules_enabled ON modules(enabled);
CREATE INDEX IF NOT EXISTS idx_modules_status ON modules(status);
CREATE INDEX IF NOT EXISTS idx_modules_type ON modules(module_type);

-- Add updated_at trigger
CREATE OR REPLACE FUNCTION update_modules_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS modules_updated_at_trigger ON modules;
CREATE TRIGGER modules_updated_at_trigger
    BEFORE UPDATE ON modules
    FOR EACH ROW
    EXECUTE FUNCTION update_modules_updated_at();

-- ============================================================================
-- MODULE METRICS HISTORY TABLE
-- ============================================================================
-- Stores historical metrics snapshots for each module
CREATE TABLE IF NOT EXISTS module_metrics (
    id SERIAL PRIMARY KEY,
    module_name VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),

    -- Trade statistics
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4) DEFAULT 0,

    -- PnL metrics
    total_pnl DECIMAL(18,8) DEFAULT 0,
    unrealized_pnl DECIMAL(18,8) DEFAULT 0,
    realized_pnl DECIMAL(18,8) DEFAULT 0,

    -- Performance metrics
    profit_factor DECIMAL(10,4) DEFAULT 0,
    sharpe_ratio DECIMAL(10,4) DEFAULT 0,
    max_drawdown DECIMAL(10,4) DEFAULT 0,

    -- Position metrics
    active_positions INTEGER DEFAULT 0,
    capital_allocated DECIMAL(18,8) DEFAULT 0,
    capital_used DECIMAL(18,8) DEFAULT 0,

    -- Additional metrics as JSON
    metrics_json JSONB,

    -- Uptime and error tracking
    uptime_seconds BIGINT DEFAULT 0,
    errors_count INTEGER DEFAULT 0,

    CONSTRAINT fk_module_metrics_module
        FOREIGN KEY (module_name)
        REFERENCES modules(name)
        ON DELETE CASCADE
);

-- Create indexes for fast time-series queries
CREATE INDEX IF NOT EXISTS idx_module_metrics_module ON module_metrics(module_name);
CREATE INDEX IF NOT EXISTS idx_module_metrics_timestamp ON module_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_module_metrics_module_time ON module_metrics(module_name, timestamp DESC);

-- ============================================================================
-- MODULE POSITIONS TABLE
-- ============================================================================
-- Stores positions for each module (for persistence across restarts)
CREATE TABLE IF NOT EXISTS module_positions (
    id SERIAL PRIMARY KEY,
    module_name VARCHAR(100) NOT NULL,
    position_id VARCHAR(100) UNIQUE NOT NULL,

    -- Position details
    symbol VARCHAR(50) NOT NULL,
    chain VARCHAR(20),
    side VARCHAR(10) NOT NULL,  -- LONG, SHORT, NEUTRAL

    -- Pricing
    entry_price DECIMAL(18,8) NOT NULL,
    current_price DECIMAL(18,8),
    liquidation_price DECIMAL(18,8),

    -- Size and leverage
    quantity DECIMAL(18,8) NOT NULL,
    cost DECIMAL(18,8) NOT NULL,
    leverage DECIMAL(5,2) DEFAULT 1.0,

    -- Risk management
    stop_loss DECIMAL(18,8),
    take_profit JSONB,  -- Array of TP levels: [{"price": 100, "size": 0.25}, ...]
    trailing_stop_pct DECIMAL(5,4),

    -- PnL
    pnl DECIMAL(18,8) DEFAULT 0,
    pnl_percentage DECIMAL(10,4) DEFAULT 0,
    fees DECIMAL(18,8) DEFAULT 0,

    -- Timestamps
    entry_time TIMESTAMP DEFAULT NOW(),
    exit_time TIMESTAMP,
    last_update TIMESTAMP DEFAULT NOW(),

    -- Status and metadata
    status VARCHAR(20) DEFAULT 'open',
    strategy VARCHAR(50),
    metadata JSONB,

    -- Audit fields
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT fk_module_positions_module
        FOREIGN KEY (module_name)
        REFERENCES modules(name)
        ON DELETE CASCADE,

    CONSTRAINT valid_side CHECK (side IN ('LONG', 'SHORT', 'NEUTRAL')),
    CONSTRAINT valid_position_status CHECK (status IN ('open', 'closing', 'closed', 'liquidated', 'error'))
);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_module_positions_module ON module_positions(module_name);
CREATE INDEX IF NOT EXISTS idx_module_positions_status ON module_positions(status);
CREATE INDEX IF NOT EXISTS idx_module_positions_symbol ON module_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_module_positions_entry_time ON module_positions(entry_time DESC);

-- Add updated_at trigger
DROP TRIGGER IF EXISTS module_positions_updated_at_trigger ON module_positions;
CREATE TRIGGER module_positions_updated_at_trigger
    BEFORE UPDATE ON module_positions
    FOR EACH ROW
    EXECUTE FUNCTION update_modules_updated_at();

-- ============================================================================
-- HELPER VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Active modules with current metrics
CREATE OR REPLACE VIEW v_active_modules AS
SELECT
    m.name,
    m.module_type,
    m.enabled,
    m.status,
    m.capital_allocation,
    m.capital_used,
    (m.capital_allocation - m.capital_used) as capital_available,
    COUNT(DISTINCT p.id) as active_positions,
    COALESCE(SUM(p.cost), 0) as total_position_value,
    COALESCE(SUM(p.pnl), 0) as total_unrealized_pnl,
    m.last_health_check,
    m.updated_at
FROM modules m
LEFT JOIN module_positions p ON m.name = p.module_name AND p.status = 'open'
WHERE m.enabled = true
GROUP BY m.id, m.name, m.module_type, m.enabled, m.status,
         m.capital_allocation, m.capital_used, m.last_health_check, m.updated_at;

-- View: Module performance summary (last 24 hours)
CREATE OR REPLACE VIEW v_module_performance_24h AS
SELECT
    module_name,
    COUNT(*) as metrics_count,
    AVG(win_rate) as avg_win_rate,
    SUM(total_trades) as total_trades_24h,
    SUM(total_pnl) as total_pnl_24h,
    AVG(sharpe_ratio) as avg_sharpe_ratio,
    MAX(max_drawdown) as worst_drawdown,
    MAX(timestamp) as last_update
FROM module_metrics
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY module_name;

-- ============================================================================
-- INITIAL DATA (Optional - For fresh installations)
-- ============================================================================

-- Insert default DEX trading module configuration if it doesn't exist
INSERT INTO modules (name, module_type, enabled, status, capital_allocation)
VALUES ('dex_trading', 'dex_trading', false, 'stopped', 500.0)
ON CONFLICT (name) DO NOTHING;

-- Insert default Solana strategies module configuration if it doesn't exist
INSERT INTO modules (name, module_type, enabled, status, capital_allocation)
VALUES ('solana_strategies', 'solana_strategies', false, 'stopped', 400.0)
ON CONFLICT (name) DO NOTHING;

-- Insert default Futures trading module configuration if it doesn't exist
INSERT INTO modules (name, module_type, enabled, status, capital_allocation)
VALUES ('futures_trading', 'futures_trading', false, 'stopped', 300.0)
ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- VERIFICATION QUERIES (For testing after migration)
-- ============================================================================

-- Uncomment these to verify the migration:

-- Check table creation
-- SELECT table_name FROM information_schema.tables
-- WHERE table_name IN ('modules', 'module_metrics', 'module_positions');

-- Check indexes
-- SELECT indexname FROM pg_indexes
-- WHERE tablename IN ('modules', 'module_metrics', 'module_positions');

-- Check views
-- SELECT table_name FROM information_schema.views
-- WHERE table_name LIKE 'v_%module%';

-- Check initial data
-- SELECT name, module_type, enabled, status, capital_allocation FROM modules;

-- ============================================================================
-- ROLLBACK (If needed - run this to remove all module support)
-- ============================================================================

-- DROP VIEW IF EXISTS v_module_performance_24h CASCADE;
-- DROP VIEW IF EXISTS v_active_modules CASCADE;
-- DROP TABLE IF EXISTS module_positions CASCADE;
-- DROP TABLE IF EXISTS module_metrics CASCADE;
-- DROP TABLE IF EXISTS modules CASCADE;
-- DROP FUNCTION IF EXISTS update_modules_updated_at() CASCADE;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- Note: Migration tracking is handled by MigrationManager (schema_migrations table)
