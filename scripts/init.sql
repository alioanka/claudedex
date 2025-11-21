-- DexScreener Trading Bot Database Schema
-- PostgreSQL with TimescaleDB Extensions

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS analytics;

-- =====================================================
-- TRADING SCHEMA TABLES
-- =====================================================

-- Token Information
CREATE TABLE IF NOT EXISTS trading.tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    address VARCHAR(42) NOT NULL UNIQUE,
    chain VARCHAR(20) NOT NULL,
    symbol VARCHAR(20),
    name VARCHAR(100),
    decimals INTEGER,
    total_supply NUMERIC,
    verified BOOLEAN DEFAULT FALSE,
    honeypot_checked BOOLEAN DEFAULT FALSE,
    honeypot_result JSONB,
    contract_analysis JSONB,
    metadata JSONB,
    first_seen_at TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_tokens_address ON trading.tokens(address);
CREATE INDEX idx_tokens_chain ON trading.tokens(chain);
CREATE INDEX idx_tokens_symbol ON trading.tokens(symbol);

-- Trading Pairs
CREATE TABLE IF NOT EXISTS trading.pairs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pair_address VARCHAR(42) NOT NULL UNIQUE,
    token_address VARCHAR(42) NOT NULL REFERENCES trading.tokens(address),
    chain VARCHAR(20) NOT NULL,
    dex VARCHAR(50) NOT NULL,
    base_token VARCHAR(42),
    quote_token VARCHAR(42),
    liquidity_usd NUMERIC,
    volume_24h NUMERIC,
    created_at TIMESTAMP DEFAULT NOW(),
    pair_created_at TIMESTAMP,
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_pairs_token ON trading.pairs(token_address);
CREATE INDEX idx_pairs_chain_dex ON trading.pairs(chain, dex);

-- Price Data (Time-series)
CREATE TABLE IF NOT EXISTS trading.price_data (
    time TIMESTAMPTZ NOT NULL,
    pair_address VARCHAR(42) NOT NULL,
    price_usd NUMERIC NOT NULL,
    price_native NUMERIC,
    volume NUMERIC,
    liquidity NUMERIC,
    trades_count INTEGER,
    buyers_count INTEGER,
    sellers_count INTEGER
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('trading.price_data', 'time', if_not_exists => TRUE);
CREATE INDEX idx_price_pair_time ON trading.price_data(pair_address, time DESC);

-- Trade Executions
CREATE TABLE IF NOT EXISTS trading.trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_address VARCHAR(42) NOT NULL,
    pair_address VARCHAR(42),
    side VARCHAR(10) NOT NULL, -- 'buy' or 'sell'
    strategy VARCHAR(50),
    amount_in NUMERIC NOT NULL,
    amount_out NUMERIC,
    price_execution NUMERIC,
    gas_used NUMERIC,
    gas_price NUMERIC,
    slippage_actual NUMERIC,
    tx_hash VARCHAR(66) UNIQUE,
    status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    executed_at TIMESTAMP,
    confirmed_at TIMESTAMP
);

CREATE INDEX idx_trades_token ON trading.trades(token_address);
CREATE INDEX idx_trades_status ON trading.trades(status);
CREATE INDEX idx_trades_created ON trading.trades(created_at DESC);

-- Positions
CREATE TABLE IF NOT EXISTS trading.positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_address VARCHAR(42) NOT NULL,
    entry_price NUMERIC NOT NULL,
    entry_amount NUMERIC NOT NULL,
    current_price NUMERIC,
    current_value NUMERIC,
    realized_pnl NUMERIC DEFAULT 0,
    unrealized_pnl NUMERIC DEFAULT 0,
    stop_loss NUMERIC,
    take_profit JSONB, -- Array of take profit levels
    status VARCHAR(20) DEFAULT 'open', -- open, closed, partial
    close_reason VARCHAR(50),
    risk_score NUMERIC,
    ml_confidence NUMERIC,
    metadata JSONB,
    opened_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP,
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_positions_status ON trading.positions(status);
CREATE INDEX idx_positions_token ON trading.positions(token_address);

-- =====================================================
-- ML SCHEMA TABLES
-- =====================================================

-- ML Predictions
CREATE TABLE IF NOT EXISTS ml.predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_address VARCHAR(42) NOT NULL,
    model_version VARCHAR(50),
    pump_probability NUMERIC,
    rug_probability NUMERIC,
    expected_return NUMERIC,
    confidence_score NUMERIC,
    time_to_pump_hours NUMERIC,
    risk_adjusted_score NUMERIC,
    model_agreements JSONB,
    feature_importance JSONB,
    features JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_predictions_token ON ml.predictions(token_address);
CREATE INDEX idx_predictions_created ON ml.predictions(created_at DESC);

-- Training Data
CREATE TABLE IF NOT EXISTS ml.training_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_address VARCHAR(42),
    features JSONB NOT NULL,
    label_pump BOOLEAN,
    label_rug BOOLEAN,
    actual_return NUMERIC,
    dataset_type VARCHAR(20), -- 'train', 'validation', 'test'
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Model Performance
CREATE TABLE IF NOT EXISTS ml.model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    accuracy NUMERIC,
    precision NUMERIC,
    recall NUMERIC,
    f1_score NUMERIC,
    auc_roc NUMERIC,
    confusion_matrix JSONB,
    feature_importance JSONB,
    training_duration_seconds INTEGER,
    test_results JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- ANALYTICS SCHEMA TABLES
-- =====================================================

-- Developer Analysis
CREATE TABLE IF NOT EXISTS analytics.developers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wallet_address VARCHAR(42) UNIQUE NOT NULL,
    total_projects INTEGER DEFAULT 0,
    successful_projects INTEGER DEFAULT 0,
    failed_projects INTEGER DEFAULT 0,
    rug_pulls INTEGER DEFAULT 0,
    average_project_lifespan_days NUMERIC,
    total_volume_generated NUMERIC,
    reputation_score NUMERIC,
    is_blacklisted BOOLEAN DEFAULT FALSE,
    blacklist_reason TEXT,
    metadata JSONB,
    first_seen TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_developers_wallet ON analytics.developers(wallet_address);
CREATE INDEX idx_developers_blacklist ON analytics.developers(is_blacklisted);

-- Token Patterns
CREATE TABLE IF NOT EXISTS analytics.token_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_name VARCHAR(100),
    token_address VARCHAR(42),
    pattern_type VARCHAR(50), -- 'pump', 'rug', 'honeypot', etc
    pattern_data JSONB,
    confidence NUMERIC,
    outcome VARCHAR(50),
    profit_loss NUMERIC,
    detected_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_patterns_token ON analytics.token_patterns(token_address);
CREATE INDEX idx_patterns_type ON analytics.token_patterns(pattern_type);

-- Performance Metrics (Time-series)
CREATE TABLE IF NOT EXISTS analytics.performance_metrics (
    time TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC NOT NULL,
    metadata JSONB
);

SELECT create_hypertable('analytics.performance_metrics', 'time', if_not_exists => TRUE);

-- Blacklists
CREATE TABLE IF NOT EXISTS analytics.blacklists (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    list_type VARCHAR(20) NOT NULL, -- 'token', 'developer', 'contract'
    address VARCHAR(42) NOT NULL,
    reason TEXT,
    severity VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
    source VARCHAR(100),
    evidence JSONB,
    added_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    UNIQUE(list_type, address)
);

CREATE INDEX idx_blacklist_type_address ON analytics.blacklists(list_type, address);

-- =====================================================
-- AGGREGATED VIEWS
-- =====================================================

-- Daily Trading Summary
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.daily_summary AS
SELECT 
    DATE(created_at) as trading_date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN side = 'buy' THEN 1 ELSE 0 END) as buy_trades,
    SUM(CASE WHEN side = 'sell' THEN 1 ELSE 0 END) as sell_trades,
    SUM(amount_in) as total_volume,
    AVG(gas_used * gas_price) as avg_gas_cost,
    AVG(slippage_actual) as avg_slippage
FROM trading.trades
WHERE status = 'completed'
GROUP BY DATE(created_at);

CREATE UNIQUE INDEX ON analytics.daily_summary(trading_date);

-- Token Performance View
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.token_performance AS
SELECT 
    t.address,
    t.symbol,
    t.name,
    COUNT(DISTINCT tr.id) as trade_count,
    AVG(p.realized_pnl) as avg_pnl,
    SUM(p.realized_pnl) as total_pnl,
    MAX(p.realized_pnl) as max_profit,
    MIN(p.realized_pnl) as max_loss,
    AVG(p.ml_confidence) as avg_ml_confidence
FROM trading.tokens t
LEFT JOIN trading.trades tr ON t.address = tr.token_address
LEFT JOIN trading.positions p ON t.address = p.token_address
GROUP BY t.address, t.symbol, t.name;

CREATE UNIQUE INDEX ON analytics.token_performance(address);

-- =====================================================
-- CONSECUTIVE LOSSES AUTO-RESET TABLES
-- =====================================================

-- Add columns to positions table for blocking state
ALTER TABLE trading.positions 
ADD COLUMN IF NOT EXISTS consecutive_losses_blocked_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS consecutive_losses_block_count INTEGER DEFAULT 0;

-- Create position manager state table (singleton)
CREATE TABLE IF NOT EXISTS trading.position_manager_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    consecutive_losses INTEGER DEFAULT 0,
    consecutive_losses_blocked_at TIMESTAMP,
    consecutive_losses_block_count INTEGER DEFAULT 0,
    last_reset_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT single_row CHECK (id = 1)
);

-- Initialize state
INSERT INTO trading.position_manager_state (id, consecutive_losses, last_reset_at)
VALUES (1, 0, NOW())
ON CONFLICT (id) DO NOTHING;

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_positions_blocked_at 
ON trading.positions(consecutive_losses_blocked_at)
WHERE consecutive_losses_blocked_at IS NOT NULL;

COMMENT ON TABLE trading.position_manager_state IS 
'Stores position manager state including consecutive losses tracking and auto-reset';

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update timestamp to tables
CREATE TRIGGER update_tokens_timestamp BEFORE UPDATE ON trading.tokens
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_pairs_timestamp BEFORE UPDATE ON trading.pairs
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_positions_timestamp BEFORE UPDATE ON trading.positions
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

-- Function to calculate position P&L
CREATE OR REPLACE FUNCTION calculate_position_pnl()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.current_price IS NOT NULL THEN
        NEW.unrealized_pnl = (NEW.current_price - NEW.entry_price) * NEW.entry_amount;
        NEW.current_value = NEW.current_price * NEW.entry_amount;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER calculate_pnl_trigger BEFORE UPDATE ON trading.positions
    FOR EACH ROW EXECUTE FUNCTION calculate_position_pnl();

-- =====================================================
-- CONTINUOUS AGGREGATES (TimescaleDB)
-- =====================================================

-- Hourly price aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS trading.price_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS hour,
    pair_address,
    AVG(price_usd) as avg_price,
    MAX(price_usd) as high_price,
    MIN(price_usd) as low_price,
    FIRST(price_usd, time) as open_price,
    LAST(price_usd, time) as close_price,
    SUM(volume) as total_volume,
    AVG(liquidity) as avg_liquidity,
    SUM(trades_count) as total_trades
FROM trading.price_data
GROUP BY hour, pair_address
WITH NO DATA;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('trading.price_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- =====================================================
-- PERMISSIONS
-- =====================================================

-- Create read-only user for dashboard
CREATE ROLE dashboard_user WITH LOGIN PASSWORD 'dashboard_password';
GRANT USAGE ON SCHEMA trading, ml, analytics TO dashboard_user;
GRANT SELECT ON ALL TABLES IN SCHEMA trading, ml, analytics TO dashboard_user;

-- Create bot user with full permissions
CREATE ROLE bot_user WITH LOGIN PASSWORD 'bot_password';
GRANT ALL PRIVILEGES ON SCHEMA trading, ml, analytics TO bot_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading, ml, analytics TO bot_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading, ml, analytics TO bot_user;

-- =====================================================
-- INITIAL DATA
-- =====================================================

-- Insert default blacklist sources
INSERT INTO analytics.blacklists (list_type, address, reason, severity, source)
VALUES 
    ('token', '0x0000000000000000000000000000000000000000', 'Null address', 'critical', 'system'),
    ('developer', '0x0000000000000000000000000000000000000000', 'Null address', 'critical', 'system')
ON CONFLICT DO NOTHING;

-- =====================================================
-- MAINTENANCE
-- =====================================================

-- Create function for data retention
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Delete old price data (keep 90 days)
    DELETE FROM trading.price_data WHERE time < NOW() - INTERVAL '90 days';
    
    -- Delete old predictions (keep 30 days)
    DELETE FROM ml.predictions WHERE created_at < NOW() - INTERVAL '30 days';
    
    -- Archive closed positions older than 1 year
    -- (In production, you'd move these to an archive table)
    DELETE FROM trading.positions 
    WHERE status = 'closed' AND closed_at < NOW() - INTERVAL '365 days';
    
    -- Refresh materialized views
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.daily_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.token_performance;
END;
$$ LANGUAGE plpgsql;

-- Schedule maintenance (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-old-data', '0 2 * * *', 'SELECT cleanup_old_data();');