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

-- Solana Trades (for Solana module persistence)
CREATE TABLE IF NOT EXISTS solana_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id VARCHAR(50),
    token_symbol VARCHAR(20) NOT NULL,
    token_mint VARCHAR(64) NOT NULL,
    strategy VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL DEFAULT 'long',
    entry_price NUMERIC NOT NULL,
    exit_price NUMERIC NOT NULL,
    amount_sol NUMERIC NOT NULL,
    amount_tokens NUMERIC,
    pnl_sol NUMERIC NOT NULL,
    pnl_usd NUMERIC,
    pnl_pct NUMERIC NOT NULL,
    fees_sol NUMERIC NOT NULL DEFAULT 0,
    exit_reason VARCHAR(50),
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP NOT NULL,
    duration_seconds INT NOT NULL DEFAULT 0,
    is_simulated BOOLEAN NOT NULL DEFAULT TRUE,
    sol_price_usd NUMERIC,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_solana_trades_token ON solana_trades(token_symbol);
CREATE INDEX IF NOT EXISTS idx_solana_trades_strategy ON solana_trades(strategy);
CREATE INDEX IF NOT EXISTS idx_solana_trades_exit_time ON solana_trades(exit_time DESC);
CREATE INDEX IF NOT EXISTS idx_solana_trades_simulated ON solana_trades(is_simulated);

-- Solana Positions (for state recovery on restart)
CREATE TABLE IF NOT EXISTS solana_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    position_id VARCHAR(50) UNIQUE NOT NULL,
    token_symbol VARCHAR(20) NOT NULL,
    token_mint VARCHAR(64) NOT NULL,
    strategy VARCHAR(20) NOT NULL,
    entry_price NUMERIC NOT NULL,
    current_price NUMERIC,
    amount_sol NUMERIC NOT NULL,
    amount_tokens NUMERIC,
    stop_loss_pct NUMERIC,
    take_profit_pct NUMERIC,
    unrealized_pnl_sol NUMERIC DEFAULT 0,
    unrealized_pnl_pct NUMERIC DEFAULT 0,
    is_simulated BOOLEAN NOT NULL DEFAULT TRUE,
    opened_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_solana_positions_token ON solana_positions(token_mint);

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
-- SELECT cron.schedule('cleanup-old-data', '0 2 * * *', 'SELECT cleanup_old_data();');-- Migration: Add Authentication Tables
-- Creates users, sessions, and audit_logs tables for secure authentication

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'operator', 'viewer')),
    email VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    require_2fa BOOLEAN DEFAULT FALSE,
    totp_secret VARCHAR(32),
    failed_login_attempts INTEGER DEFAULT 0,
    last_login TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ip_address VARCHAR(45) NOT NULL,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    last_activity TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_sessions_is_active ON sessions(is_active);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    username VARCHAR(50) NOT NULL,
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255),
    old_value TEXT,
    new_value TEXT,
    ip_address VARCHAR(45) NOT NULL,
    user_agent TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);

-- Create default admin user (password: admin123 - CHANGE IMMEDIATELY!)
-- Password hash for 'admin123' - bcrypt with cost factor 12
INSERT INTO users (username, password_hash, role, email, is_active, require_2fa)
VALUES ('admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/Lfw99hhm1qJYT8sFm', 'admin', NULL, TRUE, FALSE)
ON CONFLICT (username) DO NOTHING;

-- Clean up expired sessions (run periodically)
-- DELETE FROM sessions WHERE expires_at < NOW();

-- View for active sessions
CREATE OR REPLACE VIEW active_sessions AS
SELECT
    s.session_id,
    s.user_id,
    u.username,
    u.role,
    s.ip_address,
    s.created_at,
    s.last_activity,
    s.expires_at
FROM sessions s
JOIN users u ON s.user_id = u.id
WHERE s.is_active = TRUE
    AND s.expires_at > NOW()
ORDER BY s.last_activity DESC;

-- View for recent audit logs
CREATE OR REPLACE VIEW recent_audit_logs AS
SELECT
    a.id,
    a.user_id,
    a.username,
    a.action,
    a.resource_type,
    a.resource_id,
    a.timestamp,
    a.success,
    a.ip_address
FROM audit_logs a
ORDER BY a.timestamp DESC
LIMIT 100;

COMMENT ON TABLE users IS 'User accounts with authentication credentials';
COMMENT ON TABLE sessions IS 'Active user sessions with expiration tracking';
COMMENT ON TABLE audit_logs IS 'Audit trail for security and compliance';
-- Migration: Add configuration management tables
-- Description: Create tables for centralized configuration storage with encryption support
-- Created: 2025-11-21

-- Configuration Settings Table
-- Stores all non-sensitive configuration values
CREATE TABLE IF NOT EXISTS config_settings (
    id SERIAL PRIMARY KEY,
    config_type VARCHAR(50) NOT NULL,
    key VARCHAR(100) NOT NULL,
    value TEXT NOT NULL,
    value_type VARCHAR(20) NOT NULL DEFAULT 'string', -- string, int, float, bool, json
    description TEXT,
    is_editable BOOLEAN DEFAULT TRUE,
    requires_restart BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by INTEGER REFERENCES users(id),
    UNIQUE(config_type, key)
);

CREATE INDEX IF NOT EXISTS idx_config_settings_type ON config_settings(config_type);
CREATE INDEX IF NOT EXISTS idx_config_settings_key ON config_settings(key);
CREATE INDEX IF NOT EXISTS idx_config_settings_updated_at ON config_settings(updated_at);

-- Sensitive Configuration Table
-- Stores encrypted sensitive data (API keys, private keys, passwords)
CREATE TABLE IF NOT EXISTS config_sensitive (
    id SERIAL PRIMARY KEY,
    key VARCHAR(100) NOT NULL UNIQUE,
    encrypted_value TEXT NOT NULL,
    encryption_method VARCHAR(50) NOT NULL DEFAULT 'AES-256-GCM',
    description TEXT,
    last_rotated TIMESTAMP WITH TIME ZONE,
    rotation_interval_days INTEGER DEFAULT 30,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by INTEGER REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_config_sensitive_key ON config_sensitive(key);
CREATE INDEX IF NOT EXISTS idx_config_sensitive_active ON config_sensitive(is_active);
CREATE INDEX IF NOT EXISTS idx_config_sensitive_rotation ON config_sensitive(last_rotated) WHERE is_active = TRUE;

-- Configuration Change History
-- Tracks all configuration changes for audit
CREATE TABLE IF NOT EXISTS config_history (
    id SERIAL PRIMARY KEY,
    config_type VARCHAR(50) NOT NULL,
    key VARCHAR(100) NOT NULL,
    old_value TEXT,
    new_value TEXT,
    change_source VARCHAR(50) NOT NULL, -- 'file', 'database', 'environment', 'api', 'user'
    changed_by INTEGER REFERENCES users(id),
    changed_by_username VARCHAR(255),
    reason TEXT,
    ip_address VARCHAR(45),
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_config_history_type ON config_history(config_type);
CREATE INDEX IF NOT EXISTS idx_config_history_key ON config_history(key);
CREATE INDEX IF NOT EXISTS idx_config_history_timestamp ON config_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_config_history_user ON config_history(changed_by);

-- Configuration Validation Rules
-- Defines validation rules for configuration values
CREATE TABLE IF NOT EXISTS config_validation_rules (
    id SERIAL PRIMARY KEY,
    config_type VARCHAR(50) NOT NULL,
    key VARCHAR(100) NOT NULL,
    validation_type VARCHAR(50) NOT NULL, -- 'range', 'regex', 'enum', 'min', 'max', 'custom'
    validation_params JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(config_type, key, validation_type)
);

CREATE INDEX IF NOT EXISTS idx_config_validation_type ON config_validation_rules(config_type);

-- Insert default configurations
-- These are the default values that can be overridden

-- General Configuration
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('general', 'ml_enabled', 'false', 'bool', 'Enable machine learning features', TRUE, FALSE),
('general', 'mode', 'production', 'string', 'Application mode: development, testing, production', TRUE, TRUE),
('general', 'dry_run', 'true', 'bool', 'Enable dry run mode (no real trades)', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Portfolio Configuration
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('portfolio', 'initial_balance', '400.0', 'float', 'Initial portfolio balance in USD', TRUE, TRUE),
('portfolio', 'initial_balance_per_chain', '100.0', 'float', 'Initial balance per blockchain', TRUE, TRUE),
('portfolio', 'max_position_size_usd', '10.0', 'float', 'Maximum position size in USD', TRUE, FALSE),
('portfolio', 'min_position_size_usd', '5.0', 'float', 'Minimum position size in USD', TRUE, FALSE),
('portfolio', 'max_position_size_pct', '0.10', 'float', 'Maximum position size as % of portfolio', TRUE, FALSE),
('portfolio', 'max_positions', '40', 'int', 'Maximum number of positions', TRUE, FALSE),
('portfolio', 'max_positions_per_chain', '10', 'int', 'Maximum positions per blockchain', TRUE, FALSE),
('portfolio', 'max_concurrent_positions', '4', 'int', 'Maximum concurrent open positions', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Risk Management Configuration
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('risk_management', 'max_risk_per_trade', '0.10', 'float', 'Maximum risk per trade as %', TRUE, FALSE),
('risk_management', 'max_portfolio_risk', '0.25', 'float', 'Maximum portfolio risk as %', TRUE, FALSE),
('risk_management', 'daily_loss_limit_usd', '40.0', 'float', 'Daily loss limit in USD', TRUE, FALSE),
('risk_management', 'daily_loss_limit_pct', '0.10', 'float', 'Daily loss limit as % of portfolio', TRUE, FALSE),
('risk_management', 'stop_loss_pct', '0.12', 'float', 'Default stop loss percentage', TRUE, FALSE),
('risk_management', 'take_profit_pct', '0.24', 'float', 'Default take profit percentage', TRUE, FALSE),
('risk_management', 'position_cooldown_minutes', '30', 'int', 'Cooldown period between positions', TRUE, FALSE),
('risk_management', 'breaker_max_consecutive_losses', '5', 'int', 'Circuit breaker: max consecutive losses', TRUE, FALSE),
('risk_management', 'breaker_max_drawdown_pct', '15', 'int', 'Circuit breaker: max drawdown %', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Trading Configuration
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('trading', 'max_slippage_bps', '50', 'int', 'Maximum slippage in basis points', TRUE, FALSE),
('trading', 'expected_slippage_bps', '10', 'int', 'Expected slippage in basis points', TRUE, FALSE),
('trading', 'max_price_impact_bps', '100', 'int', 'Maximum price impact in basis points', TRUE, FALSE),
('trading', 'dex_fee_bps', '30', 'int', 'DEX fee in basis points', TRUE, FALSE),
('trading', 'min_opportunity_score', '0.25', 'float', 'Minimum opportunity score to trade', TRUE, FALSE),
('trading', 'solana_min_opportunity_score', '0.20', 'float', 'Minimum opportunity score for Solana', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Chain Configuration
-- Chains are managed via Settings > Chains Tab and stored in database
-- Monad supported by DexScreener since November 2025
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('chain', 'enabled_chains', 'ethereum,bsc,base,solana,monad,pulsechain', 'string', 'Comma-separated list of enabled chains', TRUE, TRUE),
('chain', 'default_chain', 'ethereum', 'string', 'Default blockchain network', TRUE, FALSE),
('chain', 'max_pairs_per_chain', '50', 'int', 'Maximum trading pairs per chain', TRUE, FALSE),
('chain', 'discovery_interval_seconds', '300', 'int', 'Interval between chain discovery scans', TRUE, FALSE),
-- Chain enabled flags
('chain', 'ethereum_enabled', 'true', 'bool', 'Enable Ethereum trading', TRUE, TRUE),
('chain', 'bsc_enabled', 'true', 'bool', 'Enable BSC trading', TRUE, TRUE),
('chain', 'base_enabled', 'true', 'bool', 'Enable Base trading', TRUE, TRUE),
('chain', 'arbitrum_enabled', 'false', 'bool', 'Enable Arbitrum trading (low activity)', TRUE, TRUE),
('chain', 'polygon_enabled', 'false', 'bool', 'Enable Polygon trading', TRUE, TRUE),
('chain', 'solana_enabled', 'true', 'bool', 'Enable Solana trading', TRUE, TRUE),
('chain', 'monad_enabled', 'true', 'bool', 'Enable Monad trading (DexScreener Nov 2025)', TRUE, TRUE),
('chain', 'pulsechain_enabled', 'true', 'bool', 'Enable PulseChain trading', TRUE, TRUE),
('chain', 'fantom_enabled', 'false', 'bool', 'Enable Fantom trading', TRUE, TRUE),
('chain', 'cronos_enabled', 'false', 'bool', 'Enable Cronos trading', TRUE, TRUE),
('chain', 'avalanche_enabled', 'false', 'bool', 'Enable Avalanche trading', TRUE, TRUE),
-- Chain minimum liquidity thresholds (USD)
('chain', 'ethereum_min_liquidity', '3000', 'int', 'Minimum liquidity for Ethereum pairs', TRUE, FALSE),
('chain', 'bsc_min_liquidity', '500', 'int', 'Minimum liquidity for BSC pairs', TRUE, FALSE),
('chain', 'base_min_liquidity', '2000', 'int', 'Minimum liquidity for Base pairs', TRUE, FALSE),
('chain', 'arbitrum_min_liquidity', '2500', 'int', 'Minimum liquidity for Arbitrum pairs', TRUE, FALSE),
('chain', 'polygon_min_liquidity', '500', 'int', 'Minimum liquidity for Polygon pairs', TRUE, FALSE),
('chain', 'solana_min_liquidity', '2000', 'int', 'Minimum liquidity for Solana pairs', TRUE, FALSE),
('chain', 'monad_min_liquidity', '2000', 'int', 'Minimum liquidity for Monad pairs', TRUE, FALSE),
('chain', 'pulsechain_min_liquidity', '1000', 'int', 'Minimum liquidity for PulseChain pairs', TRUE, FALSE),
('chain', 'fantom_min_liquidity', '500', 'int', 'Minimum liquidity for Fantom pairs', TRUE, FALSE),
('chain', 'cronos_min_liquidity', '500', 'int', 'Minimum liquidity for Cronos pairs', TRUE, FALSE),
('chain', 'avalanche_min_liquidity', '1000', 'int', 'Minimum liquidity for Avalanche pairs', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Position Management Configuration
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('position_management', 'default_stop_loss_percent', '12', 'int', 'Default stop loss %', TRUE, FALSE),
('position_management', 'default_take_profit_percent', '24', 'int', 'Default take profit %', TRUE, FALSE),
('position_management', 'max_hold_time_minutes', '60', 'int', 'Maximum position hold time', TRUE, FALSE),
('position_management', 'trailing_stop_enabled', 'true', 'bool', 'Enable trailing stop loss', TRUE, FALSE),
('position_management', 'trailing_stop_percent', '6', 'int', 'Trailing stop loss %', TRUE, FALSE),
('position_management', 'position_update_interval_seconds', '10', 'int', 'Position update check interval', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- API Configuration (non-sensitive parts)
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('api', 'rate_limit_requests', '100', 'int', 'API rate limit requests per window', TRUE, TRUE),
('api', 'rate_limit_window', '60', 'int', 'API rate limit window in seconds', TRUE, TRUE),
('api', 'cors_enabled', 'true', 'bool', 'Enable CORS', TRUE, TRUE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Monitoring Configuration (non-sensitive parts)
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('monitoring', 'enable_metrics', 'true', 'bool', 'Enable metrics collection', TRUE, FALSE),
('monitoring', 'log_level', 'INFO', 'string', 'Logging level', TRUE, FALSE),
('monitoring', 'log_retention_days', '30', 'int', 'Log retention period', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- ML Models Configuration
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('ml_models', 'ml_retrain_interval_hours', '24', 'int', 'Model retraining interval', TRUE, FALSE),
('ml_models', 'ml_min_confidence', '0.7', 'float', 'Minimum model confidence threshold', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Feature Flags
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('feature_flags', 'enable_experimental_features', 'false', 'bool', 'Enable experimental features', TRUE, FALSE),
('feature_flags', 'use_ai_sentiment', 'false', 'bool', 'Use AI sentiment analysis', TRUE, FALSE),
('feature_flags', 'use_whale_tracking', 'true', 'bool', 'Track whale movements', TRUE, FALSE),
('feature_flags', 'use_mempool_monitoring', 'true', 'bool', 'Monitor mempool for MEV', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Insert validation rules
INSERT INTO config_validation_rules (config_type, key, validation_type, validation_params, error_message) VALUES
('portfolio', 'max_position_size_pct', 'range', '{"min": 0.01, "max": 1.0}', 'Position size must be between 1% and 100%'),
('portfolio', 'max_positions', 'range', '{"min": 1, "max": 100}', 'Max positions must be between 1 and 100'),
('risk_management', 'stop_loss_pct', 'range', '{"min": 0.01, "max": 0.50}', 'Stop loss must be between 1% and 50%'),
('risk_management', 'take_profit_pct', 'range', '{"min": 0.01, "max": 2.0}', 'Take profit must be between 1% and 200%'),
('risk_management', 'daily_loss_limit_pct', 'range', '{"min": 0.01, "max": 1.0}', 'Daily loss limit must be between 1% and 100%'),
('trading', 'max_slippage_bps', 'range', '{"min": 1, "max": 1000}', 'Slippage must be between 1 and 1000 bps'),
('trading', 'min_opportunity_score', 'range', '{"min": 0.1, "max": 1.0}', 'Opportunity score must be between 0.1 and 1.0'),
('monitoring', 'log_level', 'enum', '{"values": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]}', 'Invalid log level'),
('general', 'mode', 'enum', '{"values": ["development", "testing", "production"]}', 'Invalid mode')
ON CONFLICT (config_type, key, validation_type) DO NOTHING;

-- Function to update config timestamp
CREATE OR REPLACE FUNCTION update_config_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic timestamp updates
DROP TRIGGER IF EXISTS update_config_settings_timestamp ON config_settings;
CREATE TRIGGER update_config_settings_timestamp
    BEFORE UPDATE ON config_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_config_updated_at();

DROP TRIGGER IF EXISTS update_config_sensitive_timestamp ON config_sensitive;
CREATE TRIGGER update_config_sensitive_timestamp
    BEFORE UPDATE ON config_sensitive
    FOR EACH ROW
    EXECUTE FUNCTION update_config_updated_at();

-- Comments for documentation
COMMENT ON TABLE config_settings IS 'Stores all non-sensitive configuration values with validation and audit support';
COMMENT ON TABLE config_sensitive IS 'Stores encrypted sensitive configuration data like API keys and private keys';
COMMENT ON TABLE config_history IS 'Audit log of all configuration changes';
COMMENT ON TABLE config_validation_rules IS 'Validation rules for configuration values';

COMMENT ON COLUMN config_settings.value_type IS 'Data type of the value: string, int, float, bool, json';
COMMENT ON COLUMN config_settings.requires_restart IS 'Whether changing this config requires application restart';
COMMENT ON COLUMN config_sensitive.encryption_method IS 'Encryption algorithm used for this value';
COMMENT ON COLUMN config_sensitive.rotation_interval_days IS 'Days between mandatory key rotations';
