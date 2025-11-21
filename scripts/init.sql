-- Migration: Add Authentication Tables
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

CREATE INDEX idx_config_settings_type ON config_settings(config_type);
CREATE INDEX idx_config_settings_key ON config_settings(key);
CREATE INDEX idx_config_settings_updated_at ON config_settings(updated_at);

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

CREATE INDEX idx_config_sensitive_key ON config_sensitive(key);
CREATE INDEX idx_config_sensitive_active ON config_sensitive(is_active);
CREATE INDEX idx_config_sensitive_rotation ON config_sensitive(last_rotated) WHERE is_active = TRUE;

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

CREATE INDEX idx_config_history_type ON config_history(config_type);
CREATE INDEX idx_config_history_key ON config_history(key);
CREATE INDEX idx_config_history_timestamp ON config_history(timestamp DESC);
CREATE INDEX idx_config_history_user ON config_history(changed_by);

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

CREATE INDEX idx_config_validation_type ON config_validation_rules(config_type);

-- Insert default configurations
-- These are the default values that can be overridden

-- General Configuration
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('general', 'ml_enabled', 'false', 'bool', 'Enable machine learning features', TRUE, FALSE),
('general', 'mode', 'production', 'string', 'Application mode: development, testing, production', TRUE, TRUE),
('general', 'dry_run', 'true', 'bool', 'Enable dry run mode (no real trades)', TRUE, FALSE);

-- Portfolio Configuration
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('portfolio', 'initial_balance', '400.0', 'float', 'Initial portfolio balance in USD', TRUE, TRUE),
('portfolio', 'initial_balance_per_chain', '100.0', 'float', 'Initial balance per blockchain', TRUE, TRUE),
('portfolio', 'max_position_size_usd', '10.0', 'float', 'Maximum position size in USD', TRUE, FALSE),
('portfolio', 'min_position_size_usd', '5.0', 'float', 'Minimum position size in USD', TRUE, FALSE),
('portfolio', 'max_position_size_pct', '0.10', 'float', 'Maximum position size as % of portfolio', TRUE, FALSE),
('portfolio', 'max_positions', '40', 'int', 'Maximum number of positions', TRUE, FALSE),
('portfolio', 'max_positions_per_chain', '10', 'int', 'Maximum positions per blockchain', TRUE, FALSE),
('portfolio', 'max_concurrent_positions', '4', 'int', 'Maximum concurrent open positions', TRUE, FALSE);

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
('risk_management', 'breaker_max_drawdown_pct', '15', 'int', 'Circuit breaker: max drawdown %', TRUE, FALSE);

-- Trading Configuration
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('trading', 'max_slippage_bps', '50', 'int', 'Maximum slippage in basis points', TRUE, FALSE),
('trading', 'expected_slippage_bps', '10', 'int', 'Expected slippage in basis points', TRUE, FALSE),
('trading', 'max_price_impact_bps', '100', 'int', 'Maximum price impact in basis points', TRUE, FALSE),
('trading', 'dex_fee_bps', '30', 'int', 'DEX fee in basis points', TRUE, FALSE),
('trading', 'min_opportunity_score', '0.25', 'float', 'Minimum opportunity score to trade', TRUE, FALSE),
('trading', 'solana_min_opportunity_score', '0.20', 'float', 'Minimum opportunity score for Solana', TRUE, FALSE);

-- Chain Configuration
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('chain', 'enabled_chains', 'ethereum,bsc,base,arbitrum,solana', 'string', 'Comma-separated list of enabled chains', TRUE, TRUE),
('chain', 'default_chain', 'ethereum', 'string', 'Default blockchain network', TRUE, FALSE),
('chain', 'ethereum_enabled', 'true', 'bool', 'Enable Ethereum trading', TRUE, TRUE),
('chain', 'bsc_enabled', 'true', 'bool', 'Enable BSC trading', TRUE, TRUE),
('chain', 'base_enabled', 'true', 'bool', 'Enable Base trading', TRUE, TRUE),
('chain', 'arbitrum_enabled', 'true', 'bool', 'Enable Arbitrum trading', TRUE, TRUE),
('chain', 'solana_enabled', 'true', 'bool', 'Enable Solana trading', TRUE, TRUE),
('chain', 'ethereum_min_liquidity', '3000', 'int', 'Minimum liquidity for Ethereum pairs', TRUE, FALSE),
('chain', 'bsc_min_liquidity', '500', 'int', 'Minimum liquidity for BSC pairs', TRUE, FALSE),
('chain', 'solana_min_liquidity', '2000', 'int', 'Minimum liquidity for Solana pairs', TRUE, FALSE);

-- Position Management Configuration
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('position_management', 'default_stop_loss_percent', '12', 'int', 'Default stop loss %', TRUE, FALSE),
('position_management', 'default_take_profit_percent', '24', 'int', 'Default take profit %', TRUE, FALSE),
('position_management', 'max_hold_time_minutes', '60', 'int', 'Maximum position hold time', TRUE, FALSE),
('position_management', 'trailing_stop_enabled', 'true', 'bool', 'Enable trailing stop loss', TRUE, FALSE),
('position_management', 'trailing_stop_percent', '6', 'int', 'Trailing stop loss %', TRUE, FALSE),
('position_management', 'position_update_interval_seconds', '10', 'int', 'Position update check interval', TRUE, FALSE);

-- API Configuration (non-sensitive parts)
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('api', 'rate_limit_requests', '100', 'int', 'API rate limit requests per window', TRUE, TRUE),
('api', 'rate_limit_window', '60', 'int', 'API rate limit window in seconds', TRUE, TRUE),
('api', 'cors_enabled', 'true', 'bool', 'Enable CORS', TRUE, TRUE);

-- Monitoring Configuration (non-sensitive parts)
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('monitoring', 'enable_metrics', 'true', 'bool', 'Enable metrics collection', TRUE, FALSE),
('monitoring', 'log_level', 'INFO', 'string', 'Logging level', TRUE, FALSE),
('monitoring', 'log_retention_days', '30', 'int', 'Log retention period', TRUE, FALSE);

-- ML Models Configuration
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('ml_models', 'ml_retrain_interval_hours', '24', 'int', 'Model retraining interval', TRUE, FALSE),
('ml_models', 'ml_min_confidence', '0.7', 'float', 'Minimum model confidence threshold', TRUE, FALSE);

-- Feature Flags
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('feature_flags', 'enable_experimental_features', 'false', 'bool', 'Enable experimental features', TRUE, FALSE),
('feature_flags', 'use_ai_sentiment', 'false', 'bool', 'Use AI sentiment analysis', TRUE, FALSE),
('feature_flags', 'use_whale_tracking', 'true', 'bool', 'Track whale movements', TRUE, FALSE),
('feature_flags', 'use_mempool_monitoring', 'true', 'bool', 'Monitor mempool for MEV', TRUE, FALSE);

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
('general', 'mode', 'enum', '{"values": ["development", "testing", "production"]}', 'Invalid mode');

-- Function to update config timestamp
CREATE OR REPLACE FUNCTION update_config_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_config_settings_timestamp
    BEFORE UPDATE ON config_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_config_updated_at();

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
