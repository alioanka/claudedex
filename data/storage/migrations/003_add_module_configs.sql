-- Migration: Add module_configs table for database-backed module configuration
-- Replaces YAML-based configuration with database storage
-- Each module (DEX, Futures, Solana) can store multiple config types

CREATE TABLE IF NOT EXISTS module_configs (
    id SERIAL PRIMARY KEY,
    module_name VARCHAR(100) NOT NULL,  -- 'dex_trading', 'futures_trading', 'solana_strategies'
    config_type VARCHAR(100) NOT NULL,  -- 'general', 'risk', 'trading', 'strategies', etc.
    config_data JSONB NOT NULL,         -- Configuration as JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),            -- User who created/modified
    version INTEGER DEFAULT 1,

    -- Ensure one config per module/type combination
    UNIQUE(module_name, config_type)
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_module_configs_lookup
ON module_configs(module_name, config_type);

-- Index for timestamp queries
CREATE INDEX IF NOT EXISTS idx_module_configs_updated
ON module_configs(updated_at DESC);

-- Table for configuration change history/audit
CREATE TABLE IF NOT EXISTS module_config_history (
    id SERIAL PRIMARY KEY,
    module_name VARCHAR(100) NOT NULL,
    config_type VARCHAR(100) NOT NULL,
    old_config JSONB,
    new_config JSONB,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    changed_by VARCHAR(100),
    change_reason TEXT
);

-- Index for history queries
CREATE INDEX IF NOT EXISTS idx_module_config_history_lookup
ON module_config_history(module_name, config_type, changed_at DESC);

-- Trigger to log config changes
CREATE OR REPLACE FUNCTION log_module_config_change()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        INSERT INTO module_config_history (
            module_name, config_type, old_config, new_config, changed_by
        ) VALUES (
            OLD.module_name, OLD.config_type, OLD.config_data, NEW.config_data, NEW.created_by
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER module_config_change_trigger
AFTER UPDATE ON module_configs
FOR EACH ROW
EXECUTE FUNCTION log_module_config_change();

-- Insert default configurations for all modules

-- DEX Trading Module defaults (preserved from existing config)
INSERT INTO module_configs (module_name, config_type, config_data) VALUES
('dex_trading', 'general', '{"mode": "production", "dry_run": true, "ml_enabled": false}'),
('dex_trading', 'risk', '{"max_risk_per_trade": 0.10, "max_portfolio_risk": 0.25, "stop_loss_pct": 0.12, "take_profit_pct": 0.24}'),
('dex_trading', 'trading', '{"max_slippage_bps": 50, "min_opportunity_score": 0.25}')
ON CONFLICT (module_name, config_type) DO NOTHING;

-- Futures Trading Module defaults
INSERT INTO module_configs (module_name, config_type, config_data) VALUES
('futures_trading', 'futures_general', '{"enabled": false, "mode": "testnet", "capital_allocation": 300.0, "max_positions": 5, "default_exchange": "binance"}'),
('futures_trading', 'futures_risk', '{"max_leverage": 3, "max_position_size_usd": 60.0, "risk_per_trade_pct": 0.02, "stop_loss_pct": 0.05, "take_profit_pct": 0.10}'),
('futures_trading', 'futures_trading', '{"min_opportunity_score": 0.70, "max_slippage_bps": 50, "partial_tp_enabled": true}'),
('futures_trading', 'futures_exchanges', '{"binance_enabled": true, "binance_testnet": true, "bybit_enabled": false}')
ON CONFLICT (module_name, config_type) DO NOTHING;

-- Solana Strategies Module defaults
INSERT INTO module_configs (module_name, config_type, config_data) VALUES
('solana_strategies', 'solana_general', '{"enabled": false, "capital_allocation": 400.0, "max_positions": 8, "use_jito": true, "priority_fee_lamports": 10000}'),
('solana_strategies', 'solana_risk', '{"max_position_size_usd": 80.0, "risk_per_trade_pct": 0.02, "stop_loss_pct": 0.10, "take_profit_pct": 0.20, "min_liquidity_usd": 10000}'),
('solana_strategies', 'solana_trading', '{"min_opportunity_score": 0.60, "use_jupiter_routing": true, "jupiter_slippage_bps": 50}'),
('solana_strategies', 'solana_pumpfun', '{"enabled": true, "monitor_new_tokens": true, "min_initial_liquidity": 5000, "max_buy_amount_sol": 0.5}'),
('solana_strategies', 'solana_jupiter', '{"enabled": true, "use_v6_api": true, "slippage_bps": 50}'),
('solana_strategies', 'solana_drift', '{"enabled": false, "use_perpetuals": true, "max_leverage": 5}')
ON CONFLICT (module_name, config_type) DO NOTHING;

-- Add comments for documentation
COMMENT ON TABLE module_configs IS 'Database-backed configuration for trading modules (DEX, Futures, Solana)';
COMMENT ON COLUMN module_configs.module_name IS 'Module identifier: dex_trading, futures_trading, or solana_strategies';
COMMENT ON COLUMN module_configs.config_type IS 'Configuration category: general, risk, trading, strategies, etc.';
COMMENT ON COLUMN module_configs.config_data IS 'Configuration as JSONB for flexibility and querying';

COMMENT ON TABLE module_config_history IS 'Audit trail for all configuration changes';
