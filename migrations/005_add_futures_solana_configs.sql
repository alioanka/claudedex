-- Migration: Add Futures and Solana Module Configurations
-- Description: Create database-backed configuration for Futures and Solana trading modules
-- Created: 2025-11-28
--
-- These configurations replace the .env-based settings for non-sensitive data.
-- Sensitive data (API keys, private keys) remains in .env

-- ============================================================================
-- FUTURES MODULE CONFIGURATION
-- ============================================================================

-- Futures General Settings
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('futures_general', 'enabled', 'false', 'bool', 'Enable the futures trading module', TRUE, TRUE),
('futures_general', 'exchange', 'binance', 'string', 'Default exchange: binance or bybit', TRUE, FALSE),
('futures_general', 'testnet', 'true', 'bool', 'Use testnet/paper trading mode', TRUE, TRUE),
('futures_general', 'contract_type', 'perpetual', 'string', 'Contract type: perpetual or coin_margined', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Futures Capital & Position Sizing
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('futures_position', 'capital_allocation', '300', 'float', 'Total capital allocated to futures (USDT)', TRUE, FALSE),
('futures_position', 'position_size_usd', '100', 'float', 'Default position size (USDT)', TRUE, FALSE),
('futures_position', 'max_position_pct', '10', 'float', 'Max position size as % of capital', TRUE, FALSE),
('futures_position', 'max_positions', '5', 'int', 'Maximum concurrent open positions', TRUE, FALSE),
('futures_position', 'min_trade_size', '10', 'float', 'Minimum trade size (USDT)', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Futures Leverage Settings
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('futures_leverage', 'default_leverage', '10', 'int', 'Default leverage for new positions', TRUE, FALSE),
('futures_leverage', 'max_leverage', '20', 'int', 'Maximum allowed leverage', TRUE, FALSE),
('futures_leverage', 'margin_mode', 'isolated', 'string', 'Margin mode: isolated or cross', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Futures Risk Management
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('futures_risk', 'stop_loss_pct', '5.0', 'float', 'Default stop loss percentage', TRUE, FALSE),
('futures_risk', 'take_profit_pct', '10.0', 'float', 'Default take profit percentage', TRUE, FALSE),
('futures_risk', 'max_daily_loss_usd', '500', 'float', 'Max daily loss before trading pauses (USDT)', TRUE, FALSE),
('futures_risk', 'max_daily_loss_pct', '10', 'float', 'Max daily loss as % of capital', TRUE, FALSE),
('futures_risk', 'liquidation_buffer', '20', 'float', 'Close position at this % from liquidation', TRUE, FALSE),
('futures_risk', 'trailing_stop_enabled', 'true', 'bool', 'Enable trailing stop loss', TRUE, FALSE),
('futures_risk', 'trailing_stop_distance', '1.5', 'float', 'Trailing stop distance %', TRUE, FALSE),
('futures_risk', 'max_consecutive_losses', '5', 'int', 'Pause after this many consecutive losses', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Futures Trading Pairs
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('futures_pairs', 'allowed_pairs', 'BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT', 'string', 'Comma-separated list of allowed trading pairs', TRUE, FALSE),
('futures_pairs', 'both_directions', 'true', 'bool', 'Allow both long and short positions', TRUE, FALSE),
('futures_pairs', 'preferred_direction', 'both', 'string', 'Preferred direction: long, short, or both', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Futures Strategy Settings
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('futures_strategy', 'signal_timeframe', '15m', 'string', 'Candle timeframe for signals: 1m, 5m, 15m, 30m, 1h, 4h', TRUE, FALSE),
('futures_strategy', 'scan_interval_seconds', '30', 'int', 'How often to scan for opportunities (seconds)', TRUE, FALSE),
('futures_strategy', 'rsi_oversold', '30', 'float', 'RSI oversold threshold (STRONG_BUY)', TRUE, FALSE),
('futures_strategy', 'rsi_overbought', '70', 'float', 'RSI overbought threshold (STRONG_SELL)', TRUE, FALSE),
('futures_strategy', 'rsi_weak_oversold', '40', 'float', 'RSI weak oversold threshold (BUY)', TRUE, FALSE),
('futures_strategy', 'rsi_weak_overbought', '60', 'float', 'RSI weak overbought threshold (SELL)', TRUE, FALSE),
('futures_strategy', 'min_signal_score', '3', 'int', 'Minimum signal score to enter (1-6, lower=more trades)', TRUE, FALSE),
('futures_strategy', 'verbose_signals', 'true', 'bool', 'Log detailed signal analysis', TRUE, FALSE),
('futures_strategy', 'cooldown_minutes', '5', 'int', 'Cooldown after closing position', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Futures Funding Rate Settings
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('futures_funding', 'funding_arbitrage_enabled', 'false', 'bool', 'Enable funding rate arbitrage', TRUE, FALSE),
('futures_funding', 'max_funding_rate', '0.1', 'float', 'Max funding rate before closing (%)', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- ============================================================================
-- SOLANA MODULE CONFIGURATION
-- ============================================================================

-- Solana General Settings
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('solana_general', 'enabled', 'false', 'bool', 'Enable the Solana trading module', TRUE, TRUE),
('solana_general', 'strategies', 'jupiter,drift,pumpfun', 'string', 'Comma-separated enabled strategies', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Solana Capital & Position Sizing
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('solana_position', 'capital_allocation_sol', '10', 'float', 'Total capital allocated (SOL)', TRUE, FALSE),
('solana_position', 'position_size_sol', '1.0', 'float', 'Default position size (SOL)', TRUE, FALSE),
('solana_position', 'max_positions', '3', 'int', 'Maximum concurrent positions', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Solana Risk Management
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('solana_risk', 'stop_loss_pct', '10.0', 'float', 'Default stop loss percentage', TRUE, FALSE),
('solana_risk', 'take_profit_pct', '50.0', 'float', 'Default take profit percentage', TRUE, FALSE),
('solana_risk', 'max_daily_loss_sol', '5.0', 'float', 'Max daily loss (SOL)', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Solana Transaction Settings
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('solana_tx', 'priority_fee', '5000', 'int', 'Priority fee in microlamports', TRUE, FALSE),
('solana_tx', 'compute_unit_price', '1000', 'int', 'Compute unit price', TRUE, FALSE),
('solana_tx', 'compute_unit_limit', '200000', 'int', 'Compute unit limit', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Jupiter Strategy Settings
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('solana_jupiter', 'enabled', 'true', 'bool', 'Enable Jupiter swaps', TRUE, FALSE),
('solana_jupiter', 'slippage_bps', '50', 'int', 'Slippage tolerance in basis points', TRUE, FALSE),
('solana_jupiter', 'api_tier', 'public', 'string', 'Jupiter API tier: lite, public, ultra', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Drift Strategy Settings
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('solana_drift', 'enabled', 'true', 'bool', 'Enable Drift perpetuals', TRUE, FALSE),
('solana_drift', 'leverage', '5', 'int', 'Default leverage for Drift', TRUE, FALSE),
('solana_drift', 'markets', 'SOL-PERP,BTC-PERP,ETH-PERP', 'string', 'Allowed Drift markets', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Pump.fun Strategy Settings
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('solana_pumpfun', 'enabled', 'true', 'bool', 'Enable Pump.fun sniping', TRUE, FALSE),
('solana_pumpfun', 'min_liquidity', '10', 'float', 'Minimum liquidity (SOL)', TRUE, FALSE),
('solana_pumpfun', 'max_age_seconds', '300', 'int', 'Max token age to consider', TRUE, FALSE),
('solana_pumpfun', 'buy_amount_sol', '0.1', 'float', 'Default buy amount (SOL)', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- ============================================================================
-- VALIDATION RULES
-- ============================================================================

-- Futures validation rules
INSERT INTO config_validation_rules (config_type, key, validation_type, validation_params, error_message) VALUES
('futures_leverage', 'default_leverage', 'range', '{"min": 1, "max": 125}', 'Leverage must be between 1x and 125x'),
('futures_leverage', 'max_leverage', 'range', '{"min": 1, "max": 125}', 'Max leverage must be between 1x and 125x'),
('futures_risk', 'stop_loss_pct', 'range', '{"min": 0.1, "max": 50}', 'Stop loss must be between 0.1% and 50%'),
('futures_risk', 'take_profit_pct', 'range', '{"min": 0.1, "max": 500}', 'Take profit must be between 0.1% and 500%'),
('futures_strategy', 'signal_timeframe', 'enum', '{"values": ["1m", "5m", "15m", "30m", "1h", "4h"]}', 'Invalid timeframe'),
('futures_strategy', 'scan_interval_seconds', 'range', '{"min": 10, "max": 300}', 'Scan interval must be between 10 and 300 seconds'),
('futures_strategy', 'min_signal_score', 'range', '{"min": 1, "max": 10}', 'Signal score must be between 1 and 10'),
('futures_general', 'exchange', 'enum', '{"values": ["binance", "bybit", "okx", "bitget"]}', 'Invalid exchange'),
('futures_leverage', 'margin_mode', 'enum', '{"values": ["isolated", "cross"]}', 'Invalid margin mode')
ON CONFLICT (config_type, key, validation_type) DO NOTHING;

-- Solana validation rules
INSERT INTO config_validation_rules (config_type, key, validation_type, validation_params, error_message) VALUES
('solana_drift', 'leverage', 'range', '{"min": 1, "max": 20}', 'Drift leverage must be between 1x and 20x'),
('solana_jupiter', 'slippage_bps', 'range', '{"min": 1, "max": 1000}', 'Slippage must be between 1 and 1000 bps'),
('solana_jupiter', 'api_tier', 'enum', '{"values": ["lite", "public", "ultra"]}', 'Invalid Jupiter API tier')
ON CONFLICT (config_type, key, validation_type) DO NOTHING;

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE config_settings IS 'Centralized configuration for all modules - non-sensitive data only';
