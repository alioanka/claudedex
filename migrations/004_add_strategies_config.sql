-- Migration: Add strategies configuration
-- Description: Add strategies configuration to config_settings table
-- Created: 2025-11-21

-- Strategies Configuration
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
-- Momentum Strategy
('strategies', 'momentum_enabled', 'true', 'bool', 'Enable momentum trading strategy', TRUE, FALSE),
('strategies', 'momentum_lookback_period', '20', 'int', 'Momentum lookback period in candles', TRUE, FALSE),
('strategies', 'momentum_min_momentum_score', '0.5', 'float', 'Minimum momentum score to trade', TRUE, FALSE),
('strategies', 'momentum_volume_threshold', '10000', 'int', 'Minimum volume threshold for momentum trades', TRUE, FALSE),

-- Scalping Strategy
('strategies', 'scalping_enabled', 'true', 'bool', 'Enable scalping trading strategy', TRUE, FALSE),
('strategies', 'scalping_profit_target', '0.02', 'float', 'Scalping profit target percentage', TRUE, FALSE),
('strategies', 'scalping_max_hold_time', '5', 'int', 'Scalping maximum hold time in minutes', TRUE, FALSE),
('strategies', 'scalping_min_spread', '0.001', 'float', 'Minimum spread for scalping trades', TRUE, FALSE),

-- AI Strategy
('strategies', 'ai_enabled', 'true', 'bool', 'Enable AI-powered trading strategy', TRUE, FALSE),
('strategies', 'ai_ml_confidence_threshold', '0.65', 'float', 'Minimum ML model confidence threshold', TRUE, FALSE),
('strategies', 'ai_min_pump_probability', '0.50', 'float', 'Minimum pump probability for AI trades', TRUE, FALSE),
('strategies', 'ai_ensemble_min_models', '3', 'int', 'Minimum number of models in ensemble', TRUE, FALSE),
('strategies', 'ai_use_lstm', 'true', 'bool', 'Use LSTM model in AI strategy', TRUE, FALSE),
('strategies', 'ai_use_xgboost', 'true', 'bool', 'Use XGBoost model in AI strategy', TRUE, FALSE),
('strategies', 'ai_use_lightgbm', 'true', 'bool', 'Use LightGBM model in AI strategy', TRUE, FALSE),

-- Strategy Selection
('strategies', 'strategy_selection_mode', 'auto', 'string', 'Strategy selection mode: auto, single, multi', TRUE, FALSE),
('strategies', 'default_strategy', 'momentum', 'string', 'Default strategy when mode is single', TRUE, FALSE),
('strategies', 'multi_strategy_enabled', 'true', 'bool', 'Allow multiple strategies per opportunity', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Insert validation rules for strategies
INSERT INTO config_validation_rules (config_type, key, validation_type, validation_params, error_message) VALUES
('strategies', 'momentum_min_momentum_score', 'range', '{"min": 0.0, "max": 1.0}', 'Momentum score must be between 0.0 and 1.0'),
('strategies', 'momentum_lookback_period', 'range', '{"min": 1, "max": 100}', 'Lookback period must be between 1 and 100'),
('strategies', 'scalping_profit_target', 'range', '{"min": 0.001, "max": 0.10}', 'Profit target must be between 0.1% and 10%'),
('strategies', 'scalping_max_hold_time', 'range', '{"min": 1, "max": 60}', 'Hold time must be between 1 and 60 minutes'),
('strategies', 'ai_ml_confidence_threshold', 'range', '{"min": 0.5, "max": 1.0}', 'Confidence threshold must be between 0.5 and 1.0'),
('strategies', 'ai_min_pump_probability', 'range', '{"min": 0.0, "max": 1.0}', 'Pump probability must be between 0.0 and 1.0'),
('strategies', 'ai_ensemble_min_models', 'range', '{"min": 1, "max": 10}', 'Ensemble models must be between 1 and 10'),
('strategies', 'strategy_selection_mode', 'enum', '{"values": ["auto", "single", "multi"]}', 'Invalid strategy selection mode'),
('strategies', 'default_strategy', 'enum', '{"values": ["momentum", "scalping", "ai"]}', 'Invalid default strategy')
ON CONFLICT (config_type, key, validation_type) DO NOTHING;
