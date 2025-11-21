-- Migration 003: Add missing Arbitrum and Base min liquidity configs
-- This migration adds the missing chain configuration settings

-- Add Base and Arbitrum min liquidity settings
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('chain', 'base_min_liquidity', '2000', 'int', 'Minimum liquidity for Base pairs', TRUE, FALSE),
('chain', 'arbitrum_min_liquidity', '2500', 'int', 'Minimum liquidity for Arbitrum pairs', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;
