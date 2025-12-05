-- Migration: Add Monad and PulseChain support
-- Description: Adds monad, pulsechain, and other new chains to config_settings
-- Date: 2025-12-05

-- Update enabled_chains to include monad and pulsechain, remove arbitrum
UPDATE config_settings
SET value = 'ethereum,bsc,base,solana,monad,pulsechain'
WHERE config_type = 'chain' AND key = 'enabled_chains';

-- Set arbitrum to disabled (low activity)
UPDATE config_settings
SET value = 'false'
WHERE config_type = 'chain' AND key = 'arbitrum_enabled';

-- Add Monad chain settings
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('chain', 'monad_enabled', 'true', 'bool', 'Enable Monad trading (DexScreener Nov 2025)', TRUE, TRUE),
('chain', 'monad_min_liquidity', '2000', 'int', 'Minimum liquidity for Monad pairs', TRUE, FALSE)
ON CONFLICT (config_type, key) DO UPDATE SET value = EXCLUDED.value;

-- Add PulseChain chain settings
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('chain', 'pulsechain_enabled', 'true', 'bool', 'Enable PulseChain trading', TRUE, TRUE),
('chain', 'pulsechain_min_liquidity', '1000', 'int', 'Minimum liquidity for PulseChain pairs', TRUE, FALSE)
ON CONFLICT (config_type, key) DO UPDATE SET value = EXCLUDED.value;

-- Add Fantom chain settings (disabled by default)
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('chain', 'fantom_enabled', 'false', 'bool', 'Enable Fantom trading', TRUE, TRUE),
('chain', 'fantom_min_liquidity', '500', 'int', 'Minimum liquidity for Fantom pairs', TRUE, FALSE)
ON CONFLICT (config_type, key) DO UPDATE SET value = EXCLUDED.value;

-- Add Cronos chain settings (disabled by default)
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('chain', 'cronos_enabled', 'false', 'bool', 'Enable Cronos trading', TRUE, TRUE),
('chain', 'cronos_min_liquidity', '500', 'int', 'Minimum liquidity for Cronos pairs', TRUE, FALSE)
ON CONFLICT (config_type, key) DO UPDATE SET value = EXCLUDED.value;

-- Add Avalanche chain settings (disabled by default)
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('chain', 'avalanche_enabled', 'false', 'bool', 'Enable Avalanche trading', TRUE, TRUE),
('chain', 'avalanche_min_liquidity', '1000', 'int', 'Minimum liquidity for Avalanche pairs', TRUE, FALSE)
ON CONFLICT (config_type, key) DO UPDATE SET value = EXCLUDED.value;

-- Add Polygon chain settings (disabled by default, already exists but ensure it's there)
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('chain', 'polygon_enabled', 'false', 'bool', 'Enable Polygon trading', TRUE, TRUE),
('chain', 'polygon_min_liquidity', '500', 'int', 'Minimum liquidity for Polygon pairs', TRUE, FALSE)
ON CONFLICT (config_type, key) DO UPDATE SET value = EXCLUDED.value;

-- Update discovery_interval_seconds if it doesn't exist
INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart) VALUES
('chain', 'discovery_interval_seconds', '300', 'int', 'Interval between chain discovery scans', TRUE, FALSE),
('chain', 'max_pairs_per_chain', '50', 'int', 'Maximum trading pairs per chain', TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;

-- Record migration in migrations table if exists
INSERT INTO migrations (version, description, applied_at)
VALUES ('007', 'Add Monad and PulseChain chains support', NOW())
ON CONFLICT (version) DO NOTHING;
