-- Migration: Add Secure Credentials Table
-- Description: Store encrypted credentials with categories and rotation tracking
-- Created: 2025-12-22
--
-- SECURITY ARCHITECTURE:
-- 1. Encryption key stored SEPARATELY in Docker secret or /secure/encryption.key
-- 2. All credential values are encrypted with Fernet before storage
-- 3. Categories allow granular access control
-- 4. Rotation tracking helps maintain security hygiene

-- ============================================================================
-- SECURE CREDENTIALS TABLE
-- Main table for storing encrypted credentials
-- ============================================================================

CREATE TABLE IF NOT EXISTS secure_credentials (
    id SERIAL PRIMARY KEY,

    -- Credential identification
    key_name VARCHAR(100) NOT NULL UNIQUE,      -- e.g., 'BINANCE_API_KEY'
    display_name VARCHAR(200),                   -- Human-readable name
    description TEXT,                            -- Description of what this credential is for

    -- Encrypted value storage
    encrypted_value TEXT NOT NULL,               -- Fernet encrypted value
    value_hash VARCHAR(64),                      -- SHA256 hash for integrity checking

    -- Categorization
    category VARCHAR(50) NOT NULL DEFAULT 'general',
    -- Categories: 'wallet', 'exchange', 'database', 'api', 'notification', 'security'

    subcategory VARCHAR(50),                     -- e.g., 'binance', 'bybit', 'telegram'
    module VARCHAR(50),                          -- Which module uses this: 'dex', 'futures', 'solana', 'all'

    -- Security metadata
    is_sensitive BOOLEAN DEFAULT TRUE,           -- If true, never show in logs
    is_encrypted BOOLEAN DEFAULT TRUE,           -- Whether value is encrypted
    encryption_version INTEGER DEFAULT 1,        -- For key rotation tracking

    -- Rotation tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_rotated_at TIMESTAMP WITH TIME ZONE,
    rotation_interval_days INTEGER DEFAULT 90,   -- Recommended rotation frequency
    next_rotation_at TIMESTAMP WITH TIME ZONE,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,              -- Soft delete support
    is_required BOOLEAN DEFAULT FALSE,           -- Whether this credential is required for operation

    -- Access control
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb           -- Additional flexible data
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_secure_creds_category ON secure_credentials(category);
CREATE INDEX IF NOT EXISTS idx_secure_creds_module ON secure_credentials(module);
CREATE INDEX IF NOT EXISTS idx_secure_creds_active ON secure_credentials(is_active);
CREATE INDEX IF NOT EXISTS idx_secure_creds_key_name ON secure_credentials(key_name);

-- ============================================================================
-- CREDENTIAL ACCESS LOG
-- Audit trail for credential access (security compliance)
-- ============================================================================

CREATE TABLE IF NOT EXISTS credential_access_log (
    id SERIAL PRIMARY KEY,
    credential_id INTEGER REFERENCES secure_credentials(id) ON DELETE CASCADE,
    key_name VARCHAR(100) NOT NULL,

    -- Access details
    access_type VARCHAR(20) NOT NULL,            -- 'read', 'write', 'rotate', 'delete'
    accessed_by VARCHAR(100),                    -- Module or user that accessed
    access_source VARCHAR(100),                  -- IP or service name

    -- Result
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,

    -- Timestamp
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cred_access_log_cred ON credential_access_log(credential_id);
CREATE INDEX IF NOT EXISTS idx_cred_access_log_time ON credential_access_log(accessed_at DESC);

-- ============================================================================
-- CREDENTIAL CATEGORIES REFERENCE
-- Reference table for valid credential categories
-- ============================================================================

CREATE TABLE IF NOT EXISTS credential_categories (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50) NOT NULL UNIQUE,
    display_name VARCHAR(100) NOT NULL,
    description TEXT,
    icon VARCHAR(50),                            -- Font Awesome icon name
    color VARCHAR(20),                           -- For UI display
    sort_order INTEGER DEFAULT 100,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default categories
INSERT INTO credential_categories (category, display_name, description, icon, color, sort_order) VALUES
('security', 'Security & Encryption', 'Encryption keys, JWT secrets, and security tokens', 'fa-shield-alt', '#dc3545', 10),
('wallet', 'Wallet Credentials', 'Private keys and wallet addresses for blockchain wallets', 'fa-wallet', '#6f42c1', 20),
('exchange', 'Exchange API Keys', 'API credentials for centralized exchanges (Binance, Bybit)', 'fa-exchange-alt', '#fd7e14', 30),
('database', 'Database & Cache', 'Database and Redis connection credentials', 'fa-database', '#20c997', 40),
('api', 'Third-Party APIs', 'API keys for external services (Helius, Jupiter, GoPlus)', 'fa-plug', '#0d6efd', 50),
('notification', 'Notifications', 'Telegram, Discord, Email, and Twitter credentials', 'fa-bell', '#ffc107', 60),
('general', 'General', 'Other miscellaneous credentials', 'fa-key', '#6c757d', 100)
ON CONFLICT (category) DO NOTHING;

-- ============================================================================
-- SEED DATA - Credential Definitions (without values)
-- Defines what credentials exist, but NOT their values
-- ============================================================================

-- Security & Encryption
INSERT INTO secure_credentials (key_name, display_name, description, category, subcategory, module, is_required, is_sensitive, encrypted_value) VALUES
('ENCRYPTION_KEY', 'Master Encryption Key', 'Fernet key for encrypting/decrypting sensitive data', 'security', 'encryption', 'all', TRUE, TRUE, 'PLACEHOLDER'),
('JWT_SECRET', 'JWT Secret', 'Secret for signing JWT tokens', 'security', 'auth', 'all', FALSE, TRUE, 'PLACEHOLDER'),
('SESSION_SECRET', 'Session Secret', 'Secret for session management', 'security', 'auth', 'all', FALSE, TRUE, 'PLACEHOLDER')
ON CONFLICT (key_name) DO NOTHING;

-- Wallet Credentials
INSERT INTO secure_credentials (key_name, display_name, description, category, subcategory, module, is_required, is_sensitive, encrypted_value) VALUES
('PRIVATE_KEY', 'EVM Private Key', 'Encrypted private key for EVM chains (Ethereum, BSC, etc.)', 'wallet', 'evm', 'dex', TRUE, TRUE, 'PLACEHOLDER'),
('WALLET_ADDRESS', 'EVM Wallet Address', 'Public wallet address for EVM chains', 'wallet', 'evm', 'dex', TRUE, FALSE, 'PLACEHOLDER'),
('SOLANA_PRIVATE_KEY', 'Solana Private Key (DEX)', 'Encrypted private key for Solana DEX trading', 'wallet', 'solana', 'dex', FALSE, TRUE, 'PLACEHOLDER'),
('SOLANA_WALLET', 'Solana Wallet (DEX)', 'Public wallet address for Solana DEX trading', 'wallet', 'solana', 'dex', FALSE, FALSE, 'PLACEHOLDER'),
('SOLANA_MODULE_PRIVATE_KEY', 'Solana Private Key (Module)', 'Encrypted private key for Solana Module', 'wallet', 'solana', 'solana', FALSE, TRUE, 'PLACEHOLDER'),
('SOLANA_MODULE_WALLET', 'Solana Wallet (Module)', 'Public wallet address for Solana Module', 'wallet', 'solana', 'solana', FALSE, FALSE, 'PLACEHOLDER'),
('FLASHBOTS_SIGNING_KEY', 'Flashbots Signing Key', 'Private key for signing Flashbots bundles', 'wallet', 'flashbots', 'dex', FALSE, TRUE, 'PLACEHOLDER')
ON CONFLICT (key_name) DO NOTHING;

-- Exchange API Keys
INSERT INTO secure_credentials (key_name, display_name, description, category, subcategory, module, is_required, is_sensitive, encrypted_value) VALUES
('BINANCE_API_KEY', 'Binance API Key (Mainnet)', 'Binance mainnet API key', 'exchange', 'binance', 'futures', FALSE, TRUE, 'PLACEHOLDER'),
('BINANCE_API_SECRET', 'Binance API Secret (Mainnet)', 'Binance mainnet API secret', 'exchange', 'binance', 'futures', FALSE, TRUE, 'PLACEHOLDER'),
('BINANCE_TESTNET_API_KEY', 'Binance API Key (Testnet)', 'Binance testnet API key', 'exchange', 'binance', 'futures', FALSE, TRUE, 'PLACEHOLDER'),
('BINANCE_TESTNET_API_SECRET', 'Binance API Secret (Testnet)', 'Binance testnet API secret', 'exchange', 'binance', 'futures', FALSE, TRUE, 'PLACEHOLDER'),
('BYBIT_API_KEY', 'Bybit API Key (Mainnet)', 'Bybit mainnet API key', 'exchange', 'bybit', 'futures', FALSE, TRUE, 'PLACEHOLDER'),
('BYBIT_API_SECRET', 'Bybit API Secret (Mainnet)', 'Bybit mainnet API secret', 'exchange', 'bybit', 'futures', FALSE, TRUE, 'PLACEHOLDER'),
('BYBIT_TESTNET_API_KEY', 'Bybit API Key (Testnet)', 'Bybit testnet API key', 'exchange', 'bybit', 'futures', FALSE, TRUE, 'PLACEHOLDER'),
('BYBIT_TESTNET_API_SECRET', 'Bybit API Secret (Testnet)', 'Bybit testnet API secret', 'exchange', 'bybit', 'futures', FALSE, TRUE, 'PLACEHOLDER')
ON CONFLICT (key_name) DO NOTHING;

-- Database & Cache
INSERT INTO secure_credentials (key_name, display_name, description, category, subcategory, module, is_required, is_sensitive, encrypted_value) VALUES
('DB_PASSWORD', 'Database Password', 'PostgreSQL database password', 'database', 'postgres', 'all', TRUE, TRUE, 'PLACEHOLDER'),
('REDIS_PASSWORD', 'Redis Password', 'Redis cache password', 'database', 'redis', 'all', FALSE, TRUE, 'PLACEHOLDER')
ON CONFLICT (key_name) DO NOTHING;

-- Third-Party APIs
INSERT INTO secure_credentials (key_name, display_name, description, category, subcategory, module, is_required, is_sensitive, encrypted_value) VALUES
('GOPLUS_API_KEY', 'GoPlus API Key', 'API key for GoPlus token security checks', 'api', 'goplus', 'dex', FALSE, TRUE, 'PLACEHOLDER'),
('1INCH_API_KEY', '1inch API Key', 'API key for 1inch DEX aggregator', 'api', '1inch', 'dex', FALSE, TRUE, 'PLACEHOLDER'),
('HELIUS_API_KEY', 'Helius API Key', 'API key for Helius Solana RPC', 'api', 'helius', 'solana', FALSE, TRUE, 'PLACEHOLDER'),
('JUPITER_API_KEY', 'Jupiter API Key', 'API key for Jupiter aggregator', 'api', 'jupiter', 'solana', FALSE, TRUE, 'PLACEHOLDER'),
('ETHERSCAN_API_KEY', 'Etherscan API Key', 'API key for Etherscan', 'api', 'etherscan', 'copytrading', FALSE, TRUE, 'PLACEHOLDER')
ON CONFLICT (key_name) DO NOTHING;

-- Notification Credentials
INSERT INTO secure_credentials (key_name, display_name, description, category, subcategory, module, is_required, is_sensitive, encrypted_value) VALUES
('TELEGRAM_BOT_TOKEN', 'Telegram Bot Token', 'Token for Telegram notification bot', 'notification', 'telegram', 'all', FALSE, TRUE, 'PLACEHOLDER'),
('TELEGRAM_CHAT_ID', 'Telegram Chat ID', 'Chat ID for Telegram notifications', 'notification', 'telegram', 'all', FALSE, FALSE, 'PLACEHOLDER'),
('DISCORD_WEBHOOK_URL', 'Discord Webhook URL', 'Webhook URL for Discord notifications', 'notification', 'discord', 'all', FALSE, TRUE, 'PLACEHOLDER'),
('TWITTER_API_KEY', 'Twitter API Key', 'Twitter API key for sentiment analysis', 'notification', 'twitter', 'ai', FALSE, TRUE, 'PLACEHOLDER'),
('TWITTER_API_SECRET', 'Twitter API Secret', 'Twitter API secret', 'notification', 'twitter', 'ai', FALSE, TRUE, 'PLACEHOLDER'),
('TWITTER_BEARER_TOKEN', 'Twitter Bearer Token', 'Twitter bearer token', 'notification', 'twitter', 'ai', FALSE, TRUE, 'PLACEHOLDER'),
('EMAIL_PASSWORD', 'Email Password', 'Password for email notifications', 'notification', 'email', 'all', FALSE, TRUE, 'PLACEHOLDER'),
('EMAIL_FROM', 'Email From Address', 'Sender email address', 'notification', 'email', 'all', FALSE, FALSE, 'PLACEHOLDER'),
('EMAIL_TO', 'Email To Address', 'Recipient email address', 'notification', 'email', 'all', FALSE, FALSE, 'PLACEHOLDER')
ON CONFLICT (key_name) DO NOTHING;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_secure_creds_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_secure_creds_updated_at ON secure_credentials;
CREATE TRIGGER update_secure_creds_updated_at
    BEFORE UPDATE ON secure_credentials
    FOR EACH ROW
    EXECUTE FUNCTION update_secure_creds_timestamp();

-- Access count update trigger
CREATE OR REPLACE FUNCTION update_credential_access()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE secure_credentials
    SET last_accessed_at = NOW(),
        access_count = access_count + 1
    WHERE id = NEW.credential_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_credential_access ON credential_access_log;
CREATE TRIGGER update_credential_access
    AFTER INSERT ON credential_access_log
    FOR EACH ROW
    WHEN (NEW.success = TRUE AND NEW.access_type = 'read')
    EXECUTE FUNCTION update_credential_access();

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE secure_credentials IS 'Encrypted storage for sensitive credentials with audit tracking';
COMMENT ON TABLE credential_access_log IS 'Audit log for all credential access';
COMMENT ON TABLE credential_categories IS 'Reference table for credential categories';

COMMENT ON COLUMN secure_credentials.encrypted_value IS 'Fernet-encrypted value. Decryption requires external ENCRYPTION_KEY';
COMMENT ON COLUMN secure_credentials.value_hash IS 'SHA256 hash for verifying integrity without decryption';
COMMENT ON COLUMN secure_credentials.is_sensitive IS 'If true, value should never be logged or displayed in full';
