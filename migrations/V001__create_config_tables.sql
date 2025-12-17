-- Configuration Settings Table
CREATE TABLE IF NOT EXISTS config_settings (
    config_type VARCHAR(50) NOT NULL,
    key VARCHAR(100) NOT NULL,
    value TEXT,
    value_type VARCHAR(20) DEFAULT 'string',
    description TEXT,
    is_editable BOOLEAN DEFAULT TRUE,
    requires_restart BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT NOW(),
    updated_by VARCHAR(50),
    PRIMARY KEY (config_type, key)
);

-- Configuration History Table
CREATE TABLE IF NOT EXISTS config_history (
    id SERIAL PRIMARY KEY,
    config_type VARCHAR(50) NOT NULL,
    key VARCHAR(100) NOT NULL,
    old_value TEXT,
    new_value TEXT,
    change_source VARCHAR(50),
    changed_by VARCHAR(50),
    changed_by_username VARCHAR(50),
    reason TEXT,
    ip_address VARCHAR(50),
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Sensitive Configuration Table (Encrypted)
CREATE TABLE IF NOT EXISTS config_sensitive (
    key VARCHAR(100) PRIMARY KEY,
    encrypted_value TEXT NOT NULL,
    description TEXT,
    last_rotated TIMESTAMP DEFAULT NOW(),
    rotation_interval_days INT DEFAULT 30,
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP DEFAULT NOW(),
    updated_by VARCHAR(50)
);
