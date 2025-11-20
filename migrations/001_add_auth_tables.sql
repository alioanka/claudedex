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
