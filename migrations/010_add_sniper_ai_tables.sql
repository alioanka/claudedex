-- Migration: Add dedicated Sniper and AI trade tables
-- Created: 2025-12-19
-- Each module should have its own trade tables for proper data isolation

-- ============================================================================
-- SNIPER TRADES TABLE
-- ============================================================================
-- Drop existing table if it has wrong schema (from partial migration)
DROP TABLE IF EXISTS sniper_trades CASCADE;

CREATE TABLE sniper_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    token_address VARCHAR(100) NOT NULL,
    chain VARCHAR(20) NOT NULL DEFAULT 'solana',

    -- Trade details
    side VARCHAR(10) NOT NULL DEFAULT 'buy',
    entry_price NUMERIC NOT NULL DEFAULT 0,
    exit_price NUMERIC,
    amount NUMERIC NOT NULL DEFAULT 0,

    -- USD values
    entry_usd NUMERIC NOT NULL DEFAULT 0,
    exit_usd NUMERIC,
    profit_loss NUMERIC,
    profit_loss_pct NUMERIC,

    -- Native token info
    native_token VARCHAR(10),  -- SOL or ETH
    native_price_at_entry NUMERIC,
    native_price_at_exit NUMERIC,
    trade_amount_native NUMERIC,

    -- Safety info
    safety_score INT,
    safety_rating VARCHAR(20),
    is_honeypot BOOLEAN DEFAULT FALSE,
    buy_tax NUMERIC,
    sell_tax NUMERIC,
    liquidity_usd NUMERIC,

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    exit_reason VARCHAR(50),
    is_simulated BOOLEAN NOT NULL DEFAULT TRUE,

    -- Timestamps
    entry_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    exit_timestamp TIMESTAMP,

    -- Transaction info
    entry_tx_hash VARCHAR(100),
    exit_tx_hash VARCHAR(100),

    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sniper_trades_timestamp ON sniper_trades(entry_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sniper_trades_status ON sniper_trades(status);
CREATE INDEX IF NOT EXISTS idx_sniper_trades_chain ON sniper_trades(chain);
CREATE INDEX IF NOT EXISTS idx_sniper_trades_token ON sniper_trades(token_address);

-- ============================================================================
-- SNIPER POSITIONS TABLE (for active snipes)
-- ============================================================================
DROP TABLE IF EXISTS sniper_positions CASCADE;

CREATE TABLE sniper_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    token_address VARCHAR(100) NOT NULL,
    chain VARCHAR(20) NOT NULL DEFAULT 'solana',
    side VARCHAR(10) NOT NULL DEFAULT 'buy',
    entry_price NUMERIC NOT NULL,
    current_price NUMERIC,
    amount NUMERIC NOT NULL,
    entry_usd NUMERIC NOT NULL DEFAULT 0,
    unrealized_pnl NUMERIC DEFAULT 0,
    unrealized_pnl_pct NUMERIC DEFAULT 0,
    safety_score INT,
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    entry_tx_hash VARCHAR(100),
    opened_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sniper_positions_status ON sniper_positions(status);
CREATE INDEX IF NOT EXISTS idx_sniper_positions_chain ON sniper_positions(chain);

-- ============================================================================
-- AI TRADES TABLE
-- ============================================================================
DROP TABLE IF EXISTS ai_trades CASCADE;

CREATE TABLE ai_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    token_symbol VARCHAR(20) NOT NULL,
    token_address VARCHAR(100),
    chain VARCHAR(20) NOT NULL DEFAULT 'ethereum',

    -- Trade details
    side VARCHAR(10) NOT NULL DEFAULT 'buy',
    entry_price NUMERIC NOT NULL DEFAULT 0,
    exit_price NUMERIC,
    amount NUMERIC NOT NULL DEFAULT 0,

    -- USD values
    entry_usd NUMERIC NOT NULL DEFAULT 0,
    exit_usd NUMERIC,
    profit_loss NUMERIC,
    profit_loss_pct NUMERIC,

    -- AI specific
    sentiment_score NUMERIC,
    confidence_score NUMERIC,
    ai_provider VARCHAR(20),  -- openai, claude, etc

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    exit_reason VARCHAR(50),
    is_simulated BOOLEAN NOT NULL DEFAULT TRUE,

    -- Timestamps
    entry_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    exit_timestamp TIMESTAMP,

    -- Transaction info
    entry_order_id VARCHAR(100),
    exit_order_id VARCHAR(100),

    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ai_trades_timestamp ON ai_trades(entry_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ai_trades_status ON ai_trades(status);
CREATE INDEX IF NOT EXISTS idx_ai_trades_symbol ON ai_trades(token_symbol);

-- ============================================================================
-- AI POSITIONS TABLE (for active AI trades)
-- ============================================================================
DROP TABLE IF EXISTS ai_positions CASCADE;

CREATE TABLE ai_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    token_symbol VARCHAR(20) NOT NULL,
    chain VARCHAR(20) NOT NULL DEFAULT 'ethereum',
    side VARCHAR(10) NOT NULL DEFAULT 'buy',
    entry_price NUMERIC NOT NULL,
    current_price NUMERIC,
    amount NUMERIC NOT NULL,
    entry_usd NUMERIC NOT NULL DEFAULT 0,
    unrealized_pnl NUMERIC DEFAULT 0,
    unrealized_pnl_pct NUMERIC DEFAULT 0,
    sentiment_score NUMERIC,
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    entry_order_id VARCHAR(100),
    opened_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ai_positions_status ON ai_positions(status);

-- ============================================================================
-- SUMMARY VIEWS
-- ============================================================================

-- Sniper Trading Summary
CREATE OR REPLACE VIEW sniper_trading_summary AS
SELECT
    is_simulated,
    chain,
    COUNT(*) as total_trades,
    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN profit_loss <= 0 THEN 1 ELSE 0 END) as losing_trades,
    CASE WHEN COUNT(*) > 0
        THEN ROUND(SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100, 2)
        ELSE 0
    END as win_rate,
    COALESCE(SUM(profit_loss), 0) as total_pnl,
    COALESCE(MAX(profit_loss), 0) as best_trade,
    COALESCE(MIN(profit_loss), 0) as worst_trade,
    COALESCE(AVG(safety_score), 0) as avg_safety_score
FROM sniper_trades
WHERE status = 'closed'
GROUP BY is_simulated, chain;

-- AI Trading Summary
CREATE OR REPLACE VIEW ai_trading_summary AS
SELECT
    is_simulated,
    COUNT(*) as total_trades,
    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN profit_loss <= 0 THEN 1 ELSE 0 END) as losing_trades,
    CASE WHEN COUNT(*) > 0
        THEN ROUND(SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100, 2)
        ELSE 0
    END as win_rate,
    COALESCE(SUM(profit_loss), 0) as total_pnl,
    COALESCE(MAX(profit_loss), 0) as best_trade,
    COALESCE(MIN(profit_loss), 0) as worst_trade,
    COALESCE(AVG(sentiment_score), 0) as avg_sentiment
FROM ai_trades
WHERE status = 'closed'
GROUP BY is_simulated;
