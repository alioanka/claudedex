-- Migration: Add dedicated Arbitrage and Copy Trading trade tables
-- Created: 2025-12-19
-- Each module should have its own trade tables for proper data isolation

-- ============================================================================
-- ARBITRAGE TRADES TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS arbitrage_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    token_address VARCHAR(100) NOT NULL,
    chain VARCHAR(20) NOT NULL DEFAULT 'ethereum',

    -- DEX info
    buy_dex VARCHAR(50) NOT NULL,
    sell_dex VARCHAR(50) NOT NULL,

    -- Trade details
    side VARCHAR(10) NOT NULL DEFAULT 'buy',
    entry_price NUMERIC NOT NULL DEFAULT 0,
    exit_price NUMERIC NOT NULL DEFAULT 0,
    amount NUMERIC NOT NULL DEFAULT 0,
    amount_eth NUMERIC NOT NULL DEFAULT 0,

    -- USD values
    entry_usd NUMERIC NOT NULL DEFAULT 0,
    exit_usd NUMERIC NOT NULL DEFAULT 0,
    profit_loss NUMERIC NOT NULL DEFAULT 0,
    profit_loss_pct NUMERIC NOT NULL DEFAULT 0,
    spread_pct NUMERIC NOT NULL DEFAULT 0,

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'closed',
    is_simulated BOOLEAN NOT NULL DEFAULT TRUE,

    -- Timestamps
    entry_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    exit_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Metadata
    tx_hash VARCHAR(100),
    eth_price_at_trade NUMERIC,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_arbitrage_trades_timestamp ON arbitrage_trades(entry_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_arbitrage_trades_status ON arbitrage_trades(status);
CREATE INDEX IF NOT EXISTS idx_arbitrage_trades_dexes ON arbitrage_trades(buy_dex, sell_dex);

-- ============================================================================
-- ARBITRAGE POSITIONS TABLE (for tracking active arbitrage if needed)
-- ============================================================================
CREATE TABLE IF NOT EXISTS arbitrage_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    token_address VARCHAR(100) NOT NULL,
    chain VARCHAR(20) NOT NULL DEFAULT 'ethereum',
    buy_dex VARCHAR(50) NOT NULL,
    sell_dex VARCHAR(50) NOT NULL,
    entry_price NUMERIC NOT NULL,
    amount NUMERIC NOT NULL,
    entry_usd NUMERIC NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    opened_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- COPY TRADING TRADES TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS copytrading_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    token_address VARCHAR(100) NOT NULL,
    chain VARCHAR(20) NOT NULL DEFAULT 'solana',

    -- Source wallet info
    source_wallet VARCHAR(100) NOT NULL,
    source_tx VARCHAR(100),

    -- Trade details
    side VARCHAR(10) NOT NULL DEFAULT 'buy',
    entry_price NUMERIC NOT NULL DEFAULT 0,
    exit_price NUMERIC NOT NULL DEFAULT 0,
    amount NUMERIC NOT NULL DEFAULT 0,

    -- USD values
    entry_usd NUMERIC NOT NULL DEFAULT 0,
    exit_usd NUMERIC NOT NULL DEFAULT 0,
    profit_loss NUMERIC NOT NULL DEFAULT 0,
    profit_loss_pct NUMERIC NOT NULL DEFAULT 0,

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    is_simulated BOOLEAN NOT NULL DEFAULT TRUE,

    -- Timestamps
    entry_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    exit_timestamp TIMESTAMP,

    -- Metadata
    tx_hash VARCHAR(100),
    native_price_at_trade NUMERIC,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_copytrading_trades_timestamp ON copytrading_trades(entry_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_copytrading_trades_status ON copytrading_trades(status);
CREATE INDEX IF NOT EXISTS idx_copytrading_trades_wallet ON copytrading_trades(source_wallet);
CREATE INDEX IF NOT EXISTS idx_copytrading_trades_chain ON copytrading_trades(chain);

-- ============================================================================
-- COPY TRADING POSITIONS TABLE (for active positions)
-- ============================================================================
CREATE TABLE IF NOT EXISTS copytrading_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    token_address VARCHAR(100) NOT NULL,
    chain VARCHAR(20) NOT NULL DEFAULT 'solana',
    source_wallet VARCHAR(100) NOT NULL,
    side VARCHAR(10) NOT NULL DEFAULT 'buy',
    entry_price NUMERIC NOT NULL,
    current_price NUMERIC,
    amount NUMERIC NOT NULL,
    entry_usd NUMERIC NOT NULL DEFAULT 0,
    unrealized_pnl NUMERIC DEFAULT 0,
    unrealized_pnl_pct NUMERIC DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    opened_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_copytrading_positions_wallet ON copytrading_positions(source_wallet);
CREATE INDEX IF NOT EXISTS idx_copytrading_positions_status ON copytrading_positions(status);

-- ============================================================================
-- SUMMARY VIEWS
-- ============================================================================

-- Arbitrage Trading Summary
CREATE OR REPLACE VIEW arbitrage_trading_summary AS
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
    COALESCE(AVG(spread_pct), 0) as avg_spread,
    COALESCE(MAX(profit_loss), 0) as best_trade,
    COALESCE(MIN(profit_loss), 0) as worst_trade
FROM arbitrage_trades
GROUP BY is_simulated;

-- Copy Trading Summary
CREATE OR REPLACE VIEW copytrading_summary AS
SELECT
    is_simulated,
    chain,
    COUNT(*) as total_trades,
    COUNT(DISTINCT source_wallet) as unique_wallets,
    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN profit_loss <= 0 THEN 1 ELSE 0 END) as losing_trades,
    CASE WHEN COUNT(*) > 0
        THEN ROUND(SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100, 2)
        ELSE 0
    END as win_rate,
    COALESCE(SUM(profit_loss), 0) as total_pnl,
    COALESCE(MAX(profit_loss), 0) as best_trade,
    COALESCE(MIN(profit_loss), 0) as worst_trade
FROM copytrading_trades
GROUP BY is_simulated, chain;
