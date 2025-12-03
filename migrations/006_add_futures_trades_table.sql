-- Migration: Add Futures trades and positions tables for persistence
-- Created: 2025-11-29

-- Futures Trades (closed positions)
CREATE TABLE IF NOT EXISTS futures_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'long' or 'short'
    entry_price NUMERIC NOT NULL,
    exit_price NUMERIC NOT NULL,
    size NUMERIC NOT NULL,
    notional_value NUMERIC NOT NULL,
    leverage INT NOT NULL DEFAULT 1,
    pnl NUMERIC NOT NULL,
    pnl_pct NUMERIC NOT NULL,
    fees NUMERIC NOT NULL DEFAULT 0,
    net_pnl NUMERIC NOT NULL,
    exit_reason VARCHAR(50),
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP NOT NULL,
    duration_seconds INT NOT NULL DEFAULT 0,
    is_simulated BOOLEAN NOT NULL DEFAULT TRUE,
    exchange VARCHAR(20) NOT NULL DEFAULT 'binance',
    network VARCHAR(20) NOT NULL DEFAULT 'testnet',
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_futures_trades_symbol ON futures_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_futures_trades_exit_time ON futures_trades(exit_time DESC);
CREATE INDEX IF NOT EXISTS idx_futures_trades_simulated ON futures_trades(is_simulated);

-- Futures Active Positions (for state recovery on restart)
CREATE TABLE IF NOT EXISTS futures_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL UNIQUE,
    side VARCHAR(10) NOT NULL,  -- 'long' or 'short'
    entry_price NUMERIC NOT NULL,
    current_price NUMERIC,
    size NUMERIC NOT NULL,
    notional_value NUMERIC NOT NULL,
    leverage INT NOT NULL DEFAULT 1,
    stop_loss NUMERIC,
    take_profit NUMERIC,
    unrealized_pnl NUMERIC DEFAULT 0,
    unrealized_pnl_pct NUMERIC DEFAULT 0,
    is_simulated BOOLEAN NOT NULL DEFAULT TRUE,
    exchange VARCHAR(20) NOT NULL DEFAULT 'binance',
    network VARCHAR(20) NOT NULL DEFAULT 'testnet',
    opened_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_futures_positions_symbol ON futures_positions(symbol);

-- View for futures trading summary
CREATE OR REPLACE VIEW futures_trading_summary AS
SELECT
    is_simulated,
    exchange,
    network,
    COUNT(*) as total_trades,
    SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN net_pnl <= 0 THEN 1 ELSE 0 END) as losing_trades,
    CASE WHEN COUNT(*) > 0
        THEN ROUND(SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100, 2)
        ELSE 0
    END as win_rate,
    COALESCE(SUM(net_pnl), 0) as total_pnl,
    COALESCE(SUM(fees), 0) as total_fees,
    COALESCE(AVG(CASE WHEN net_pnl > 0 THEN net_pnl END), 0) as avg_win,
    COALESCE(AVG(CASE WHEN net_pnl <= 0 THEN net_pnl END), 0) as avg_loss,
    COALESCE(MAX(net_pnl), 0) as best_trade,
    COALESCE(MIN(net_pnl), 0) as worst_trade,
    COALESCE(AVG(duration_seconds), 0) as avg_duration_seconds
FROM futures_trades
GROUP BY is_simulated, exchange, network;
