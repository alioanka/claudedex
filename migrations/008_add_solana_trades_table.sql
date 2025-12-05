-- Migration: Add Solana trades table for persistence
-- Description: Creates solana_trades table to persist Solana module trades across restarts
-- Date: 2025-12-05

-- Solana Trades (closed positions)
CREATE TABLE IF NOT EXISTS solana_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id VARCHAR(50),
    token_symbol VARCHAR(20) NOT NULL,
    token_mint VARCHAR(64) NOT NULL,
    strategy VARCHAR(20) NOT NULL,  -- 'jupiter', 'pumpfun', 'drift'
    side VARCHAR(10) NOT NULL DEFAULT 'long',  -- Solana trades are always long
    entry_price NUMERIC NOT NULL,
    exit_price NUMERIC NOT NULL,
    amount_sol NUMERIC NOT NULL,
    amount_tokens NUMERIC,
    pnl_sol NUMERIC NOT NULL,
    pnl_usd NUMERIC,
    pnl_pct NUMERIC NOT NULL,
    fees_sol NUMERIC NOT NULL DEFAULT 0,
    exit_reason VARCHAR(50),  -- 'tp', 'sl', 'time', 'manual', 'trailing_stop'
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP NOT NULL,
    duration_seconds INT NOT NULL DEFAULT 0,
    is_simulated BOOLEAN NOT NULL DEFAULT TRUE,
    sol_price_usd NUMERIC,  -- SOL price at trade time for USD conversion
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_solana_trades_token ON solana_trades(token_symbol);
CREATE INDEX IF NOT EXISTS idx_solana_trades_strategy ON solana_trades(strategy);
CREATE INDEX IF NOT EXISTS idx_solana_trades_exit_time ON solana_trades(exit_time DESC);
CREATE INDEX IF NOT EXISTS idx_solana_trades_simulated ON solana_trades(is_simulated);
CREATE INDEX IF NOT EXISTS idx_solana_trades_created ON solana_trades(created_at DESC);

-- Solana Active Positions (for state recovery on restart)
CREATE TABLE IF NOT EXISTS solana_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    position_id VARCHAR(50) UNIQUE NOT NULL,
    token_symbol VARCHAR(20) NOT NULL,
    token_mint VARCHAR(64) NOT NULL,
    strategy VARCHAR(20) NOT NULL,
    entry_price NUMERIC NOT NULL,
    current_price NUMERIC,
    amount_sol NUMERIC NOT NULL,
    amount_tokens NUMERIC,
    stop_loss_pct NUMERIC,
    take_profit_pct NUMERIC,
    unrealized_pnl_sol NUMERIC DEFAULT 0,
    unrealized_pnl_pct NUMERIC DEFAULT 0,
    is_simulated BOOLEAN NOT NULL DEFAULT TRUE,
    opened_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_solana_positions_token ON solana_positions(token_mint);

-- View for Solana trading summary
CREATE OR REPLACE VIEW solana_trading_summary AS
SELECT
    is_simulated,
    strategy,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl_sol > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN pnl_sol <= 0 THEN 1 ELSE 0 END) as losing_trades,
    CASE WHEN COUNT(*) > 0
        THEN ROUND(SUM(CASE WHEN pnl_sol > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100, 2)
        ELSE 0
    END as win_rate,
    COALESCE(SUM(pnl_sol), 0) as total_pnl_sol,
    COALESCE(SUM(pnl_usd), 0) as total_pnl_usd,
    COALESCE(SUM(fees_sol), 0) as total_fees,
    COALESCE(AVG(CASE WHEN pnl_sol > 0 THEN pnl_sol END), 0) as avg_win_sol,
    COALESCE(AVG(CASE WHEN pnl_sol <= 0 THEN pnl_sol END), 0) as avg_loss_sol,
    COALESCE(MAX(pnl_sol), 0) as best_trade_sol,
    COALESCE(MIN(pnl_sol), 0) as worst_trade_sol,
    COALESCE(AVG(duration_seconds), 0) as avg_duration_seconds
FROM solana_trades
GROUP BY is_simulated, strategy;

-- Record migration
INSERT INTO migrations (version, description, applied_at)
VALUES ('008', 'Add Solana trades and positions tables', NOW())
ON CONFLICT (version) DO NOTHING;
