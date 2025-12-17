-- Arbitrage Tables
CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_address VARCHAR(100),
    dex_a VARCHAR(50),
    dex_b VARCHAR(50),
    spread_pct NUMERIC,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Copy Trading Tables
CREATE TABLE IF NOT EXISTS copy_wallets (
    address VARCHAR(100) PRIMARY KEY,
    label VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    performance_score NUMERIC
);

CREATE TABLE IF NOT EXISTS copy_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    wallet_address VARCHAR(100) REFERENCES copy_wallets(address),
    tx_hash VARCHAR(100),
    token_address VARCHAR(100),
    side VARCHAR(10),
    timestamp TIMESTAMP DEFAULT NOW()
);
