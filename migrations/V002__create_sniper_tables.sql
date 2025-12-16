CREATE TABLE IF NOT EXISTS sniper_targets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_address VARCHAR(100) NOT NULL,
    chain VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    liquidity_usd NUMERIC,
    detected_at TIMESTAMP DEFAULT NOW(),
    executed_at TIMESTAMP,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS sniper_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_id UUID REFERENCES sniper_targets(id),
    entry_price NUMERIC,
    exit_price NUMERIC,
    pnl NUMERIC,
    pnl_pct NUMERIC,
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);
