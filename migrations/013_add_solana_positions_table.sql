-- Migration: Replace solana_positions table for open-position persistence
-- Description: Old solana_positions schema (from 008) was created but never wired
-- into solana_engine.py — recreate with the columns the engine actually needs
-- so open positions survive restarts and can be reconciled against on-chain
-- SPL balances.
-- Date: 2026-05-12

-- Drop the unused legacy table (no production code reads/writes it).
DROP TABLE IF EXISTS solana_positions;

-- Solana open positions (mirrors active_positions dict in solana_engine.py).
-- token_mint is UNIQUE so the open path is naturally idempotent under retry:
-- ON CONFLICT (token_mint) DO UPDATE keeps the latest write.
CREATE TABLE IF NOT EXISTS solana_positions (
    position_id   TEXT PRIMARY KEY,
    token_mint    TEXT NOT NULL UNIQUE,
    token_symbol  TEXT,
    strategy      TEXT,
    entry_price   DOUBLE PRECISION,
    amount        DOUBLE PRECISION,         -- token amount (human-readable)
    value_sol     DOUBLE PRECISION,
    stop_loss     DOUBLE PRECISION,
    take_profit   DOUBLE PRECISION,
    is_simulated  BOOLEAN NOT NULL DEFAULT FALSE,
    tx_signature  TEXT,
    opened_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata      JSONB
);

CREATE INDEX IF NOT EXISTS idx_solana_positions_opened_at ON solana_positions (opened_at);
CREATE INDEX IF NOT EXISTS idx_solana_positions_mint ON solana_positions (token_mint);

-- Note: Migration tracking is handled by MigrationManager (schema_migrations table)
