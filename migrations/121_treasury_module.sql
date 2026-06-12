-- Migration 121: treasury — config seeds + treasury_snapshots table
--
-- Phase 1 OBSERVE-ONLY wallet/gas/inventory observer (modules/treasury/).
-- Polls native + key token balances of the bot''s public wallet addresses,
-- reconciles against the trade ledgers, writes snapshots, and logs alerts
-- (gas_low / hot_wallet_high / reconcile_drift / ledger_unbacked). It NEVER
-- signs, NEVER transfers, and NEVER touches logs/.killswitch. Applying this
-- migration changes NO behavior: the module only runs when
-- TREASURY_MODULE_ENABLED=true (default false), and even then it only READS
-- balances and WRITES snapshot rows. All defaults are observe-safe.
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('treasury', 'poll_interval_seconds', '300', 'int',
     'Seconds between balance-observation ticks (slow loop; default 5 min).',
     NOW(), NOW()),

    ('treasury', 'evm_chains', 'ethereum,arbitrum,base', 'string',
     'Comma-separated EVM chains on which the shared EVM EOA (WALLET_ADDRESS) '
     'is observed. Must match pool_engine provider types <CHAIN>_RPC.',
     NOW(), NOW()),

    ('treasury', 'gas_floor_default', '0.005', 'float',
     'Default per-chain native low-water mark (native units). Below this the '
     'module logs a gas_low alert; severity escalates to ERROR when the trade '
     'ledgers show open LIVE positions on that chain (exit gas starvation).',
     NOW(), NOW()),

    ('treasury', 'gas_floor_ethereum', '0.02', 'float',
     'Ethereum mainnet native low-water mark in ETH (mainnet exits cost the '
     'most gas; keep this above one worst-case emergency exit).', NOW(), NOW()),

    ('treasury', 'gas_floor_arbitrum', '0.005', 'float',
     'Arbitrum native low-water mark in ETH.', NOW(), NOW()),

    ('treasury', 'gas_floor_base', '0.005', 'float',
     'Base native low-water mark in ETH.', NOW(), NOW()),

    ('treasury', 'gas_floor_solana', '0.05', 'float',
     'Solana native low-water mark in SOL (covers fees + Jito tips + ATA rent '
     'for an emergency exit).', NOW(), NOW()),

    ('treasury', 'hot_wallet_ceiling_usd', '1000', 'float',
     'Stablecoin balance (USD) above which the module logs a hot_wallet_high '
     'alert: unswept realized profit sitting as hot-wallet risk. Observation '
     'only — Phase 1 never sweeps.', NOW(), NOW()),

    ('treasury', 'reconcile_drift_usd', '250', 'float',
     'Absolute change in a wallet''s stablecoin balance between two snapshots '
     'above which a reconcile_drift alert is logged (unexplained inflow/outflow '
     'to reconcile against the trade ledgers).', NOW(), NOW()),

    ('treasury', 'native_dust', '0.0005', 'float',
     'Native balance at or below this counts as an empty wallet for the '
     'ledger_unbacked check (ledgers imply open LIVE positions but the wallet '
     'holds nothing).', NOW(), NOW()),

    ('treasury', 'extra_wallets', '[]', 'json',
     'Additional PUBLIC addresses to observe, e.g. '
     '[{"chain": "base", "address": "0x...", "label": "cold"}]. '
     'Addresses only — the module never reads or holds keys.', NOW(), NOW()),

    ('treasury', 'evm_tokens',
     '{"ethereum": [{"symbol": "USDC", "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "decimals": 6, "stable": true}], '
     '"arbitrum": [{"symbol": "USDC", "address": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831", "decimals": 6, "stable": true}], '
     '"base": [{"symbol": "USDC", "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913", "decimals": 6, "stable": true}]}',
     'json',
     'Key ERC20 tokens observed per EVM chain (canonical USDC by default). '
     'stable=true balances count toward hot_wallet_ceiling_usd.', NOW(), NOW()),

    ('treasury', 'solana_tokens',
     '{"solana": [{"symbol": "USDC", "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "decimals": 6, "stable": true}]}',
     'json',
     'Key SPL tokens observed on Solana (canonical USDC mint by default). '
     'stable=true balances count toward hot_wallet_ceiling_usd.', NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- One row per wallet per tick (full observation history; the latest row per
-- (chain, address) is the current state; alerts are embedded for audit).
CREATE TABLE IF NOT EXISTS treasury_snapshots (
    id                     BIGSERIAL PRIMARY KEY,
    wallet_group           TEXT NOT NULL,            -- 'evm' | 'dex_solana' | 'solana_module' | extra label
    chain                  TEXT NOT NULL,
    address                TEXT NOT NULL,
    native_balance         DOUBLE PRECISION,         -- native units (ETH/SOL/...)
    native_symbol          TEXT,
    tokens                 JSONB,                    -- {symbol: balance}
    stable_usd             DOUBLE PRECISION,         -- sum of stable-marked token balances
    ledger_open_positions  INTEGER NOT NULL DEFAULT 0,
    ledger_exposure_usd    DOUBLE PRECISION NOT NULL DEFAULT 0,
    ledger_detail          JSONB,                    -- {module: {open, usd}}
    alerts                 JSONB,                    -- [{type, severity, message}]
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_treasury_snapshots_wallet_created
    ON treasury_snapshots(chain, address, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_treasury_snapshots_created
    ON treasury_snapshots(created_at DESC);

COMMIT;
