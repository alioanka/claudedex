-- Migration 127: execution_gateway — config seeds + sends-audit table
--
-- Shared EVM send-policy service (modules/execution_gateway/). The execution
-- sibling of pool_engine: pool_engine answers 'which RPC do I READ from',
-- the gateway answers 'how do I SEND safely' (private order flow with
-- public-RPC fallback, centralized nonce + gas policy, should_skip_live
-- asserted at the send boundary). Applying this migration changes NO
-- behavior: every private_send_enabled_* flag defaults to false, so even a
-- module already wired to the gateway keeps sending via the public mempool
-- exactly as today. No module is rewired by this migration.
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    -- Per-chain private order-flow switches: ALL OFF by default.
    ('execution_gateway', 'private_send_enabled_ethereum', 'false', 'bool',
     'Route Ethereum mainnet sends through the private relay RPC (Flashbots '
     'Protect / MEV-Blocker style). false = public mempool, today''s behavior.',
     NOW(), NOW()),

    ('execution_gateway', 'private_send_enabled_bsc', 'false', 'bool',
     'Route BSC sends through a private relay RPC when one is configured '
     '(e.g. bloXroute). Default off.', NOW(), NOW()),

    ('execution_gateway', 'private_send_enabled_base', 'false', 'bool',
     'Route Base sends privately. Base is sequencer-ordered (no public '
     'mempool to sandwich from), so this should usually stay off.',
     NOW(), NOW()),

    ('execution_gateway', 'private_send_enabled_arbitrum', 'false', 'bool',
     'Route Arbitrum sends privately. Sequencer-ordered; usually stays off.',
     NOW(), NOW()),

    ('execution_gateway', 'private_send_enabled_polygon', 'false', 'bool',
     'Route Polygon sends privately when a relay is configured. Default off.',
     NOW(), NOW()),

    -- Private relay URLs (URL present + flag off = still public; the flag
    -- is the single switch). Ethereum falls back to env FLASHBOTS_RPC when
    -- this row is empty.
    ('execution_gateway', 'private_rpc_url_ethereum', 'https://rpc.flashbots.net/fast', 'string',
     'Private order-flow RPC for Ethereum mainnet. Empty = use env '
     'FLASHBOTS_RPC. Inert while private_send_enabled_ethereum=false.',
     NOW(), NOW()),

    ('execution_gateway', 'private_rpc_url_bsc', '', 'string',
     'Private order-flow RPC for BSC (operator-provisioned, e.g. bloXroute). '
     'Empty = no private route on BSC.', NOW(), NOW()),

    ('execution_gateway', 'private_rpc_url_base', '', 'string',
     'Private order-flow RPC for Base. Empty = no private route.', NOW(), NOW()),

    ('execution_gateway', 'private_rpc_url_arbitrum', '', 'string',
     'Private order-flow RPC for Arbitrum. Empty = no private route.', NOW(), NOW()),

    ('execution_gateway', 'private_rpc_url_polygon', '', 'string',
     'Private order-flow RPC for Polygon. Empty = no private route.', NOW(), NOW()),

    -- Route behavior
    ('execution_gateway', 'public_fallback_enabled', 'true', 'bool',
     'When a private send fails, fall back to the pool_engine public RPC. '
     'true = fill rate over secrecy (recommended); false = fail closed.',
     NOW(), NOW()),

    ('execution_gateway', 'private_send_timeout_seconds', '30', 'int',
     'HTTP timeout for the private relay submission before fallback/error.',
     NOW(), NOW()),

    ('execution_gateway', 'private_min_notional_usd', '0', 'float',
     'Sends below this USD notional skip the private route (latency not '
     'worth it on dust). 0 = no floor. Callers may override per call.',
     NOW(), NOW()),

    -- Gas policy knobs (consumed by core/gas_policy.compute_gas)
    ('execution_gateway', 'priority_fee_gwei_default', '1.5', 'float',
     'Default EIP-1559 priority-fee floor in gwei; per-chain override via '
     'priority_fee_gwei_<chain> rows.', NOW(), NOW()),

    ('execution_gateway', 'priority_fee_gwei_ethereum', '1.5', 'float',
     'Ethereum mainnet priority-fee floor in gwei.', NOW(), NOW()),

    ('execution_gateway', 'max_fee_multiplier', '2.0', 'float',
     'maxFeePerGas = base_fee * this + priority (headroom for base-fee '
     'spikes between quote and inclusion).', NOW(), NOW()),

    ('execution_gateway', 'gas_ceiling_gwei', '150', 'float',
     'Hard clamp on maxFeePerGas in gwei. A capped quote may be slow to '
     'include; that is the honest outcome, the gateway never exceeds it.',
     NOW(), NOW()),

    ('execution_gateway', 'replacement_bump_pct', '15', 'float',
     'Fee bump percent for replacing a stuck tx (floored at the node '
     'minimum of 12.5).', NOW(), NOW()),

    -- Diagnostics
    ('execution_gateway', 'audit_enabled', 'true', 'bool',
     'Write one row per gateway send (incl. simulated) to '
     'execution_gateway_sends. Fail-soft: audit errors never block a send.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- One row per gateway send attempt (simulated ones included so DRY_RUN
-- routing can be inspected before any live flip).
CREATE TABLE IF NOT EXISTS execution_gateway_sends (
    id                        BIGSERIAL PRIMARY KEY,
    module                    TEXT NOT NULL,
    chain                     TEXT NOT NULL,
    direction                 TEXT,             -- 'entry' | 'exit'
    route                     TEXT,             -- 'private' | 'public'
    status                    TEXT NOT NULL,    -- 'ok' | 'error'
    tx_hash                   TEXT,
    fallback_used             BOOLEAN NOT NULL DEFAULT FALSE,
    simulated                 BOOLEAN NOT NULL DEFAULT FALSE,
    nonce                     BIGINT,
    max_fee_per_gas           NUMERIC,
    max_priority_fee_per_gas  NUMERIC,
    notional_usd              NUMERIC,
    tag                       TEXT,
    error                     TEXT,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_egw_sends_module_created ON execution_gateway_sends(module, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_egw_sends_created ON execution_gateway_sends(created_at DESC);

COMMIT;
