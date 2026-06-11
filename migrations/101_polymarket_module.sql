-- Migration 101: Polymarket module — config seeds + shadow-record tables
--
-- New prediction-market module (modules/polymarket/). Trading-capable but
-- LIVE execution is GATED OFF by default (shadow_mode=true AND
-- live_execution_enabled=false), mirroring the arbitrage module: every
-- detected opportunity is recorded as a SIMULATED row (is_simulated=true);
-- the live CLOB path (py-clob-client) is only reached when shadow_mode=false
-- AND live_execution_enabled=true AND not should_skip_live AND
-- RiskManager.validate_trade passes. Applying this migration changes NO
-- behavior on its own — the module only runs when POLYMARKET_MODULE_ENABLED=true.
--
-- Data source: read-only Gamma API (https://gamma-api.polymarket.com) — no key,
-- no on-chain risk. Network: Polygon PoS (chainId 137), USDC.e collateral.
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

-- ── config seeds (defaults mirror the in-code config.get(...) fallbacks) ──
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('polymarket_config', 'shadow_mode', 'true', 'bool',
     'Shadow mode (default ON): record every detected opportunity as a '
     'simulated polymarket_trades row (is_simulated=true); never reach the '
     'live CLOB order path. Turn OFF only when ready to consider live USDC '
     'execution.', NOW(), NOW()),

    ('polymarket_config', 'live_execution_enabled', 'false', 'bool',
     'Explicit LIVE opt-in: even with shadow_mode off and module dry_run '
     'false, placing a real CLOB order requires this true. Fail-safe '
     'default false.', NOW(), NOW()),

    ('polymarket_config', 'gamma_base_url', 'https://gamma-api.polymarket.com', 'string',
     'Read-only Polymarket Gamma API base (markets + events, no auth).',
     NOW(), NOW()),

    ('polymarket_config', 'gamma_max_requests_per_minute', '30', 'int',
     'Client-side rate cap on Gamma API requests per minute (politeness).',
     NOW(), NOW()),

    ('polymarket_config', 'clob_base_url', 'https://clob.polymarket.com', 'string',
     'Polymarket CLOB API base (live order path via py-clob-client; only '
     'used when live_execution_enabled=true).', NOW(), NOW()),

    ('polymarket_config', 'chain_id', '137', 'int',
     'EVM chain id for execution/signing. Polygon PoS = 137.', NOW(), NOW()),

    ('polymarket_config', 'usdc_address', '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174', 'string',
     'USDC.e collateral on Polygon (6 decimals). Operator-verifiable.',
     NOW(), NOW()),

    ('polymarket_config', 'ctf_address', '0x4D97DCd97eC945f40cF65F87097ACe5EA0476045', 'string',
     'Conditional Tokens Framework (ERC-1155) on Polygon. Operator-verifiable.',
     NOW(), NOW()),

    ('polymarket_config', 'ctf_exchange_address', '0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E', 'string',
     'Polymarket CTF Exchange on Polygon. Operator-verifiable.', NOW(), NOW()),

    ('polymarket_config', 'poll_interval_s', '60', 'int',
     'Seconds between Gamma poll/strategy cycles.', NOW(), NOW()),

    ('polymarket_config', 'max_markets_per_poll', '200', 'int',
     'Max active markets pulled per cycle.', NOW(), NOW()),

    ('polymarket_config', 'category_filter', '', 'string',
     'Optional comma-separated category allowlist (empty = all).', NOW(), NOW()),

    ('polymarket_config', 'max_position_size_usd', '50', 'float',
     'Per-trade notional cap (USD). Applies to simulated and live sizing.',
     NOW(), NOW()),

    ('polymarket_config', 'min_arb_edge_bps', '100', 'int',
     'Risk-free arb floor: only flag a market when (1 - (YES_ask+NO_ask)) '
     'clears this many bps AFTER fee_gas_buffer_bps. 100 = 1%.', NOW(), NOW()),

    ('polymarket_config', 'fee_gas_buffer_bps', '100', 'int',
     'Conservative fee+gas buffer (bps) subtracted from the gross YES+NO<1 '
     'edge before it counts as risk-free.', NOW(), NOW()),

    ('polymarket_config', 'momentum_min_liquidity_usd', '10000', 'int',
     'Momentum signal only considers markets with at least this liquidity.',
     NOW(), NOW()),

    ('polymarket_config', 'momentum_min_move_frac', '0.05', 'float',
     'Min short-window YES-price move (fraction) to emit a momentum signal.',
     NOW(), NOW()),

    ('polymarket_config', 'momentum_min_score', '0.3', 'float',
     'Min transparent momentum score [0,1] to publish the advice signal.',
     NOW(), NOW()),

    ('polymarket_config', 'shadow_record_interval_s', '300', 'int',
     'Per (signal_type, market) throttle: at most one recorded row per this '
     'many seconds (keeps the shadow track honest, avoids row spam).',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- ── shadow/advice + (future) live trade ledger ──
CREATE TABLE IF NOT EXISTS polymarket_trades (
    id               BIGSERIAL PRIMARY KEY,
    market_id        TEXT NOT NULL,
    market_question  TEXT,
    strategy         TEXT NOT NULL,
    side             TEXT,
    outcome          TEXT,
    token_id         TEXT,
    price            DOUBLE PRECISION,
    size_usd         DOUBLE PRECISION,
    expected_edge_bps DOUBLE PRECISION,
    status           TEXT,
    order_id         TEXT,
    skip_reason      TEXT,
    is_simulated     BOOLEAN NOT NULL DEFAULT TRUE,
    details          JSONB,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_polymarket_trades_market ON polymarket_trades(market_id);
CREATE INDEX IF NOT EXISTS idx_polymarket_trades_sim_created ON polymarket_trades(is_simulated, created_at DESC);

-- ── advice/signal stream (always simulated/advice) ──
CREATE TABLE IF NOT EXISTS polymarket_signals (
    id               BIGSERIAL PRIMARY KEY,
    signal_type      TEXT NOT NULL,
    market_id        TEXT NOT NULL,
    market_question  TEXT,
    direction        TEXT,
    yes_price        DOUBLE PRECISION,
    no_price         DOUBLE PRECISION,
    edge_bps         DOUBLE PRECISION,
    score            DOUBLE PRECISION,
    details          JSONB,
    is_simulated     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_polymarket_signals_type_created ON polymarket_signals(signal_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_polymarket_signals_market ON polymarket_signals(market_id);

COMMIT;
