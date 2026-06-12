-- Migration 130: intent_solver module — config seeds + shadow opportunities table
--
-- New SHADOW-ONLY EXPERIMENTAL scaffold (modules/intent_solver/). Doc verdict
-- (docs/agents/NEW_MODULE_IDEAS.md #11): PARK IT — full CoW solving / UniswapX
-- filling is structurally uncompetitive for this bot (staked bond, colocated
-- sub-100ms quoting). This scaffold ONLY measures whether fillable edge exists
-- net of costs: it polls open intents from free public APIs, compares each
-- intent''s limit to a reference DEX quote minus gas, and records simulated
-- rows (is_simulated=true). There is NO live path in the module — no keys, no
-- signing, no settlement. Applying this migration changes NO behavior on its
-- own — the module only runs when INTENT_SOLVER_MODULE_ENABLED=true (default
-- false).
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

-- ── config seeds (defaults mirror the in-code config.get(...) fallbacks) ──
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('intent_solver', 'shadow_mode', 'true', 'bool',
     'Documentation flag only: this scaffold has NO live path regardless of '
     'any setting. Kept for fleet-wide config consistency; flipping it does '
     'nothing in v1.', NOW(), NOW()),

    ('intent_solver', 'live_execution_enabled', 'false', 'bool',
     'Reserved fail-safe-off gate for a hypothetical future live filler. '
     'NO-OP in v1 — the module contains no order, signing, or settlement '
     'code path.', NOW(), NOW()),

    ('intent_solver', 'poll_interval_s', '60', 'int',
     'Seconds between intent-polling / shadow-evaluation cycles.', NOW(), NOW()),

    ('intent_solver', 'cow_enabled', 'true', 'bool',
     'Poll the free CoW Protocol orderbook auction for open intents.',
     NOW(), NOW()),

    ('intent_solver', 'cow_base_url', 'https://api.cow.fi', 'string',
     'CoW orderbook API base (free, read-only, no key).', NOW(), NOW()),

    ('intent_solver', 'cow_chains', 'mainnet', 'string',
     'CSV of CoW chain slugs to poll. Allowed: mainnet, xdai, arbitrum_one, '
     'base.', NOW(), NOW()),

    ('intent_solver', 'uniswapx_enabled', 'false', 'bool',
     'Poll the public UniswapX open-orders API (default OFF; less stable '
     'surface than CoW).', NOW(), NOW()),

    ('intent_solver', 'uniswapx_base_url', 'https://api.uniswap.org', 'string',
     'UniswapX order API base (free, read-only, no key).', NOW(), NOW()),

    ('intent_solver', 'uniswapx_chain_ids', '1', 'string',
     'CSV of EVM chain ids to poll on UniswapX. Only ids with a CoW quote '
     'source are evaluated: 1, 100, 42161, 8453.', NOW(), NOW()),

    ('intent_solver', 'max_orders_per_poll', '200', 'int',
     'Cap on raw open orders pulled per source per cycle.', NOW(), NOW()),

    ('intent_solver', 'max_quotes_per_cycle', '10', 'int',
     'Reference-quote budget per cycle (respects the free CoW /quote API; '
     'quoting every open order would be abusive and pointless in shadow).',
     NOW(), NOW()),

    ('intent_solver', 'max_requests_per_minute', '30', 'int',
     'Client-side rate cap per upstream API.', NOW(), NOW()),

    ('intent_solver', 'quote_from_address', '0x1111111111111111111111111111111111111111', 'string',
     'Dummy from-address for CoW /quote requests (the endpoint rejects the '
     'zero address; no key or balance involved).', NOW(), NOW()),

    ('intent_solver', 'min_edge_bps', '30', 'int',
     'Record floor: net edge (gross minus gas minus safety buffer) must '
     'clear this many bps of the limit amount.', NOW(), NOW()),

    ('intent_solver', 'min_edge_usd', '5', 'float',
     'Record floor in USD (applied only when notional is priceable).',
     NOW(), NOW()),

    ('intent_solver', 'safety_buffer_bps', '20', 'int',
     'Haircut for quote staleness, adverse selection, and slippage between '
     'quote and hypothetical settlement.', NOW(), NOW()),

    ('intent_solver', 'settlement_gas_units', '350000', 'int',
     'Assumed settlement gas for one fill (conservative single-order CoW '
     'settlement / UniswapX reactor fill).', NOW(), NOW()),

    ('intent_solver', 'gas_price_gwei', '5', 'float',
     'Assumed gas price for the gas haircut. Deliberately config-static in '
     'v1 (no RPC calls at all); wire pool_engine eth_gasPrice only if the '
     'scaffold ever graduates.', NOW(), NOW()),

    ('intent_solver', 'native_usd_fallback', '3000', 'float',
     'Native-token USD price used when the DexScreener lookup fails.',
     NOW(), NOW()),

    ('intent_solver', 'fallback_gas_bps', '50', 'int',
     'Gas haircut in bps applied when the order notional cannot be priced '
     'in USD (conservative: overstates gas on large orders).', NOW(), NOW()),

    ('intent_solver', 'min_validity_s', '30', 'int',
     'Skip intents expiring sooner than this (a real fill could not land).',
     NOW(), NOW()),

    ('intent_solver', 'shadow_record_interval_s', '300', 'int',
     'Per-order-uid throttle: at most one evaluation per this many seconds '
     '(avoids row spam on long-lived limit orders).', NOW(), NOW()),

    ('intent_solver', 'dexscreener_base_url', 'https://api.dexscreener.com', 'string',
     'Free public price API used only for native-token USD conversion.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- ── shadow opportunity ledger (always simulated in v1) ──
CREATE TABLE IF NOT EXISTS intent_fill_opportunities (
    id                  BIGSERIAL PRIMARY KEY,
    source              TEXT NOT NULL,
    chain               TEXT NOT NULL,
    order_uid           TEXT NOT NULL,
    order_kind          TEXT NOT NULL,
    sell_token          TEXT NOT NULL,
    buy_token           TEXT NOT NULL,
    sell_amount         TEXT NOT NULL,
    buy_amount          TEXT NOT NULL,
    quote_amount        TEXT NOT NULL,
    gross_edge_bps      DOUBLE PRECISION,
    gas_bps             DOUBLE PRECISION,
    buffer_bps          DOUBLE PRECISION,
    net_edge_bps        DOUBLE PRECISION,
    notional_usd        DOUBLE PRECISION,
    edge_usd            DOUBLE PRECISION,
    gas_cost_usd        DOUBLE PRECISION,
    partially_fillable  BOOLEAN NOT NULL DEFAULT FALSE,
    valid_to            TIMESTAMPTZ,
    is_simulated        BOOLEAN NOT NULL DEFAULT TRUE,
    details             JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_intent_fill_opps_source_created
    ON intent_fill_opportunities(source, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_intent_fill_opps_order_uid
    ON intent_fill_opportunities(order_uid);
CREATE INDEX IF NOT EXISTS idx_intent_fill_opps_sim_created
    ON intent_fill_opportunities(is_simulated, created_at DESC);

COMMIT;
