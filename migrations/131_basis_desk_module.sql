-- Migration 131: basis_desk module — config seeds + carry-suggestions table
--
-- NEW self-contained ADVISORY/SHADOW module (modules/basis_desk/): costs out
-- the COMPLETE delta-neutral funding-carry structure per symbol per venue
-- (perp leg + the equal-base-quantity SPOT hedge leg that futures
-- funding-carry v2 leaves to a manual operator action) and records
-- actionable suggestions as SIMULATED rows (is_simulated=true). The live
-- order path is gated (shadow_mode AND live_execution_enabled AND
-- should_skip_live AND RiskManager) AND intentionally unwired — even with
-- every gate green the executor records status=''live_blocked''. Applying
-- this migration changes NO behavior on its own — the module only runs when
-- BASIS_DESK_MODULE_ENABLED=true (default false). Health port 8103.
--
-- Data sources: free key-less public REST (Bybit V5 tickers, Binance
-- premiumIndex + spot ticker). No paid API, no LLM spend.
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

-- ── config seeds (defaults mirror the in-code config.get(...) fallbacks) ──
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('basis_desk', 'shadow_mode', 'true', 'bool',
     'Shadow mode (default ON): every suggestion is recorded simulated '
     '(is_simulated=true); the live order path is never reached. The live '
     'path is additionally UNWIRED by design in this module version.',
     NOW(), NOW()),

    ('basis_desk', 'live_execution_enabled', 'false', 'bool',
     'Explicit LIVE opt-in. Fail-safe default false. NOTE: this module '
     'version has no live order wiring — even true here only moves the '
     'recorded skip_reason to live_path_not_implemented.', NOW(), NOW()),

    ('basis_desk', 'poll_interval_s', '300', 'int',
     'Seconds between venue poll / carry-evaluation cycles.', NOW(), NOW()),

    ('basis_desk', 'symbols', 'BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT', 'string',
     'Comma-separated perp/spot symbol universe (venue-native format).',
     NOW(), NOW()),

    ('basis_desk', 'venues', 'bybit,binance', 'string',
     'Comma-separated venue allowlist. Supported: bybit, binance.',
     NOW(), NOW()),

    ('basis_desk', 'funding_interval_hours', '8', 'float',
     'Assumed funding interval (hours). Bybit/Binance USDT perps default '
     'to 8h; some Bybit symbols pay 4h/1h — conservative to assume 8h.',
     NOW(), NOW()),

    ('basis_desk', 'min_net_carry_bps', '10', 'float',
     'Suggestion floor: NET carry at horizon_intervals (after the full '
     'two-leg cost model) must clear this many bps of notional.', NOW(), NOW()),

    ('basis_desk', 'horizon_intervals', '6', 'float',
     'Holding horizon, in funding intervals, used for the net-carry gate '
     '(6 x 8h = 2 days).', NOW(), NOW()),

    ('basis_desk', 'max_breakeven_intervals', '3', 'float',
     'Reject structures needing more than this many funding intervals just '
     'to cover round-trip costs (funding can decay before breakeven).',
     NOW(), NOW()),

    ('basis_desk', 'perp_taker_fee_bps', '6', 'float',
     'Perp taker fee per fill, bps of notional (charged twice: open+close).',
     NOW(), NOW()),

    ('basis_desk', 'spot_taker_fee_bps', '10', 'float',
     'Spot taker fee per fill, bps of notional (charged twice: open+close).',
     NOW(), NOW()),

    ('basis_desk', 'perp_slippage_bps', '5', 'float',
     'Modeled round-trip slippage on the perp leg, bps.', NOW(), NOW()),

    ('basis_desk', 'spot_slippage_bps', '5', 'float',
     'Modeled round-trip slippage on the spot leg, bps.', NOW(), NOW()),

    ('basis_desk', 'liquidation_premium_bps', '2', 'float',
     'Margin-buffer / forced-exit reserve charged to every structure, bps '
     '(the perp leg can be liquidated even when the book is hedged).',
     NOW(), NOW()),

    ('basis_desk', 'max_notional_usd', '200', 'float',
     'Per-structure notional cap (USD) used to size the suggested legs.',
     NOW(), NOW()),

    ('basis_desk', 'allow_short_spot', 'false', 'bool',
     'Allow LONG-perp + SHORT-spot suggestions (negative funding). Requires '
     'spot margin/borrow the stack does not have — default false; those '
     'structures are still evaluated but recorded non-actionable.',
     NOW(), NOW()),

    ('basis_desk', 'confirm_polls', '2', 'int',
     'Consecutive actionable polls required before a suggestion is recorded '
     '(one funding print can mean-revert before the first payment).',
     NOW(), NOW()),

    ('basis_desk', 'suggest_interval_s', '1800', 'int',
     'Per (venue, symbol) record throttle: at most one suggestion row per '
     'this many seconds.', NOW(), NOW()),

    ('basis_desk', 'hedge_epsilon_frac', '0.02', 'float',
     'Reconciliation tolerance for a live hedged book: '
     '|perp_qty - spot_qty| / perp_qty must stay <= this every tick, else '
     'flatten — a leg-out book is directional, not carry.', NOW(), NOW()),

    ('basis_desk', 'bybit_base_url', 'https://api.bybit.com', 'string',
     'Bybit V5 public REST base (key-less market data).', NOW(), NOW()),

    ('basis_desk', 'binance_fapi_base_url', 'https://fapi.binance.com', 'string',
     'Binance USDT-perp public REST base (premiumIndex).', NOW(), NOW()),

    ('basis_desk', 'binance_spot_base_url', 'https://api.binance.com', 'string',
     'Binance spot public REST base (ticker/price).', NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- ── advisory/shadow carry-suggestion ledger ──
CREATE TABLE IF NOT EXISTS basis_carry_suggestions (
    id                            BIGSERIAL PRIMARY KEY,
    venue                         TEXT NOT NULL,
    symbol                        TEXT NOT NULL,
    direction                     TEXT NOT NULL,  -- SHORT_PERP_LONG_SPOT | LONG_PERP_SHORT_SPOT
    perp_side                     TEXT,           -- SELL | BUY
    spot_side                     TEXT,           -- BUY | SELL
    funding_bps                   DOUBLE PRECISION,
    funding_interval_hours        DOUBLE PRECISION,
    perp_price                    DOUBLE PRECISION,
    spot_price                    DOUBLE PRECISION,
    basis_bps                     DOUBLE PRECISION,
    perp_qty                      DOUBLE PRECISION,
    spot_qty                      DOUBLE PRECISION,
    notional_usd                  DOUBLE PRECISION,
    gross_carry_bps_per_interval  DOUBLE PRECISION,
    round_trip_cost_bps           DOUBLE PRECISION,
    total_cost_bps                DOUBLE PRECISION,
    breakeven_intervals           DOUBLE PRECISION,
    horizon_intervals             DOUBLE PRECISION,
    net_carry_bps_at_horizon      DOUBLE PRECISION,
    apr_gross_pct                 DOUBLE PRECISION,
    needs_borrow                  BOOLEAN NOT NULL DEFAULT FALSE,
    status                        TEXT,           -- advised | live_blocked
    skip_reason                   TEXT,
    is_simulated                  BOOLEAN NOT NULL DEFAULT TRUE,
    details                       JSONB,
    created_at                    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_basis_carry_suggestions_symbol_created
    ON basis_carry_suggestions(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_basis_carry_suggestions_sim_created
    ON basis_carry_suggestions(is_simulated, created_at DESC);

COMMIT;
