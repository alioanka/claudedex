-- Migration: arb_realized_slippage — rolling 7-day per-(chain, dex, pair)
-- realized-slippage estimator. Wave-3 ARBITRAGE deliverable. Replaces the
-- static CHAIN_CONFIGS[*]['default_slippage_pct'] in the
-- _check_arb_opportunity gate and _log_arb_trade PnL accounting once a
-- sample exists; falls back to the static value for cold-start keys.
--
-- One row per (chain, dex_pair, pair_symbol). dex_pair is "buy_dex->sell_dex"
-- because the EVM spatial-arb path takes both legs through the same
-- (buy_dex, sell_dex) ordering; we don't yet have per-leg fill prices to
-- attribute slippage to a single venue.
--
-- Refreshed hourly by EVMArbitrageEngine._refresh_realized_slippage from
-- arbitrage_trades rows in the trailing 7d window. Reads are best-effort
-- (a missing row just falls back to the static CHAIN_CONFIGS default).
--
-- Date: 2026-05-19

CREATE TABLE IF NOT EXISTS arb_realized_slippage (
    id            BIGSERIAL PRIMARY KEY,
    chain         TEXT        NOT NULL,
    dex_pair      TEXT        NOT NULL,           -- "uniswap_v2->sushiswap"
    pair_symbol   TEXT        NOT NULL,           -- "USDC/WETH" or "DAI/WETH"
    sample_count  INTEGER     NOT NULL DEFAULT 0,
    median_pct    DOUBLE PRECISION NOT NULL,      -- median realized slippage, fraction (0.005 = 0.5%)
    p90_pct       DOUBLE PRECISION NOT NULL,      -- p90 realized slippage, fraction
    window_start  TIMESTAMPTZ NOT NULL,           -- first sample timestamp included
    window_end    TIMESTAMPTZ NOT NULL,           -- last sample timestamp included
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_arb_slip_chain_dex_pair
    ON arb_realized_slippage (chain, dex_pair, pair_symbol);

CREATE INDEX IF NOT EXISTS idx_arb_slip_updated
    ON arb_realized_slippage (updated_at DESC);
