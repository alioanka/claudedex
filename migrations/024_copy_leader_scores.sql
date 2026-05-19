-- Wave-2 COPY_TRADING enhancement: persistent leader-scoring table.
--
-- The Phase-1 quant audit (docs/agents/reports/COPY_TRADING_quant.md
-- section 2.1) flagged that copy_engine selects leaders by membership
-- in config_settings.copytrading_config.target_wallets ONLY — there
-- is no scoring, no edge filter, no sample-size adjustment. Operators
-- have no way to compare leaders objectively.
--
-- This table is written by:
--   modules/copy_trading/leader_scorer.py         (compute_score)
--   modules/copy_trading/wallet_discovery.py      (cache discovered candidates)
--
-- It is read by:
--   dashboard /copytrading/leaders                (top-N ranked view)
--   modules/copy_trading/copy_engine.py           (Kelly-fraction sizing)
--
-- All fields are nullable except identity / timestamps because a wallet
-- can be discovered (address known) before it has been scored, and a
-- score can be recomputed before we have 30d of history. The
-- dashboard treats NULL metrics as "not yet computed" rather than 0.

CREATE TABLE IF NOT EXISTS copy_leader_scores (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chain               VARCHAR(20) NOT NULL,            -- 'solana' | 'ethereum' | 'base' | ...
    wallet_address      VARCHAR(100) NOT NULL,
    label               VARCHAR(120),                    -- optional human label (e.g. 'GMGN smart-money #3')
    source              VARCHAR(40) NOT NULL,            -- 'dexscreener' | 'birdeye' | 'gmgn' | 'helius' | 'manual'

    -- Scoring inputs (all in USD where applicable, days where time)
    realized_pnl_usd_30d    NUMERIC(20, 4),
    realized_pnl_usd_90d    NUMERIC(20, 4),
    sharpe_30d              NUMERIC(8, 4),               -- daily-return Sharpe, sqrt(365) annualised
    hit_rate                NUMERIC(5, 4),               -- 0..1 closed-trade win ratio
    avg_hold_seconds        BIGINT,                      -- average position duration
    max_drawdown_pct        NUMERIC(6, 4),               -- 0..1 (e.g. 0.42 = 42% DD)
    trade_count_30d         INTEGER,
    avg_trade_size_usd      NUMERIC(20, 4),

    -- Composite score (0..100). Computed by leader_scorer.compute_score.
    -- Higher = better. NULL until first scoring run.
    score                   NUMERIC(6, 2),

    -- Kelly inputs derived from score + history (CT-Q-02). Operator
    -- can override the fraction in the dashboard.
    kelly_fraction          NUMERIC(6, 4),               -- 0..1 capped to 0.25 (quarter-Kelly) in engine

    -- Provenance / freshness
    sample_window_days      INTEGER,                     -- how many days of history fed compute_score
    last_scored_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    discovered_at           TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Raw discovery payload (for audit + reproducing the score).
    raw_metrics             JSONB,

    UNIQUE (chain, wallet_address)
);

CREATE INDEX IF NOT EXISTS idx_copy_leader_scores_score
    ON copy_leader_scores (chain, score DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS idx_copy_leader_scores_freshness
    ON copy_leader_scores (last_scored_at DESC);

CREATE INDEX IF NOT EXISTS idx_copy_leader_scores_source
    ON copy_leader_scores (source);
