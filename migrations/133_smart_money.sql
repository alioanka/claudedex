-- Migration 133: smart_money — config seeds + events/scores/signals tables
--
-- On-chain smart-money flow follower (modules/smart_money/). ADVISORY ONLY:
-- it detects wallets that repeatedly front profitable moves (scored by
-- realized FORWARD return — an event is only ever scored after every horizon
-- has fully elapsed, so there is no look-ahead by construction), clusters
-- their fresh accumulation, and writes advisory rows to smart_money_signals.
-- It NEVER places a trade and has no broadcast path. Applying this migration
-- changes NO behavior: the module only runs when SMART_MONEY_MODULE_ENABLED=true
-- (default false).
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('smart_money', 'poll_interval_seconds', '300', 'int',
     'Seconds between smart-money scan cycles (default 5 min).', NOW(), NOW()),

    ('smart_money', 'chains', 'ethereum,base,arbitrum', 'str',
     'CSV of chains to scan. v1 coverage is EVM-only (eth_getLogs swap '
     'decoding); a solana entry is ignored until the Solana parser ships.',
     NOW(), NOW()),

    ('smart_money', 'watch_pairs_per_chain', '12', 'int',
     'Top DexScreener pairs (by 24h volume) watched per chain per tick.',
     NOW(), NOW()),

    ('smart_money', 'min_pair_liquidity_usd', '100000', 'float',
     'Pairs below this liquidity are not watched (thin pools = noisy prices).',
     NOW(), NOW()),

    ('smart_money', 'min_pair_volume_24h_usd', '250000', 'float',
     'Pairs below this 24h volume are not watched.', NOW(), NOW()),

    ('smart_money', 'min_event_usd', '2000', 'float',
     'Swaps below this USD size are ignored — small flow is not smart-money '
     'evidence and wallet attribution costs an RPC call per tx.', NOW(), NOW()),

    ('smart_money', 'max_event_age_minutes', '30', 'int',
     'Logs older than this are skipped at ingest so price_usd_at_event (the '
     'current DexScreener price) stays an honest at-event mark.', NOW(), NOW()),

    ('smart_money', 'max_blocks_per_tick', '600', 'int',
     'Hard cap on the block range scanned per chain per tick (RPC budget).',
     NOW(), NOW()),

    ('smart_money', 'initial_lookback_blocks', '300', 'int',
     'First-run / post-restart scan depth (the events unique key dedupes).',
     NOW(), NOW()),

    ('smart_money', 'max_wallet_lookups_per_tick', '60', 'int',
     'Cap on eth_getTransactionByHash wallet attributions per chain per tick '
     '(largest swaps first; free-tier RPC discipline).', NOW(), NOW()),

    ('smart_money', 'max_mark_tokens_per_tick', '60', 'int',
     'Cap on distinct tokens price-marked per tick (DexScreener batches of 30).',
     NOW(), NOW()),

    ('smart_money', 'forward_horizons_minutes', '60,360,1440', 'str',
     'CSV of forward-return horizons. An event is scored ONLY after its '
     'longest horizon has fully elapsed — the no-look-ahead gate.', NOW(), NOW()),

    ('smart_money', 'scoring_window_hours', '336', 'int',
     'Rolling window (default 14d) of past events used to score a wallet.',
     NOW(), NOW()),

    ('smart_money', 'min_events_per_wallet', '3', 'int',
     'Wallets with fewer fully-elapsed buy events than this are not scored.',
     NOW(), NOW()),

    ('smart_money', 'score_half_life_days', '14', 'int',
     'Exponential age-decay half-life on event weight — smart-money edges '
     'decay; stale wins must stop carrying a wallet.', NOW(), NOW()),

    ('smart_money', 'n_target_events', '12', 'int',
     'Scored-event count at which wallet-score confidence saturates to 1.0.',
     NOW(), NOW()),

    ('smart_money', 'min_wallet_score', '0.55', 'float',
     'Wallet score (0..1) at or above which a wallet counts as smart in a '
     'cluster.', NOW(), NOW()),

    ('smart_money', 'min_forward_return_pct', '3.0', 'float',
     'Minimum historical avg forward return of the smart wallets for a '
     'cluster to emit a signal.', NOW(), NOW()),

    ('smart_money', 'cluster_window_minutes', '45', 'int',
     'Trailing window in which distinct-wallet buys of one token form an '
     'accumulation cluster.', NOW(), NOW()),

    ('smart_money', 'cluster_min_wallets', '3', 'int',
     'Minimum DISTINCT wallets buying one token in the window to call it a '
     'cluster.', NOW(), NOW()),

    ('smart_money', 'min_smart_wallets', '2', 'int',
     'Minimum scored-smart wallets inside a cluster to emit a signal.',
     NOW(), NOW()),

    ('smart_money', 'crowding_soft_cap_wallets', '12', 'int',
     'Above this many cluster participants, signal strength decays as '
     'soft_cap/n — crowded accumulation is late accumulation.', NOW(), NOW()),

    ('smart_money', 'signal_cooldown_minutes', '240', 'int',
     'Per (chain, token) dedupe window between two emitted signals.',
     NOW(), NOW()),

    ('smart_money', 'event_retention_days', '45', 'int',
     'Disk discipline: wallet events older than this are purged.', NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- Observed large swaps with at-event price and LATE forward-return marks.
-- The unique key makes ingest idempotent across restarts/rescans.
CREATE TABLE IF NOT EXISTS smart_money_wallet_events (
    id                  BIGSERIAL PRIMARY KEY,
    chain               TEXT NOT NULL,
    wallet              TEXT NOT NULL,
    token               TEXT NOT NULL,
    token_symbol        TEXT,
    side                TEXT NOT NULL,            -- 'buy' | 'sell'
    amount_usd          DOUBLE PRECISION NOT NULL,
    price_usd_at_event  DOUBLE PRECISION NOT NULL,
    tx_hash             TEXT NOT NULL,
    log_index           INTEGER NOT NULL DEFAULT 0,
    block_time          TIMESTAMPTZ NOT NULL,     -- estimated event time
    observed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    fwd_returns         JSONB NOT NULL DEFAULT '{}',  -- horizon_minutes -> pct
    fully_marked        BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT uq_smart_money_event UNIQUE (chain, tx_hash, log_index)
);
CREATE INDEX IF NOT EXISTS idx_sm_events_wallet
    ON smart_money_wallet_events (chain, wallet, block_time DESC);
CREATE INDEX IF NOT EXISTS idx_sm_events_unmarked
    ON smart_money_wallet_events (block_time) WHERE NOT fully_marked;
CREATE INDEX IF NOT EXISTS idx_sm_events_token_time
    ON smart_money_wallet_events (chain, token, block_time DESC);

-- One current score row per (chain, wallet); upserted every tick.
CREATE TABLE IF NOT EXISTS smart_money_wallet_scores (
    id                  BIGSERIAL PRIMARY KEY,
    chain               TEXT NOT NULL,
    wallet              TEXT NOT NULL,
    events_scored       INTEGER NOT NULL DEFAULT 0,
    total_buy_usd       DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_fwd_return_pct  DOUBLE PRECISION,         -- decay-weighted, realized
    hit_rate            DOUBLE PRECISION,
    confidence          DOUBLE PRECISION,
    score               DOUBLE PRECISION,         -- 0..1 final
    details             JSONB,
    scored_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_smart_money_wallet UNIQUE (chain, wallet)
);
CREATE INDEX IF NOT EXISTS idx_sm_scores_score
    ON smart_money_wallet_scores (score DESC NULLS LAST);

-- ADVISORY accumulation signals — never traded by this module.
CREATE TABLE IF NOT EXISTS smart_money_signals (
    id                  BIGSERIAL PRIMARY KEY,
    chain               TEXT NOT NULL,
    token               TEXT NOT NULL,
    token_symbol        TEXT,
    signal_type         TEXT NOT NULL DEFAULT 'accumulation',
    strength            DOUBLE PRECISION,         -- 0..1
    cluster_wallets     INTEGER,
    smart_wallets       INTEGER,
    avg_wallet_score    DOUBLE PRECISION,
    avg_fwd_return_pct  DOUBLE PRECISION,         -- historical, of smart wallets
    crowding_factor     DOUBLE PRECISION,
    total_buy_usd       DOUBLE PRECISION,
    advisory            BOOLEAN NOT NULL DEFAULT TRUE,
    details             JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sm_signals_created
    ON smart_money_signals (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sm_signals_token
    ON smart_money_signals (chain, token, created_at DESC);

COMMIT;
