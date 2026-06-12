-- Migration 132: Stat-arb module — config seeds + shadow-record tables
--
-- New market-neutral pairs mean-reversion module (modules/stat_arb/), per the
-- stat-arb item in docs/agents/NEW_STRATEGY_BACKLOG.md. Trading-capable but
-- LIVE execution is GATED OFF by default (shadow_mode=true AND
-- live_execution_enabled=false), mirroring the polymarket module: every
-- paired suggestion is recorded as a SIMULATED row (is_simulated=true); the
-- live path (ccxt Bybit linear perps) is only reached when shadow_mode=false
-- AND live_execution_enabled=true AND not should_skip_live AND
-- RiskManager.validate_trade passes. Applying this migration changes NO
-- behavior on its own — the module only runs when STAT_ARB_MODULE_ENABLED=true.
--
-- Tail-risk policy baked into the seeds: pair breaks are fat-tailed, so the
-- HARD z-stop (stop_z) and the post-stop cooldown are the primary risk
-- controls — there is NO averaging-down and NO stop-widening code path.
--
-- Data source: FREE — market_data_warehouse reader when populated, else
-- Bybit v5 / Binance USD-M public klines (no keys).
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

-- ── config seeds (defaults mirror the in-code config.get(...) fallbacks) ──
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('stat_arb', 'shadow_mode', 'true', 'bool',
     'Shadow mode (default ON): record every paired suggestion as a simulated '
     'stat_arb_trades row (is_simulated=true); never reach the live order '
     'path. Turn OFF only after a measured 4+ week shadow season.', NOW(), NOW()),

    ('stat_arb', 'live_execution_enabled', 'false', 'bool',
     'Explicit LIVE opt-in: even with shadow_mode off and module dry_run '
     'false, placing a real perp order requires this true. Fail-safe default '
     'false.', NOW(), NOW()),

    ('stat_arb', 'universe', 'BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,DOGE/USDT,ADA/USDT,AVAX/USDT,LINK/USDT,DOT/USDT', 'string',
     'CSV of liquid USDT-perp symbols scanned pairwise (auto mode). Keep '
     'liquid majors only — thin perps make the spread math lie.', NOW(), NOW()),

    ('stat_arb', 'pairs', '', 'string',
     'Optional explicit pair list, e.g. ETH/USDT|BTC/USDT,SOL/USDT|AVAX/USDT. '
     'Empty = scan all universe combinations.', NOW(), NOW()),

    ('stat_arb', 'timeframe', '1h', 'string',
     'Bar timeframe for spread estimation (15m/30m/1h/4h/1d). Thesis horizon '
     'is hours-days, so 1h is the default.', NOW(), NOW()),

    ('stat_arb', 'lookback_bars', '240', 'int',
     'Rolling window (bars) for OLS hedge ratio, z-score and the '
     'mean-reversion test. 240 x 1h = 10 days.', NOW(), NOW()),

    ('stat_arb', 'min_correlation', '0.6', 'float',
     'Gate 1: minimum Pearson correlation of log returns over the window.',
     NOW(), NOW()),

    ('stat_arb', 'adf_tstat_max', '-2.9', 'float',
     'Gate 2: AR(1) t-stat of the spread must be <= this (approx 5% ADF '
     'critical value, no-trend case). More negative = stricter.', NOW(), NOW()),

    ('stat_arb', 'min_half_life_bars', '4', 'int',
     'Gate 3 lower bound: spread half-life in bars. Below = microstructure '
     'noise, not a tradable reversion.', NOW(), NOW()),

    ('stat_arb', 'max_half_life_bars', '120', 'int',
     'Gate 3 upper bound: half-life above this means reversion too slow to '
     'carry (funding + fee drag eats the edge).', NOW(), NOW()),

    ('stat_arb', 'entry_z', '2.0', 'float',
     'Enter (long cheap leg / short rich leg) when |spread z| >= this. '
     'Entries are refused entirely when |z| >= stop_z (blown-out spread is a '
     'pair break, not an opportunity).', NOW(), NOW()),

    ('stat_arb', 'exit_z', '0.5', 'float',
     'Take-profit when z reverts to within this band (or overshoots past it).',
     NOW(), NOW()),

    ('stat_arb', 'stop_z', '4.0', 'float',
     'HARD z-stop: adverse |z| at/past this closes the pair immediately and '
     'puts it on cooldown. THE tail-risk control for pair breaks — never '
     'widened, never averaged into.', NOW(), NOW()),

    ('stat_arb', 'max_hold_hours', '96', 'int',
     'Time stop: a pair still open after this many hours is closed. The '
     'thesis is hours-days reversion; stale trades are cut, not nursed.',
     NOW(), NOW()),

    ('stat_arb', 'max_concurrent_pairs', '3', 'int',
     'Max simultaneously open pairs (concurrency/notional cap).', NOW(), NOW()),

    ('stat_arb', 'notional_per_leg_usd', '100', 'float',
     'Fixed per-leg notional (USD), dollar-neutral two legs. Positions are '
     'never resized after open.', NOW(), NOW()),

    ('stat_arb', 'pair_cooldown_hours', '48', 'int',
     'After a HARD z-stop the pair is barred from re-entry for this long — '
     'the stop said the relationship broke; require a fresh window to '
     're-qualify.', NOW(), NOW()),

    ('stat_arb', 'retest_fail_exits', '6', 'int',
     'Close an open pair after this many consecutive cycles where its spread '
     'can no longer be evaluated (data gone / fit broken).', NOW(), NOW()),

    ('stat_arb', 'taker_fee_bps', '5.5', 'float',
     'Modeled taker fee (bps per fill) used in simulated PnL — 4 fills per '
     'round trip, so shadow PnL is pessimistic by design.', NOW(), NOW()),

    ('stat_arb', 'slippage_bps', '3', 'float',
     'Modeled slippage (bps per fill) added on top of the taker fee in '
     'simulated PnL.', NOW(), NOW()),

    ('stat_arb', 'poll_interval_s', '300', 'int',
     'Seconds between scan/manage cycles.', NOW(), NOW()),

    ('stat_arb', 'spread_state_record_interval_s', '900', 'int',
     'Per-pair throttle on stat_arb_spread_state rows (avoids row spam while '
     'keeping the diagnostic ledger honest).', NOW(), NOW()),

    ('stat_arb', 'max_candidate_pairs', '45', 'int',
     'Hard cap on pairs evaluated per cycle (10-symbol universe = 45 combos).',
     NOW(), NOW()),

    ('stat_arb', 'request_spacing_s', '0.25', 'float',
     'Client-side spacing between free public-API kline requests (politeness).',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- ── paired suggestions / (future) live trade ledger ──
CREATE TABLE IF NOT EXISTS stat_arb_trades (
    id                    BIGSERIAL PRIMARY KEY,
    pair_key              TEXT NOT NULL,
    long_symbol           TEXT NOT NULL,
    short_symbol          TEXT NOT NULL,
    beta                  DOUBLE PRECISION,
    notional_per_leg_usd  DOUBLE PRECISION,
    z_at_entry            DOUBLE PRECISION,
    z_at_exit             DOUBLE PRECISION,
    entry_long_price      DOUBLE PRECISION,
    entry_short_price     DOUBLE PRECISION,
    exit_long_price       DOUBLE PRECISION,
    exit_short_price      DOUBLE PRECISION,
    status                TEXT NOT NULL DEFAULT 'open',
    exit_reason           TEXT,
    pnl_usd               DOUBLE PRECISION,
    skip_reason           TEXT,
    is_simulated          BOOLEAN NOT NULL DEFAULT TRUE,
    details               JSONB,
    opened_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at             TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_stat_arb_trades_pair ON stat_arb_trades(pair_key);
CREATE INDEX IF NOT EXISTS idx_stat_arb_trades_status ON stat_arb_trades(status);
CREATE INDEX IF NOT EXISTS idx_stat_arb_trades_sim_opened ON stat_arb_trades(is_simulated, opened_at DESC);

-- ── spread diagnostics ledger (always simulated/diagnostic) ──
CREATE TABLE IF NOT EXISTS stat_arb_spread_state (
    id              BIGSERIAL PRIMARY KEY,
    pair_key        TEXT NOT NULL,
    y_symbol        TEXT NOT NULL,
    x_symbol        TEXT NOT NULL,
    alpha           DOUBLE PRECISION,
    beta            DOUBLE PRECISION,
    spread          DOUBLE PRECISION,
    spread_mean     DOUBLE PRECISION,
    spread_std      DOUBLE PRECISION,
    zscore          DOUBLE PRECISION,
    correlation     DOUBLE PRECISION,
    ar1_tstat       DOUBLE PRECISION,
    half_life_bars  DOUBLE PRECISION,
    tradable        BOOLEAN NOT NULL DEFAULT FALSE,
    reason          TEXT,
    is_simulated    BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_stat_arb_spread_state_pair_created ON stat_arb_spread_state(pair_key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_stat_arb_spread_state_tradable ON stat_arb_spread_state(tradable, created_at DESC);

COMMIT;
