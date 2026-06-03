-- Migration 076: advisor "New Gems" discovery layer
--
-- Adds a discovery / candidate-screening layer to the ADVICE-ONLY advisor so it
-- surfaces promising tickers BEYOND the operator's watchlists, for CRYPTO, US
-- EQUITIES, and BIST. Discovery pulls a free candidate universe per market,
-- screens/dedupes/ranks it locally (NO LLM, NO paid data), and hands the top-N
-- NEW symbols to the EXISTING analyzer pipeline so they get normal advice,
-- flagged as origin='discovery' (vs 'watchlist').
--
-- COST DISCIPLINE: discovery defaults to rule-based rationale (NO LLM). Only the
-- top advisor_discovery_llm_max (default 0 = none) may be LLM-narrated, and that
-- ALWAYS goes through the global daily paid-LLM budget (core/llm_budget) so the
-- hard daily cap is the ceiling. Discovery can never blow the budget.
--
-- DEFAULT OFF: advisor_discovery_enabled='false' — nothing is discovered, and
-- nothing spends, until the operator explicitly enables it.
--
-- Schema:
--   * advisor_advice gains an origin VARCHAR(16) DEFAULT 'watchlist' column
--     (existing rows backfilled to 'watchlist'); discovered rows write 'discovery'.
--   * seeds all new advisor_config keys (idempotent ON CONFLICT DO NOTHING).
--
-- Surface every key in the Advisor Settings page (dashboard agent).
-- ADVICE-ONLY. Idempotent. No data destroyed.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. origin column on advisor_advice
-- ---------------------------------------------------------------------------
ALTER TABLE advisor_advice
    ADD COLUMN IF NOT EXISTS origin VARCHAR(16) NOT NULL DEFAULT 'watchlist';

-- Backfill any pre-existing NULLs (defensive; DEFAULT already covers new rows).
UPDATE advisor_advice SET origin = 'watchlist' WHERE origin IS NULL;

CREATE INDEX IF NOT EXISTS idx_advisor_advice_origin
    ON advisor_advice (origin, created_at DESC);

-- ---------------------------------------------------------------------------
-- 2. Discovery config keys (advisor_config) — DEFAULT OFF + cost-safe.
-- ---------------------------------------------------------------------------
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('advisor_config', 'advisor_discovery_enabled', 'false', 'bool',
     'Master toggle for the New Gems discovery layer. When false (default), the '
     'advisor analyzes ONLY the configured watchlists and nothing is discovered '
     'or spent. Turn on to surface trending candidates beyond your watchlists.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_discovery_markets', 'crypto,us_equities,bist', 'string',
     'Comma-separated markets to run discovery for (crypto|us_equities|bist). '
     'Other markets (fx, midas_funds) have no discovery source.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_discovery_max_per_market', '5', 'int',
     'Max NEW symbols discovered per market per discovery pass (top-N by score). '
     'Each gets normal analyzer advice flagged origin=discovery.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_discovery_refresh_hours', '12', 'int',
     'Discovery cadence in hours. Discovery runs at most once per this many hours '
     '(NOT every advice cycle) to limit data-source load and noise. Default 12.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_discovery_llm_max', '0', 'int',
     'How many of the top discovery candidates per pass may be LLM-narrated. '
     'Default 0 = NONE (rule-based rationale only — free). Any LLM use still '
     'draws from the global daily paid-LLM budget (advisor_llm_daily_max_calls), '
     'which is the hard ceiling. Keep low to stay cheap.',
     NOW(), NOW()),

    -- crypto screening thresholds
    ('advisor_config', 'advisor_discovery_crypto_min_quote_vol_usd', '1000000', 'float',
     'Crypto discovery: minimum 24h quote-volume (USD) for a candidate pair.',
     NOW(), NOW()),
    ('advisor_config', 'advisor_discovery_crypto_min_market_cap_usd', '0', 'float',
     'Crypto discovery: minimum market cap (USD) if known. 0 = no floor.',
     NOW(), NOW()),
    ('advisor_config', 'advisor_discovery_crypto_max_market_cap_usd', '0', 'float',
     'Crypto discovery: maximum market cap (USD) if known. 0 = no ceiling.',
     NOW(), NOW()),
    ('advisor_config', 'advisor_discovery_crypto_min_abs_change_pct', '3', 'float',
     'Crypto discovery: minimum absolute 24h % move to be considered a mover.',
     NOW(), NOW()),
    ('advisor_config', 'advisor_discovery_crypto_universe_cap', '400', 'int',
     'Crypto discovery: cap the ranked universe to the top-N by volume before '
     'screening, to bound compute.',
     NOW(), NOW()),

    -- us equities screening thresholds
    ('advisor_config', 'advisor_discovery_us_min_price', '2', 'float',
     'US-equity discovery: minimum share price (avoid sub-$1 penny junk).',
     NOW(), NOW()),
    ('advisor_config', 'advisor_discovery_us_min_dollar_vol', '5000000', 'float',
     'US-equity discovery: minimum daily dollar volume (price*shares).',
     NOW(), NOW()),
    ('advisor_config', 'advisor_discovery_us_min_abs_change_pct', '2', 'float',
     'US-equity discovery: minimum absolute % move to be considered a mover.',
     NOW(), NOW()),
    ('advisor_config', 'advisor_discovery_us_screens', 'day_gainers,most_actives,undervalued_growth_stocks', 'string',
     'US-equity discovery: comma-sep Yahoo predefined screener ids to pull '
     '(free, no key; endpoint is undocumented/rate-limited — fail-soft).',
     NOW(), NOW()),
    ('advisor_config', 'advisor_discovery_us_screen_count', '50', 'int',
     'US-equity discovery: rows to request per Yahoo screener (max 100).',
     NOW(), NOW()),

    -- bist screening thresholds (degraded/fragile)
    ('advisor_config', 'advisor_discovery_bist_min_abs_change_pct', '2', 'float',
     'BIST discovery: minimum absolute % move. BIST discovery is best-effort and '
     'commonly returns nothing — that is acceptable (fail-soft).',
     NOW(), NOW()),
    ('advisor_config', 'advisor_discovery_bist_universe_cap', '100', 'int',
     'BIST discovery: cap the candidate universe size.',
     NOW(), NOW()),
    ('advisor_config', 'advisor_discovery_bist_scrape_url', '', 'string',
     'BIST discovery: optional operator-supplied JSON movers URL (array of '
     '{symbol,change,price,volume}). Empty (default) = borsapy-only / no scrape.',
     NOW(), NOW()),

    -- internal: last-run timestamp persistence (per pass)
    ('advisor_config', 'advisor_discovery_last_run_at', '', 'string',
     'Internal: ISO-8601 UTC timestamp of the last discovery pass. Managed by '
     'the advice engine to enforce advisor_discovery_refresh_hours cadence. '
     'Leave blank.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;

-- ---------------------------------------------------------------------------
-- DOWN (manual rollback)
-- ---------------------------------------------------------------------------
-- BEGIN;
-- DROP INDEX IF EXISTS idx_advisor_advice_origin;
-- ALTER TABLE advisor_advice DROP COLUMN IF EXISTS origin;
-- DELETE FROM config_settings WHERE config_type='advisor_config'
--   AND key LIKE 'advisor_discovery_%';
-- COMMIT;
