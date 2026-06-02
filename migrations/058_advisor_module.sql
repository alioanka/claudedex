-- Migration 058: Financial Advisor module — Wave-20 scaffold
-- Creates: advisor_advice, advisor_sim_positions, advisor_portfolio
-- Seeds:   config_settings rows for config_type='advisor_config'
-- Idempotent: CREATE TABLE IF NOT EXISTS + ON CONFLICT DO NOTHING
-- Safe to re-run; no destructive changes.

BEGIN;

-- =========================================================================
-- 1. advisor_advice
--    One row per published advice event.
--    operator_notes: operator adds notes post-advice ("bought at X").
-- =========================================================================
CREATE TABLE IF NOT EXISTS advisor_advice (
    id                  BIGSERIAL PRIMARY KEY,
    market              VARCHAR(32)   NOT NULL,          -- Market enum value
    symbol              VARCHAR(64)   NOT NULL,
    horizon             VARCHAR(16)   NOT NULL,          -- short/mid/long
    direction           VARCHAR(16)   NOT NULL,          -- long/short/neutral
    entry_low           NUMERIC(20,8),
    entry_high          NUMERIC(20,8),
    target_price        NUMERIC(20,8),
    stop_price          NUMERIC(20,8),
    confidence          NUMERIC(5,4)  NOT NULL DEFAULT 0,
    rationale           TEXT          NOT NULL DEFAULT '',
    model_id            VARCHAR(128)  NOT NULL DEFAULT '',
    kronos_signal       NUMERIC(12,6),                   -- NULL if Kronos not loaded
    data_source_status  VARCHAR(32)   NOT NULL DEFAULT 'available',
    sim_enabled         BOOLEAN       NOT NULL DEFAULT FALSE,
    sim_amount_usd      NUMERIC(12,2) NOT NULL DEFAULT 1000,
    extra               JSONB         NOT NULL DEFAULT '{}',
    operator_notes      TEXT,
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_advisor_advice_symbol
    ON advisor_advice (symbol, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_advisor_advice_market_horizon
    ON advisor_advice (market, horizon, created_at DESC);

-- =========================================================================
-- 2. advisor_sim_positions
--    Dry-run position tracking: seeded from advisor_advice when sim_enabled.
--    Marked-to-market daily by the advice engine.
--    close_reason: "manual" | "target_hit" | "stop_hit" | "expired" | "operator"
-- =========================================================================
CREATE TABLE IF NOT EXISTS advisor_sim_positions (
    id              BIGSERIAL PRIMARY KEY,
    advice_id       BIGINT REFERENCES advisor_advice(id) ON DELETE SET NULL,
    symbol          VARCHAR(64)   NOT NULL,
    market          VARCHAR(32)   NOT NULL,
    direction       VARCHAR(16)   NOT NULL,
    horizon         VARCHAR(16)   NOT NULL,
    entry_price     NUMERIC(20,8) NOT NULL,
    current_price   NUMERIC(20,8),
    target_price    NUMERIC(20,8),
    stop_price      NUMERIC(20,8),
    notional_usd    NUMERIC(12,2) NOT NULL DEFAULT 1000,
    exit_price      NUMERIC(20,8),
    pnl_pct         NUMERIC(10,4),                       -- in percent, e.g. 3.25
    pnl_usd         NUMERIC(12,2),
    status          VARCHAR(16)   NOT NULL DEFAULT 'open', -- open|closed|expired
    close_reason    VARCHAR(64),
    opened_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_advisor_sim_status
    ON advisor_sim_positions (status, opened_at DESC);

CREATE INDEX IF NOT EXISTS idx_advisor_sim_symbol
    ON advisor_sim_positions (symbol, status);

-- =========================================================================
-- 3. advisor_portfolio
--    Operator-reported holdings on Midas (or elsewhere).
--    Operator enters/updates via /advisor/portfolio dashboard page.
--    Used by advice_engine for context ("you already hold AAPL at 20%").
-- =========================================================================
CREATE TABLE IF NOT EXISTS advisor_portfolio (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(64)   NOT NULL,
    market          VARCHAR(32)   NOT NULL,
    quantity        NUMERIC(20,8) NOT NULL DEFAULT 0,
    avg_cost        NUMERIC(20,8) NOT NULL DEFAULT 0,     -- per unit in USD
    current_price   NUMERIC(20,8),
    notes           TEXT,
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (symbol, market)
);

-- =========================================================================
-- 4. advisor_config seeds (config_type='advisor_config')
--    All advisor settings live here. Values are strings; value_type
--    tells the config manager how to cast.
-- =========================================================================

-- Core LLM
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_anthropic_model',
     'claude-opus-4-8', 'string',
     'Anthropic model ID for LLM rationale generation. Bump here when a newer model ships. '
     'Loud WARNING logged on 404 — never silent fallback.')
ON CONFLICT (config_type, key) DO NOTHING;

-- Markets + horizons
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'enabled_markets',
     'crypto,us_equities', 'string',
     'Comma-sep list of active market analyzers: crypto|us_equities|bist|fx|midas_funds. '
     'BIST/FX/midas_funds require separate data source config before enabling.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'enabled_horizons',
     'short,mid,long', 'string',
     'Comma-sep horizons to generate advice for: short|mid|long. '
     'short=1day-1wk, mid=1wk-3mo, long=3mo-2yr.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'run_interval_minutes',
     '60', 'integer',
     'How often (minutes) to run an advice cycle. Default 60.')
ON CONFLICT (config_type, key) DO NOTHING;

-- Risk / sim
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'min_confidence',
     '0.35', 'float',
     'Minimum confidence [0.0-1.0] for an advice to be published. Default 0.35.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'max_sim_positions',
     '20', 'integer',
     'Maximum number of open sim positions at any time. Default 20.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'blocked_symbols',
     '', 'string',
     'Comma-sep symbol blocklist. Advice for these symbols is suppressed.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'sim_default_enabled',
     'false', 'boolean',
     'If true, every published advice automatically opens a sim position. Default false.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'sim_default_amount_usd',
     '1000.0', 'float',
     'Default notional USD for auto-opened sim positions. Default 1000.')
ON CONFLICT (config_type, key) DO NOTHING;

-- Watchlists (per-market, comma-separated symbols)
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'watchlist_crypto',
     'BTC/USDT,ETH/USDT,SOL/USDT', 'string',
     'Crypto pairs to analyze (ccxt format, e.g. BTC/USDT).')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'watchlist_us_equities',
     'AAPL,MSFT,NVDA,TSLA,AMZN', 'string',
     'US equity tickers to analyze (yfinance format, e.g. AAPL).')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'watchlist_bist',
     '', 'string',
     'BIST tickers to analyze. Format depends on advisor_bist_data_source: '
     'yfinance → THYAO.IS; Matriks → THYAO. Requires data source config.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'watchlist_fx',
     'EURUSD=X,GBPUSD=X,XAUUSD=X,XAGUSD=X', 'string',
     'FX pairs and metals (yfinance format). XAUUSD=X = gold, XAGUSD=X = silver. '
     'Requires advisor_fx_data_source to be set.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'watchlist_midas_funds',
     '', 'string',
     'Turkish fund codes to analyze (Tefas FONKODU format). '
     'E.g. TPP (Tera Para Piyasasi), AKP (AK Para Piyasasi Katilim). '
     'Requires advisor_midas_data_source to be set.')
ON CONFLICT (config_type, key) DO NOTHING;

-- Data sources
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_crypto_exchange',
     'binance', 'string',
     'ccxt exchange ID for crypto OHLCV data. Default binance (public, no key).')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_bist_data_source',
     '', 'string',
     'BIST data source: yfinance (degraded, free) | matriks (paid, ADVISOR_BIST_API_KEY required) | empty = disabled. '
     'Add ADVISOR_BIST_API_KEY to Secure Credentials for Matriks.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_fx_data_source',
     'yfinance', 'string',
     'FX data source: yfinance (free, default) | alphavantage (ADVISOR_FX_ALPHAVANTAGE_KEY required) | stooq (free, no SLA).')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_midas_data_source',
     '', 'string',
     'Midas fund data source: tefas_scrape (undocumented POST, fragile, free) | manual (operator enters NAV in dashboard) | empty = disabled.')
ON CONFLICT (config_type, key) DO NOTHING;

-- Telegram (separate advisor bot — NOT the shared trading-module bot)
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_telegram_enabled',
     'true', 'boolean',
     'Master toggle for advisor Telegram notifications. Default true (requires token+chat to be set).')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_telegram_bot_token',
     '', 'string',
     'SEPARATE advisor bot token. Store in Secure Credentials as ADVISOR_TELEGRAM_BOT_TOKEN. '
     'Do NOT use the shared TELEGRAM_BOT_TOKEN — this is a different bot.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_telegram_chat_id',
     '', 'string',
     'Target chat/group/channel ID for advisor messages. Store as ADVISOR_TELEGRAM_CHAT_ID in Secure Credentials.')
ON CONFLICT (config_type, key) DO NOTHING;

-- Kronos forecaster (off by default — weights must be manually downloaded)
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_kronos_enabled',
     'false', 'boolean',
     'Enable Kronos K-line foundation model for directional forecast overlay. '
     'Requires weights at ADVISOR_KRONOS_WEIGHTS_PATH. Default false.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_kronos_variant',
     'Kronos-mini', 'string',
     'Kronos model variant: Kronos-mini (4.1M, CPU-OK) | Kronos-small (24.7M) | Kronos-base (102.3M, GPU recommended).')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_kronos_device',
     'cpu', 'string',
     'PyTorch device for Kronos inference: cpu | cuda. Use cuda only if GPU is available.')
ON CONFLICT (config_type, key) DO NOTHING;

-- ML loop (off by default — ML agent implements in Wave-22)
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_ml_enabled',
     'false', 'boolean',
     'Enable ML self-improvement loop (Wave-22 feature). Default false.')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('advisor_config', 'advisor_ml_daily_learning',
     'false', 'boolean',
     'Run daily outcome ingestion + model refinement (Wave-22 feature). Default false.')
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;

-- DOWN (reversible rollback):
-- BEGIN;
-- DROP TABLE IF EXISTS advisor_sim_positions;
-- DROP TABLE IF EXISTS advisor_portfolio;
-- DROP TABLE IF EXISTS advisor_advice;
-- DELETE FROM config_settings WHERE config_type = 'advisor_config';
-- COMMIT;
