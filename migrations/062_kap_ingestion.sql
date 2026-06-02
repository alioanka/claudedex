-- Migration 062: KAP Ingestion Service -- Wave-23
--
-- Creates:
--   kap_disclosures       -- raw disclosure events from KAP
--   kap_returns           -- forward returns keyed to a disclosure
--   kap_company_profiles  -- ticker -> sector/name mapping for cohort stats
--
-- Seeds config_settings (config_type='advisor_config'):
--   advisor_kap_enabled          default 'false'  (operator opts in)
--   advisor_kap_poll_interval_s  default '60'
--   kap_return_windows           default '1,3,5,10,30'
--
-- NOTE: kap_classifications (063) is owned by the sibling classifier agent.
--       This migration creates kap_disclosures.classification_id (FK placeholder)
--       only as a nullable BIGINT -- the FK constraint is added by 063 after it
--       creates kap_classifications.  This avoids a hard ordering dependency.
--
-- Idempotent: CREATE TABLE IF NOT EXISTS + ON CONFLICT DO NOTHING
-- Reversible: DOWN block at bottom.

BEGIN;

-- =========================================================================
-- 1. kap_company_profiles
--    Static ticker -> company metadata for cohort filtering.
--    Populated by kap_archive_crawler.py and kap_listener.py (upsert).
-- =========================================================================
CREATE TABLE IF NOT EXISTS kap_company_profiles (
    id              BIGSERIAL PRIMARY KEY,
    ticker          VARCHAR(32)   NOT NULL,   -- e.g. 'THYAO', 'GARAN'
    company_name    TEXT          NOT NULL DEFAULT '',
    company_id      VARCHAR(64),              -- KAP internal mkkMemberOid
    sector          VARCHAR(128),             -- KAP sector/industry string
    market_cap_usd  NUMERIC(20,2),            -- optional, operator-filled
    is_active       BOOLEAN       NOT NULL DEFAULT TRUE,
    extra           JSONB         NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_kap_company_profiles_ticker
    ON kap_company_profiles (ticker);

COMMENT ON TABLE kap_company_profiles IS
    'BIST company metadata: ticker->sector mapping for cohort stats. '
    'Populated by kap_archive_crawler; updated on each new disclosure seen.';


-- =========================================================================
-- 2. kap_disclosures
--    One row per disclosure fetched from KAP.
--    Deduped by disclosure_id (KAP disclosureIndex).
--
-- Intraday-anchor convention (documented here):
--    price_anchor_type = 'next_session_open'
--    T+0 is defined as the OPENING PRICE of the NEXT trading session after
--    the disclosure timestamp (Turkey timezone, UTC+3).
--    Rationale: disclosures after market close (e.g. 17:50 Istanbul time)
--    are not actionable until the next morning open.  Disclosures during
--    trading hours (09:05) use the SAME session's open price (already past),
--    so T+0 is the next-session open for consistency.
--    This convention avoids look-ahead bias (never uses intraday prices
--    between disclosure and next open).
-- =========================================================================
CREATE TABLE IF NOT EXISTS kap_disclosures (
    id                  BIGSERIAL PRIMARY KEY,
    disclosure_id       VARCHAR(64)   NOT NULL,   -- KAP disclosureIndex (dedup key)
    ticker              VARCHAR(32),              -- primary ticker (NULL if multi-company)
    tickers             TEXT[],                   -- all affected tickers
    company_name        TEXT          NOT NULL DEFAULT '',
    subject             TEXT          NOT NULL DEFAULT '',   -- human-readable subject
    disclosure_type     VARCHAR(32)   NOT NULL DEFAULT '',   -- FAR/FR/ODA/DG/etc.
    summary             TEXT          NOT NULL DEFAULT '',
    full_text           TEXT          NOT NULL DEFAULT '',
    url                 TEXT          NOT NULL DEFAULT '',
    disclosed_at        TIMESTAMPTZ   NOT NULL,              -- KAP publishDate
    fetched_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    -- Intraday anchor (see convention above)
    price_anchor_type   VARCHAR(32)   NOT NULL DEFAULT 'next_session_open',

    -- Classification FK placeholder (constraint added by migration 063)
    -- Sibling classifier writes to kap_classifications.disclosure_id.
    -- We do NOT add classification columns here; sibling owns 063.

    -- Source tracking
    source              VARCHAR(32)   NOT NULL DEFAULT 'pykap',  -- 'pykap' or 'scrape'
    raw_payload         JSONB         NOT NULL DEFAULT '{}',      -- full API response

    created_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_kap_disclosures_disclosure_id
    ON kap_disclosures (disclosure_id);

CREATE INDEX IF NOT EXISTS idx_kap_disclosures_ticker_disclosed
    ON kap_disclosures (ticker, disclosed_at DESC);

CREATE INDEX IF NOT EXISTS idx_kap_disclosures_disclosed_at
    ON kap_disclosures (disclosed_at DESC);

CREATE INDEX IF NOT EXISTS idx_kap_disclosures_disclosure_type
    ON kap_disclosures (disclosure_type, disclosed_at DESC);

COMMENT ON TABLE kap_disclosures IS
    'Raw KAP disclosures. Deduped by disclosure_id (KAP disclosureIndex). '
    'Price anchor convention: next_session_open (see column comment). '
    'source=pykap uses PyKap v0.2.0 BISTCompany.get_disclosures(). '
    'source=scrape uses direct kap.org.tr/tr/api/ calls as fallback.';

COMMENT ON COLUMN kap_disclosures.price_anchor_type IS
    'Intraday anchor convention for forward return computation. '
    'next_session_open: T+0 = opening price of the next trading session '
    'after disclosed_at (Istanbul time UTC+3). Avoids look-ahead bias.';


-- =========================================================================
-- 3. kap_returns
--    Forward returns for each disclosure, keyed by disclosure_id.
--    One row per disclosure (not per window) -- window values are columns.
--    Filled incrementally by forward_return_accumulator.py.
--
-- Window semantics:
--    return_Nd = (close_at_T+N - T0_open) / T0_open
--    where T0_open = next trading session open after disclosed_at.
--    N is measured in TRADING DAYS (not calendar days).
--    A NULL value means the window has not matured yet (today < T+N).
-- =========================================================================
CREATE TABLE IF NOT EXISTS kap_returns (
    id              BIGSERIAL PRIMARY KEY,
    disclosure_id   VARCHAR(64)   NOT NULL,   -- FK to kap_disclosures.disclosure_id
    ticker          VARCHAR(32)   NOT NULL,

    -- T+0 anchor price (next session open)
    anchor_price    NUMERIC(20,8),            -- NULL until price data available
    anchor_date     DATE,                     -- trading date of T0 open

    -- Forward returns (NULL until window matured)
    return_1d       NUMERIC(12,6),
    return_3d       NUMERIC(12,6),
    return_5d       NUMERIC(12,6),
    return_10d      NUMERIC(12,6),
    return_30d      NUMERIC(12,6),

    -- Price data source used for return computation
    price_source    VARCHAR(32)   NOT NULL DEFAULT 'borsapy',  -- 'borsapy' or 'yfinance'

    -- Tracking
    last_computed   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    windows_complete BOOLEAN      NOT NULL DEFAULT FALSE,  -- TRUE when return_30d filled

    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_kap_returns_disclosure_ticker
    ON kap_returns (disclosure_id, ticker);

CREATE INDEX IF NOT EXISTS idx_kap_returns_ticker_anchor
    ON kap_returns (ticker, anchor_date DESC);

CREATE INDEX IF NOT EXISTS idx_kap_returns_incomplete
    ON kap_returns (windows_complete, last_computed)
    WHERE windows_complete = FALSE;

COMMENT ON TABLE kap_returns IS
    'Forward returns for KAP disclosures. One row per (disclosure, ticker). '
    'return_Nd = (close_T+N - anchor_price) / anchor_price. '
    'N is trading days. anchor_price = next-session open after disclosed_at. '
    'Filled by forward_return_accumulator.py (runs daily). '
    'windows_complete flips TRUE when return_30d is populated.';


-- =========================================================================
-- 4. config_settings seeds (advisor_config)
-- =========================================================================
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('advisor_config', 'advisor_kap_enabled', 'false', 'bool',
     'Enable KAP disclosure ingestion listener. '
     'Set true to start polling for new disclosures. '
     'Requires no API key (PyKap uses public KAP endpoints). '
     'Run scripts/crawl_kap_history.py separately to backfill history.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_kap_poll_interval_s', '60', 'int',
     'Seconds between KAP disclosure poll cycles. '
     'Minimum recommended: 60 (polite crawling). '
     'KAP typically publishes a few hundred disclosures per day; '
     '60s is more than sufficient to catch them near-real-time.',
     NOW(), NOW()),

    ('advisor_config', 'kap_return_windows', '1,3,5,10,30', 'string',
     'Comma-separated list of forward return windows in trading days. '
     'These windows are computed by forward_return_accumulator.py. '
     'Changing this after data collection starts requires a manual migration '
     'to add/remove columns in kap_returns.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;


-- =========================================================================
-- DOWN (reversible) -- run manually to undo
-- =========================================================================
-- BEGIN;
-- DELETE FROM config_settings
--   WHERE config_type='advisor_config'
--     AND key IN ('advisor_kap_enabled','advisor_kap_poll_interval_s','kap_return_windows');
-- DROP TABLE IF EXISTS kap_returns;
-- DROP TABLE IF EXISTS kap_disclosures;
-- DROP TABLE IF EXISTS kap_company_profiles;
-- COMMIT;
