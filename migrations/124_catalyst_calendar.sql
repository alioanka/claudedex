-- Migration 124: catalyst_calendar — config seeds + catalysts table
--
-- New ADVISORY module (modules/catalyst_calendar/): aggregates known forward
-- catalysts — token unlock cliffs (DefiLlama, best-effort), exchange
-- listing/delisting announcements (Binance CMS, scrape-class), and scheduled
-- macro events (static FOMC schedule + key-gated FMP) — into the catalysts
-- table for other modules and the operator to read. It NEVER trades, never
-- writes pause/killswitch flags, and requires no keys. Applying this migration
-- changes NO behavior: the module only runs when
-- CATALYST_CALENDAR_MODULE_ENABLED=true (default false), and every seed below
-- is read-only feed configuration.
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('catalyst_calendar', 'refresh_interval_seconds', '3600', 'int',
     'Seconds between calendar refresh cycles (default 1h — catalyst feeds '
     'move slowly; keep free sources unstressed).', NOW(), NOW()),

    ('catalyst_calendar', 'lookahead_days', '30', 'int',
     'Forward window: only events within this many days are written.',
     NOW(), NOW()),

    ('catalyst_calendar', 'listing_recent_window_hours', '72', 'int',
     'Exchange announcements released within this window are kept as active '
     'catalysts (the announcement itself is the event; go-live times in '
     'titles are not reliably parseable).', NOW(), NOW()),

    ('catalyst_calendar', 'purge_after_days', '90', 'int',
     'Events whose event_time is older than this are deleted so the table '
     'stays a forward calendar.', NOW(), NOW()),

    ('catalyst_calendar', 'max_requests_per_minute', '10', 'int',
     'TOTAL outbound HTTP request cap across all calendar sources '
     '(client-side throttle; all sources are free tiers).', NOW(), NOW()),

    ('catalyst_calendar', 'source_defillama_unlocks_enabled', 'true', 'bool',
     'Token unlock/vesting cliffs from the unofficial DefiLlama emissions '
     'endpoint. Best-effort coverage; a schema change degrades to zero '
     'events (confidence: medium).', NOW(), NOW()),

    ('catalyst_calendar', 'source_binance_listings_enabled', 'true', 'bool',
     'Listing/delisting announcements parsed from the Binance CMS endpoint. '
     'Scrape-class and FRAGILE — may break or be geo-blocked at any time; '
     'failures degrade to zero events (confidence: low).', NOW(), NOW()),

    ('catalyst_calendar', 'source_macro_static_enabled', 'true', 'bool',
     'Built-in static macro schedule (official FOMC 2026 decision dates). '
     'EXPIRES 2026-12-31 — past that it emits NOTHING until the operator '
     'refreshes the code list or sets macro_static_override.', NOW(), NOW()),

    ('catalyst_calendar', 'source_fmp_macro_enabled', 'false', 'bool',
     'KEY-GATED macro calendar via FMP (FMP_API_KEY through secrets_manager '
     'or env). Default OFF; without a key it degrades to zero events.',
     NOW(), NOW()),

    ('catalyst_calendar', 'macro_static_override', '', 'string',
     'Optional JSON list replacing the built-in macro schedule, e.g. '
     '[{date: 2027-01-27T19:00:00, event_type: macro_fomc, title: FOMC rate '
     'decision}] (keys/values quoted as JSON). Empty = use the built-in list.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- Forward catalyst calendar. One row per (source, symbol, event_type,
-- event_time); re-seen events bump last_seen_at so consumers can judge feed
-- freshness per row before trusting it (a stale calendar is worse than none).
-- symbol '*' = market-wide (macro). magnitude: unlocks only — fraction of max
-- supply unlocking (0..1), NULL when not derivable. confidence: source-trust
-- label ('high' | 'medium' | 'low').
CREATE TABLE IF NOT EXISTS catalysts (
    id            BIGSERIAL PRIMARY KEY,
    source        TEXT NOT NULL,          -- 'defillama_unlocks' | 'binance_announcements' | 'static_macro' | 'fmp_macro'
    symbol        TEXT NOT NULL,          -- upper-cased asset symbol, or '*' = market-wide
    event_type    TEXT NOT NULL,          -- 'token_unlock' | 'exchange_listing' | 'exchange_delisting' | 'macro_fomc' | 'macro_cpi' | 'macro_other'
    event_time    TIMESTAMPTZ NOT NULL,
    title         TEXT,
    magnitude     DOUBLE PRECISION,
    confidence    TEXT,
    url           TEXT,
    raw           JSONB,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_catalysts_identity UNIQUE (source, symbol, event_type, event_time)
);
CREATE INDEX IF NOT EXISTS idx_catalysts_event_time ON catalysts(event_time);
CREATE INDEX IF NOT EXISTS idx_catalysts_symbol_time ON catalysts(symbol, event_time);

COMMIT;
