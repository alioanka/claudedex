-- Migration 083: advisor — Fonoloji ACTIVATION + Developer-plan quota keys.
--
-- ROOT CAUSE the operator hit for a week: the Fonoloji integration existed but
-- was effectively INACTIVE because activation depended on flipping several
-- advisor_config rows that were never flipped:
--   * advisor_bist_universe was seeded 'watchlist' (migration 078) => BIST
--     still analyzed ONLY the 4 watchlist tickers even with a key present.
--   * advisor_midas_data_source / advisor_bist_data_source only auto-preferred
--     Fonoloji when EXACTLY '' or 'fonoloji' — an old 'tefas_scrape'/'yfinance'
--     value silently kept the dead Tefas endpoint / degraded yfinance path.
-- This migration plus the code change makes Fonoloji the DEFAULT whenever the
-- key is present (advisor_fonoloji_auto_prefer, default true), while explicit
-- opt-outs keep working ('manual' midas source, 'matriks' bist source, the
-- auto_prefer master switch itself, and explicit universe presets).
--
-- Also seeds the Developer-plan (30,000/month, 3,000/day, 60/min) quota knobs
-- and drops the cache TTL 6h -> 3h (only where the operator left the old
-- default untouched). Idempotent. Reserved migration number 083.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('advisor_config', 'advisor_fonoloji_auto_prefer', 'true', 'boolean',
     'Master switch: when the Fonoloji key is present, Fonoloji is tried FIRST '
     'for BIST data, the BIST universe, Midas NAV, KAP ticker prices and the '
     'forward-return accumulator — WITHOUT flipping each data-source setting. '
     'Explicit opt-outs still win: advisor_midas_data_source=''manual'', '
     'advisor_bist_data_source=''matriks'', explicit universe presets '
     '(''bist30''/''bist50''/''custom''/''watchlist''), or set this to false to '
     'restore the old only-when-source-empty behaviour. FAIL-SOFT: every '
     'Fonoloji miss falls through to the previous source chain. ADVICE-ONLY.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_fonoloji_daily_budget', '3000', 'integer',
     'Local per-UTC-day request budget for the Fonoloji client (Developer plan '
     'allows 3,000/day server-side). When the local counter reaches this the '
     'client serves stale cache only until UTC midnight — protects the quota '
     'from a misconfiguration. Designed steady-state usage is well under '
     '2,000/day with the 3h TTL.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_fonoloji_daily_warn_pct', '0.8', 'float',
     'Fraction of advisor_fonoloji_daily_budget at which to log ONE warning '
     'per day (default 0.8 = warn at 2,400 of 3,000).',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- Developer plan: drop the base cache TTL 6h -> 3h, but ONLY where the
-- operator left the migration-080 default untouched (explicit choices win).
UPDATE config_settings
   SET value = '10800', updated_at = NOW()
 WHERE config_type = 'advisor_config'
   AND key = 'advisor_fonoloji_cache_ttl_s'
   AND value = '21600';

UPDATE config_settings
   SET description = 'TTL (seconds) of the process-wide Fonoloji response '
       'cache. Default 10800 (3h, Developer plan 30k/month). Static endpoints '
       '(stocks/list, recommendations, cpi, percentile) hold a 24h floor; '
       'intraday endpoints (live-estimate, market/live) a 1h ceiling.',
       updated_at = NOW()
 WHERE config_type = 'advisor_config'
   AND key = 'advisor_fonoloji_cache_ttl_s';

-- ACTIVATION: the never-flipped universe default 'watchlist' becomes 'auto'
-- (= Fonoloji live BIST list when a key is present, else watchlist-only,
-- exactly as before). An operator who explicitly wants watchlist-only can set
-- it back to 'watchlist' — that value is now treated as an explicit override
-- and is NOT auto-upgraded again because this UPDATE only fires while the
-- pre-083 seeded value is still in place.
UPDATE config_settings
   SET value = 'auto', updated_at = NOW()
 WHERE config_type = 'advisor_config'
   AND key = 'advisor_bist_universe'
   AND value = 'watchlist';

UPDATE config_settings
   SET description = 'BIST scan universe: auto (DEFAULT — Fonoloji live '
       '/stocks/list when a key is present, falling back to the BIST-50 '
       'snapshot, else watchlist-only) | watchlist (explicit watchlist-only) | '
       'bist30 | bist50 | fonoloji (live list) | custom. Watchlist tickers are '
       'always included; advisor_universe_max caps the total.',
       updated_at = NOW()
 WHERE config_type = 'advisor_config'
   AND key = 'advisor_bist_universe';

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('advisor_config', 'advisor_bist_universe', 'auto', 'string',
     'BIST scan universe: auto (DEFAULT — Fonoloji live /stocks/list when a '
     'key is present, falling back to the BIST-50 snapshot, else '
     'watchlist-only) | watchlist | bist30 | bist50 | fonoloji | custom.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;
