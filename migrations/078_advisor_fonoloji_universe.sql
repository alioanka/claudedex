-- Migration 078: advisor — Fonoloji data source + BIST/FX universe scan +
--                 discovery robustness + KAP-sim moderate polarity.
--
-- Covers four operator requests:
--   1. Midas funds via the Fonoloji API (https://fonoloji.com/api-docs) — an
--      OPTIONAL, key-gated, fail-soft data source preferred over the
--      tefas-crawler chain when ADVISOR_FONOLOJI_API_KEY is present.
--   2. BIST + FX UNIVERSE scan (BIST-30/50, FX majors/extended) so the whole
--      universe is analyzed (free local math), not just the watchlist.
--   3. GEMS discovery robustness for US (yfinance local-movers fallback) and
--      BIST (Fonoloji movers + maintained BIST-50 universe fallback).
--   4. KAP-driven sims: optional moderate-polarity firing + synthetic
--      last-price fallback knobs.
--
-- ADVICE-ONLY. All keys default to the prior behaviour (no key / 'watchlist' /
-- off) so an un-tuned deployment is UNCHANGED. Surface these in Advisor
-- Settings (dashboard agent owns the UI).
--
-- Idempotent: ON CONFLICT (config_type, key) DO NOTHING.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Fonoloji data source (Midas funds; also feeds BIST movers if configured).
-- ---------------------------------------------------------------------------
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('advisor_config', 'advisor_fonoloji_base_url', 'https://fonoloji.com', 'string',
     'Fonoloji API origin. Override only if the operator account uses a '
     'different host (e.g. api.fonoloji.com). Used for Midas NAV + optional '
     'BIST movers.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_fonoloji_nav_path', '', 'string',
     'OPTIONAL Fonoloji NAV history path template, tried FIRST before the '
     'built-in candidates. Placeholders: {code} {start} {end} (dates YYYY-MM-DD). '
     'Set this to the EXACT path from https://fonoloji.com/api-docs once the '
     'operator confirms it. Empty => the analyzer probes its default candidate '
     'paths.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_fonoloji_auth_header', 'Authorization', 'string',
     'HTTP header NAME carrying the Fonoloji API key. Common values: '
     'Authorization (with Bearer scheme), X-API-Key, apikey.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_fonoloji_auth_scheme', 'Bearer', 'string',
     'Prefix for the Fonoloji auth header value. Use "Bearer" for '
     'Authorization: Bearer <key>; set EMPTY for a bare key (e.g. with '
     'X-API-Key).',
     NOW(), NOW()),

    ('advisor_config', 'advisor_fonoloji_bist_movers_path', '', 'string',
     'OPTIONAL Fonoloji BIST movers/gainers path for GEMS discovery. Empty => '
     'not used. Set to the confirmed movers endpoint to feed BIST gems from '
     'Fonoloji.',
     NOW(), NOW()),

-- Note: advisor_midas_data_source already exists (migration 058). Operator sets
-- it to 'fonoloji' OR leaves it empty (auto-prefer Fonoloji when a key is set).

-- ---------------------------------------------------------------------------
-- 2. BIST + FX universe scan.
-- ---------------------------------------------------------------------------
    ('advisor_config', 'advisor_bist_universe', 'watchlist', 'string',
     'BIST scan universe: watchlist (default, unchanged) | bist30 | bist50 | '
     'custom. Presets are maintained constituent lists; watchlist tickers are '
     'ALWAYS included. Analysis is free local math (only LLM rationale costs '
     'and it is budget-capped), so scanning a large universe is cost-safe.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_bist_universe_custom', '', 'string',
     'Comma-separated extra BIST tickers (bare, e.g. THYAO,GARAN) to add to the '
     'chosen preset, or to use alone with advisor_bist_universe=custom.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_fx_universe', 'watchlist', 'string',
     'FX/metals scan universe: watchlist (default) | majors | extended. '
     'Watchlist symbols always included.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_fx_universe_custom', '', 'string',
     'Comma-separated extra FX/metal symbols (yfinance format, e.g. '
     'USDTRY=X,GC=F) to add to the chosen FX preset.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_universe_max', '60', 'integer',
     'Hard cap on total symbols per market for the universe scan. Watchlist '
     'tickers survive the cap.',
     NOW(), NOW()),

-- ---------------------------------------------------------------------------
-- 3. Discovery robustness.
-- ---------------------------------------------------------------------------
    ('advisor_config', 'advisor_discovery_us_fallback_universe', '', 'string',
     'OPTIONAL comma-separated US ticker universe used by the yfinance '
     'local-movers fallback when the Yahoo screener JSON returns nothing. Empty '
     '=> a built-in liquid large/mid-cap list is used.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_discovery_bist_use_universe', 'true', 'boolean',
     'When true, BIST GEMS discovery falls back to surfacing the maintained '
     'BIST-50 universe when no movers feed (borsapy/Fonoloji/scrape) yields '
     'data — so BIST gems are never empty. The analyzer still produces the real '
     'directional advice; dedupe drops watchlist/open-sim names.',
     NOW(), NOW()),

-- ---------------------------------------------------------------------------
-- 4. KAP-driven sim knobs.
-- ---------------------------------------------------------------------------
    ('advisor_config', 'advisor_kap_sim_include_moderate', 'false', 'boolean',
     'When true, KAP-driven sims also fire on MODERATE polarity (POSITIVE -> '
     'LONG, NEGATIVE -> SHORT), not only STRONG_POSITIVE/VERY_NEGATIVE. Useful '
     'when strong-polarity disclosures are too rare to ever open a sim. Still '
     'gated by advisor_kap_alert_min_confidence + advisor_kap_sim_enabled.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 5. Fonoloji Secure Credentials card (so it appears in the dashboard panel).
-- ---------------------------------------------------------------------------
INSERT INTO secure_credentials (key_name, display_name, description, category, subcategory, module, is_required, is_sensitive, encrypted_value) VALUES
('ADVISOR_FONOLOJI_API_KEY', 'Advisor — Fonoloji API Key',
 'Fonoloji (fonoloji.com) API key for Turkish TEFAS fund NAV history (Midas '
 'funds) and optional BIST movers. Optional; when set, Fonoloji is preferred '
 'over the free tefas-crawler chain. Without it, Midas falls back to '
 'tefas-crawler/tefasfon/manual. Create an account at fonoloji.com/api-docs.',
 'api', 'fonoloji', 'advisor', FALSE, TRUE, 'PLACEHOLDER')
ON CONFLICT (key_name) DO NOTHING;

COMMIT;

-- ---------------------------------------------------------------------------
-- DOWN (manual rollback)
-- ---------------------------------------------------------------------------
-- BEGIN;
-- DELETE FROM config_settings WHERE config_type='advisor_config' AND key IN (
--   'advisor_fonoloji_base_url','advisor_fonoloji_nav_path',
--   'advisor_fonoloji_auth_header','advisor_fonoloji_auth_scheme',
--   'advisor_fonoloji_bist_movers_path',
--   'advisor_bist_universe','advisor_bist_universe_custom',
--   'advisor_fx_universe','advisor_fx_universe_custom','advisor_universe_max',
--   'advisor_discovery_us_fallback_universe','advisor_discovery_bist_use_universe',
--   'advisor_kap_sim_include_moderate');
-- DELETE FROM secure_credentials WHERE key_name='ADVISOR_FONOLOJI_API_KEY'
--   AND encrypted_value='PLACEHOLDER';
-- COMMIT;
