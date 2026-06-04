-- Migration 080: advisor — Fonoloji VERIFIED contract + shared-client config.
--
-- The prior migration 078 seeded a GUESSED Fonoloji contract
-- (base https://fonoloji.com, Authorization: Bearer). The operator's official
-- api-docs verify a DIFFERENT contract:
--   * Base URL   : https://fonoloji.com/v1
--   * Auth       : header X-API-Key: <key>  (NO Bearer scheme)
--   * Free tier  : 15,000 req/MONTH (~500/day) + per-minute/day caps.
--
-- This migration CORRECTS the three guessed keys (UPDATE — they already exist
-- from 078, so ON CONFLICT DO NOTHING would not change them) and ADDS the new
-- keys the shared FonolojiClient + the BIST/FX/gems wiring read:
--   * advisor_fonoloji_cache_ttl_s        (TTL cache, default 6h)
--   * advisor_bist_data_source='fonoloji' / advisor_fx_data_source='fonoloji'
--     are now accepted values (the *_data_source keys already exist; this only
--     documents them — see the analyzer headers).
--   * advisor_bist_universe='fonoloji'    (live BIST list — value, no new key)
--   * advisor_discovery_bist_* screener knobs for the gems screener batch.
--
-- ADVICE-ONLY. EVERYTHING is gated on ADVISOR_FONOLOJI_API_KEY presence — with
-- NO key the behaviour is UNCHANGED. Surface the new keys in Advisor Settings
-- (dashboard agent owns the UI). Idempotent.
--
-- Reserved migration number 080.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. CORRECT the guessed contract values seeded by migration 078.
--    These rows ALREADY EXIST, so we UPDATE (not INSERT) to the verified values.
--    Only overwrite the stale guessed defaults; an operator who already hand-set
--    a non-default value keeps it.
-- ---------------------------------------------------------------------------
UPDATE config_settings
   SET value = 'https://fonoloji.com/v1', updated_at = NOW()
 WHERE config_type = 'advisor_config'
   AND key = 'advisor_fonoloji_base_url'
   AND value = 'https://fonoloji.com';

UPDATE config_settings
   SET value = 'X-API-Key', updated_at = NOW()
 WHERE config_type = 'advisor_config'
   AND key = 'advisor_fonoloji_auth_header'
   AND value = 'Authorization';

-- X-API-Key carries a BARE key (no scheme prefix). The 078 default was 'Bearer'.
UPDATE config_settings
   SET value = '', updated_at = NOW()
 WHERE config_type = 'advisor_config'
   AND key = 'advisor_fonoloji_auth_scheme'
   AND value = 'Bearer';

-- ---------------------------------------------------------------------------
-- 2. NEW keys for the shared client + verified wiring.
-- ---------------------------------------------------------------------------
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('advisor_config', 'advisor_fonoloji_cache_ttl_s', '21600', 'int',
     'Fonoloji shared-client TTL cache, seconds (default 21600 = 6h). Fund NAV '
     'and BIST bars are DAILY data, so a long TTL keeps usage far under the '
     '15,000 req/month free tier. Raise it to be even more conservative.',
     NOW(), NOW()),

    -- Defensively (re)assert the verified base/header in case 078 never ran or
    -- the keys were deleted. ON CONFLICT DO NOTHING leaves any operator value
    -- (and the UPDATEs in section 1 already corrected the stale 078 defaults).
    ('advisor_config', 'advisor_fonoloji_base_url', 'https://fonoloji.com/v1', 'string',
     'Fonoloji API base URL (VERIFIED: https://fonoloji.com/v1). Override only '
     'if the operator account uses a different host.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_fonoloji_auth_header', 'X-API-Key', 'string',
     'HTTP header carrying the Fonoloji API key (VERIFIED: X-API-Key, bare key, '
     'no scheme). The key itself is ADVISOR_FONOLOJI_API_KEY in Secure '
     'Credentials.',
     NOW(), NOW()),

    -- GEMS screener batch knobs (BIST high-ROE / low-PE value tilt).
    ('advisor_config', 'advisor_discovery_bist_screener_enabled', 'true', 'bool',
     'When a Fonoloji key is present, also pull a /screener/bist batch (in '
     'addition to /market/stock-movers) so BIST gems are not empty. Set false '
     'to use movers only.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_discovery_bist_pe_max', '25', 'float',
     'BIST gems /screener/bist max P/E filter (empty = no PE cap).',
     NOW(), NOW()),

    ('advisor_config', 'advisor_discovery_bist_roe_min', '15', 'float',
     'BIST gems /screener/bist min ROE % filter (empty = no ROE floor).',
     NOW(), NOW()),

    ('advisor_config', 'advisor_discovery_bist_pb_max', '', 'float',
     'BIST gems /screener/bist max P/B filter (empty = none).',
     NOW(), NOW()),

    ('advisor_config', 'advisor_discovery_bist_div_min', '', 'float',
     'BIST gems /screener/bist min dividend-yield % filter (empty = none).',
     NOW(), NOW()),

    ('advisor_config', 'advisor_discovery_bist_sort_by', 'roe', 'string',
     'BIST gems /screener/bist sort field (e.g. roe, pe, market_cap).',
     NOW(), NOW()),

    ('advisor_config', 'advisor_discovery_bist_sort_order', 'desc', 'string',
     'BIST gems /screener/bist sort order (asc|desc).',
     NOW(), NOW()),

    ('advisor_config', 'advisor_discovery_bist_screener_limit', '30', 'int',
     'BIST gems /screener/bist result limit per refresh.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;
