-- Migration 077: advisor sim CHANNELS — per-channel sim tables + caps
--
-- Replaces the per-MARKET sim cap (issue #13, migration 071) with a per-CHANNEL
-- model. A "channel" is an independent sim bucket, each with its own cap, so the
-- operator can run 15 crypto + 15 gems + 15 kap sims simultaneously without one
-- channel starving another.
--
-- Seven channels: crypto, us_equities, bist, fx, midas_funds, gems, kap.
--   * For watchlist advice, channel == the underlying market value.
--   * For discovered advice (origin='discovery', the "New Gems" layer), channel
--     is ALWAYS 'gems' regardless of the underlying market — so a gem on a US
--     ticker counts against the gems cap, NOT us_equities.
--   * For KAP-driven sims (strong-polarity BIST disclosures), channel == 'kap'.
--
-- Schema:
--   * advisor_sim_positions gains a channel VARCHAR(16) column (indexed).
--     Existing rows backfilled: channel = market (and any sim whose source
--     advice has origin='discovery' -> 'gems', where derivable via advice_id).
--   * seeds advisor_sim_cap_per_channel (int, default 15). The old
--     max_sim_positions key is kept as a working FALLBACK/alias (risk_engine
--     reads the per-channel key first, then falls back to max_sim_positions).
--
-- ADVICE-ONLY: sim positions are dry-run bookkeeping; no orders are placed.
-- Additive, idempotent, safe to re-run. No data destroyed.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. channel column on advisor_sim_positions
-- ---------------------------------------------------------------------------
ALTER TABLE advisor_sim_positions
    ADD COLUMN IF NOT EXISTS channel VARCHAR(16);

-- Backfill: default every existing row's channel to its market value.
UPDATE advisor_sim_positions
SET channel = market
WHERE channel IS NULL;

-- Re-route any existing sim whose originating advice was discovered into the
-- 'gems' channel (best-effort; only where the FK + origin column are present).
UPDATE advisor_sim_positions s
SET channel = 'gems'
FROM advisor_advice a
WHERE s.advice_id = a.id
  AND a.origin = 'discovery'
  AND s.channel IS DISTINCT FROM 'gems';

CREATE INDEX IF NOT EXISTS idx_advisor_sim_channel
    ON advisor_sim_positions (channel, status);

-- ---------------------------------------------------------------------------
-- 2. Per-channel sim cap config key (advisor_config).
-- ---------------------------------------------------------------------------
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('advisor_config', 'advisor_sim_cap_per_channel', '15', 'int',
     'Maximum number of OPEN sim positions PER CHANNEL. Channels are independent '
     'sim buckets (crypto, us_equities, bist, fx, midas_funds, gems, kap), each '
     'capped at this value — e.g. 15 crypto + 15 gems + 15 kap simultaneously. '
     'Replaces the per-market max_sim_positions cap (kept as a fallback alias). '
     'Sim positions are dry-run bookkeeping only; no orders are placed.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- Refresh the legacy key's description to note it is now a fallback alias.
UPDATE config_settings
SET description = 'LEGACY/FALLBACK for advisor_sim_cap_per_channel. Used as the '
                 'per-channel cap only when advisor_sim_cap_per_channel is absent. '
                 'Sim positions are dry-run bookkeeping; no orders are placed.',
    updated_at = NOW()
WHERE config_type = 'advisor_config'
  AND key = 'max_sim_positions';

COMMIT;

-- ---------------------------------------------------------------------------
-- DOWN (manual rollback)
-- ---------------------------------------------------------------------------
-- BEGIN;
-- DROP INDEX IF EXISTS idx_advisor_sim_channel;
-- ALTER TABLE advisor_sim_positions DROP COLUMN IF EXISTS channel;
-- DELETE FROM config_settings WHERE config_type='advisor_config'
--   AND key='advisor_sim_cap_per_channel';
-- COMMIT;
