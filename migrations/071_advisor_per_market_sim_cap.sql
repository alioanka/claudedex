-- Migration 071: advisor sim cap is now PER-MARKET — issue #13
--
-- max_sim_positions (config_type='advisor_config') was enforced as ONE GLOBAL
-- cap across all markets (count_open_sims() summed every market). The advice
-- engine now passes the PER-MARKET open-sim count to the risk gate, so the same
-- key is the cap PER market/strategy (e.g. 10 crypto + 10 BIST + 10 US, each
-- independent). The config key is unchanged — only its meaning narrowed.
--
-- The operator wants 10 per market. We update the value to 10 ONLY if it is
-- still at the old shipped default of 20, so we do NOT clobber a value the
-- operator has already customised. Idempotent and safe to re-run.
--
-- Does NOT alter the shipped seed in migration 058.
-- ADVICE-ONLY: sim positions are dry-run bookkeeping; no orders are placed.

BEGIN;

UPDATE config_settings
SET value = '10', updated_at = NOW()
WHERE config_type = 'advisor_config'
  AND key = 'max_sim_positions'
  AND value = '20';

-- Refresh the description to reflect the per-market semantics (no value change).
UPDATE config_settings
SET description = 'Maximum number of OPEN sim positions PER MARKET/STRATEGY '
                 '(e.g. 10 crypto + 10 BIST + 10 US, enforced independently per '
                 'market — NOT one global cap). Sim positions are dry-run '
                 'bookkeeping only; no orders are placed.',
    updated_at = NOW()
WHERE config_type = 'advisor_config'
  AND key = 'max_sim_positions';

COMMIT;

-- ---------------------------------------------------------------------------
-- DOWN (manual rollback) — restores the global-cap default value only.
-- ---------------------------------------------------------------------------
-- BEGIN;
-- UPDATE config_settings SET value='20', updated_at=NOW()
--   WHERE config_type='advisor_config' AND key='max_sim_positions' AND value='10';
-- COMMIT;
