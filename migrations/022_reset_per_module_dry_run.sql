-- Phase 4 follow-up: seed per-module dry_run rows from the global
-- DRY_RUN env var (effectively).
--
-- Why this exists:
-- The operator hit a case where arbitrage_config.dry_run='False' in
-- DB (from an earlier branch's experiments) while their .env had
-- DRY_RUN=true globally. The /full-dashboard correctly showed
-- arbitrage as LIVE based on the DB row, but the operator expected
-- DRY because they only configured the global flag. This row resets
-- every per-module dry_run to 'true' (safe default) so a fresh
-- deploy starts uniformly DRY across all modules.
--
-- Safe to re-run: ON CONFLICT updates only existing rows that say
-- 'False' (rows already 'true' stay 'true' — non-destructive). To
-- explicitly opt one module into LIVE, set <MODULE>_DRY_RUN=false in
-- .env OR POST /api/modules/{m}/dry-run {dry_run: false} AFTER this
-- migration has run.

INSERT INTO config_settings (config_type, key, value, value_type) VALUES
  ('sniper_config',       'dry_run', 'true', 'bool'),
  ('arbitrage_config',    'dry_run', 'true', 'bool'),
  ('copytrading_config',  'dry_run', 'true', 'bool'),
  ('futures_config',      'dry_run', 'true', 'bool'),
  ('solana_config',       'dry_run', 'true', 'bool'),
  ('dex_config',          'dry_run', 'true', 'bool'),
  ('ai_config',           'dry_run', 'true', 'bool')
ON CONFLICT (config_type, key) DO UPDATE
  SET value = 'true'
  WHERE config_settings.value <> 'true';
