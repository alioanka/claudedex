-- ============================================================================
-- One-time repair: advisor + KAP schema (fixes 058/062/063 "already exists")
-- ============================================================================
-- Symptom (scripts/migrate_database.py):
--   058 -> duplicate key value violates unique constraint "pg_class_relname_nsp_index"
--          DETAIL: Key (relname, ...)=(advisor_advice_id_seq, ...) already exists.
--   062 -> ... (kap_company_profiles_id_seq, ...) already exists.
--   063 -> duplicate key ... "pg_type_typname_nsp_index" (kap_event_taxonomy) already exists.
--
-- Cause: a prior, UNRECORDED migration run left the advisor/KAP schema in a
-- mixed partial state. Some tables exist (e.g. advisor_sim_positions, which
-- migration 060 successfully ALTERed), while others are absent but left an
-- orphaned sequence / rowtype behind. `CREATE TABLE IF NOT EXISTS` only checks
-- for the TABLE in pg_class, so when the table is absent it proceeds and then
-- collides creating the already-present sequence/type.
--
-- Safety: the advisor module has NEVER started successfully (it crash-looped
-- on a config-typing bug), and the KAP listener never ran, so advisor_advice /
-- advisor_sim_positions / all kap_* tables hold ZERO rows and are safe to drop
-- and recreate. advisor_portfolio (operator-entered holdings, populated only
-- via the dashboard) is PRESERVED here — only its orphaned sequence is cleaned,
-- and only if the table itself is absent.
--
-- AFTER running this, re-run the migrator to recreate everything in order:
--   docker exec trading-bot python scripts/migrate_database.py
--
-- To apply this repair:
--   docker exec -i trading-postgres psql \
--     -U "$(docker exec trading-postgres cat /run/secrets/db_user)" \
--     -d tradingbot < scripts/repair_advisor_kap_schema.sql
-- ============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Regenerable advisor tables (empty) + any orphaned sequences.
-- ---------------------------------------------------------------------------
DROP TABLE    IF EXISTS advisor_advice          CASCADE;
DROP SEQUENCE IF EXISTS advisor_advice_id_seq   CASCADE;

DROP TABLE    IF EXISTS advisor_sim_positions        CASCADE;
DROP SEQUENCE IF EXISTS advisor_sim_positions_id_seq CASCADE;

-- advisor_portfolio: PRESERVE the table (it may hold operator-entered
-- holdings). Only clean an orphaned sequence if the table is absent.
DO $$
BEGIN
    IF to_regclass('public.advisor_portfolio') IS NULL THEN
        DROP SEQUENCE IF EXISTS advisor_portfolio_id_seq CASCADE;
    END IF;
END $$;

-- ---------------------------------------------------------------------------
-- 2. KAP tables (empty) + orphaned sequences / rowtype. Drop in FK-dependency
--    order; CASCADE covers any remaining dependencies.
-- ---------------------------------------------------------------------------
DROP TABLE    IF EXISTS kap_classifications          CASCADE;
DROP SEQUENCE IF EXISTS kap_classifications_id_seq   CASCADE;

DROP TABLE    IF EXISTS kap_returns                  CASCADE;
DROP SEQUENCE IF EXISTS kap_returns_id_seq           CASCADE;

DROP TABLE    IF EXISTS kap_disclosures              CASCADE;
DROP SEQUENCE IF EXISTS kap_disclosures_id_seq       CASCADE;

DROP TABLE    IF EXISTS kap_company_profiles         CASCADE;
DROP SEQUENCE IF EXISTS kap_company_profiles_id_seq  CASCADE;

DROP TABLE    IF EXISTS kap_event_taxonomy           CASCADE;
DROP TYPE     IF EXISTS kap_event_taxonomy           CASCADE;

-- ---------------------------------------------------------------------------
-- 3. Un-record every advisor/KAP migration that carries DDL we just dropped,
--    so the migrator RE-RUNS them and recreates the tables in order:
--      058 -> advisor_advice + advisor_sim_positions (+ advisor_portfolio,
--             which CREATE TABLE IF NOT EXISTS skips since we preserved it)
--      060 -> ALTER advisor_sim_positions ADD horizon_end_date
--      062 -> kap_company_profiles / kap_disclosures / kap_returns
--      063 -> kap_event_taxonomy / kap_classifications
--    All four are idempotent (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS /
--    ON CONFLICT), so replay is safe. 061 and 064 are config-only seeds
--    (ON CONFLICT) and need no replay.
--
--    NOTE: these may or may not currently be recorded depending on how many
--    times the migrator has run — DELETE is a no-op when absent, so this is
--    safe to run repeatedly.
-- ---------------------------------------------------------------------------
DELETE FROM migrations WHERE version IN (
    '058_advisor_module',
    '060_advisor_ml_sim_config',
    '062_kap_ingestion',
    '063_kap_classifications'
);

COMMIT;
