-- Migration 136: low-priority follow-up tuning seeds
--
-- clmm_lp.max_annual_vol: ceiling on the realized-vol estimator so a noisy
-- thin-pool price series can't produce absurd net-APR (e.g. -192417%). The
-- code default is 5.0 (=500% annualized, already extreme); this seeds the row
-- so it is visible + tunable on the /config/clmm_lp page. Idempotent.
--
-- (The other follow-ups in this wave — sentinel single-source depeg filter,
-- catalyst dead-source suppression, smart_money ingest diagnostics — are pure
-- code changes and need no new config.)
--
-- Single-quoted SQL literals only ('' escapes apostrophes). Idempotent.

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('clmm_lp', 'max_annual_vol', '5.0', 'float',
     'Ceiling on the realized annualized-vol estimate (a noisy thin-pool price '
     'series can spike vol > 20 and produce nonsense net-APR). 5.0 = 500%, '
     'already extreme; readings above are capped (logged as realized_capped).',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;
