-- Migration 103: advisor sim-performance panel config seeds (dashboard v2)
--
-- Seeds the two knobs consumed by GET /api/advisor/performance
-- (monitoring/enhanced_dashboard.py), whose math lives in the pure,
-- self-tested helpers in modules/advisor/core/performance.py.
-- ADVICE-ONLY observability: these keys only shape how dry-run sim
-- bookkeeping is summarized; nothing here enables execution.
--
-- Idempotent: ON CONFLICT (config_type, key) DO NOTHING — operator
-- overrides and re-runs are safe.
-- NOTE: all SQL string literals below are single-quoted; apostrophes are
-- escaped as '' (double quotes are identifiers in Postgres and would be a
-- syntax error here).

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
(
    'advisor_config',
    'advisor_perf_lookback_days',
    '90',
    'int',
    'Default lookback window (days over closed_at) for the advisor sim performance panel and /api/advisor/performance. 0 = all history. OPEN sims are always included regardless of the window. The dashboard''s lookback selector overrides per request.'
),
(
    'advisor_config',
    'advisor_perf_equity_max_points',
    '500',
    'int',
    'Maximum number of points returned in the advisor equity curve. Longer histories are stride-downsampled; the final point always carries the exact end-state equity. Floor of 10 enforced in code.'
)
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;

-- Rollback (manual):
-- BEGIN;
-- DELETE FROM config_settings WHERE config_type='advisor_config'
--   AND key IN ('advisor_perf_lookback_days','advisor_perf_equity_max_points');
-- COMMIT;
