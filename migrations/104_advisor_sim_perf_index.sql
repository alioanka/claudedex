-- Migration 104: index for advisor sim-performance lookback queries
--
-- /api/advisor/performance filters terminal sims by closed_at inside a
-- lookback window (status = 'open' OR closed_at >= NOW() - N days) and the
-- equity curve orders decided sims by closed_at. Migration 058 indexed
-- (status) and 077 indexed (channel, status); neither covers closed_at, so
-- the lookback degraded to a sequential scan as the sim book grows.
--
-- Partial index on terminal rows only: open rows are matched by the
-- existing status index, and terminal rows are exactly the ones probed by
-- closed_at. Idempotent: CREATE INDEX IF NOT EXISTS; safe to re-run.
-- NOTE: single-quoted string literals only (double quotes = identifiers).

BEGIN;

CREATE INDEX IF NOT EXISTS idx_advisor_sim_perf_closed_at
    ON advisor_sim_positions (closed_at DESC)
    WHERE status IN ('closed', 'expired');

COMMIT;

-- Rollback (manual):
-- BEGIN;
-- DROP INDEX IF EXISTS idx_advisor_sim_perf_closed_at;
-- COMMIT;
