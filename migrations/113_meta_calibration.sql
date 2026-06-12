-- Migration 113: meta_controller — self-improvement calibration table
--
-- The meta controller scores how well its PAST decisions matched the forward
-- realized PnL that followed them (a PAUSE was right if the next window lost
-- money; a KEEP/ACTIVATE was right if it did not), and records a simple
-- hit-rate + per-module detail here every tick. Pure bookkeeping — the
-- operator reads this to judge the controller and tune the meta_config
-- thresholds. It actuates nothing on its own.
--
-- Single-quoted SQL literals only ('' escapes apostrophes). Idempotent.

BEGIN;

CREATE TABLE IF NOT EXISTS meta_calibration (
    id                BIGSERIAL PRIMARY KEY,
    window_hours      INTEGER NOT NULL,
    decisions_scored  INTEGER NOT NULL,
    hit_rate          DOUBLE PRECISION,   -- fraction of prior decisions that matched forward PnL
    detail            JSONB,              -- per-module {prior_decision, forward_pnl_usd, forward_trades, correct}
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_meta_calibration_created ON meta_calibration(created_at DESC);

COMMIT;
