-- Migration: ai_confidence_calibration — track LLM-predicted sentiment/confidence
-- vs realised PnL per AI trade. Source for /api/ai/calibration reliability
-- diagram + Brier score endpoint (A6 E2). Open-row written by
-- sentiment_engine._execute_trade; realised columns backfilled by
-- _close_position.
-- Date: 2026-05-19

CREATE TABLE IF NOT EXISTS ai_confidence_calibration (
    id                   BIGSERIAL PRIMARY KEY,
    trade_id             TEXT NOT NULL,
    provider             TEXT,                  -- 'openai' / 'anthropic' / 'both' / null
    model                TEXT,                  -- model_id at decision time
    predicted_score      DOUBLE PRECISION,      -- raw sentiment score at entry, -1..1
    predicted_confidence DOUBLE PRECISION,      -- |predicted_score|, 0..1 (denormalised for query speed)
    realized_pnl_pct     DOUBLE PRECISION,      -- backfilled at close
    realized_won         BOOLEAN,               -- backfilled at close
    quorum_required      BOOLEAN,               -- whether quorum gate was active at the time
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at            TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_ai_calibration_trade
    ON ai_confidence_calibration (trade_id);
CREATE INDEX IF NOT EXISTS idx_ai_calibration_created
    ON ai_confidence_calibration (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_calibration_closed
    ON ai_confidence_calibration (closed_at DESC)
    WHERE realized_won IS NOT NULL;
