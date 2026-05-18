-- Phase 3 follow-up: ML training data capture.
--
-- Every operator decision (approve/reject) on an orchestrator
-- recommendation is a labeled (features, action, label) triple we
-- want for training a future model. The scorer's metrics blob is
-- the feature vector; the operator's decision is the label. We
-- already store all of this in orchestrator_recommendations — this
-- view + index just makes it easy to query for ML training without
-- changing the underlying schema.
--
-- We capture only ACTUALLY-DECIDED rows (approved IS NOT NULL).
-- 'hold' rows that the operator never touched are excluded because
-- they're not informative (the orchestrator already chose 'hold';
-- operator inaction != reject).

CREATE OR REPLACE VIEW orchestrator_training_data AS
SELECT
    id::text                   AS rec_id,
    created_at,
    module,
    recommended,
    confidence,
    metrics,
    -- Feature columns extracted from the JSONB blob for easy
    -- pandas/sklearn ingestion. Falls back to NULL when a metric
    -- isn't present (older rows pre-Sharpe).
    (metrics->>'closed_trades')::int                          AS closed_trades,
    (metrics->>'total_pnl_usd')::numeric                      AS total_pnl_usd,
    (metrics->>'live_trades')::int                            AS live_trades,
    (metrics->>'btc_24h_change_pct')::numeric                 AS btc_24h_change_pct,
    (metrics->'components'->>'win_rate')::numeric             AS win_rate,
    (metrics->'components'->>'pnl_signal')::numeric           AS pnl_signal,
    (metrics->'components'->>'volume_factor')::numeric        AS volume_factor,
    (metrics->'components'->>'regime_signal')::numeric        AS regime_signal,
    (metrics->'components'->>'sharpe_signal')::numeric        AS sharpe_signal,
    (metrics->'components'->>'sharpe')::numeric               AS sharpe,
    (metrics->>'score')::numeric                              AS score,
    -- The label: did the operator agree with the recommendation?
    approved                  AS operator_agreed,
    approved_at,
    approved_by
FROM orchestrator_recommendations
WHERE approved IS NOT NULL
  AND recommended <> 'hold';

-- Speed up ML training queries that filter by module.
CREATE INDEX IF NOT EXISTS idx_orch_recs_approved_module
    ON orchestrator_recommendations(module, approved, approved_at DESC)
    WHERE approved IS NOT NULL;
