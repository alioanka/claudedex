-- Migration: ai_feature_store — persisted feature vectors for offline ML training
-- Description: AIStrategy._extract_features writes a row on every signal-generation
-- cycle. Training scripts (train_ai_strategy_scaler.py, train_rug_classifier.py,
-- train_pump_predictor.py) gain a --from-feature-store flag in a separate
-- follow-up that queries this table instead of the synthetic _generate_*
-- helpers. Schema is JSONB-heavy so future feature-shape changes don't
-- require migrations.
-- Date: 2026-05-12

CREATE TABLE IF NOT EXISTS ai_feature_store (
    id              BIGSERIAL PRIMARY KEY,
    token_address   TEXT,                          -- nullable; some flows don't have a single token
    chain           TEXT NOT NULL DEFAULT 'unknown',
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    feature_vector  JSONB NOT NULL,                -- { "scaler_v1": [...], "rug_features": {...}, "pump_features": [...] }
    outcome_label   JSONB,                         -- filled later by a trade-close hook (separate follow-up); e.g. {"pnl_pct": ..., "is_pump": ..., "is_rug": ...}
    side            TEXT,                          -- 'buy' / 'sell' / null
    metadata        JSONB                          -- strategy name, model versions used, regime tag, etc.
);

CREATE INDEX IF NOT EXISTS idx_ai_feature_store_token
    ON ai_feature_store (token_address, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ai_feature_store_timestamp
    ON ai_feature_store (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ai_feature_store_with_outcome
    ON ai_feature_store (timestamp DESC)
    WHERE outcome_label IS NOT NULL;

-- Note: Migration tracking is handled by MigrationManager (schema_migrations table)
