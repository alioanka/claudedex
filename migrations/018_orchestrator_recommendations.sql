-- Phase 3 D2: AI/ML orchestrator recommendations table.
--
-- The orchestrator subprocess (modules/orchestrator_ai/) computes a
-- per-module performance score from recent *_runtime_stats + closed
-- *_trades rows + market state, then writes a recommendation into
-- this table. The dashboard surfaces pending rows; the operator
-- approves each one explicitly. Approval flips the corresponding
-- config_settings.<module>_config.dry_run row (or _MODULE_ENABLED env).
--
-- The orchestrator is ADVISORY-ONLY at this stage. No auto-action
-- until we have ≥30 days of operator-approval history showing the
-- recommendations are well-calibrated.

CREATE TABLE IF NOT EXISTS orchestrator_recommendations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    module          VARCHAR(32) NOT NULL,             -- 'sniper' / 'arbitrage' / ...
    recommended     VARCHAR(16) NOT NULL,             -- 'enable' | 'disable' | 'to_dry' | 'to_live' | 'hold'
    confidence      NUMERIC(4, 3) NOT NULL,           -- 0.000..1.000
    reason          TEXT NOT NULL,                    -- short explanation
    metrics         JSONB,                            -- snapshot of the scoring inputs
    -- approval state. NULL = pending; TRUE = approved + applied;
    -- FALSE = explicitly rejected by operator (kept for audit).
    approved        BOOLEAN,
    approved_at     TIMESTAMP,
    approved_by     VARCHAR(64),
    -- whether the orchestrator has expired this recommendation (e.g.
    -- market moved before operator acted). NULL = still actionable.
    superseded_at   TIMESTAMP,
    superseded_by   UUID REFERENCES orchestrator_recommendations(id)
);

CREATE INDEX IF NOT EXISTS idx_orch_recs_pending
    ON orchestrator_recommendations(module, created_at DESC)
    WHERE approved IS NULL AND superseded_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_orch_recs_module_time
    ON orchestrator_recommendations(module, created_at DESC);
