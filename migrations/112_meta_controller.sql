-- Migration 112: meta_controller — config seeds + meta_decisions table
--
-- The self-deciding advisory layer (modules/meta_controller/). Reads every
-- trading module's rolling DRY_RUN + LIVE performance, writes a transparent
-- ACTIVATE / KEEP / PAUSE decision per module to meta_decisions, and (only
-- when meta_autopilot_enabled=true) actuates pause/resume flags with a dwell
-- guard. ADVISORY BY DEFAULT — applying this migration changes NO behavior;
-- the module only runs when META_CONTROLLER_MODULE_ENABLED=true, and even then
-- it only WRITES decision rows until autopilot is explicitly enabled. It never
-- trades and never touches logs/.killswitch.
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('meta_config', 'meta_autopilot_enabled', 'false', 'bool',
     'Master autopilot switch. When false (default) the meta controller only '
     'WRITES advisory decisions. When true it may write/clear logs/.pause_<module> '
     'to actuate PAUSE/ACTIVATE (subject to the dwell guard). It NEVER places a '
     'trade and NEVER touches the killswitch.', NOW(), NOW()),

    ('meta_config', 'tick_interval_seconds', '900', 'int',
     'Seconds between meta decision cycles (default 15 min).', NOW(), NOW()),

    ('meta_config', 'lookback_hours', '24', 'int',
     'Rolling performance window used for scoring + calibration.', NOW(), NOW()),

    ('meta_config', 'autopilot_dwell_minutes', '720', 'int',
     'Minimum minutes between two autopilot actuations of the SAME module '
     '(anti-flap; default 12h). Decisions are still recorded every tick.',
     NOW(), NOW()),

    ('meta_config', 'pause_score', '0.35', 'float',
     'Blended health score (0..1) at or below which a module is judged PAUSE.',
     NOW(), NOW()),

    ('meta_config', 'activate_score', '0.55', 'float',
     'Health score at or above which a PAUSED module is judged ACTIVATE. The '
     'gap to pause_score is anti-flap hysteresis.', NOW(), NOW()),

    ('meta_config', 'min_confidence', '0.5', 'float',
     'Below this data-sufficiency confidence, a PAUSE/ACTIVATE is downgraded to '
     'an advisory KEEP (thin data never actuates).', NOW(), NOW()),

    ('meta_config', 'pause_loss_usd', '25.0', 'float',
     'Hard rule: LIVE window loss beyond this (or 2x on a dry-only track) forces '
     'PAUSE regardless of score.', NOW(), NOW()),

    ('meta_config', 'min_trades', '10', 'int',
     'Closed-trade floor in the window below which the decision is always KEEP '
     '(confidence 0, never actuated).', NOW(), NOW()),

    ('meta_config', 'live_weight', '0.7', 'float',
     'Blend weight of the LIVE track when both LIVE and DRY exist (LIVE evidence '
     'dominates real-money judgement).', NOW(), NOW()),

    ('meta_config', 'n_target', '50', 'int',
     'Closed-trade count at which scoring confidence saturates to 1.0.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- One row per module per tick (full audit history; dashboard reads the latest).
CREATE TABLE IF NOT EXISTS meta_decisions (
    id            BIGSERIAL PRIMARY KEY,
    module        TEXT NOT NULL,
    decision      TEXT NOT NULL,           -- 'activate' | 'keep' | 'pause'
    health_score  DOUBLE PRECISION,
    confidence    DOUBLE PRECISION,
    reason        TEXT,
    components    JSONB,
    actuated      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_meta_decisions_module_created ON meta_decisions(module, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_meta_decisions_created ON meta_decisions(created_at DESC);

COMMIT;
