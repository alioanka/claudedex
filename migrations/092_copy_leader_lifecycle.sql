-- Migration 092: COPY_TRADING leader lifecycle — dead-leader flagging,
-- periodic leader refresh, auto-discovery candidates, and the
-- signal-age coherence buffer.
--
-- Context (operator-reported): the COPY module produced 6 trades in a
-- week from 33 leader wallets. Funnel diagnosis found three chokepoints:
--   (1) copy_max_signal_age_s=5s vs copy_poll_interval_s=15s — the
--       staleness gate rejected nearly every detected swap. Fixed in
--       copy_engine._effective_signal_age_s(); the buffer is seeded here.
--   (2) Dead leader wallets — many of the 33 targets no longer trade.
--       This migration adds activity/dead-flag columns so the engine's
--       periodic leader refresh can surface a "dead leader" list for
--       operator-driven pruning. The engine NEVER auto-removes targets.
--   (3) No pipeline for replacing dead leaders — copy_leader_candidates
--       receives discovery proposals (status='pending') for OPERATOR
--       APPROVAL. An unapproved candidate is never traded; approval
--       means the operator manually adds the wallet to target_wallets.

-- ---------------------------------------------------------------------
-- 1. Leader activity / dead-flag columns on copy_leader_scores
-- ---------------------------------------------------------------------
ALTER TABLE copy_leader_scores
    ADD COLUMN IF NOT EXISTS last_activity_at  TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS is_dead           BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS dead_flagged_at   TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS dead_reason       TEXT;

COMMENT ON COLUMN copy_leader_scores.last_activity_at IS
    'Most recent on-chain tx blockTime observed for this leader by the copy monitor (any tx, not only mirrored ones)';
COMMENT ON COLUMN copy_leader_scores.is_dead IS
    'Auto-flagged by the engine leader-refresh job when no on-chain activity for copy_dead_leader_days. Advisory: suggests removal from target_wallets; the engine never auto-removes.';

CREATE INDEX IF NOT EXISTS idx_copy_leader_scores_dead
    ON copy_leader_scores (is_dead) WHERE is_dead;

-- ---------------------------------------------------------------------
-- 2. Discovery candidates awaiting operator approval
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS copy_leader_candidates (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chain           VARCHAR(20)  NOT NULL,
    wallet_address  VARCHAR(100) NOT NULL,
    source          VARCHAR(40)  NOT NULL,            -- discovery source id
    label           VARCHAR(120),
    score           NUMERIC(6, 2),                    -- leader_scorer composite at proposal time
    metrics         JSONB,                            -- raw LeaderMetrics snapshot
    status          VARCHAR(20)  NOT NULL DEFAULT 'pending',  -- pending | approved | rejected
    proposed_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    reviewed_at     TIMESTAMPTZ,
    reviewed_by     VARCHAR(80),
    UNIQUE (chain, wallet_address)
);

COMMENT ON TABLE copy_leader_candidates IS
    'Auto-discovery proposals for new copy-trading leaders. NEVER traded directly: the engine only mirrors wallets in copytrading_config.target_wallets, which the operator edits manually after reviewing a candidate.';

CREATE INDEX IF NOT EXISTS idx_copy_leader_candidates_status
    ON copy_leader_candidates (status, score DESC NULLS LAST);

-- ---------------------------------------------------------------------
-- 3. Config seeds (ON CONFLICT DO NOTHING: operator overrides survive)
-- ---------------------------------------------------------------------
INSERT INTO config_settings (config_type, key, value, value_type)
VALUES
    -- Chokepoint (1): indexing-lag buffer for the effective staleness
    -- limit max(copy_max_signal_age_s, copy_poll_interval_s + buffer).
    ('copytrading_config', 'copy_signal_age_buffer_s',            '10',    'number'),
    -- Chokepoint (2): periodic leader refresh + dead-leader flagging.
    ('copytrading_config', 'copy_leader_refresh_enabled',         'true',  'boolean'),
    ('copytrading_config', 'copy_leader_refresh_interval_hours',  '6',     'number'),
    ('copytrading_config', 'copy_dead_leader_days',               '7',     'number'),
    -- Chokepoint (3): auto-discovery proposals. OFF by default —
    -- proposals only; never trades an unapproved wallet either way.
    ('copytrading_config', 'copy_auto_discovery_enabled',         'false', 'boolean'),
    ('copytrading_config', 'copy_auto_discovery_interval_hours',  '24',    'number'),
    ('copytrading_config', 'copy_auto_discovery_min_score',       '60',    'number'),
    ('copytrading_config', 'copy_auto_discovery_max_candidates',  '10',    'number')
ON CONFLICT (config_type, key) DO NOTHING;
