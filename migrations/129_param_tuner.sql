-- Migration 129: param_tuner — config seeds + proposals + bandit-state tables
--
-- Bandit-based SHADOW-ONLY self-tuning of registered module knobs
-- (modules/param_tuner/). A deterministic UCB1 bandit per registered
-- tunable accumulates rolling-performance evidence per discretized value
-- and writes value-change PROPOSALS to param_proposals for operator
-- approval. Applying this migration changes NO behavior: the module only
-- runs when PARAM_TUNER_MODULE_ENABLED=true, and even then it only
-- WRITES proposal rows until auto_apply_enabled (seeded false here) is
-- explicitly flipped. Auto-apply, when ever enabled, may only update a
-- config_settings value inside the operator-set [min, max] bounds below,
-- only for kind=''exploit'' evidence-backed proposals, logged to
-- config_history and reversible. It never trades, never touches
-- logs/.killswitch, and risk knobs are hard-excluded in code.
--
-- HONESTY: rewards come from whatever value was actually running, so
-- paper-tuned proposals overfit the window they were learned on.
-- Proposals are hypotheses requiring DRY_RUN validation, not verified
-- improvements — that is why shadow-only is the default posture.
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes
-- are identifiers and previously halted a deploy). Idempotent.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('param_tuner', 'auto_apply_enabled', 'false', 'bool',
     'Master auto-apply switch. When false (default) the tuner ONLY writes '
     'proposal rows to param_proposals. When true it may update a '
     'config_settings value, but only within the operator min/max bounds in '
     'tunable_registry, only for evidence-backed (exploit) proposals, with '
     'cooldown, logged to config_history and reversible. It NEVER touches '
     'code, env flags, risk knobs, or the killswitch.', NOW(), NOW()),

    ('param_tuner', 'tick_interval_seconds', '3600', 'int',
     'Seconds between tuner cycles (default 1h; knob evidence accrues '
     'slowly — there is no value in fast ticks).', NOW(), NOW()),

    ('param_tuner', 'lookback_hours', '24', 'int',
     'Rolling closed-trade window used to score the owning module when '
     'rewarding the bandit arm of the currently configured value.',
     NOW(), NOW()),

    ('param_tuner', 'reward_window_hours', '6', 'float',
     'Minimum hours between two reward observations of the SAME tunable '
     '(overlapping windows are correlated; rewarding every tick would '
     'multiply-count the same trades).', NOW(), NOW()),

    ('param_tuner', 'reward_min_trades', '5', 'int',
     'Closed-trade floor in the window below which no reward is recorded '
     '(thin tape teaches the bandit nothing but noise).', NOW(), NOW()),

    ('param_tuner', 'n_target', '50', 'int',
     'Closed-trade count at which the track score saturates (same '
     'convention as meta_config.n_target).', NOW(), NOW()),

    ('param_tuner', 'exploration_c', '0.5', 'float',
     'UCB1 exploration constant c in mean + c*sqrt(2 ln N / n). Higher = '
     'more explore-type proposals for under-sampled values.', NOW(), NOW()),

    ('param_tuner', 'min_pulls_exploit', '3', 'int',
     'Minimum reward observations an alternative value needs before a '
     'proposal may be labeled exploit (the only kind auto-apply may ever '
     'touch). Below this it is labeled explore = hypothesis only.',
     NOW(), NOW()),

    ('param_tuner', 'improvement_margin', '0.05', 'float',
     'Absolute mean-reward gain (scores live in 0..1) an alternative must '
     'show over the current value before an exploit proposal is written '
     '(anti-churn).', NOW(), NOW()),

    ('param_tuner', 'max_open_proposals_per_key', '1', 'int',
     'Pending-proposal cap per tunable so an unattended tuner cannot spam '
     'the approval queue.', NOW(), NOW()),

    ('param_tuner', 'auto_apply_cooldown_hours', '24', 'int',
     'Minimum hours between two auto-applies of the SAME tunable '
     '(anti-flap; irrelevant while auto_apply_enabled=false).', NOW(), NOW()),

    ('param_tuner', 'tunable_registry', '[
  {"module": "dex", "config_type": "trading", "key": "min_vol_liq_ratio",
   "min": 0.01, "max": 0.50, "steps": 7, "value_type": "float"},
  {"module": "ai", "config_type": "ai_config", "key": "confidence_threshold",
   "min": 0.20, "max": 0.60, "steps": 5, "value_type": "float"},
  {"module": "sniper", "config_type": "sniper_config", "key": "max_hold_minutes",
   "min": 15, "max": 240, "steps": 6, "value_type": "int"}
]', 'json',
     'Operator-owned whitelist of tunables the bandit may reason about, '
     'each with HARD min/max bounds the tuner can never exceed. Entry '
     'fields: module (trade-table owner for reward scoring), config_type + '
     'key (the config_settings row), min/max (bounds), steps (grid size), '
     'value_type (float or int). NON-RISK knobs only — entries targeting '
     'risk_management/security/wallets or stop-loss/leverage/live-execution '
     'keys are rejected in code even if listed here. Seeded with three '
     'documented entry-side knobs: DEX vol/liq gate, AI confidence gate, '
     'sniper max hold.', NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- One row per proposed value change. status lifecycle:
--   pending -> operator_applied | dismissed | superseded (operator actions)
--   pending -> auto_applied (only when auto_apply_enabled; old_value is
--              preserved in components for one-step reversal) -> reverted
CREATE TABLE IF NOT EXISTS param_proposals (
    id             BIGSERIAL PRIMARY KEY,
    config_type    TEXT NOT NULL,
    key            TEXT NOT NULL,
    module         TEXT NOT NULL,
    current_value  TEXT NOT NULL,
    proposed_value TEXT NOT NULL,
    bound_min      DOUBLE PRECISION NOT NULL,
    bound_max      DOUBLE PRECISION NOT NULL,
    kind           TEXT NOT NULL,           -- 'exploit' | 'explore'
    reason         TEXT,
    components     JSONB NOT NULL DEFAULT '{}'::jsonb,
    status         TEXT NOT NULL DEFAULT 'pending',
    applied_at     TIMESTAMPTZ,
    reverted_at    TIMESTAMPTZ,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_param_proposals_key_created
    ON param_proposals(config_type, key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_param_proposals_status
    ON param_proposals(status, created_at DESC);

-- Durable bandit state, one row per registered tunable. state is the
-- JSON round-trip of modules.param_tuner.core.bandit.to_dict (arms with
-- per-value pulls/reward sums); last_reward_at throttles correlated
-- reward windows across restarts.
CREATE TABLE IF NOT EXISTS param_bandit_state (
    config_type    TEXT NOT NULL,
    key            TEXT NOT NULL,
    module         TEXT NOT NULL,
    state          JSONB NOT NULL,
    last_reward_at TIMESTAMPTZ,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (config_type, key)
);

COMMIT;
