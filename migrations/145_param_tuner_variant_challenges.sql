-- Migration 145: param_tuner variant challenges — out-of-sample acceptance
--
-- Challenge/variant evaluation pattern (adapted from HKUDS/AI-Trader, MIT)
-- as the acceptance gate for bandit proposals: instead of moving a proposal
-- straight to the pending-operator-approval queue on counterfactual reward
-- alone, the tuner opens a CHALLENGE — BASELINE (current value) vs VARIANT
-- (proposed value) scored side by side against the owning module''s REAL
-- closed trades entered AFTER the challenge opened (out-of-sample by
-- construction; no counterfactual replay). Only a PASSED challenge moves
-- the proposal to ''pending''; a failed one auto-expires as
-- ''challenge_failed''. This answers the mig-129 honesty note that
-- counterfactual rewards overfit the window they were learned on.
--
-- Applying this migration changes NO live behavior: the module only runs
-- when PARAM_TUNER_MODULE_ENABLED=true; challenges only make proposal
-- acceptance STRICTER (an extra out-of-sample gate before the operator
-- queue); auto_apply_enabled stays false and challenge-routed proposals
-- are NEVER auto-applied even when it is true. Knobs without an honest
-- observable proxy (documented in code: modules/param_tuner/core/
-- challenge.py CHALLENGE_EXEMPT) bypass the challenge and keep the old
-- direct-to-pending behavior, logged.
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes
-- are identifiers and previously halted a deploy). Idempotent.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('param_tuner', 'challenge_enabled', 'true', 'bool',
     'When true (default), bandit proposals for knobs with an honest '
     'out-of-sample proxy open a baseline-vs-variant challenge scored on '
     'the module''s real closed trades before reaching the pending queue. '
     'Strictly an EXTRA acceptance gate: false restores the old '
     'direct-to-pending flow. Never affects auto-apply semantics '
     '(challenge-routed proposals are never auto-applied).', NOW(), NOW()),

    ('param_tuner', 'challenge_window_hours', '72', 'float',
     'Out-of-sample scoring window of a variant challenge. Both baseline '
     'and variant are marked against closed trades entered inside this '
     'window; at window end the verdict is final (no extensions — thin '
     'tape fails the challenge, it does not pass it).', NOW(), NOW()),

    ('param_tuner', 'challenge_min_edge_pct', '5', 'float',
     'Relative edge the variant''s mean out-of-sample reward must show '
     'over the baseline''s (percent of baseline mean) for a PASS verdict. '
     'Conservative on purpose: ties and small edges fail.', NOW(), NOW()),

    ('param_tuner', 'challenge_min_samples', '20', 'int',
     'Minimum out-of-sample closed-trade observations a challenge needs '
     'by window end. Below this the verdict is challenge_failed — '
     'insufficient evidence is a fail, never a pass.', NOW(), NOW()),

    ('param_tuner', 'challenge_retry_cooldown_hours', '72', 'int',
     'After a failed challenge, the same (knob, proposed value) pair may '
     'not open a new challenge for this many hours. The bandit is '
     'deterministic, so without this it would re-propose the identical '
     'value next tick and churn the challenge table.', NOW(), NOW()),

    ('param_tuner', 'challenge_pnl_cap_pct', '20', 'float',
     'Per-trade realized P&L percent at which the challenge''s reward '
     'mapping saturates (reward = 0.5 + 0.5 * clamp(pnl_pct/cap, -1, 1)). '
     'Keeps one moonshot or one rug from single-handedly deciding a '
     'verdict.', NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- One row per opened challenge. Running stats are accumulated tick by
-- tick (last_scored_at is the incremental cursor over trade exit times;
-- only trades ENTERED after started_at ever count — out-of-sample cut).
-- verdict lifecycle: NULL (running) -> passed | challenge_failed |
-- abandoned (proposal dismissed/vanished mid-challenge).
CREATE TABLE IF NOT EXISTS param_variant_challenges (
    id                  BIGSERIAL PRIMARY KEY,
    proposal_id         BIGINT NOT NULL REFERENCES param_proposals(id),
    config_type         TEXT NOT NULL,
    key                 TEXT NOT NULL,
    module              TEXT NOT NULL,
    baseline_value      TEXT NOT NULL,
    variant_value       TEXT NOT NULL,
    n_samples           INT NOT NULL DEFAULT 0,
    baseline_reward_sum DOUBLE PRECISION NOT NULL DEFAULT 0,
    variant_reward_sum  DOUBLE PRECISION NOT NULL DEFAULT 0,
    last_scored_at      TIMESTAMPTZ,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    window_hours        DOUBLE PRECISION NOT NULL DEFAULT 72,
    resolved_at         TIMESTAMPTZ,
    verdict             TEXT,
    components          JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_param_variant_challenges_open
    ON param_variant_challenges(resolved_at) WHERE resolved_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_param_variant_challenges_key
    ON param_variant_challenges(config_type, key, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_param_variant_challenges_proposal
    ON param_variant_challenges(proposal_id);

COMMIT;
