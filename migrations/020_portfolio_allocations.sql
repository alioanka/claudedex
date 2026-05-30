-- Phase 4B: per-module capital allocation history.
--
-- Each row is one allocation snapshot, either:
--   - proposed_by='allocator': the rebalance engine's recommendation
--   - proposed_by='operator' : an operator-driven manual allocation
--
-- The current effective allocation per module is the row with the
-- max approved_at (and effective_until either NULL or in the future).
-- We persist proposals + approvals so the operator can see "the
-- allocator wanted to bump sniper to 35% but I capped it at 25%".

CREATE TABLE IF NOT EXISTS portfolio_allocations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    module          VARCHAR(32) NOT NULL,
    pct_of_book     NUMERIC(5, 2) NOT NULL,            -- 0.00 .. 100.00
    usd_amount      NUMERIC(20, 4) NOT NULL,
    proposed_by     VARCHAR(16) NOT NULL,              -- 'allocator' | 'operator'
    reason          TEXT,
    metrics         JSONB,                             -- inputs the proposal was based on
    -- approval state. NULL = proposed but not approved; non-NULL = applied.
    approved_at     TIMESTAMP,
    approved_by     VARCHAR(64),
    -- Optional expiry. When set, this allocation only applies until this time;
    -- after that, the next-most-recent approved row wins.
    effective_until TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_portfolio_allocations_module_approved
    ON portfolio_allocations(module, approved_at DESC NULLS LAST)
    WHERE approved_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_portfolio_allocations_pending
    ON portfolio_allocations(module, created_at DESC)
    WHERE approved_at IS NULL;
