-- Wave-3 COPY_TRADING enhancements (CT-W3-01..03)
--
-- 1. copy_slippage_observations -- per-mirrored-trade record of
--    (leader_fill_price, our_fill_price, delta_ms). Drives the
--    /api/copytrading/slippage rolling-median view and the
--    discovery-page slippage chart so operators can see which
--    leaders are too fast to mirror profitably.
--
-- 2. copy_leader_scores.on_probation + probation_until + probation_reason --
--    set by the engine when (a) a leader's composite score drops below the
--    operator threshold OR (b) a mirrored trade goes worse than the
--    operator-configured negative-PnL threshold. While `on_probation`
--    is TRUE and `probation_until` is in the future, the engine refuses
--    new BUYs from this leader (SELLs still allowed -- they reduce
--    exposure). Re-entry is automatic on `probation_until` expiry.
--
-- Both pieces are additive -- existing readers and writers see no schema
-- breakage. NULL columns degrade to "no probation" / "no slippage data"
-- gracefully.

-- ----- 1. slippage observations --------------------------------------
CREATE TABLE IF NOT EXISTS copy_slippage_observations (
    id                      BIGSERIAL PRIMARY KEY,
    chain                   VARCHAR(20)  NOT NULL,
    leader_wallet           VARCHAR(100) NOT NULL,
    token_address           VARCHAR(100) NOT NULL,
    side                    VARCHAR(10)  NOT NULL,        -- 'buy' | 'sell'
    leader_tx_hash          VARCHAR(120),
    our_tx_hash             VARCHAR(120),

    leader_fill_price_usd   NUMERIC(28, 12),              -- USD per token at leader fill
    our_fill_price_usd      NUMERIC(28, 12),              -- USD per token at our fill
    -- Signed price-decay bps: positive = we filled worse than leader on
    -- the buy side / better on the sell side (i.e. price moved against
    -- us). Negative = we got a better fill than the leader (rare).
    slippage_bps            NUMERIC(10, 2),

    leader_fill_ts          TIMESTAMPTZ,
    our_fill_ts             TIMESTAMPTZ,
    delta_ms                BIGINT,                       -- our_fill_ts - leader_fill_ts in ms

    is_simulated            BOOLEAN NOT NULL DEFAULT TRUE,
    recorded_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes                   JSONB
);

CREATE INDEX IF NOT EXISTS idx_copy_slippage_leader_recent
    ON copy_slippage_observations (chain, leader_wallet, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_copy_slippage_recorded_at
    ON copy_slippage_observations (recorded_at DESC);

-- ----- 2. probation state on copy_leader_scores ----------------------
ALTER TABLE copy_leader_scores
    ADD COLUMN IF NOT EXISTS on_probation        BOOLEAN     NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS probation_until     TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS probation_reason    VARCHAR(120),
    ADD COLUMN IF NOT EXISTS probation_set_at    TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_copy_leader_scores_probation
    ON copy_leader_scores (on_probation, probation_until)
    WHERE on_probation = TRUE;

-- ----- 3. config-settings seeds (only if missing) --------------------
-- All Wave-3 tunables default to safe values. Operators can override
-- via the dashboard Settings page -- the dashboard agent owns the UI;
-- the engine consumes via ConfigManager.
INSERT INTO config_settings (config_type, key, value, value_type)
VALUES
    ('copytrading_config', 'probation_score_threshold',     '30',   'number'),
    ('copytrading_config', 'probation_loss_pct_threshold',  '15',   'number'),
    ('copytrading_config', 'probation_days',                '7',    'number'),
    ('copytrading_config', 'probation_gate_enabled',        'true', 'boolean'),
    ('copytrading_config', 'cross_module_exposure_check_enabled', 'true',  'boolean'),
    ('copytrading_config', 'cross_module_exposure_cap_usd',        '5000', 'number'),
    ('copytrading_config', 'slippage_tracking_enabled',     'true', 'boolean')
ON CONFLICT (config_type, key) DO NOTHING;
