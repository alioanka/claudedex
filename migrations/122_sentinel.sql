-- Migration 122: sentinel — config seeds + anomaly-events + actions tables
--
-- Cross-module anomaly detection + auto-freeze advisor (modules/sentinel/).
-- Watches minutes-scale distributions no per-module gate sees (stablecoin
-- depeg, cross-source price divergence, silent module death, 100%-rejection
-- patterns, abnormal loss velocity, correlated drawdown) and records graded
-- events to sentinel_anomalies. ADVISORY BY DEFAULT — applying this migration
-- changes NO behavior; the module only runs when SENTINEL_MODULE_ENABLED=true,
-- and even then it only RECORDS anomalies until sentinel_autopilot_enabled is
-- explicitly flipped. The autopilot may ONLY write/clear logs/.pause_<module>
-- (dwell-guarded, clears only its own freezes). It never trades and never
-- touches logs/.killswitch.
--
-- Boundary: meta_controller scores PERFORMANCE over days; sentinel detects
-- ANOMALIES over minutes.
--
-- Single-quoted SQL literals only ('' escapes apostrophes; double-quotes are
-- identifiers and previously halted a deploy). Idempotent.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('sentinel', 'sentinel_autopilot_enabled', 'false', 'bool',
     'Master autopilot switch. When false (default) sentinel only RECORDS '
     'anomalies. When true, CRITICAL anomalies from the freeze-eligible class '
     '(loss_velocity, correlated_drawdown) may write logs/.pause_<module> to '
     'freeze the offending module''s entries (dwell-guarded; sentinel clears '
     'only freezes it wrote itself). It NEVER places a trade and NEVER touches '
     'the killswitch.', NOW(), NOW()),

    ('sentinel', 'tick_interval_seconds', '60', 'int',
     'Seconds between sentinel detection cycles (minutes-scale watchdog).',
     NOW(), NOW()),

    ('sentinel', 'autopilot_dwell_minutes', '360', 'int',
     'Minimum minutes between two autopilot actions (freeze OR unfreeze) on '
     'the SAME module (anti-flap; default 6h). Anomalies are still recorded '
     'every tick.', NOW(), NOW()),

    ('sentinel', 'unfreeze_clear_minutes', '60', 'int',
     'A sentinel-written freeze is cleared only after NO critical '
     'freeze-eligible anomaly has fired for this module for this many minutes '
     '(and the dwell guard allows).', NOW(), NOW()),

    ('sentinel', 'anomaly_refire_minutes', '30', 'int',
     'Dedup window: a persisting anomaly (same detector+subject+severity) '
     'within this window updates last_seen_at/fire_count on the open row '
     'instead of inserting a new one.', NOW(), NOW()),

    ('sentinel', 'alive_window_hours', '24', 'int',
     'A stale-heartbeat module is only flagged silent if it also traded '
     'within this window — a disabled module never alarms.', NOW(), NOW()),

    ('sentinel', 'price_fetch_timeout_seconds', '8', 'int',
     'Per-request timeout for the free two-source price board (Coinbase '
     'Exchange + Kraken public REST; keyless, fail-soft).', NOW(), NOW()),

    ('sentinel', 'depeg_enabled', 'true', 'bool',
     'Stablecoin depeg detector (USDT/USDC/DAI vs the 1.0 peg). '
     'Detection-only; never freeze-eligible.', NOW(), NOW()),

    ('sentinel', 'depeg_warn_bps', '50', 'float',
     'Median peg deviation (bps) at which a stablecoin depeg is WARN.',
     NOW(), NOW()),

    ('sentinel', 'depeg_critical_bps', '150', 'float',
     'Median peg deviation (bps) at which a depeg is CRITICAL (capped at WARN '
     'when only one price source responded — one feed alone is the likelier '
     'failure).', NOW(), NOW()),

    ('sentinel', 'divergence_enabled', 'true', 'bool',
     'Cross-source price divergence detector (BTC/ETH/SOL, Coinbase vs '
     'Kraken). Detection-only; never freeze-eligible.', NOW(), NOW()),

    ('sentinel', 'divergence_warn_bps', '100', 'float',
     'Two-source divergence (bps of mid) at which the deviation is WARN.',
     NOW(), NOW()),

    ('sentinel', 'divergence_critical_bps', '300', 'float',
     'Two-source divergence (bps of mid) at which the deviation is CRITICAL '
     '(oracle/venue break class).', NOW(), NOW()),

    ('sentinel', 'silent_module_enabled', 'true', 'bool',
     'Silent-module detector: *_runtime_stats heartbeat age vs the module''s '
     'expected refresh interval. Detection-only; never freeze-eligible '
     '(freezing cannot fix a dead process).', NOW(), NOW()),

    ('sentinel', 'heartbeat_warn_factor', '3', 'float',
     'Heartbeat age as a multiple of the expected refresh interval at which '
     'a recently-alive module is WARN-silent.', NOW(), NOW()),

    ('sentinel', 'heartbeat_critical_factor', '10', 'float',
     'Heartbeat age multiple at which silence is CRITICAL.', NOW(), NOW()),

    ('sentinel', 'full_rejection_enabled', 'true', 'bool',
     '100%-rejection detector: a module that evaluated many candidates and '
     'accepted ZERO since process start (the DEX vol/liq and Futures '
     'volume-gate bug class). Detection-only; never freeze-eligible.',
     NOW(), NOW()),

    ('sentinel', 'rejection_min_seen', '25', 'int',
     'Minimum candidates evaluated (with 0 accepted) before the '
     'full-rejection anomaly fires WARN; CRITICAL at 4x this sample.',
     NOW(), NOW()),

    ('sentinel', 'loss_velocity_enabled', 'true', 'bool',
     'Abnormal loss-velocity detector: realized window loss converted to '
     'USD/hour per module (DRY + LIVE combined). Freeze-eligible at CRITICAL '
     'when autopilot is on.', NOW(), NOW()),

    ('sentinel', 'loss_velocity_window_minutes', '60', 'int',
     'Trailing window over each module''s trade table for loss velocity and '
     'correlated drawdown.', NOW(), NOW()),

    ('sentinel', 'loss_velocity_warn_usd_per_hr', '15', 'float',
     'Loss velocity (USD/hour) at which a module is WARN.', NOW(), NOW()),

    ('sentinel', 'loss_velocity_critical_usd_per_hr', '50', 'float',
     'Loss velocity (USD/hour) at which a module is CRITICAL '
     '(freeze-eligible).', NOW(), NOW()),

    ('sentinel', 'corr_drawdown_enabled', 'true', 'bool',
     'Correlated-drawdown detector: several modules losing in the same '
     'window (fleet crypto-beta tail). Freeze-eligible at CRITICAL when '
     'autopilot is on (freezes the listed losers'' entries only).',
     NOW(), NOW()),

    ('sentinel', 'corr_drawdown_min_modules', '3', 'int',
     'Minimum simultaneously-losing modules for the correlated-drawdown '
     'anomaly to fire.', NOW(), NOW()),

    ('sentinel', 'corr_drawdown_module_loss_usd', '5', 'float',
     'Per-module window loss (USD) that counts a module as a loser in the '
     'correlated-drawdown check.', NOW(), NOW()),

    ('sentinel', 'corr_drawdown_critical_total_usd', '100', 'float',
     'Combined loser loss (USD) at which correlated drawdown is CRITICAL '
     '(freeze-eligible).', NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- Anomaly events: one open row per persisting (detector, subject, severity)
-- condition; refires within anomaly_refire_minutes bump last_seen_at and
-- fire_count instead of inserting. Dashboard reads the latest per detector.
CREATE TABLE IF NOT EXISTS sentinel_anomalies (
    id             BIGSERIAL PRIMARY KEY,
    detector       TEXT NOT NULL,            -- 'stable_depeg' | 'price_divergence' | 'silent_module' | 'full_rejection' | 'loss_velocity' | 'correlated_drawdown'
    subject        TEXT NOT NULL,            -- module short key, asset symbol, or 'fleet'
    severity       TEXT NOT NULL,            -- 'info' | 'warn' | 'critical'
    value          DOUBLE PRECISION,
    threshold      DOUBLE PRECISION,
    message        TEXT,
    details        JSONB,
    module_scoped  BOOLEAN NOT NULL DEFAULT FALSE,
    actuated       BOOLEAN NOT NULL DEFAULT FALSE,
    first_seen_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    fire_count     INT NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_sentinel_anomalies_detector_seen
    ON sentinel_anomalies (detector, subject, last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_sentinel_anomalies_seen
    ON sentinel_anomalies (last_seen_at DESC);

-- Autopilot action audit: every freeze/unfreeze, also the dwell-guard source.
CREATE TABLE IF NOT EXISTS sentinel_actions (
    id          BIGSERIAL PRIMARY KEY,
    module      TEXT NOT NULL,
    action      TEXT NOT NULL,               -- 'freeze' | 'unfreeze'
    detector    TEXT,
    reason      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sentinel_actions_module_created
    ON sentinel_actions (module, created_at DESC);

COMMIT;

-- down:
-- BEGIN;
-- DROP TABLE IF EXISTS sentinel_actions;
-- DROP TABLE IF EXISTS sentinel_anomalies;
-- DELETE FROM config_settings WHERE config_type = 'sentinel';
-- COMMIT;
