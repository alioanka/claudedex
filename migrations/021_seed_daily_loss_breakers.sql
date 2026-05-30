-- Phase 4C: per-module daily-loss circuit breaker thresholds.
--
-- Each module gets a row in config_settings. If a LIVE module loses
-- more than `daily_loss_breaker_pct` of its allocated capital in 24h,
-- the orchestrator auto-flips it back to DRY_RUN (and triggers a
-- subprocess restart so the new dry_run state takes effect).
--
-- 5% default — conservative. Operator can raise per module via the
-- /global-settings page or the dashboard. Set to a very high value
-- like 100 to effectively disable the breaker for a specific module.

INSERT INTO config_settings (config_type, key, value, value_type)
VALUES
  ('sniper_config',       'daily_loss_breaker_pct', '5.0', 'float'),
  ('arbitrage_config',    'daily_loss_breaker_pct', '5.0', 'float'),
  ('copytrading_config',  'daily_loss_breaker_pct', '5.0', 'float'),
  ('futures_config',      'daily_loss_breaker_pct', '5.0', 'float'),
  ('solana_config',       'daily_loss_breaker_pct', '5.0', 'float'),
  ('dex_config',          'daily_loss_breaker_pct', '5.0', 'float'),
  ('ai_config',           'daily_loss_breaker_pct', '5.0', 'float')
ON CONFLICT (config_type, key) DO NOTHING;

-- Audit trail for circuit-breaker events. Each row is one auto-trip:
-- module + pnl_loss + threshold + the resulting action.
CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tripped_at        TIMESTAMP NOT NULL DEFAULT NOW(),
    module            VARCHAR(32) NOT NULL,
    pnl_loss_usd      NUMERIC(20, 4) NOT NULL,
    capital_usd       NUMERIC(20, 4) NOT NULL,
    pct_loss          NUMERIC(8, 4) NOT NULL,    -- e.g. -7.5 means 7.5% loss
    threshold_pct     NUMERIC(8, 4) NOT NULL,
    action_taken      VARCHAR(32) NOT NULL,      -- 'flipped_to_dry' | 'logged_only'
    notes             TEXT,
    -- After 24h elapses since the trip, the orchestrator can write a
    -- companion 'cleared' event when conditions normalize. Operator
    -- can also clear manually.
    cleared_at        TIMESTAMP,
    cleared_by        VARCHAR(64)
);

CREATE INDEX IF NOT EXISTS idx_circuit_breaker_events_module_time
    ON circuit_breaker_events(module, tripped_at DESC);

CREATE INDEX IF NOT EXISTS idx_circuit_breaker_events_active
    ON circuit_breaker_events(module, tripped_at DESC)
    WHERE cleared_at IS NULL;
