-- Migration 151: Wave-F7 — smart RPC/API governor (per-module AIMD limiter)
--
-- Root cause (docs/agents/wave-f6/01_rate_limiting.md, operator-verified):
-- the pool engine had ONE token bucket per provider_type shared by ALL
-- modules, so one spamming module (sniper tx-enrich ~10/s, copy polling)
-- drained the bucket and EVERY module showed RATE_LIMITED on /settings/
-- rpc-api — with no per-module attribution, no adaptive backoff, no
-- priority, no coalescing.
--
-- This migration ships:
--   1. rpc_governor_status — per-(module, provider) consumption + throttle
--      rows. Each module subprocess's pool engine flushes its own rows every
--      ~30s; the dashboard reads them for the "Per-Module Consumption"
--      panel on /settings/rpc-api (rows older than 15 min are ignored).
--   2. config_type='rpc_governor' knobs consumed by config/rpc_governor.py
--      via pool_engine._load_governor_config().
--
-- SAFETY: the governor only ever DELAYS/PACES outbound requests — it never
-- changes trade logic, flips no live/paid flag, and every wait is capped
-- (background 15s / execution 2s) so nothing can hang. governor_enabled=
-- false falls back to the legacy shared per-provider bucket. DRY_RUN-safe.
--
-- Idempotent: CREATE TABLE IF NOT EXISTS + INSERT ON CONFLICT DO NOTHING
-- (operator overrides of knob values are preserved on re-run).

BEGIN;

-- ---------------------------------------------------------------------
-- 1. Per-(module, provider) governor status (dashboard visibility)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rpc_governor_status (
    module            TEXT        NOT NULL,
    provider_type     TEXT        NOT NULL,
    requests_per_min  NUMERIC     DEFAULT 0,
    requests_per_hour NUMERIC     DEFAULT 0,
    credits_per_min   NUMERIC     DEFAULT 0,
    rate429_pct       NUMERIC     DEFAULT 0,
    current_rps       NUMERIC     DEFAULT 0,
    ceiling_rps       NUMERIC,
    state             TEXT        DEFAULT 'OK',      -- OK | THROTTLED | CLAMPED
    clamped           BOOLEAN     DEFAULT FALSE,
    throttle_events   INTEGER     DEFAULT 0,
    updated_at        TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (module, provider_type)
);

CREATE INDEX IF NOT EXISTS idx_rpc_governor_status_updated
    ON rpc_governor_status (updated_at);

-- ---------------------------------------------------------------------
-- 2. Governor knobs (config/rpc_governor.py defaults mirrored here; the
--    code runs with identical built-in defaults if these rows are absent)
-- ---------------------------------------------------------------------
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('rpc_governor', 'governor_enabled', 'true', 'boolean',
     'Master switch for the per-(module,provider) adaptive RPC governor. '
     'false = fall back to the legacy single shared bucket per provider. '
     'The governor only paces/delays requests - never trade logic.'),
    ('rpc_governor', 'governor_min_rps', '0.2', 'float',
     'AIMD floor: no module-provider pair is ever throttled below this '
     'rate, so even a clamped spammer keeps a trickle and can recover.'),
    ('rpc_governor', 'governor_decrease_factor', '0.5', 'float',
     'Multiplicative decrease applied to the offending module''s rps on '
     'each 429/quota event attributed to it (0.5 = halve).'),
    ('rpc_governor', 'governor_recover_step', '0.25', 'float',
     'Additive recovery in rps per clean second (no 429 in the clean '
     'window) up to the provider ceiling.'),
    ('rpc_governor', 'governor_clean_window_s', '20', 'float',
     'Seconds without a 429 on the pair before additive recovery starts.'),
    ('rpc_governor', 'governor_module_max_rps', '10', 'float',
     'SPAM CLAMP: a module whose 10s request rate on one provider exceeds '
     'this is hard-clamped to governor_min_rps with a named WARN once/min.'),
    ('rpc_governor', 'governor_module_credit_budget_per_min', '3000', 'float',
     'SPAM CLAMP: per-module per-provider credit budget per minute '
     '(method-weighted cost, e.g. Helius enhanced-tx=100).'),
    ('rpc_governor', 'governor_exec_reserve_pct', '30', 'float',
     'Share of each provider ceiling reserved for execution/quote-class '
     'calls. Background scans draw from the remaining share only, so a '
     'scan can never rate-limit a live send.'),
    ('rpc_governor', 'governor_max_wait_s', '15', 'float',
     'Hard cap on how long a background call can be delayed by pacing '
     '(fail-soft: after the cap the call proceeds).'),
    ('rpc_governor', 'governor_exec_max_wait_s', '2', 'float',
     'Hard cap on execution-class pacing delay.'),
    ('rpc_governor', 'governor_default_ceiling_rps', '8', 'float',
     'Default provider ceiling when no configure_rate_limiter call has '
     'set one (matches the pool''s historic 8 rps default).'),
    ('rpc_governor', 'governor_coalesce_ttl_s', '2.0', 'float',
     'TTL for the identical-read coalescing map (in-flight sharing for '
     'idempotent reads like getTransaction/getLogs).'),
    ('rpc_governor', 'governor_method_costs',
     '{"default": 1, "gethealth": 1, "getslot": 1, "eth_blocknumber": 1, "getbalance": 1, "gettransaction": 10, "getsignaturesforaddress": 10, "getparsedtransaction": 50, "getprogramaccounts": 25, "eth_getlogs": 25, "enhanced_tx": 100, "parse_transactions": 100}',
     'json',
     'Method-weighted credit cost table (merged over code defaults; keys '
     'lowercased). Heavy calls burn the per-module credit budget faster.'),
    ('rpc_governor', 'governor_exec_methods',
     '["sendtransaction", "sendrawtransaction", "eth_sendrawtransaction", "simulatetransaction", "eth_estimategas", "quote", "swap"]',
     'json',
     'Method names auto-classified as execution priority (in addition to '
     'the explicit priority=execution kwarg / governed_call context).')
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;

-- DOWN (manual rollback):
-- BEGIN;
-- DELETE FROM config_settings WHERE config_type = 'rpc_governor';
-- DROP TABLE IF EXISTS rpc_governor_status;
-- COMMIT;
