-- Migration 037: Sniper compound index + max_hold_minutes seed
--
-- PART 1 — Compound index on sniper_trades(chain, status)
--   Context: _effective_active_count() (sniper_engine.py) issues:
--     SELECT COUNT(*) FROM sniper_trades WHERE status='open'
--   and monitoring loops filter on (chain, status) for per-chain metrics.
--   Migration 010 already created single-column indexes idx_sniper_trades_status
--   and idx_sniper_trades_chain but not a compound one. As sniper_trades grows
--   past ~100k rows the compound scan becomes measurably cheaper than the
--   single-column bitmap-and-scan path.
--   IF NOT EXISTS is used so this is idempotent.
--
-- PART 2 — Seed sniper_config.max_hold_minutes (default 60)
--   The sniper engine reads this key via ConfigManager and enforces a per-
--   position time-stop (position is closed synthetically if held longer than
--   max_hold_minutes). The code was already shipping this check but the key
--   was never seeded in any migration, so the engine defaulted to 0 (disabled)
--   on a fresh DB -- meaning positions that never hit TP or SL held indefinitely,
--   tying up active-position cap slots permanently.
--   60 minutes (24x/day turnover) is a conservative default; operator can
--   lower to 30 for faster churn or 0 to disable the time-stop.
--
-- NOTE -- no CHECK constraint on profit_loss_pct:
--   Agent 6 (sniper) noted that profit_loss_pct is unbounded in the DDL and
--   suggested a CHECK (profit_loss_pct BETWEEN -100 AND 300). This migration
--   intentionally omits that constraint: existing live rows may contain values
--   outside that range due to the price-unit mismatch bug fixed in Wave-13
--   (sniper_engine.py _monitor_active_snipes phantom-price guard). Adding a
--   CHECK against a column that already has violating rows would cause
--   ALTER TABLE to fail on production. The application-layer clamp shipped in
--   Wave-12 (_simulate_sell, +-200% cap) and the Wave-13 phantom-price guard
--   together enforce the bound at write time; a CHECK can be added in a
--   future wave after the violating rows are cleaned up via the optional
--   DELETE in docs/agents/wave13/wave13_db_queries.sh.
--
-- Idempotent: IF NOT EXISTS for DDL; ON CONFLICT DO NOTHING for seed.
--
-- Date: 2026-05-30

-- ============================================================================
-- PART 1: Compound index
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_sniper_trades_chain_status
    ON sniper_trades (chain, status);

-- ============================================================================
-- PART 2: max_hold_minutes config seed
-- ============================================================================

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('sniper_config', 'max_hold_minutes', '60', 'int',
     'Per-position time-stop for the sniper: positions still open after this '
     'many minutes are closed synthetically (DRY_RUN) or via Jupiter sell '
     '(LIVE). Set to 0 to disable the time-stop. Default 60 (24x daily '
     'turnover). Lower to 30 for faster cap-slot churn; raise to 120 for '
     'momentum trades that need more time.')
ON CONFLICT (config_type, key) DO NOTHING;

-- down:
-- DROP INDEX IF EXISTS idx_sniper_trades_chain_status;
-- DELETE FROM config_settings
-- WHERE config_type = 'sniper_config' AND key = 'max_hold_minutes';
