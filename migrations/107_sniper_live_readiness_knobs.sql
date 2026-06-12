-- Migration 107: Sniper LIVE-readiness knobs
--
-- Two knobs backing the LIVE-flip hardening in modules/sniper:
--
-- 1. sniper_confirm_timeout_secs — budget (seconds) for the new LIVE
--    Solana send-confirmation poll in trade_executor.py
--    (_confirm_solana_tx: getSignatureStatuses + on-chain err check).
--    Previously a sent-but-failed tx (blockhash expiry, slippage error)
--    was recorded as a successful LIVE buy. DRY_RUN never reaches this
--    code path.
--
-- 2. sniper_reconcile_on_start — when true (default), the engine reloads
--    mode-matched open sniper_trades rows into active_snipes at startup
--    (_reconcile_open_positions). Previously a restart orphaned every
--    open position: in LIVE the held tokens had no SL/TP monitoring and
--    exits were never broadcast again. Mode-mismatched rows
--    (is_simulated != current mode) are skipped and surfaced for
--    operator review, never auto-traded.
--
-- Engine reads are fail-soft (missing row -> code defaults 45 / true).
-- Idempotent: ON CONFLICT (config_type, key) DO NOTHING.

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
  ('sniper_config', 'sniper_confirm_timeout_secs', '45', 'float',
   'LIVE Solana send-confirmation budget (seconds) for sniper buys/sells. The executor polls getSignatureStatuses and treats on-chain err or timeout as a FAILED trade (no fill recorded). Fail-soft code default 45.'),
  ('sniper_config', 'sniper_reconcile_on_start', 'true', 'bool',
   'Restart reconcile: reload mode-matched open sniper_trades rows into the in-memory monitor at startup so SL/TP exits resume after a crash/restart. Mode-mismatched rows are skipped and logged for operator review. Fail-soft code default true.')
ON CONFLICT (config_type, key) DO NOTHING;
