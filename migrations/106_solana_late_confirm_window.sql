-- Migration 106: Solana late-confirmation rescue window (LIVE-readiness)
--
-- Context: JupiterHelper.execute_swap returns None when confirm_transaction
-- times out (90s), but the broadcast tx can still land AFTER the timeout.
-- Previously the engine recorded the entry as failed, leaving the bought
-- tokens untracked in the wallet (no position row, no SL/TP monitoring,
-- no exit) until a restart reconcile. The engine now runs one bounded
-- extra confirmation window (_late_confirm_rescue in solana_engine.py)
-- before declaring the buy failed.
--
-- Knob semantics:
--   solana_late_confirm_window_s > 0  -> extra getSignatureStatuses window
--                                        (seconds) after the primary timeout
--   solana_late_confirm_window_s = 0  -> rescue disabled (pre-106 behavior)
--
-- Engine read is fail-soft: missing row / DB outage -> code default 45.
-- DRY_RUN is unaffected (the rescue only runs on the LIVE broadcast path).
-- Idempotent: ON CONFLICT (config_type, key) DO NOTHING.

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
  ('solana_general', 'solana_late_confirm_window_s', '45', 'float',
   'Extra confirmation window (seconds) re-checking a broadcast-but-unconfirmed Jupiter buy before declaring it failed. Prevents late-landing buys from leaving untracked tokens in the wallet. 0 disables. Fail-soft code default 45.')
ON CONFLICT (config_type, key) DO NOTHING;
