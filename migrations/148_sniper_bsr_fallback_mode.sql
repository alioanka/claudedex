-- Migration 148: Wave-F6 item 2 — SNIPER missing-BSR fallback mode.
--
-- Context: sniper_fail_closed_missing_bsr=true + an unavailable Birdeye
-- buy/sell-ratio source rejected 99.5% of candidates (85,122 of 85,561 in
-- 3 days, 0 entries) — the Wave-13 "100%-block" class. The new knob is
-- AUTHORITATIVE for the missing-BSR branch in sniper_engine.py:
--   'skip_gate' (default): when BSR data is unavailable, skip ONLY the BSR
--       gate; honeypot quorum, liquidity, tax, safety-score, holder and
--       dev-holding gates all still run. A genuinely LOW BSR still rejects.
--   'reject': legacy fail-closed posture (restores Wave-F5 behavior).
--
-- Idempotent + operator-safe: INSERT only if the row is absent
-- (ON CONFLICT DO NOTHING) — an operator-set value is never clobbered.
-- No live/paid flag is touched; DRY_RUN posture unchanged.

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('sniper_config', 'sniper_bsr_fallback_mode', 'skip_gate', 'string',
     'Wave-F6: behavior when buy/sell-ratio data is unavailable. '
     'skip_gate = skip ONLY the BSR gate (all other safety gates still '
     'run); reject = fail-closed (rejects every candidate while the BSR '
     'source is down). A genuinely low BSR rejects in both modes.')
ON CONFLICT (config_type, key) DO NOTHING;
