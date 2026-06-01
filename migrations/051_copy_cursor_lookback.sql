-- Migration 051: seed copy_cursor_lookback_minutes for wave-16 startup catch-up.
--
-- Root cause of zero mirrored trades: the wave-15 cursor seeded to the
-- most-recent signature on first run and returned 0 (skipped ALL history).
-- Combined with copy_max_signal_age_s=5 s, any leader trade older than 5 s
-- was dropped by the execution staleness guard on every subsequent cycle.
-- If none of the 33 tracked leaders traded in the last 5 s since the process
-- started, copy sat idle indefinitely.
--
-- Fix (wave-16): on first run, process sigs within copy_cursor_lookback_minutes
-- of now (default 15 min) using the lookback window as the staleness gate
-- instead of copy_max_signal_age_s. This catches genuinely-recent leader
-- trades at startup without bypassing the run-time execution guard (which
-- continues to apply on all subsequent cycles unchanged).
--
-- Policy:
--   copy_cursor_lookback_minutes — startup catch-up window (first run only).
--     0 restores W15 behaviour (skip all history). Clamp [0, 120].
--   copy_max_signal_age_s — run-time execution staleness guard (every cycle
--     after the first). Unchanged at 5 s default. Operators running slower
--     strategies may raise this via DB update.
--
-- These two keys serve DIFFERENT roles and are independently tunable.
-- A 15-min lookback does NOT raise the 5 s real-time guard, so the
-- profitability argument in migration 042 (memecoin edge evaporates in
-- 1-2 s) is fully preserved for all ongoing monitoring cycles.

INSERT INTO config_settings (config_type, key, value)
VALUES (
    'copytrading_config',
    'copy_cursor_lookback_minutes',
    '15.0'
)
ON CONFLICT (config_type, key) DO NOTHING;
