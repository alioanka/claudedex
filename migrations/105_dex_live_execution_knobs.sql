-- Migration 105: DEX LIVE-execution knobs (live-readiness wave, 2026-06-12)
--
-- Context: the DEX live-readiness audit hardened the previously unreachable
-- LIVE broadcast path in core/engine.py (RiskManager.validate_trade gate,
-- kill-switch/pause enforcement at the broadcast boundary, correct
-- native-unit order construction, dict-result handling). Three previously
-- HARDCODED execution constants are now DB-backed TradingConfig fields so the
-- operator can tune them before flipping LIVE. Defaults mirror the old
-- hardcoded values exactly -> this migration changes NO behavior by itself.
--
-- Idempotent: ON CONFLICT (config_type, key) DO NOTHING — operator-tuned
-- values are never clobbered. Safe to re-run.
-- All string literals single-quoted; embedded quotes escaped as ''.

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('trading', 'live_entry_slippage_pct', '0.05', 'float',
     'Max slippage tolerance (fraction) on LIVE DEX entry swaps. Was '
     'hardcoded 0.05 in core/engine.py _execute_opportunity. The executor '
     'derives min_amount_out from it, so it bounds the worst accepted fill. '
     'Sanity-clamped in code to (0, 0.5]; out-of-range values fall back to '
     '0.05. 5% is loose for majors — consider 0.01-0.02 once LIVE.'),

    ('trading', 'live_exit_slippage_pct', '0.05', 'float',
     'Max slippage tolerance (fraction) on LIVE DEX exit (sell) swaps. Was '
     'hardcoded 0.05 in core/engine.py _close_position. Exits should stay '
     'looser than entries: a stop-loss sell that reverts on slippage burns '
     'gas and leaves the position bleeding. Clamped to (0, 0.5].'),

    ('trading', 'live_max_execute_retries', '3', 'int',
     'Executor in-flight retry count for LIVE swaps (clamped 1..5). Default '
     '3 mirrors the legacy executor behavior. WARNING: the retry loop can '
     'RE-BROADCAST the swap after a receipt-timeout, i.e. a duplicate buy '
     '(double-spend surface) — 1 is the recommended LIVE setting until the '
     'executor''s retry path is made idempotent.'),

    ('trading', 'live_reconcile_enabled', 'true', 'bool',
     'Startup reconcile of restored LIVE positions against on-chain ERC20 '
     'balances (executor''s chain only). Observability-only: a >5% deficit '
     'logs CRITICAL, alerts, and flags metadata.reconcile_onchain_deficit — '
     'it never auto-closes. No effect under DRY_RUN.')

ON CONFLICT (config_type, key) DO NOTHING;

-- Verification:
--   SELECT key, value FROM config_settings
--   WHERE config_type = 'trading' AND key LIKE 'live_%' ORDER BY key;
