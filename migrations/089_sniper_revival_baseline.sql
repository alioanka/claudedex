-- Migration 089: SNIPER revival baseline
--
-- Context (2026-06-11): module disabled since wave-18 (budget=0). Code fixes
-- shipped alongside this migration restore the entry pipeline (W17 watchlist
-- promote loop), the killswitch poller, and the allocation-guard wiring.
--
-- PART 1 — seed the BASE sniper_config keys.
--   The engine reads these in _load_settings but NO migration ever seeded
--   them; they only existed on databases where the operator created them via
--   the dashboard. On a fresh DB the engine silently fell back to in-code
--   defaults. Seeding makes the configuration explicit, dashboard-editable,
--   and consistent across environments. ON CONFLICT DO NOTHING — existing
--   operator-tuned values are never clobbered.
--
--   trade_amount 0.05 (SOL) is deliberately SMALL: with stop_loss 20% the
--   worst-case loss per snipe is ~0.01 SOL + fees. Raise only after a
--   profitable burn-in window.
--
-- PART 2 — force safety_check_enabled = 'true'.
--   This is the long-pending "final step before LIVE" flip documented since
--   Phase 2 (it was set false purely to measure raw WSS volume). It must be
--   ON for the burn-in too: gates 1/3/4 (holder count, dev holding, safety
--   score) only run when a SafetyReport exists, so leaving it off makes the
--   burn-in stats dishonest about what LIVE would do. ON CONFLICT DO UPDATE
--   (a DO NOTHING would silently leave it false on the production DB).
--
-- NOT touched here: allocation_guard_config.budget_usd_sniper stays 0 —
-- LIVE entries remain hard-blocked. The engine now allows DRY_RUN simulated
-- entries through the budget=0 gate, so the burn-in needs no budget change.
--
-- Idempotent: safe to re-run.

-- PART 1: base config seeds (preserve existing values)
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('sniper_config', 'trade_amount', '0.05', 'float',
     'Entry size per snipe in chain-native units (SOL on Solana). '
     'Conservative revival default 0.05 SOL; worst case per snipe at '
     'stop_loss 20% is ~0.01 SOL + fees. Raise only after a profitable '
     'DRY_RUN burn-in.'),
    ('sniper_config', 'slippage', '10.0', 'float',
     'Per-snipe slippage cap (%). 10% is required for fresh pools; '
     'lower values fail to fill on launch volatility.'),
    ('sniper_config', 'priority_fee', '5000', 'int',
     'Solana prioritization fee in lamports for snipe txs. 5000 is '
     'minimal; raise for LIVE fill-rate during congestion.'),
    ('sniper_config', 'max_buy_tax', '15.0', 'float',
     'Reject tokens whose buy tax exceeds this %.'),
    ('sniper_config', 'max_sell_tax', '15.0', 'float',
     'Reject tokens whose sell tax exceeds this %.'),
    ('sniper_config', 'min_liquidity', '1000.0', 'float',
     'Minimum pool liquidity gate (USD) at safety-check time.'),
    ('sniper_config', 'take_profit_pct', '50.0', 'float',
     'Full take-profit threshold (%). Partial-take at '
     'sniper_partial_take_pct fires first when enabled.'),
    ('sniper_config', 'stop_loss_pct', '20.0', 'float',
     'Stop-loss threshold (%, positive number).'),
    ('sniper_config', 'chain', 'solana', 'string',
     'Target chain routing. solana = primary (EVM additionally gated by '
     'sniper_evm_enabled, default false per migration 043).'),
    ('sniper_config', 'test_mode', 'false', 'bool',
     'Relaxed safety gates for testing. Engine REFUSES to start with '
     'test_mode=true in LIVE mode.')
ON CONFLICT (config_type, key) DO NOTHING;

-- PART 2: force the pre-LIVE safety flip (Phase-2 measurement toggle off)
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('sniper_config', 'safety_check_enabled', 'true', 'bool',
     'Honeypot/tax/liquidity safety pipeline (GoPlus + Honeypot.is quorum '
     'on EVM, RugCheck on Solana). MUST be true for LIVE (engine refuses '
     'to start otherwise) and should be true for burn-in so collected '
     'stats reflect the gates LIVE would apply. Forced true by migration '
     '089 (was false for Phase-2 volume measurement).')
ON CONFLICT (config_type, key) DO UPDATE
    SET value = excluded.value,
        description = excluded.description;

-- down:
-- UPDATE config_settings SET value='false'
--   WHERE config_type='sniper_config' AND key='safety_check_enabled';
-- DELETE FROM config_settings WHERE config_type='sniper_config'
--   AND key IN ('trade_amount','slippage','priority_fee','max_buy_tax',
--               'max_sell_tax','min_liquidity','take_profit_pct',
--               'stop_loss_pct','chain','test_mode');
