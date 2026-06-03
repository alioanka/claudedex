-- Migration 066: Wave-24 futures strategy review — NEUTRALIZE futures entries
--
-- Operator decision (2026-06-03), backed by the wave-24 deep strategy review
-- (docs/agents/wave24/futures_strategy_review.md):
--
-- The futures momentum stack is structurally unprofitable on the current
-- setup. 17 of 18 symbols were net-negative over the measured 15h window
-- (only BCH positive); win rates 27-64% sit BELOW the ~62.5% break-even win
-- rate IMPLIED BY THE PAYOFF GEOMETRY itself:
--
--   * SL = 1.2% (full position) ; TP1 = 1.8% closing 40%, then stop->breakeven.
--   * The dominant win path therefore banks ~0.40 * 1.8% = 0.72% of notional
--     while a full loss is 1.2% of notional. Win:loss payoff = 0.6 : 1, so
--     even before fees the strategy needs 1.2/(1.2+0.72) = 62.5% wins just to
--     break even. Observed win rates are mostly below that.
--   * On top of that, ~$0.09/trade round-trip fees (Bybit taker 0.06% x2 on a
--     ~$100-500 notional, charged AGAIN on each of up to 4 partial TPs) is pure
--     drag the 0.72% gross win cannot reliably cover.
--   * The signal stack mixes mean-reversion (RSI extremes) with trend-following
--     (Bollinger breakout, EMA) ADDITIVELY, so it has no single coherent edge
--     thesis — it is as likely to fade a move as to follow it.
--
-- Six prior tuning waves (2,4,5,7,13,14: confluence gate, ATR SL/TP, cool-off,
-- edge gate, regime gate, leverage cut, exit controls) did NOT restore an edge.
-- Per the working-rule "when unsure, choose the conservative/neutralize path",
-- this migration NEUTRALIZES new futures entries rather than attempting a
-- seventh round of tuning.
--
-- HOW NEUTRALIZATION WORKS (two independent levers, both seeded here):
--
--   1. futures_config / entries_suppressed = 'true'  (PRIMARY, load-bearing)
--      Read by FuturesTradingEngine at startup (self.entries_suppressed). When
--      true, _trading_cycle skips _scan_opportunities AND the funding-carry
--      scan, so NO new momentum or carry positions are opened. Existing
--      positions are still monitored and exited normally (SL/TP/time/manual).
--      This is DRY_RUN-safe and live-safe: it only removes the OPEN path.
--
--   2. allocation_guard_config / budget_usd_futures = '0'  (DOCUMENTATION/BELT)
--      Mirrors wave-18 (migration 055) for sniper/arb. NOTE: on its own this
--      is a NO-OP for futures, because core/allocation_guard.py treats
--      budget_usd == 0 as "unlimited" (it only enforces when budget_usd > 0),
--      and the futures engine does not consult the allocation guard on the
--      open path. It is set here only so the operator-facing budget reads 0 and
--      matches the neutralized intent; lever (1) is what actually suppresses.
--
-- Idempotent: entries_suppressed uses ON CONFLICT DO NOTHING so a pre-existing
-- operator override survives a re-run; budget_usd_futures uses DO UPDATE to
-- force it to 0 (it already exists at 200.0 from migration 046, so DO NOTHING
-- would be silently ineffective — same rationale as migration 055).
--
-- TO RE-ENABLE futures entries (fully reversible):
--   UPDATE config_settings SET value='false'
--     WHERE config_type='futures_config' AND key='entries_suppressed';
--   UPDATE config_settings SET value='200'
--     WHERE config_type='allocation_guard_config' AND key='budget_usd_futures';

-- UP

-- Lever 1 (PRIMARY): engine entry-suppression switch.
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES (
    'futures_config',
    'entries_suppressed',
    'true',
    'bool',
    'FUT-RM-26 (wave-24): when true the futures engine opens NO new momentum/carry '
    'positions (existing positions still monitored + exited). Neutralization of a '
    'structurally-unprofitable momentum stack — see wave-24 strategy review. '
    'Set to false to re-enable entries.'
)
ON CONFLICT (config_type, key) DO NOTHING;

-- Lever 2 (DOCUMENTATION/BELT): force the operator-facing budget to 0.
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES (
    'allocation_guard_config',
    'budget_usd_futures',
    '0',
    'float',
    'USD budget for futures module. 0 = neutralized (wave-24 decision: momentum '
    'stack structurally unprofitable). NOTE: 0 is treated as unlimited by the '
    'allocation guard, so the load-bearing suppression is '
    'futures_config.entries_suppressed=true. Set to a positive value (e.g. 200) '
    'to restore the budget reading when re-enabling.'
)
ON CONFLICT (config_type, key) DO UPDATE
    SET value = excluded.value,
        description = excluded.description;

-- DOWN (re-enable futures entries)
-- UPDATE config_settings SET value='false' WHERE config_type='futures_config' AND key='entries_suppressed';
-- UPDATE config_settings SET value='200.0' WHERE config_type='allocation_guard_config' AND key='budget_usd_futures';
