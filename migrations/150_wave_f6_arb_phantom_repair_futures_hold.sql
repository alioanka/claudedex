-- Migration 150: Wave-F6 arbitrage phantom-PnL data repair + futures hold-window seed
--
-- Two independent, fully idempotent parts. Single-quoted literals only.
-- No live/paid flag is flipped; both parts are DRY_RUN-safe.
--
-- PART A — one-off arbitrage data repair (docs/agents/wave-f6/02_arbitrage_audit.md).
--   Every dollar of the control-center's arbitrage PnL (+1085 USD 7D /
--   +3250 USD all-time, 100% win, Sharpe 3.00) was booked by the triangular
--   engine's DRY_RUN branch: gross quote-time spread, zero gas/fees, ~2,400x
--   notional inflation (1 CRV mislabeled as 1 ETH), on a path that cannot
--   execute live at all (MB-05 atomic-receiver guard). Mirror of the Solana
--   mig-140A pattern: tag the rows metadata.excluded=true so dashboard PnL
--   aggregates (which filter on that tag) stop counting fabricated wins.
--   Rows are KEPT for audit — only the tag is added; nothing is deleted.
--   Re-runnable: the NOT COALESCE(excluded) guard makes it a no-op the
--   second time. The triangular write path itself now books net-of-gas and
--   self-tags DRY rows excluded at insert time (Wave-F6 code fix), so this
--   repair covers only pre-fix history.
--
-- PART B — FUTURES geometry, next single lever (F5 A/B discipline).
--   Wave-F5 mig 139 lowered atr_tp_rr_ratio 2.0 -> 1.0 to make TP1 reachable;
--   after 3 days fresh DRY_RUN there are STILL zero take_profit exits
--   (exit mix: time_limit 17, SL 10, TSL 7, signal 3, TP 0; PF stuck at 0.55
--   — docs/agents/wave-f6/03_trading_sweep.md). 17/37 closes died on the
--   240-min time limit, i.e. the hold window expires before TP1 can be hit.
--   Next lever (ONE change only, trailing arm kept): max_hold_minutes
--   240 -> 480. Conditional UPDATE only fires if the row still holds the old
--   default '240'; an operator override never matches and is preserved.
--   VALIDATION: needs 2+ weeks fresh DRY_RUN before judging — do not stack
--   further geometry changes inside that window.

-- =====================================================================
-- PART A: arbitrage phantom triangular DRY-fill tagging
-- =====================================================================

UPDATE arbitrage_trades
SET metadata = jsonb_set(
        jsonb_set(COALESCE(metadata, '{}'::jsonb), '{excluded}', 'true'::jsonb, true),
        '{exclusion_reason}',
        '"wave_f6_tri_phantom_pnl_dry_unexecutable_mb05"'::jsonb, true)
WHERE is_simulated = TRUE
  AND (side = 'triangular' OR metadata->>'type' = 'triangular')
  AND NOT COALESCE((metadata->>'excluded')::boolean, false);

-- =====================================================================
-- PART B: futures_risk.max_hold_minutes 240 -> 480
-- =====================================================================

UPDATE config_settings SET value = '480', updated_at = NOW()
 WHERE config_type = 'futures_risk' AND key = 'max_hold_minutes'
   AND value = '240';

INSERT INTO config_settings (config_type, key, value, description, updated_at)
VALUES (
    'futures_risk',
    'max_hold_minutes',
    '480',
    'Wave-F6: intraday max-hold cap in minutes (FUT-RM-23). Raised 240->480: '
    'rr=1.0 (mig 139) alone still produced 0 take_profit exits in 3 days '
    '(17/37 closes were time_limit) — give TP1 time to hit. ONE geometry '
    'lever per wave; judge after 2+ weeks fresh DRY_RUN. 0 = disabled.',
    NOW()
)
ON CONFLICT (config_type, key) DO NOTHING;
