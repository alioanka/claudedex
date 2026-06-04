-- Migration 079: exclude flagged bad-exit rows from solana_trading_summary
--
-- Agent fix (378a456 + bc8f6e6) tags corrupt-exit-price rows (e.g. the PYTH
-- $0.0352 -> $153.52 / +2000% row) with metadata.excluded=true so they don't
-- skew analytics. But the solana_trading_summary view (migration 008) still
-- aggregated ALL rows, so excluded trades kept poisoning win-rate / total-PnL.
-- Recreate the view with a guard that drops excluded rows.
--
-- Idempotent (CREATE OR REPLACE VIEW). Column list unchanged from 008 so any
-- consumer keeps working. ADVICE-/ANALYTICS-only.

BEGIN;

CREATE OR REPLACE VIEW solana_trading_summary AS
SELECT
    is_simulated,
    strategy,
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl_sol > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN pnl_sol <= 0 THEN 1 ELSE 0 END) as losing_trades,
    CASE WHEN COUNT(*) > 0
        THEN ROUND(SUM(CASE WHEN pnl_sol > 0 THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100, 2)
        ELSE 0
    END as win_rate,
    COALESCE(SUM(pnl_sol), 0) as total_pnl_sol,
    COALESCE(SUM(pnl_usd), 0) as total_pnl_usd,
    COALESCE(SUM(fees_sol), 0) as total_fees,
    COALESCE(AVG(CASE WHEN pnl_sol > 0 THEN pnl_sol END), 0) as avg_win_sol,
    COALESCE(AVG(CASE WHEN pnl_sol <= 0 THEN pnl_sol END), 0) as avg_loss_sol,
    COALESCE(MAX(pnl_sol), 0) as best_trade_sol,
    COALESCE(MIN(pnl_sol), 0) as worst_trade_sol,
    COALESCE(AVG(duration_seconds), 0) as avg_duration_seconds
FROM solana_trades
WHERE NOT COALESCE((metadata->>'excluded')::boolean, false)
GROUP BY is_simulated, strategy;

COMMIT;
