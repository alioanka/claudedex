-- Migration 050: Solana Wave-16 PnL-scale config seeds
-- Seeds new DB config keys for:
--   (a) position_size_sol raise: 0.05 -> 0.15 SOL conservative default
--   (b) Jupiter partial-take + trail toggle
--   (c) max_positions raise: 3 -> 10 (budget-gated by AllocationGuard)
--   (d) pumpfun_max_positions raise: 2 -> 3
--   (e) kill-switch thresholds surfaced in DB
--
-- All values are conservative defaults; operator overrides via dashboard or
-- direct UPDATE config_settings ... WHERE config_type='solana_*' AND key=...

-- ============================================================================
-- 1. Raise default position size: 0.05 -> 0.15 SOL
--    Rationale: 3000 DRY_RUN trades at 71%+ WR confirm positive expectancy.
--    0.15 SOL (~$12 at $80/SOL) x 10 concurrent = 1.5 SOL max exposure,
--    well within the $150 solana allocation-guard budget.
--    Operator may raise further (to 0.5 SOL) once LIVE confirms fill quality.
-- ============================================================================
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
  ('solana_general', 'position_size',        '0.15',  'float',
   'Base SOL committed per entry (wave-16 scale: was 0.05). Budget: max_positions x this <= allocation_guard solana budget.')
ON CONFLICT (config_type, key) DO NOTHING;

-- ============================================================================
-- 2. Raise max concurrent positions: 3 -> 10
--    Rationale: Dashboard shows only 6 ACTIVE vs 30 max but only 6 signals
--    fire per cycle across pumpfun+jupiter. Raising engine max_positions to 10
--    allows more concurrent without blowing the budget (1.5 SOL total).
--    pumpfun_max_positions stays at 3 (separate slot pool).
-- ============================================================================
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
  ('solana_general', 'max_positions',        '10',    'int',
   'Total concurrent positions across all strategies (wave-16: raised from 3 to 10).')
ON CONFLICT (config_type, key) DO NOTHING;

-- ============================================================================
-- 3. Jupiter: partial-take + trail remainder at TP
--    New keys allow exiting a portion at TP then trailing the rest to capture
--    larger moves on winning trades without changing entry logic or WR.
--    All OFF by default; operator flips once LIVE confirms fill quality.
-- ============================================================================
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
  ('solana_jupiter', 'jupiter_partial_take_enabled',        'false', 'bool',
   'Wave-16: sell jupiter_partial_take_pct at TP level then trail remainder. Raises avg win without touching WR.'),
  ('solana_jupiter', 'jupiter_partial_take_pct',            '50.0',  'float',
   'Wave-16: pct of Jupiter position to realize at TP hit (default 50%). Remainder held with trail stop.'),
  ('solana_jupiter', 'jupiter_trail_after_partial_enabled', 'false', 'bool',
   'Wave-16: trail the remaining Jupiter position after partial TP exit.'),
  ('solana_jupiter', 'jupiter_trail_after_partial_pct',     '3.0',   'float',
   'Wave-16: trail stop pct below peak price after Jupiter partial TP exit (default 3%).')
ON CONFLICT (config_type, key) DO NOTHING;

-- ============================================================================
-- 4. Pumpfun: raise concurrent slots 2 -> 3
--    Rationale: 2189 pumpfun trades = 73% of total volume. At 0.15 SOL each,
--    3 concurrent pumpfun = 0.45 SOL exposure, within budget.
-- ============================================================================
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
  ('solana_pumpfun', 'pumpfun_max_positions', '3',    'int',
   'Wave-16: raise pumpfun concurrent slots from 2 to 3 (budget allows 0.45 SOL at 0.15 SOL/trade).')
ON CONFLICT (config_type, key) DO NOTHING;

-- ============================================================================
-- 5. Kill-switch thresholds surfaced in DB
-- ============================================================================
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
  ('solana_general', 'solana_max_drawdown_pct',       '15.0', 'float',
   'Wave-16: kill-switch: disable new entries if realized daily PnL falls below -this% of starting capital.'),
  ('solana_general', 'solana_max_consecutive_losses', '8',    'int',
   'Wave-16: kill-switch: pause strategy after N consecutive losses.')
ON CONFLICT (config_type, key) DO NOTHING;
