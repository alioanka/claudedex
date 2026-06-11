-- Migration 095: DEX week-1 performance-tuning seeds + DB-first exit watchdog keys
--
-- Context (2026-06-11): commit 254cf11 added the week-1 DEX tunables as real
-- ConfigManager model fields (RiskManagementConfig.sl_trigger_buffer_pct /
-- sl_fast_poll_band_pct, TradingConfig.max_trades_per_day / chain_weights,
-- MLModelsConfig.ml_quality_auc_floor) and referenced a "migration 086" that
-- was never created. This migration delivers those seed rows under the
-- reserved number 095, plus the entry/exit policy rows the shared engine
-- already reads but no migration ever seeded (on a fresh DB the engine
-- silently falls back to in-code defaults and the dashboard Settings page has
-- nothing to edit), plus three new dex_module keys consumed by the new
-- DB-first exit watchdog in modules/dex_trading/position_service.py.
--
-- Week-1 live DRY_RUN evidence (1638 trades, $1041 PnL, 42.1% WR):
--   * realized stop-loss averaged -14.51% against a 12% configured stop
--     (polling latency + gap on thin pairs) -> sl_trigger_buffer_pct fires the
--     stop early by the measured overshoot; sl_fast_poll_band_pct arms rapid
--     polling near the stop so detection latency stops eating ~2.5pts/loss.
--   * solana avg -2.24/trade (n=388), monad avg -16.64 (n=2) -> chain_weights
--     defaults both to 0 (disabled); operator re-enables by raising the weight.
--   * rug ensemble head CV AUC 0.5498 (a coin flip) -> ml_quality_auc_floor
--     0.6 keeps untrustworthy heads advisory-only, never vetoing entries.
--
-- Idempotent: ON CONFLICT (config_type, key) DO NOTHING everywhere — existing
-- operator-tuned values are never clobbered. Safe to re-run.
-- All string literals single-quoted; no double-quoted literals.

-- ============================================================================
-- PART 1: week-1 tunables (ConfigManager fields landed in 254cf11; rows
-- make them dashboard-editable and explicit on fresh databases)
-- ============================================================================

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('risk_management', 'sl_trigger_buffer_pct', '0.02', 'float',
     'Stop-loss early-fire buffer. The stop TRIGGERS at -(stop_loss_pct - '
     'buffer) so the REALIZED loss lands near stop_loss_pct. Week-1 realized '
     'SL averaged -14.51% vs the 12% config; 0.02 = the measured overshoot.'),

    ('risk_management', 'sl_fast_poll_band_pct', '0.04', 'float',
     'When an open position PnL is within this band ABOVE the stop trigger, '
     'position monitoring drops to rapid polling to cut stop-detection '
     'latency. 0.04 = arm fast polling within 4 points of the stop.'),

    ('trading', 'max_trades_per_day', '100', 'int',
     'Daily DEX entry budget across all chains. 0 = unlimited. Entries are '
     'processed best-score-first, so the budget keeps the highest-conviction '
     'trades. Week-1 ran ~234 trades/day; 100 trims the low-score tail.'),

    ('trading', 'chain_weights', '{"ethereum": 1.0, "base": 1.0, "bsc": 1.0, "pulsechain": 1.0, "solana": 0.0, "monad": 0.0}', 'json',
     'Per-chain capital weights. 0 disables the chain; (0,1] scales position '
     'size. Chains absent from the map default to 1.0. Week-1: solana avg '
     '-2.24/trade (n=388), monad -16.64 (n=2) -> both default 0.'),

    ('ml_models', 'ml_quality_auc_floor', '0.6', 'float',
     'Per-head ensemble quality gate. A pump/rug head may only VOTE on '
     'entries when its persisted cross-validated AUC clears this floor; '
     'below it the head is advisory-only (logged, never gating). Week-1 rug '
     'head CV AUC was 0.5498 — a coin flip must not veto trades.')

ON CONFLICT (config_type, key) DO NOTHING;

-- ============================================================================
-- PART 2: entry/exit policy rows the shared engine ALREADY reads
-- (core/engine.py) but which no migration ever seeded. Defaults mirror the
-- in-code defaults exactly — this migration changes no behavior by itself,
-- it makes the knobs visible and editable from the dashboard Settings page.
-- ============================================================================

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('trading', 'min_opportunity_score', '0.25', 'float',
     'Minimum composite opportunity score (0-1) for a DEX candidate to '
     'become a trade. Raise toward 0.30-0.35 to trim the weakest entries '
     'once per-score-bucket win rates are available from DRY_RUN data.'),

    ('risk_management', 'stop_loss_pct', '0.12', 'float',
     'Configured stop-loss fraction used for position sizing (size = risk '
     'capital / stop) and by the DEX DB-first exit watchdog. The watchdog '
     'triggers at -(stop_loss_pct - sl_trigger_buffer_pct).'),

    ('risk_management', 'take_profit_pct', '0.24', 'float',
     'Configured take-profit fraction enforced by the DEX DB-first exit '
     'watchdog. NOTE: the in-engine in-memory monitor still hardcodes 0.30; '
     'the watchdog makes this DB value the authoritative operator surface.'),

    ('risk_management', 'risk_per_trade_pct', '0.02', 'float',
     'Fraction of portfolio value risked per trade before the Kelly '
     'adjustment in position sizing. 0.02 = 2% of portfolio at risk.'),

    ('position_management', 'max_hold_time_minutes', '60', 'int',
     'Maximum holding time per DEX position in minutes. The in-engine '
     'monitor exits at this age; the DB-first watchdog exits orphaned rows '
     'at this age plus watchdog_max_hold_grace_pct.'),

    ('position_management', 'trailing_stop_enabled', 'true', 'bool',
     'Enable the ratchet trailing stop on DEX positions (activates after '
     'trailing_stop_activation percent profit).'),

    ('position_management', 'trailing_stop_percent', '6', 'int',
     'Base trailing distance in percent once the trail is active. The '
     'ratchet tightens it automatically: 5 at 15%+ peak, 3 at 30%+, 2 at '
     '50%+ peak profit.'),

    ('position_management', 'trailing_stop_activation', '10', 'int',
     'Peak profit percent at which the trailing stop activates.'),

    ('portfolio', 'max_positions', '40', 'int',
     'Maximum simultaneous open DEX positions across all chains.'),

    ('portfolio', 'max_position_size_usd', '10.0', 'float',
     'Hard per-position USD cap applied after dynamic sizing.'),

    ('portfolio', 'min_position_size_usd', '5.0', 'float',
     'Per-position USD floor applied after dynamic sizing.')

ON CONFLICT (config_type, key) DO NOTHING;

-- ============================================================================
-- PART 3: DEX DB-first exit watchdog (modules/dex_trading/position_service.py)
-- New keys, read directly from config_settings by the DEX subprocess (they
-- are intentionally NOT ConfigManager model fields — per campaign rules no
-- new fields may be added to config/config_manager.py).
-- ============================================================================

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('dex_module', 'db_exit_watchdog_enabled', 'true', 'bool',
     'Enable the DB-first exit watchdog in the DEX position service. It '
     'enforces stop/take-profit/trailing/max-hold on OPEN trades rows in '
     'DRY_RUN only (simulated closes; live exits remain engine-owned). '
     'Covers restart-orphaned positions the in-memory monitor misses.'),

    ('dex_module', 'watchdog_fast_poll_seconds', '5', 'int',
     'Refresh interval while any open position PnL sits within '
     'sl_fast_poll_band_pct above the stop trigger (normal interval 30s). '
     'Fast mode is skipped when more than 25 positions are open to respect '
     'price-API rate limits.'),

    ('dex_module', 'watchdog_max_hold_grace_pct', '25', 'int',
     'Grace margin in percent added to max_hold_time_minutes before the '
     'watchdog time-limit exit fires, so the in-engine monitor (when alive) '
     'always exits first. 25 -> watchdog closes at 75min for a 60min cap.')

ON CONFLICT (config_type, key) DO NOTHING;

-- down:
-- DELETE FROM config_settings
-- WHERE (config_type = 'risk_management' AND key IN
--        ('sl_trigger_buffer_pct', 'sl_fast_poll_band_pct'))
--    OR (config_type = 'trading' AND key IN
--        ('max_trades_per_day', 'chain_weights'))
--    OR (config_type = 'ml_models' AND key = 'ml_quality_auc_floor')
--    OR (config_type = 'dex_module' AND key IN
--        ('db_exit_watchdog_enabled', 'watchdog_fast_poll_seconds',
--         'watchdog_max_hold_grace_pct'));
-- (PART 2 rows mirror in-code defaults; deleting them is optional.)
