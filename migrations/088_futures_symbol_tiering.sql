-- Migration 088: FUT-RM-27 (Wave 25) — futures per-symbol tiering + rolling
-- performance gate.
--
-- Driven by ONE WEEK of live (sim/testnet, 5x) data: 442 trades, +$31.05
-- total, 55.7% WR overall, but with a hard winner/loser split:
--
--   Winners: BCH +39.11 (75% WR), AAVE +15.80 (68.8%), ETH +9.15,
--            ADA +5.77, DOGE +4.07, JUP +2.11, BNB +1.52, XRP +0.97
--   Losers:  SUI -18.08 (30% WR), NEAR -16.06, ZEC -15.82 (42 trades!),
--            DOT -15.49 (35% WR), AVAX -11.16, FIL -6.09, LTC -2.95,
--            LINK -2.80, BTC -0.75, SOL -0.21
--
-- Cutting the six-symbol loser tier (~-$83 combined) roughly TRIPLES the
-- weekly PnL. ZEC additionally shows the cool-off (FUT-RM-17) failure
-- mode: 42 trades/week kept re-entering through the expiring 4h cool-off.
--
-- Two layers (both read by FuturesRiskManager via FuturesRiskConfig):
--
--   1. symbol_size_weights (STATIC, operator-curated): per-symbol size
--      multiplier. 0 = symbol disabled; 0<w<1 = reduced size; a symbol
--      missing from the map trades at full size (1.0). Seeded below with
--      the loser tier at 0 — the operator can re-enable any symbol from
--      the settings page or by editing the JSON.
--
--   2. rolling_gate_* (DYNAMIC): auto-BENCH a symbol when its trailing
--      rolling_gate_window (20) closed trades have
--          net PnL < rolling_gate_max_net_pnl_usd (-5.0 USD)
--        AND win rate < rolling_gate_max_win_rate (0.45),
--      for rolling_gate_bench_minutes (1440 = 24h). On expiry the symbol
--      auto-UNBENCHES into a probation window: its trailing window is
--      cleared and it trades at rolling_gate_probation_weight (0.5x) size
--      for rolling_gate_min_trades (10) trades. The window is warmed from
--      futures_trades at startup, so benches survive restarts.
--
-- Threshold sanity check against the live week: SUI (30% WR, -$18) and
-- DOT (35% WR, -$15.49) would have benched quickly; BCH (75% WR) and
-- AAVE (68.8% WR) can never bench while winning. The AND-condition means
-- a low-WR symbol that is still net-positive (lottery-payoff profile)
-- keeps trading, as does a high-WR symbol in a temporary drawdown.
--
-- Idempotent: ON CONFLICT DO NOTHING everywhere — operator overrides and
-- re-runs are safe. config_settings.is_editable defaults TRUE (mig 002),
-- which is required for FuturesConfigManager._load_config_from_db to read
-- these rows.
--
-- TO RE-ENABLE a disabled symbol: edit the JSON value of
--   config_type='futures_risk', key='symbol_size_weights'
-- and set the symbol's weight to a positive number (e.g. 0.5 or 1.0).
-- TO DISABLE the whole feature: set symbol_tiering_enabled='false'.

-- UP

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
(
    'futures_risk',
    'symbol_tiering_enabled',
    'true',
    'bool',
    'FUT-RM-27: master switch for per-symbol tiering (static size weights) '
    'and the rolling per-symbol performance gate. false = both layers off.'
),
(
    'futures_risk',
    'symbol_size_weights',
    '{"BCH/USDT": 1.0, "AAVE/USDT": 1.0, "ETH/USDT": 1.0, "ADA/USDT": 1.0, "DOGE/USDT": 1.0, "JUP/USDT": 1.0, "BNB/USDT": 1.0, "XRP/USDT": 1.0, "BTC/USDT": 1.0, "SOL/USDT": 1.0, "LINK/USDT": 1.0, "LTC/USDT": 1.0, "SUI/USDT": 0.0, "NEAR/USDT": 0.0, "ZEC/USDT": 0.0, "DOT/USDT": 0.0, "AVAX/USDT": 0.0, "FIL/USDT": 0.0}',
    'json',
    'FUT-RM-27: per-symbol position-size multiplier. 0 = symbol disabled, '
    '0<w<1 = reduced size, missing symbol = full size (1.0). Seeded from '
    'the 2026-06 live week: loser tier SUI/NEAR/ZEC/DOT/AVAX/FIL disabled '
    '(combined -$83.70 of a +$31.05 week). Edit to re-enable.'
),
(
    'futures_risk',
    'rolling_gate_enabled',
    'true',
    'bool',
    'FUT-RM-27: auto-bench symbols on trailing-N underperformance so the '
    'tiering stays current without manual curation.'
),
(
    'futures_risk',
    'rolling_gate_window',
    '20',
    'int',
    'FUT-RM-27: trailing trade-count window per symbol for the rolling gate.'
),
(
    'futures_risk',
    'rolling_gate_min_trades',
    '10',
    'int',
    'FUT-RM-27: minimum closed trades in the window before the gate can '
    'bench a symbol; also the length of the post-unbench probation window.'
),
(
    'futures_risk',
    'rolling_gate_max_net_pnl_usd',
    '-5.0',
    'float',
    'FUT-RM-27: bench a symbol when trailing-window net PnL is below this '
    'USD value AND win rate is below rolling_gate_max_win_rate.'
),
(
    'futures_risk',
    'rolling_gate_max_win_rate',
    '0.45',
    'float',
    'FUT-RM-27: bench requires win rate below this (fraction, 0-1) in '
    'addition to the net-PnL condition.'
),
(
    'futures_risk',
    'rolling_gate_bench_minutes',
    '1440',
    'int',
    'FUT-RM-27: bench duration in minutes (1440 = 24h). After expiry the '
    'symbol re-enters on probation at rolling_gate_probation_weight size.'
),
(
    'futures_risk',
    'rolling_gate_probation_weight',
    '0.5',
    'float',
    'FUT-RM-27: size multiplier during the post-unbench probation window '
    '(rolling_gate_min_trades trades). Set 1.0 to disable probation sizing.'
)
ON CONFLICT (config_type, key) DO NOTHING;

-- DOWN (feature off, keep rows for audit)
-- UPDATE config_settings SET value='false'
--   WHERE config_type='futures_risk' AND key='symbol_tiering_enabled';
