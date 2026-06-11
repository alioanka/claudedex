-- Migration 102: FUT-QC-01 funding-carry v2 + AI-QC-01 confirmation signal
--
-- Seeds the feature flags (BOTH default false -- zero behavior change on
-- existing deployments) and their tunables for two new default-OFF
-- strategies:
--
--   1. FUT-QC-01 (config_type futures_funding, loaded by
--      FuturesConfigManager -> FuturesFundingConfig): bidirectional,
--      stability-gated funding carry. Planner logic + cost model in
--      modules/futures_trading/strategies/funding_carry.py; engine wiring in
--      modules/futures_trading/core/futures_engine.py (_scan_funding_carry_v2).
--      Entries route through _open_position so every existing risk gate
--      applies (FUT-RM-27 tiering + rolling gate, FUT-RM-19 edge gate,
--      FUT-RM-05 funding gate, FuturesRiskManager.validate_new_position,
--      DRY_RUN/killswitch/pause). carry_max_hold_minutes already exists
--      (migration 041) and is reused by v2.
--
--   2. AI-QC-01 (config_type ai_config, loaded by
--      SentimentEngine._load_settings): free cross-source confirmation gate.
--      Pure logic in modules/ai_analysis/core/confirmation_signal.py; wiring
--      in core/sentiment_engine.py (_confirmation_gate inside _execute_trade,
--      ahead of the executor and the downstream RiskManager.validate_trade).
--
-- Idempotent: ON CONFLICT (config_type, key) DO NOTHING everywhere, so
-- operator overrides and re-runs are safe. config_settings.is_editable
-- defaults TRUE (migration 002), which FuturesConfigManager requires.
-- NOTE: all SQL string literals below are single-quoted; double quotes are
-- identifiers in Postgres and would be a syntax error here.

-- ---------------------------------------------------------------------------
-- FUTURES: funding-carry v2 (FUT-QC-01)
-- ---------------------------------------------------------------------------

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
(
    'futures_funding',
    'futures_funding_carry_enabled',
    'false',
    'bool',
    'FUT-QC-01: master switch for funding-carry v2 (bidirectional, stability-gated). Default off; flip only after observing [carry-v2] DRY_RUN log lines. Independent of the v1 funding_carry_enabled flag.'
),
(
    'futures_funding',
    'carry_min_abs_funding_bps',
    '10',
    'float',
    'FUT-QC-01: entry threshold on absolute per-interval funding (bps), applied to the WEAKEST sample in the stability window. 10 bps breaks even in under 2 intervals vs the ~17 bps round-trip taker+slippage cost.'
),
(
    'futures_funding',
    'carry_funding_stability_window',
    '3',
    'int',
    'FUT-QC-01: minimum number of spaced funding samples (engine records at most one per 300s cache TTL) required inside the planner stability window before an entry can arm. Window-span coverage and same-sign persistence are additionally enforced by the planner.'
),
(
    'futures_funding',
    'carry_max_carry_positions',
    '3',
    'int',
    'FUT-QC-01: cap on simultaneous carry-v2 positions. Independent of the v1 carry_max_positions cap and the momentum max_positions cap. Start at 1 during initial DRY_RUN observation.'
),
(
    'futures_funding',
    'carry_max_hold_minutes',
    '960',
    'int',
    'Max hold for carry positions (minutes). 960 = 2 x 8h funding intervals. Shared by FUT-RM-25 (v1) and FUT-QC-01 (v2); seeded in migration 041, re-seeded here idempotently.'
)
ON CONFLICT (config_type, key) DO NOTHING;

-- ---------------------------------------------------------------------------
-- AI: cross-source confirmation signal (AI-QC-01)
-- ---------------------------------------------------------------------------

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
(
    'ai_config',
    'ai_confirmation_signal_enabled',
    'false',
    'bool',
    'AI-QC-01: master switch for the cross-source confirmation gate. When true, every per-symbol sentiment entry must also be confirmed by that symbol''s own free tape (price momentum + volume vs baseline, closed candles only). Default off; measure via [ai-confirm] lines and the confirmation_not_met skip reason before enabling.'
),
(
    'ai_config',
    'ai_confirmation_min_confidence',
    '0.45',
    'float',
    'AI-QC-01: minimum combined confidence (|0.5*sentiment + 0.5*momentum| scaled by the volume factor) for the gate to pass. Raise for fewer, higher-precision entries.'
),
(
    'ai_config',
    'ai_confirmation_momentum_weight',
    '0.5',
    'float',
    'AI-QC-01: weight of tape momentum in the combined directional score; the remainder goes to the LLM sentiment. 0 = sentiment-only, 1 = tape-only.'
),
(
    'ai_config',
    'ai_confirmation_max_opposing_momentum',
    '0.3',
    'float',
    'AI-QC-01: hard veto -- refuse the entry when normalized tape momentum opposes the sentiment direction by more than this, regardless of the combined score. Catches the bullish-headlines-falling-knife case.'
)
ON CONFLICT (config_type, key) DO NOTHING;
