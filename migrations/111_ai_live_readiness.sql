-- Migration 111: AI_ANALYSIS live-readiness tunables
--
-- Seeds the knobs consumed by modules/ai_analysis/core/sentiment_engine.py
-- (_load_settings). Defaults preserve current behavior exactly — applying
-- this migration changes nothing until an operator edits a value.
--
--   ai_trade_amount_max_usd   NEW hard ceiling on per-trade notional;
--                             trade_amount_usd is clamped to
--                             [1, ai_trade_amount_max_usd] at load so a
--                             fat-finger DB value cannot size an
--                             oversized futures entry. Default 1000 is
--                             20x the default trade_amount_usd (50).
--
--   llm_daily_max_calls       Operator-visible mirror of the bot-wide
--                             paid-LLM daily call cap enforced by
--                             core/llm_budget.py. NOTE: the engine reads
--                             the cap from env BOT_LLM_DAILY_MAX_CALLS
--                             first, then this config row when a config
--                             dict is passed to try_consume, else the
--                             code default (25). Seeded here so the
--                             dashboard can surface the intended value;
--                             <= 0 disables the cap entirely.
--
-- Idempotent: ON CONFLICT (config_type, key) DO NOTHING, so operator
-- overrides and re-runs are safe. All SQL string literals single-quoted;
-- embedded apostrophes escaped as ''.

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
(
    'ai_config',
    'ai_trade_amount_max_usd',
    '1000',
    'float',
    'Hard ceiling on the AI module''s per-trade notional in USD. trade_amount_usd is clamped to [1, this] at every settings reload, so a fat-finger value cannot size an oversized futures entry. Clamped 1..100000 at load.'
),
(
    'ai_config',
    'llm_daily_max_calls',
    '25',
    'int',
    'Bot-wide daily cap on PAID LLM API calls (shared across AI sentiment + Advisor via core/llm_budget.py and logs/.llm_budget.json). Resolution order in code: env BOT_LLM_DAILY_MAX_CALLS, then this key when passed, then default 25. A value <= 0 disables the cap. On exhaustion every caller falls back to neutral/rule-based output with NO paid call until UTC rollover.'
)
ON CONFLICT (config_type, key) DO NOTHING;
