-- Migration 049: AI multi-symbol config seeds (Wave-16)
--
-- Context:
--   Wave-16 reworks the AI advisor to trade a configurable symbol set instead
--   of the hardcoded 'ETH'.  Two new ai_config keys control behavior:
--
--   ai_symbols       — comma-separated symbols to consider each cycle.
--                      Default 'BTC,ETH,SOL'. The LLM produces a market-wide
--                      sentiment score; the engine opens one position per
--                      symbol that is not already held and below the position cap.
--
--   ai_max_positions — maximum number of concurrent open AI positions.
--                      Default 3 (one per default symbol).
--
--   take_profit_pct / stop_loss_pct / max_hold_hours — previously only set as
--   Python __init__ defaults (never persisted to DB), so operators had no
--   dashboard-visible handle on them and a restart silently reset them.  Seed
--   them now so they are DB-managed and editable from the Settings page.
--   Note: stop_loss_pct is stored as a positive magnitude '3.0'; the engine
--   converts to negative internally via -abs(raw).
--
-- Idempotent: all inserts use ON CONFLICT (config_type, key) DO NOTHING, so
-- re-running never overwrites an operator-tuned value.
--
-- Date: 2026-06-01

INSERT INTO config_settings (config_type, key, value)
VALUES ('ai_config', 'ai_symbols', 'BTC,ETH,SOL')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value)
VALUES ('ai_config', 'ai_max_positions', '3')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value)
VALUES ('ai_config', 'take_profit_pct', '5.0')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value)
VALUES ('ai_config', 'stop_loss_pct', '3.0')
ON CONFLICT (config_type, key) DO NOTHING;

INSERT INTO config_settings (config_type, key, value)
VALUES ('ai_config', 'max_hold_hours', '24')
ON CONFLICT (config_type, key) DO NOTHING;

-- down:
-- DELETE FROM config_settings
--  WHERE config_type = 'ai_config'
--    AND key IN ('ai_symbols', 'ai_max_positions', 'take_profit_pct', 'stop_loss_pct', 'max_hold_hours');
