-- Migration 061: Advisor Wave-22 — composite signal weights + OpenAI dual-advice config
--
-- Seeds config_settings rows (config_type = 'advisor_config') for:
--   1. Signal-layer weights (tunable via dashboard Settings > Advisor Settings)
--   2. OpenAI dual-advice keys
--
-- All rows use INSERT ... ON CONFLICT DO NOTHING so re-running is safe.
--
-- DASHBOARD NOTE: These keys should be surfaced in the dashboard Advisor Settings
-- panel under a "Signal Engine" section (wave-22 dashboard task).
--   signal_weight_trend / signal_weight_momentum / signal_weight_volume /
--   signal_weight_volatility / signal_weight_regime  -> numeric sliders [0.0 - 1.0]
--   advisor_dual_advice_mode -> dropdown: off / both / consensus
--   advisor_openai_model     -> text field (shows current model, warn on 404)

BEGIN;

-- -----------------------------------------------------------------------
-- Signal layer weights
-- Tunable: values sum should be close to 1.0; normalised at runtime if not.
-- -----------------------------------------------------------------------

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
  ('advisor_config', 'signal_weight_trend',      '0.30', 'float',
   'Trend layer weight for composite signal (EMA alignment + ADX)', NOW(), NOW()),
  ('advisor_config', 'signal_weight_momentum',   '0.25', 'float',
   'Momentum layer weight (RSI + ROC + MFI)', NOW(), NOW()),
  ('advisor_config', 'signal_weight_volume',     '0.20', 'float',
   'Volume layer weight (OBV + rel-volume + delta)', NOW(), NOW()),
  ('advisor_config', 'signal_weight_volatility', '0.15', 'float',
   'Volatility layer weight (ATR + BB width + hist-vol) — acts as risk penalty', NOW(), NOW()),
  ('advisor_config', 'signal_weight_regime',     '0.10', 'float',
   'Regime layer weight (TRENDING/RANGING/PANIC/EUPHORIA/ACCUMULATION/DISTRIBUTION)', NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- -----------------------------------------------------------------------
-- OpenAI dual-advice config
-- -----------------------------------------------------------------------

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
  ('advisor_config', 'advisor_openai_model',     'gpt-4o', 'string',
   'OpenAI model for dual-advice rationale. Update here (not in code) when a newer model ships. WARN logged on 404.', NOW(), NOW()),
  ('advisor_config', 'advisor_dual_advice_mode', 'off',    'string',
   'Dual-advice mode: off (Anthropic-only), both (show two rationales side-by-side), consensus (flag disagreement + adjust confidence). Requires ADVISOR_OPENAI_API_KEY in Secure Credentials.', NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;
