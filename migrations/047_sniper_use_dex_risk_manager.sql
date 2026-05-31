-- Migration 047: sniper_use_dex_risk_manager config flag
--
-- Wave-15 risk-gate fix.
--
-- Problem:
--   RiskManager.validate_trade sets liquidity_risk=1.0 for any token with
--   <$10k liquidity, then rejects with "Insufficient liquidity". Brand-new
--   pump.fun / Raydium launches always have <$10k in their first seconds —
--   that is the nature of sniping — so 100% of targets were blocked.
--   Live log: "SNIPER STATS: Analyzed: 122 | Passed: 0 (0.0%)"
--
-- Fix:
--   New flag sniper_use_dex_risk_manager (default FALSE). When FALSE, the
--   sniper skips the DEX liquidity/honeypot/token-risk analysis and only
--   runs the capital-protection half (circuit breakers + allocation guard).
--   The sniper's own _check_filters + TokenSafetyChecker remain the memecoin
--   risk layer. When TRUE, full validate_trade is restored (legacy behavior).
--
-- Idempotent: ON CONFLICT DO NOTHING.
-- Date: 2026-05-31

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES (
    'sniper_config',
    'sniper_use_dex_risk_manager',
    'false',
    'bool',
    'Wave-15 risk-gate. FALSE (default): skip DEX-oriented liquidity/honeypot '
    'analysis in RiskManager.validate_trade — which rejects 100% of new-launch '
    'targets with <$10k liquidity — and only run capital-protection checks '
    '(circuit breakers + allocation guard). The sniper''s own _check_filters + '
    'TokenSafetyChecker remain the memecoin risk layer. TRUE: restore full '
    'validate_trade behavior (DEX / mature-token sniping). Default FALSE.'
)
ON CONFLICT (config_type, key) DO NOTHING;

-- down:
-- DELETE FROM config_settings WHERE config_type = 'sniper_config' AND key = 'sniper_use_dex_risk_manager';
