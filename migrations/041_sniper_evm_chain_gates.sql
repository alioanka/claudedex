-- Migration 041: Sniper per-chain EVM enable switch + tighter EVM-specific filters
--
-- Context: 21h DRY_RUN data (3686 closed EVM trades) shows:
--   EVM:    12.7% win-rate / -0.09% avg P&L  (negative expectancy)
--   Solana: 50.8% win-rate / +2.81% avg P&L  (positive expectancy)
--
-- Three new sniper_config keys are seeded:
--
--   sniper_evm_enabled (bool, default false)
--     Master toggle for the EVM listener. When false, the EVMListener is not
--     instantiated at all — no connections, no scanning, no EVM snipes fired.
--     Solana path is completely unaffected. Default FALSE because EVM has
--     negative expectancy at current filter settings. Flip to true once the
--     operator has verified a profitable EVM sub-segment exists (e.g. by
--     tightening evm_min_liquidity / evm_min_safety_score until WR > ~35%).
--
--   evm_min_liquidity (numeric USD, default 50000)
--     EVM-only liquidity floor applied on top of (never below) the global
--     min_liquidity. The global default is $1,000 which passes almost every
--     EVM pair. Raising to $50,000 restricts to established pairs where
--     slippage and rug risk are lower. Operator can lower if data shows a
--     profitable sub-segment at lower liquidity.
--
--   evm_min_safety_score (int 0-100, default 70)
--     EVM-only score floor. TokenSafetyChecker._calculate_score is a
--     subtractive-penalty model: 100 = perfect, 0 = DANGER. Setting this
--     to 70 rejects CAUTION-band tokens (typically 40-69) that make up the
--     bulk of EVM losses without a honeypot flag. 0 disables the gate.
--
-- Idempotent: ON CONFLICT DO NOTHING on all inserts.
-- Date: 2026-05-31

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('sniper_config', 'sniper_evm_enabled', 'false', 'bool',
     'Master toggle for EVM sniping. false=EVMListener not started (Solana unaffected). '
     '21h DRY_RUN: EVM 12.7% WR / -0.09% avg vs Solana 50.8% / +2.81%. '
     'Set true only after confirming a profitable EVM sub-segment via tighter '
     'evm_min_liquidity + evm_min_safety_score thresholds.'),

    ('sniper_config', 'evm_min_liquidity', '50000', 'float',
     'EVM-only liquidity floor (USD). Applied on top of global min_liquidity — '
     'effective threshold = max(min_liquidity, evm_min_liquidity). '
     '0 = inherit global value. Default 50000 restricts EVM to established '
     'pairs where slippage and rug risk are substantially lower.'),

    ('sniper_config', 'evm_min_safety_score', '70', 'int',
     'EVM-only safety score floor (0-100, subtractive-penalty scale). '
     'Tokens with score < this value are rejected even if not DANGER-rated. '
     'Targets the CAUTION band (score 40-69) that drives most EVM losses. '
     '0 = gate disabled (use global DANGER/CAUTION rules only). '
     'Default 70 keeps only HIGH/SAFE-rated EVM tokens.')
ON CONFLICT (config_type, key) DO NOTHING;

-- down:
-- DELETE FROM config_settings
-- WHERE config_type = 'sniper_config'
--   AND key IN ('sniper_evm_enabled', 'evm_min_liquidity', 'evm_min_safety_score');
