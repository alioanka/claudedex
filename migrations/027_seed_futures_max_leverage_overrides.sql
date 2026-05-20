-- Migration 027: seed FUT-RM-08 max_leverage_overrides key.
--
-- Wave 3 deliverable: per-symbol leverage cap table. Operator can configure
-- different caps per pair (e.g. max 5x on PEPE/USDT but 10x on BTC/USDT) via
-- the Futures settings page. Stored as JSON map; FuturesRiskManager normalizes
-- symbol form (case, slash) on lookup. Empty {} = use global max_leverage for
-- every pair (current pre-Wave-3 behavior).
--
-- Idempotent: ON CONFLICT DO NOTHING. Existing operator overrides survive
-- a re-run.
--
-- Schema note: config_settings has (config_type, key, value, value_type,
-- description, is_editable, requires_restart). It does NOT have is_sensitive
-- (sensitive values live in the separate sensitive_configs table). Earlier
-- draft of this migration referenced is_sensitive and failed with
-- UndefinedColumnError on a fresh DB.

INSERT INTO config_settings (config_type, key, value, value_type, description, is_editable, requires_restart)
VALUES
    ('futures_leverage', 'max_leverage_overrides', '{}', 'json',
     'Per-symbol leverage cap overrides (JSON map). Wins over max_leverage. '
     'Example: {"BTC/USDT": 10, "PEPE/USDT": 5}. Empty {} = use global cap for all.',
     TRUE, FALSE)
ON CONFLICT (config_type, key) DO NOTHING;
