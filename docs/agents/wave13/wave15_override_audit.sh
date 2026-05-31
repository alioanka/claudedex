#!/usr/bin/env bash
# =============================================================================
# ClaudeDex — DB config-override audit (Wave-14/15)
# =============================================================================
# Read-only. Finds config keys whose LIVE DB value differs from what the
# Wave-14/15 migrations intended to seed. Because those migrations use
# `ON CONFLICT (config_type,key) DO NOTHING`, any key that already existed
# kept its OLD value — silently shadowing the shipped fix (exactly what
# happened with arbitrage.flash_loan_amount).
#
# Run on the VPS:  bash docs/agents/wave13/wave15_override_audit.sh
# Any row printed under "MISMATCH" = a fix being shadowed by stale config.
# =============================================================================

PG() { docker exec trading-postgres psql -U "$(docker exec trading-postgres cat /run/secrets/db_user)" -d tradingbot -P pager=off "$@"; }

echo "############################################################"
echo "## EXPECTED vs LIVE — Wave-14/15 seeded keys"
echo "## (expected = what the migration intended to seed)"
echo "############################################################"

# A temp table of (config_type, key, expected_value) then LEFT JOIN live config.
PG -c "
WITH expected(config_type, key, expected_value) AS (VALUES
    -- 040 futures exit controls
    ('futures_risk','max_hold_minutes','240'),
    ('futures_risk','signal_reversal_threshold','4'),
    -- 041 futures funding carry
    ('futures_funding','funding_carry_enabled','false'),
    ('futures_funding','carry_min_funding_bps','8.0'),
    ('futures_funding','carry_exit_funding_bps','3.0'),
    ('futures_funding','carry_max_positions','2'),
    ('futures_funding','carry_max_hold_minutes','960'),
    -- 042 / 044 copy
    ('copytrading_config','copy_max_signal_age_s','5.0'),
    ('copytrading_config','copy_max_concurrent_wallets','5'),
    -- 043 sniper evm gate
    ('sniper_config','sniper_evm_enabled','false'),
    ('sniper_config','evm_min_liquidity','50000'),
    ('sniper_config','evm_min_safety_score','70'),
    -- 045 sniper entry/exit rework
    ('sniper_config','sniper_min_holder_count','10'),
    ('sniper_config','sniper_min_token_age_seconds','30'),
    ('sniper_config','sniper_max_dev_holding_pct','30.0'),
    ('sniper_config','sniper_min_buy_sell_ratio','1.5'),
    ('sniper_config','sniper_min_safety_score','40'),
    ('sniper_config','sniper_partial_take_pct','20.0'),
    ('sniper_config','sniper_partial_take_size_pct','50.0'),
    ('sniper_config','sniper_trail_after_partial','true'),
    -- 047 sniper risk-gate flag
    ('sniper_config','sniper_use_dex_risk_manager','false'),
    -- 046 allocation guard
    ('allocation_guard_config','enabled','true'),
    ('allocation_guard_config','use_orchestrator_budget','true'),
    ('allocation_guard_config','global_evm_wallet_cap_usd','700.0'),
    ('allocation_guard_config','global_solana_wallet_cap_usd','700.0'),
    ('allocation_guard_config','global_total_cap_usd','1000.0'),
    ('allocation_guard_config','budget_usd_sniper','150.0'),
    ('allocation_guard_config','budget_usd_arbitrage','200.0'),
    ('allocation_guard_config','budget_usd_copy_trading','150.0'),
    ('allocation_guard_config','budget_usd_futures','200.0'),
    ('allocation_guard_config','budget_usd_solana','150.0'),
    ('allocation_guard_config','budget_usd_dex','100.0'),
    ('allocation_guard_config','budget_usd_ai','100.0'),
    -- 039 AI (wave-13) — confirm threshold + model took
    ('ai_config','confidence_threshold','0.35'),
    ('ai_config','claude_model','claude-haiku-4-5-20251001'),
    -- 036 DEX scoring fix
    ('trading','min_vol_liq_ratio','0.05')
)
SELECT e.config_type, e.key,
       e.expected_value AS expected,
       c.value          AS live,
       CASE
         WHEN c.value IS NULL THEN '⚠️ MISSING (code default applies)'
         WHEN c.value = e.expected_value THEN 'ok'
         ELSE '❌ MISMATCH — fix shadowed'
       END AS status
FROM expected e
LEFT JOIN config_settings c
  ON c.config_type = e.config_type AND c.key = e.key
ORDER BY (CASE WHEN c.value IS DISTINCT FROM e.expected_value THEN 0 ELSE 1 END),
         e.config_type, e.key;
"

echo
echo "############################################################"
echo "## ARBITRAGE flash_loan_amount override (the known case)"
echo "## If present + = 10, the L2 right-sizing fix is shadowed."
echo "############################################################"
PG -c "SELECT config_type, key, value FROM config_settings WHERE key='flash_loan_amount';"

echo
echo "############################################################"
echo "## Any OTHER arbitrage keys that could gate trades"
echo "############################################################"
PG -c "SELECT key, value FROM config_settings WHERE config_type IN ('arbitrage','arbitrage_config') AND key IN ('min_profit_spread','min_profit_threshold','solana_enabled','triangular_enabled','use_flash_loans') ORDER BY key;"
