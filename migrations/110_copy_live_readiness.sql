-- Migration 110: COPY_TRADING live-readiness execution tunables
--
-- Seeds the five knobs consumed by modules/copy_trading/copy_engine.py
-- (_load_settings). Every default preserves the previously hardcoded
-- behavior EXACTLY — applying this migration changes nothing until an
-- operator edits a value. Loads are clamped in code against fat-finger
-- values (see copy_engine.py for the clamp bands).
--
--   copy_wallet_cooldown_s          was hardcoded 300 s per leader
--   copy_evm_slippage_pct           was hardcoded 10 (% minOut tolerance)
--   copy_solana_buy_slippage_bps    was hardcoded 100 (Jupiter BUY quote)
--   copy_solana_sell_slippage_bps   was hardcoded 300 (Jupiter SELL quote
--                                   + dashboard manual close)
--   copy_max_open_per_leader        NEW BUY-only per-leader cap; 0 keeps
--                                   the previous uncapped-per-leader
--                                   behavior (global max_active_positions
--                                   still applies)
--
-- Idempotent: ON CONFLICT (config_type, key) DO NOTHING, so operator
-- overrides and re-runs are safe. All SQL string literals single-quoted;
-- embedded apostrophes escaped as ''.

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
(
    'copytrading_config',
    'copy_wallet_cooldown_s',
    '300',
    'float',
    'Per-leader cooldown between mirrored copies, in seconds. 0 disables the cooldown. Clamped 0..86400 at load. Was hardcoded 300 s before migration 110.'
),
(
    'copytrading_config',
    'copy_evm_slippage_pct',
    '10',
    'float',
    'EVM swap minOut tolerance in percent (e.g. 10 = accept 90% of the quoted output). Applies to BUY and SELL mirroring plus dashboard manual closes. Clamped 0.1..49 at load so a fat-finger can neither zero the sandwich protection nor permit a >49% slip.'
),
(
    'copytrading_config',
    'copy_solana_buy_slippage_bps',
    '100',
    'int',
    'Jupiter slippage for mirrored Solana BUYs, in basis points (100 = 1%). Clamped 10..2000 at load. Was hardcoded 100 before migration 110.'
),
(
    'copytrading_config',
    'copy_solana_sell_slippage_bps',
    '300',
    'int',
    'Jupiter slippage for mirrored Solana SELLs and manual closes, in basis points (300 = 3%, memecoin-tolerant). Clamped 10..5000 at load — wider ceiling than BUY because an exit must not strand the position.'
),
(
    'copytrading_config',
    'copy_max_open_per_leader',
    '0',
    'int',
    'BUY-only cap on simultaneous open mirrored positions per leader wallet. 0 = disabled (previous behavior). Bounds a single hyperactive leader from consuming the whole global max_active_positions budget. Refusals log [replay] reason=leader_position_cap. Clamped 0..100 at load.'
)
ON CONFLICT (config_type, key) DO NOTHING;
