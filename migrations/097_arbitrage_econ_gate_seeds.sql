-- Migration 097: seed the arbitrage economics-gate config keys
--
-- arbitrage_engine.py (economics-gates commit) reads these keys from
-- config_settings (config_type='arbitrage_config') via main_arbitrage.py and
-- its in-code comments referenced a "migration 090" that was never created
-- (the same never-shipped-seed failure mode as the DEX "086"/095 case). The
-- in-code defaults are fail-safe (shadow_mode=true, live_execution_enabled=
-- false), so behavior is unchanged today — but without rows the knobs are
-- invisible on the dashboard Settings surface and undiscoverable on a fresh
-- DB. This seeds them with values matching the code defaults exactly, so
-- applying this migration changes NO behavior on its own.
--
-- Idempotent: ON CONFLICT (config_type, key) DO NOTHING.

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('arbitrage_config', 'shadow_mode', 'true', 'boolean',
     'Shadow mode (default ON): record every detected opportunity with a '
     'simulated outcome to arbitrage_trades(is_simulated=true); never enter '
     'the execute path. Turn off only when ready to consider live execution.',
     NOW(), NOW()),

    ('arbitrage_config', 'live_execution_enabled', 'false', 'boolean',
     'Explicit LIVE opt-in: even with shadow_mode off and dry_run false, '
     'broadcasting an arbitrage transaction requires this to be true. '
     'Fail-safe default false.',
     NOW(), NOW()),

    ('arbitrage_config', 'min_net_spread_bps_ethereum', '0', 'integer',
     'Operator floor on NET spread (bps, after the per-chain breakeven model: '
     'live gas + Aave 5bps flash fee + slippage buffer) required to act on an '
     'Ethereum opportunity. 0 = breakeven model alone decides.',
     NOW(), NOW()),

    ('arbitrage_config', 'min_net_spread_bps_arbitrum', '0', 'integer',
     'Operator floor on NET spread (bps, after breakeven) on Arbitrum. '
     '0 = breakeven model alone decides.',
     NOW(), NOW()),

    ('arbitrage_config', 'min_net_spread_bps_base', '0', 'integer',
     'Operator floor on NET spread (bps, after breakeven) on Base. '
     '0 = breakeven model alone decides.',
     NOW(), NOW()),

    ('arbitrage_config', 'pair_dormancy_minutes', '240', 'integer',
     'When a pair''s rolling-24h p90 cross-DEX spread cannot clear full '
     'breakeven, stop scanning it for this many minutes (saves RPC quota).',
     NOW(), NOW()),

    ('arbitrage_config', 'dormancy_min_samples', '30', 'integer',
     'Minimum observed spread samples before a pair may be put dormant.',
     NOW(), NOW()),

    ('arbitrage_config', 'shadow_record_interval_s', '60', 'integer',
     'Shadow-mode record throttle: at most one simulated arbitrage_trades row '
     'per pair per this many seconds.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;
