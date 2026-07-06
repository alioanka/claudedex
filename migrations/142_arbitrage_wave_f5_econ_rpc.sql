-- Migration 142: Wave-F5 arbitrage economics + RPC-refresh seeds
--
-- RC-A2 (economics): the DB-configured flash_loan_amount of 10 ETH pushed
-- every round-trip quote through thin V2 pools with ~-162bps of size-driven
-- price impact (observed median best spread -1.62% over 3 weeks, zero
-- positive samples). Shrink the scan/borrow size to 1 ETH so price impact
-- stops dominating the spread math. Conditional UPDATE only rewrites the
-- untouched default ('10'); an operator who deliberately set another value
-- keeps it.
--
-- RC-A1 (RPC): seed arb_rpc_refresh_minutes (periodic pool_engine endpoint
-- re-resolve; the URL used to be pinned at startup for the process lifetime)
-- and arb_max_price_impact_bps (reserve-math pre-filter threshold).
--
-- DRY_RUN-safe: no live flag is touched; all keys are scan-economics or
-- infra knobs. Idempotent: conditional UPDATE + ON CONFLICT DO NOTHING.

UPDATE config_settings
SET value = '1', updated_at = NOW()
WHERE config_type = 'arbitrage_config'
  AND key = 'flash_loan_amount'
  AND value = '10';

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('arbitrage_config', 'flash_loan_amount', '1', 'float',
     'Flash-loan scan/borrow size in ETH. Wave-F5: 10 ETH produced ~-162bps '
     'of round-trip price impact on thin V2 pools, drowning the real 1-30bps '
     'cross-DEX divergence; 1 ETH keeps impact subordinate to the spread.',
     NOW(), NOW()),

    ('arbitrage_config', 'arb_rpc_refresh_minutes', '15', 'integer',
     'Periodically re-resolve the chain RPC endpoint from pool_engine every '
     'N minutes (0 disables). Wave-F5 RC-A1: the endpoint was pinned at '
     'startup, so a dead/over-quota key was never swapped out.',
     NOW(), NOW()),

    ('arbitrage_config', 'arb_max_price_impact_bps', '50', 'integer',
     'Reserve-math price-impact pre-filter: skip quoting a DEX whose pool '
     'reserves imply more than this many bps of one-way price impact at the '
     'configured flash_loan_amount. Saves RPC quota and keeps spread stats '
     'honest (impact-dominated quotes are not opportunities).',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;
