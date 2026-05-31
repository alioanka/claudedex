-- Migration 044: seed copy_max_concurrent_wallets for wave-15 fan-out cap.
--
-- Root cause of Helius 429 storm: wave-14 asyncio.gather fires all N leader
-- wallets concurrently (33 wallets = 33 simultaneous Helius REST calls per
-- 15 s cycle).  Helius free/developer plans rate-limit at ~10-30 RPS per
-- API key, so 33 concurrent calls trigger sustained 429 bursts.
--
-- This key caps the asyncio.Semaphore slot count in _monitor_solana_wallets.
-- Default 5: allows 5 concurrent Helius calls at a time, spreading 33
-- wallets across ~7 batches per cycle (~6-8 s total at 8 s timeout each).
-- Operator can raise to 10 on paid Helius plan or lower to 2 on free tier.
-- Clamped [1, 20] in copy_engine._load_settings.

INSERT INTO config_settings (config_type, key, value)
VALUES (
    'copytrading_config',
    'copy_max_concurrent_wallets',
    '5'
)
ON CONFLICT (config_type, key) DO NOTHING;
