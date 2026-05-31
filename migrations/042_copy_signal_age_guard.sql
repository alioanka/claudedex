-- Migration 042: seed copy_max_signal_age_s for wave-14 staleness guard.
--
-- Profitability basis: measured p50 fill delay is 23,443 ms; Solana memecoin
-- edge evaporates within ~1-2 s of confirmation.  Any signal older than 5 s
-- is structurally unprofitable to mirror — we absorb price impact on entry
-- with no edge remaining.  This guard hard-gates those late signals before
-- they reach the executor.
--
-- Operator tuning: update the value column to raise/lower the threshold.
-- Setting >= 3600 effectively disables the guard (fail-open, not recommended
-- for live memecoins).  Setting < 1 is clamped to 1.0 in copy_engine.py.

INSERT INTO config_settings (config_type, key, value)
VALUES (
    'copytrading_config',
    'copy_max_signal_age_s',
    '5.0'
)
ON CONFLICT (config_type, key) DO NOTHING;
