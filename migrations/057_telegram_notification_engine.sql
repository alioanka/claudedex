-- Migration 057: Telegram notification engine config
-- Wave-19: topic-routed Telegram notifications with per-module topics,
-- periodic dashboard/summary jobs, rate-limiting, and error de-dup.
-- All keys in config_type='telegram_config'.
-- Idempotent: ON CONFLICT DO NOTHING.

BEGIN;

-- Master enable/disable
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('telegram_config', 'notifications_enabled',
     'true', 'boolean',
     'Master toggle for all Telegram notifications')
ON CONFLICT (config_type, key) DO NOTHING;

-- Telegram forum-topic group ID (numeric supergroup ID, e.g. -1001234567890)
-- Leave empty to keep single-DM fallback (existing behavior unchanged)
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('telegram_config', 'telegram_group_id',
     '', 'string',
     'Supergroup ID with Topics enabled. Empty = single-DM fallback.')
ON CONFLICT (config_type, key) DO NOTHING;

-- Per-module topic thread IDs (empty = use group/DM fallback)
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('telegram_config', 'topic_thread_id_dex',
     '', 'string', 'Forum topic thread_id for DEX trade alerts'),
    ('telegram_config', 'topic_thread_id_futures',
     '', 'string', 'Forum topic thread_id for Futures trade alerts'),
    ('telegram_config', 'topic_thread_id_solana',
     '', 'string', 'Forum topic thread_id for Solana trade alerts'),
    ('telegram_config', 'topic_thread_id_ai',
     '', 'string', 'Forum topic thread_id for AI trade alerts'),
    ('telegram_config', 'topic_thread_id_sniper',
     '', 'string', 'Forum topic thread_id for Sniper trade alerts'),
    ('telegram_config', 'topic_thread_id_arbitrage',
     '', 'string', 'Forum topic thread_id for Arbitrage trade alerts'),
    ('telegram_config', 'topic_thread_id_copy',
     '', 'string', 'Forum topic thread_id for Copy Trading trade alerts')
ON CONFLICT (config_type, key) DO NOTHING;

-- Special-purpose topic thread IDs
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('telegram_config', 'topic_thread_id_dashboard',
     '', 'string', 'Forum topic thread_id for the periodic full dashboard message'),
    ('telegram_config', 'topic_thread_id_summary',
     '', 'string', 'Forum topic thread_id for the compact periodic summary'),
    ('telegram_config', 'topic_thread_id_error',
     '', 'string', 'Forum topic thread_id for de-duped error/warning messages')
ON CONFLICT (config_type, key) DO NOTHING;

-- Periodic job intervals
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('telegram_config', 'dashboard_interval_hours',
     '3', 'integer',
     'How often (hours) to post the full dashboard summary to the Dashboard topic'),
    ('telegram_config', 'summary_interval_hours',
     '6', 'integer',
     'How often (hours) to post the compact summary to the Summary topic')
ON CONFLICT (config_type, key) DO NOTHING;

-- Error de-duplication window (seconds).
-- Same module+error-signature is suppressed within this window; a rollup is
-- sent when the window expires showing the repeat count.
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('telegram_config', 'error_dedup_window_s',
     '900', 'integer',
     'Error de-dup window in seconds. Same module+error hash suppressed within window.')
ON CONFLICT (config_type, key) DO NOTHING;

-- Per-module verbosity: all | summary | off
-- 'all'     = every trade notification forwarded
-- 'summary' = only periodic summary posted; individual trades suppressed
-- 'off'     = module silent (errors still reach error topic)
-- Solana defaults to 'summary' due to ~1000 msgs / 2-3 days volume
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('telegram_config', 'notify_dex_mode',
     'all', 'string', 'DEX trade notification verbosity: all|summary|off'),
    ('telegram_config', 'notify_futures_mode',
     'all', 'string', 'Futures trade notification verbosity: all|summary|off'),
    ('telegram_config', 'notify_solana_mode',
     'summary', 'string', 'Solana trade notification verbosity: all|summary|off (default summary due to volume)'),
    ('telegram_config', 'notify_ai_mode',
     'all', 'string', 'AI trade notification verbosity: all|summary|off'),
    ('telegram_config', 'notify_sniper_mode',
     'all', 'string', 'Sniper trade notification verbosity: all|summary|off'),
    ('telegram_config', 'notify_arbitrage_mode',
     'all', 'string', 'Arbitrage trade notification verbosity: all|summary|off'),
    ('telegram_config', 'notify_copy_mode',
     'all', 'string', 'Copy Trading trade notification verbosity: all|summary|off')
ON CONFLICT (config_type, key) DO NOTHING;

-- Per-module trade-notification throttle (minimum seconds between consecutive
-- trade messages per module). 0 = no throttle.
-- Solana default 60s to cap flood even in 'all' mode.
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('telegram_config', 'throttle_dex_s',
     '0', 'integer', 'Min seconds between DEX trade notifications (0=no throttle)'),
    ('telegram_config', 'throttle_futures_s',
     '0', 'integer', 'Min seconds between Futures trade notifications'),
    ('telegram_config', 'throttle_solana_s',
     '60', 'integer', 'Min seconds between Solana trade notifications (default 60)'),
    ('telegram_config', 'throttle_ai_s',
     '0', 'integer', 'Min seconds between AI trade notifications'),
    ('telegram_config', 'throttle_sniper_s',
     '0', 'integer', 'Min seconds between Sniper trade notifications'),
    ('telegram_config', 'throttle_arbitrage_s',
     '0', 'integer', 'Min seconds between Arbitrage trade notifications'),
    ('telegram_config', 'throttle_copy_s',
     '0', 'integer', 'Min seconds between Copy Trading trade notifications')
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;

-- DOWN (reversible):
-- BEGIN;
-- DELETE FROM config_settings WHERE config_type = 'telegram_config';
-- COMMIT;
