-- Migration 065: Financial Advisor module — Secure Credentials placeholder rows
--
-- The dashboard's Secure Credentials page (/credentials) lists rows from the
-- secure_credentials table. The advisor's API keys were never seeded there, so
-- they had no cards in the UI even though the module reads them. This seeds the
-- advisor keys as PLACEHOLDER rows (module='advisor') so they appear as
-- not-configured cards the operator can fill in.
--
-- Resolution: main_advisor.py injects ADVISOR_ANTHROPIC/OPENAI/FX/BIST keys from
-- secrets_manager (encrypted DB) into the advisor config at startup; the advisor
-- Telegram bot resolves ADVISOR_TELEGRAM_* via secrets_manager directly. All keep
-- an os.getenv/.env fallback. None are trade-capable (advice-only module).
--
-- Idempotent: ON CONFLICT (key_name) DO NOTHING.
--
-- NOTE: ADVISOR_KRONOS_WEIGHTS_PATH is intentionally NOT a credential — it is a
-- filesystem path (not a secret) read via os.getenv only, so it belongs in .env
-- (see docs/ADVISOR_SETUP_AND_USAGE.md §5), not the Secure Credentials store.

INSERT INTO secure_credentials (key_name, display_name, description, category, subcategory, module, is_required, is_sensitive, encrypted_value) VALUES
('ADVISOR_ANTHROPIC_API_KEY',   'Advisor — Anthropic API Key',   'Anthropic key for advisor LLM advice rationale (all markets). May reuse your existing Anthropic key. Without it, rationale falls back to rule-based.', 'api', 'anthropic', 'advisor', FALSE, TRUE, 'PLACEHOLDER'),
('ADVISOR_OPENAI_API_KEY',      'Advisor — OpenAI API Key',      'OpenAI key for the advisor dual-advice second opinion. Set this, then enable dual-advice in Advisor Settings. Optional.', 'api', 'openai', 'advisor', FALSE, TRUE, 'PLACEHOLDER'),
('ADVISOR_FX_ALPHAVANTAGE_KEY', 'Advisor — Alpha Vantage Key',   'Alpha Vantage key for real-time FX/metals (free tier ~25 req/day). Optional; only used when advisor_fx_data_source=alphavantage.', 'api', 'alphavantage', 'advisor', FALSE, TRUE, 'PLACEHOLDER'),
('ADVISOR_BIST_API_KEY',        'Advisor — BIST (Matriks) Key',  'Paid BIST (Matriks) data key for full Borsa Istanbul coverage. Optional; free borsapy/yfinance works without it.', 'api', 'matriks', 'advisor', FALSE, TRUE, 'PLACEHOLDER'),
('ADVISOR_TELEGRAM_BOT_TOKEN',  'Advisor — Telegram Bot Token',  'Token for a SEPARATE advisor Telegram bot (from @BotFather), distinct from the trading bot. Optional.', 'notification', 'telegram', 'advisor', FALSE, TRUE, 'PLACEHOLDER'),
('ADVISOR_TELEGRAM_CHAT_ID',    'Advisor — Telegram Chat ID',    'Chat/channel ID that receives advisor Telegram messages. Optional.', 'notification', 'telegram', 'advisor', FALSE, FALSE, 'PLACEHOLDER')
ON CONFLICT (key_name) DO NOTHING;

-- DOWN (reversible):
-- DELETE FROM secure_credentials WHERE module='advisor' AND encrypted_value='PLACEHOLDER';
