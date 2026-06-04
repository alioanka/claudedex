-- Migration 082: seed ADVISOR_FONOLOJI_API_KEY as a Secure Credentials card
--
-- The Fonoloji integration (migrations 078/080/081) resolves
-- ADVISOR_FONOLOJI_API_KEY via secrets_manager (encrypted DB) -> os.getenv
-- fallback, and main_advisor.py injects it at startup. But the key was never
-- seeded as a card on the dashboard Secure Credentials page, so the operator
-- had no UI field to enter + encrypt it. This adds it (PLACEHOLDER row), mirroring
-- the other advisor keys from migration 065.
--
-- Fonoloji (https://fonoloji.com) is a Turkish TEFAS/BIST analytics API. Free
-- tier 15k req/month. Powers Midas-fund NAV, BIST equities, BIST gems,
-- FX/gold spot, AI fund summaries, and broker recommendations — all key-gated
-- and fail-soft (no key => existing free sources unchanged). ADVICE-ONLY.
--
-- Idempotent: ON CONFLICT (key_name) DO NOTHING.

INSERT INTO secure_credentials
    (key_name, display_name, description, category, subcategory, module, is_required, is_sensitive, encrypted_value)
VALUES
('ADVISOR_FONOLOJI_API_KEY', 'Advisor — Fonoloji API Key',
 'Fonoloji (fonoloji.com) key for Turkish data: Midas-fund NAV history, BIST '
 'equity prices + universe + screener gems, FX/gold spot, free AI fund '
 'summaries, and broker AL/TUT/SAT recommendations. Free tier 15k req/month. '
 'Optional; without it the advisor uses its existing free fallbacks. After '
 'adding, set advisor_midas_data_source=''fonoloji'' and '
 'advisor_bist_data_source=''fonoloji'' in Advisor Settings.',
 'api', 'fonoloji', 'advisor', FALSE, TRUE, 'PLACEHOLDER')
ON CONFLICT (key_name) DO NOTHING;

-- DOWN (reversible):
-- DELETE FROM secure_credentials WHERE key_name='ADVISOR_FONOLOJI_API_KEY' AND encrypted_value='PLACEHOLDER';
