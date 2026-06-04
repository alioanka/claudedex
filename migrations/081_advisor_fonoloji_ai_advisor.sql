-- Migration 081: advisor — Fonoloji AI-advisor layer (ADVICE-ONLY).
--
-- Builds on the shared FonolojiClient (migrations 078/080, verified contract:
-- base https://fonoloji.com/v1, X-API-Key, 15,000 req/MONTH free tier). Adds
-- the AI/analyst layer that uses Fonoloji's FREE (within quota — no per-token
-- cost) endpoints to:
--   (a) REPLACE the paid Anthropic rationale for Turkish funds with the free
--       /funds/{code}/ai-summary  (a SAVING, not just a swap), and
--   (b) fold a real broker-consensus VOTE (AL/TUT/SAT + target upside) from
--       /stocks/{ticker}/recommendations into the BIST directional signal — a
--       genuine signal not available from yfinance — as ONE input alongside the
--       existing SMA/RSI/BB technical votes (it does NOT dominate).
--   (c) an optional, light AI daily market digest (/market/digest).
--
-- ALL gated on the Fonoloji key (ADVISOR_FONOLOJI_API_KEY in Secure
-- Credentials). With NO key the behaviour is UNCHANGED. ai-summary + digest are
-- a readonly DB cache that 404s until warmed by a fonoloji.com page visit —
-- the client treats 404 as None (fail-soft), so funds fall back to the existing
-- build_rationale() path and BIST falls back to the unchanged technical signal.
--
-- COST: ai-summary + recommendations are cached 6h and count toward the
-- 15k/month quota. Estimate: BIST-50 recs (~50/day) + N funds ai-summary
-- (~10/day) + digest (~4/day) ≈ a few hundred calls/day — well under 15k/month.
-- Crucially ai-summary REPLACES paid Anthropic calls for funds (net saving).
--
-- Surface the three keys in Advisor Settings (dashboard agent owns the UI).
-- Idempotent. Reserved migration number 081.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('advisor_config', 'advisor_fonoloji_ai_summary_enabled', 'true', 'boolean',
     'When a Fonoloji key is present, use the FREE /funds/{code}/ai-summary as '
     'the Turkish-fund advice rationale INSTEAD of the paid Anthropic '
     'build_rationale() — a net saving of paid LLM spend. 404 (summary not yet '
     'warmed on fonoloji.com) / disabled / no key => falls back to the existing '
     'rationale path. ADVICE-ONLY.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_bist_use_analyst_recommendations', 'true', 'boolean',
     'When a Fonoloji key is present, fold a broker-consensus VOTE from '
     '/stocks/{ticker}/recommendations (AL=+1 / TUT=0 / SAT=-1, plus target '
     'price upside) into the BIST directional signal as ONE input alongside the '
     'SMA/RSI/BB technical votes (it does NOT dominate). A genuine signal not '
     'available from yfinance. The raw summary is stored in '
     'extra[''analyst_consensus''] for the dashboard. FAIL-SOFT: no data => BIST '
     'signal unchanged. ADVICE-ONLY.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_fonoloji_market_digest_enabled', 'false', 'boolean',
     'Optional/light: when a Fonoloji key is present, fetch the AI daily market '
     'digest (/market/digest, cached 6h) once per cycle and stash it in the BIST '
     'advice extra[''market_digest''] for a dashboard hook. Default OFF. 404 (not '
     'yet warmed) / no key => omitted. ADVICE-ONLY.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;
