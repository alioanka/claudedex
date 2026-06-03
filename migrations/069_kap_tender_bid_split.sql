-- Migration 069: KAP tender taxonomy split — TENDER_BID vs TENDER_WIN
--
-- Issue #4: the classifier conflated "entered/participated in a tender"
-- (ihaleye katılım / teklif verilmesi / 1. oturum, outcome UNKNOWN) with
-- "won a tender" (ihale kazanıldı / üzerinde kalmıştır + ihale bedeli),
-- labelling a mere bid as TENDER_WIN / STRONG_POSITIVE. This adds a new
-- TENDER_BID event type (NEUTRAL — outcome unknown) and re-narrows TENDER_WIN
-- so it requires explicit AWARD evidence.
--
-- The live classifier reads its taxonomy from the Python TAXONOMY dict
-- (modules/advisor/core/kap/taxonomy.py); this table is the reference/seed
-- copy AND, critically, the FK target for kap_classifications.event_type.
-- TENDER_BID MUST exist here or the classifier's INSERT into
-- kap_classifications would violate the FK and fail (fail-soft -> no write).
--
-- Idempotent: TENDER_BID via INSERT ... ON CONFLICT DO NOTHING; TENDER_WIN via
-- UPDATE. Does NOT alter the already-shipped migration 063.
-- ADVICE-ONLY: base_polarity is a documented PRIOR, not a market-impact score.

BEGIN;

-- New: tender participation / bid submitted (outcome unknown).
INSERT INTO kap_event_taxonomy
    (event_type, base_polarity, notes, param_keys, turkish_triggers, priority,
     created_at, updated_at)
VALUES
    ('TENDER_BID', 'NEUTRAL',
     'Tender PARTICIPATION / bid submitted: the company entered a public or '
     'private tender (katılım, teklif verilmesi, 1. oturum) but the OUTCOME IS '
     'UNKNOWN at disclosure time (İhale Sonucu / İhale Bedeli fields empty). '
     'NEUTRAL prior: entering a tender is not winning one. Promote to TENDER_WIN '
     'only when an award is explicitly stated (kazanılmıştır / üzerinde '
     'kalmıştır / ihaleyi kazandı + ihale bedeli).',
     '["contract_value"]',
     '["ihaleye katıldı", "teklif verilmesi", "1. oturuma katıldı", '
     '"İhaleye Teklif Verme Tarihi"]',
     4,
     NOW(), NOW())
ON CONFLICT (event_type) DO NOTHING;

-- Re-narrow TENDER_WIN to require explicit award evidence.
UPDATE kap_event_taxonomy
SET notes = 'Government or public tender WIN — an AWARD is confirmed '
            '(ihale kazanıldı / üzerinde kalmıştır / uhdemizde kaldı). Strong '
            'positive: lower counterparty risk, multi-year revenue visibility; '
            'BIST construction, defence, infrastructure react strongly. Distinct '
            'from TENDER_BID (mere participation, outcome unknown).',
    turkish_triggers = '["ihale kazanıldı", "ihaleyi kazandı", '
                       '"ihale üzerinde kalmıştır", "uhdemizde kaldı"]',
    priority = 4,
    updated_at = NOW()
WHERE event_type = 'TENDER_WIN';

COMMIT;

-- ---------------------------------------------------------------------------
-- DOWN (manual rollback)
-- ---------------------------------------------------------------------------
-- BEGIN;
-- DELETE FROM kap_classifications WHERE event_type = 'TENDER_BID';
-- DELETE FROM kap_event_taxonomy  WHERE event_type = 'TENDER_BID';
-- UPDATE kap_event_taxonomy
--   SET turkish_triggers = '["ihale kazanıldı", "ihale sonucu", "kamu ihale"]',
--       priority = 1
--   WHERE event_type = 'TENDER_WIN';
-- COMMIT;
