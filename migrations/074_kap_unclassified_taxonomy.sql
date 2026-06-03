-- Migration 074: seed the UNCLASSIFIED taxonomy sentinel (FK fix)
--
-- The KAP classifier emits event_type='UNCLASSIFIED' as its fail-soft fallback
-- (no rule matched AND the LLM was unavailable / budget exhausted). But
-- migration 063 deliberately did NOT add an UNCLASSIFIED row to
-- kap_event_taxonomy (see modules/advisor/core/kap/taxonomy.py: "UNCLASSIFIED is
-- not in TAXONOMY"). Since kap_classifications.event_type has a FOREIGN KEY to
-- kap_event_taxonomy(event_type), every UNCLASSIFIED classification was rejected
-- with a foreign-key violation and never persisted:
--   Key (event_type)=(UNCLASSIFIED) is not present in table "kap_event_taxonomy".
--
-- UNCLASSIFIED is a legitimate, storable outcome (it records "we saw this
-- disclosure and could not classify it") and carries NO market-impact prior, so
-- it is NEUTRAL by definition. Adding it satisfies the FK so these rows persist.
--
-- Idempotent: ON CONFLICT (event_type) DO NOTHING. ADVICE-ONLY.

BEGIN;

INSERT INTO kap_event_taxonomy
    (event_type, base_polarity, notes, param_keys, turkish_triggers, priority)
VALUES
    ('UNCLASSIFIED', 'NEUTRAL',
     'Sentinel for disclosures the rule stage did not match and the LLM could '
     'not classify (or the daily LLM budget was exhausted). Carries NO '
     'market-impact prior — NEUTRAL by definition. Required because '
     'kap_classifications.event_type FK-references this table and the classifier '
     'emits UNCLASSIFIED as its fail-soft fallback. Never rule-matched (it IS the '
     'no-match outcome), so priority is irrelevant.',
     '[]'::jsonb, '[]'::jsonb, 0)
ON CONFLICT (event_type) DO NOTHING;

COMMIT;

-- ---------------------------------------------------------------------------
-- DOWN (manual rollback)
-- ---------------------------------------------------------------------------
-- BEGIN;
-- DELETE FROM kap_classifications WHERE event_type='UNCLASSIFIED';
-- DELETE FROM kap_event_taxonomy WHERE event_type='UNCLASSIFIED';
-- COMMIT;
