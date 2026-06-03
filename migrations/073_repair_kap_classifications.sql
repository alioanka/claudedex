-- Migration 073: REPAIR kap_classifications (idempotent re-create)
--
-- Migration 063 is recorded as APPLIED in the migrations table, but the
-- kap_classifications table does not actually exist at runtime (it was dropped
-- at some point while the 063 migration record stayed). Because the migrator
-- skips 063 ("already applied"), the table never comes back, so the KAP
-- classifier worker logs "kap_classifications table not found" every cycle and
-- classifications never persist (which also kept the LLM re-classifying the
-- same disclosures until the daily budget cap stopped it).
--
-- This forward-only migration re-asserts the table with CREATE TABLE IF NOT
-- EXISTS, so it is a no-op if the table is already present and a clean repair
-- if it is missing. DDL copied verbatim from 063 §2 to guarantee schema match.
-- kap_event_taxonomy is also re-asserted (table only, no seed — 069 confirmed
-- it exists and is populated) so the FK target is guaranteed before the FK.
--
-- Idempotent. No data loss. ADVICE-ONLY (KAP is a polarity-prior engine).

BEGIN;

-- FK target guard (no-op if it already exists; seed lives in 063/069).
CREATE TABLE IF NOT EXISTS kap_event_taxonomy (
    event_type          VARCHAR(64)   PRIMARY KEY,
    base_polarity       VARCHAR(32)   NOT NULL,
    description         TEXT          NOT NULL DEFAULT '',
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kap_classifications (
    id                  BIGSERIAL     PRIMARY KEY,
    disclosure_id       BIGINT        NOT NULL,
    event_type          VARCHAR(64)   NOT NULL REFERENCES kap_event_taxonomy(event_type),
    base_polarity       VARCHAR(32)   NOT NULL,
    params              JSONB         NOT NULL DEFAULT '{}',
    classifier_stage    VARCHAR(16)   NOT NULL DEFAULT 'unclassified',
    confidence          NUMERIC(4,3)  NOT NULL DEFAULT 0,
    raw_subject         TEXT          NOT NULL DEFAULT '',
    extra               JSONB         NOT NULL DEFAULT '{}',
    classified_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (disclosure_id)
);

CREATE INDEX IF NOT EXISTS idx_kap_clf_disclosure
    ON kap_classifications (disclosure_id);
CREATE INDEX IF NOT EXISTS idx_kap_clf_event_type
    ON kap_classifications (event_type, classified_at DESC);
CREATE INDEX IF NOT EXISTS idx_kap_clf_polarity
    ON kap_classifications (base_polarity, classified_at DESC);
CREATE INDEX IF NOT EXISTS idx_kap_clf_stage
    ON kap_classifications (classifier_stage, classified_at DESC);

COMMIT;
