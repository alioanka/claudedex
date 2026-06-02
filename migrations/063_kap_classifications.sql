-- Migration 063: KAP Event Classifier — Wave-23
-- Creates: kap_classifications, kap_event_taxonomy
-- Seeds:   full taxonomy reference (24 event types) from classifier taxonomy.py
-- Depends: migration 062 (kap_disclosures) by sibling kap-storage agent.
-- Idempotent: CREATE TABLE IF NOT EXISTS + INSERT ON CONFLICT DO NOTHING
-- Safe to re-run; no destructive changes.
--
-- DESIGN NOTE: base_polarity is a documented PRIOR, not a quantitative
-- impact prediction.  Numeric impact estimates require historical return
-- statistics from kap_returns (accumulated over months — not yet available).
-- Do NOT add impact_score columns until sample sizes are adequate.

BEGIN;

-- =========================================================================
-- 1. kap_event_taxonomy
--    Reference table seeded from classifier/taxonomy.py definitions.
--    Used by the dashboard to display the full event taxonomy.
--    Populated below; do not edit rows manually (re-run migration to refresh).
-- =========================================================================
CREATE TABLE IF NOT EXISTS kap_event_taxonomy (
    event_type          VARCHAR(64)   PRIMARY KEY,
    base_polarity       VARCHAR(32)   NOT NULL,
    notes               TEXT          NOT NULL DEFAULT '',
    param_keys          JSONB         NOT NULL DEFAULT '[]',
    turkish_triggers    JSONB         NOT NULL DEFAULT '[]',
    priority            SMALLINT      NOT NULL DEFAULT 1,
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE kap_event_taxonomy IS
    'Reference table of KAP disclosure event types with Turkish trigger patterns '
    'and qualitative polarity priors. Seeded by migration 063. '
    'base_polarity is a documented prior, NOT a quantitative impact prediction. '
    'See modules/advisor/core/kap/taxonomy.py for authoritative source.';

COMMENT ON COLUMN kap_event_taxonomy.base_polarity IS
    'Qualitative market-impact prior: STRONG_POSITIVE, POSITIVE, NEUTRAL, NEGATIVE, VERY_NEGATIVE. '
    'Not a quantitative prediction. Impact estimation requires historical return statistics '
    'which accumulate in kap_returns over months.';

-- Seed taxonomy reference rows (idempotent)
-- Source of truth: modules/advisor/core/kap/taxonomy.py
-- Re-run migration to refresh if taxonomy changes.

INSERT INTO kap_event_taxonomy
    (event_type, base_polarity, notes, param_keys, turkish_triggers, priority)
VALUES

-- Capital structure / equity events
('BONUS_ISSUE', 'POSITIVE',
 'Bedelsiz (free) capital increase: shareholders receive new shares from retained '
 'earnings/revaluation reserves. Market cap unchanged; price adjusts down by '
 'dilution factor. Neutral to mildly positive (retail momentum, signals healthy '
 'reserves). Do NOT confuse with RIGHTS_ISSUE (bedelli = dilution cash call).',
 '["bonus_ratio"]',
 '["bedelsiz sermaye artırımı", "bedelsiz pay dağıtımı", "iç kaynaktan sermaye artırımı"]',
 2),

('RIGHTS_ISSUE', 'NEGATIVE',
 'Bedelli (paid) capital increase: company issues new shares to raise cash. '
 'Existing shareholders diluted unless they exercise pre-emptive rights (rüçhan hakkı). '
 'Prior is NEGATIVE because dilution is certain and use-of-proceeds quality is '
 'uncertain at disclosure time. Do NOT confuse with BONUS_ISSUE (bedelsiz = free).',
 '["amount_tl"]',
 '["bedelli sermaye artırımı", "nakdi sermaye artırımı", "rüçhan hakkı"]',
 2),

('SHARE_BUYBACK', 'POSITIVE',
 'Share buyback programme. Positive: signals management confidence, EPS accretion. '
 'Actual execution rate may be lower than announced.',
 '["amount_tl", "pct_stake"]',
 '["pay geri alım programı", "hisse geri alım programı", "öz hisse alımı"]',
 1),

('CAPITAL_REDUCTION', 'NEGATIVE',
 'Registered capital reduction. May signal accumulated losses requiring absorption '
 '(Turkish Commercial Code Article 376: losses exceed half of capital).',
 '["amount_tl"]',
 '["sermaye azaltımı", "sermaye indirimi"]',
 1),

-- Income events
('DIVIDEND', 'POSITIVE',
 'Cash or in-kind dividend announcement. Positive: signals profitability and '
 'willingness to return capital. Size relative to share price drives actual impact.',
 '["dividend_per_share"]',
 '["temettü", "kar payı dağıtımı", "nakit temettü"]',
 1),

-- Business wins / commercial
('NEW_CONTRACT', 'POSITIVE',
 'New material commercial contract. Positive: adds revenue visibility. '
 'Size relative to market cap drives actual impact. contract_value param captured.',
 '["contract_value"]',
 '["yeni sözleşme imzalandı", "sipariş alındı", "tedarik sözleşmesi"]',
 1),

('TENDER_WIN', 'STRONG_POSITIVE',
 'Government or public tender win. Strong positive: lower counterparty risk, '
 'multi-year revenue visibility. BIST construction, defence, infrastructure '
 'sectors see especially strong reactions.',
 '["contract_value"]',
 '["ihale kazanıldı", "ihale sonucu", "kamu ihale"]',
 1),

('EXPORT_AGREEMENT', 'POSITIVE',
 'Export or international trade agreement. Positive: FX revenue diversification, '
 'especially valuable during TRY depreciation periods.',
 '["contract_value"]',
 '["ihracat sözleşmesi", "ihracat anlaşması", "uluslararası sözleşme"]',
 1),

-- Capacity / operations
('CAPACITY_INCREASE', 'POSITIVE',
 'Capacity expansion investment decision. Positive: growth signal, but short-term '
 'capex impact can weigh on FCF. Duration of project matters for horizon alignment.',
 '["amount_tl"]',
 '["kapasite artışı", "yeni üretim hattı", "genişletme yatırımı"]',
 1),

('FACTORY_OPENING', 'POSITIVE',
 'New facility officially opened or commissioned. Positive: transition from '
 'capex spend phase to revenue generation phase.',
 '[]',
 '["fabrika açılışı", "tesis açılışı", "üretim tesisi devreye alındı"]',
 1),

('PRODUCTION_HALT', 'VERY_NEGATIVE',
 'Production halt, facility shutdown, or strike. Very negative: direct revenue '
 'loss with no offsetting benefit announced.',
 '[]',
 '["üretim durduruldu", "tesis kapatıldı", "grev"]',
 1),

-- Corporate actions
('ACQUISITION', 'POSITIVE',
 'Acquisition or merger. Positive prior: growth signal. Actual impact depends '
 'on price paid and strategic fit. Acquirer may dip initially (integration risk '
 'premium); target typically rises sharply.',
 '["amount_tl", "amount_usd", "pct_stake"]',
 '["devralma kararı", "satın alma kararı", "birleşme anlaşması"]',
 1),

('PARTNERSHIP', 'POSITIVE',
 'Strategic partnership or JV (non-acquisition). Positive: market validation, '
 'shared risk signal.',
 '["pct_stake"]',
 '["ortaklık anlaşması", "iş birliği anlaşması", "ortak girişim"]',
 1),

-- Management / governance
('CEO_CHANGE', 'NEUTRAL',
 'Senior executive appointment or resignation. Neutral prior: unexpected '
 'departures can be negative; planned succession neutral; high-profile external '
 'hire can be positive.',
 '["exec_name", "exec_role"]',
 '["genel müdür değişikliği", "genel müdür atandı", "üst yönetim değişikliği"]',
 1),

('BOARD_MEETING', 'NEUTRAL',
 'Board of directors meeting notice or minutes. Neutral: meeting itself '
 'informational; decisions (dividend, capital increase) classified separately.',
 '[]',
 '["yönetim kurulu toplantısı", "yönetim kurulu kararı"]',
 1),

('GENERAL_ASSEMBLY', 'NEUTRAL',
 'General assembly (AGM or EGM) notice or minutes. Neutral: agenda items '
 'classified separately when they match higher-priority entries.',
 '[]',
 '["olağan genel kurul", "olağanüstü genel kurul", "genel kurul toplantısı"]',
 1),

-- Investor / insider events
('INSIDER_PURCHASE', 'POSITIVE',
 'Insider (director or major shareholder) purchases shares. Positive: '
 'skin-in-the-game conviction signal.',
 '["exec_name", "exec_role", "amount_tl", "pct_stake"]',
 '["yönetici pay alımı", "ortak pay alımı", "içeriden alım"]',
 1),

('INSIDER_SALE', 'NEGATIVE',
 'Insider sells shares. Negative prior but weaker than purchase signal: '
 'insiders may sell for personal liquidity reasons unrelated to outlook.',
 '["exec_name", "exec_role", "amount_tl", "pct_stake"]',
 '["yönetici pay satışı", "ortak pay satışı", "içeriden satış"]',
 1),

-- Regulatory / credit events
('SPK_INVESTIGATION', 'VERY_NEGATIVE',
 'SPK (capital markets regulator) investigation or administrative action. '
 'Very negative: regulatory uncertainty, possible trading halt, reputational damage.',
 '[]',
 '["SPK soruşturma", "SPK inceleme", "manipülasyon iddia"]',
 1),

('CREDIT_RATING', 'NEUTRAL',
 'Credit rating action. Base is NEUTRAL because upgrades are positive and '
 'downgrades are negative — direction requires reading the content. '
 'rating_from / rating_to params capture direction.',
 '["rating_from", "rating_to"]',
 '["kredi derecelendirme", "kredi notu değişti", "fitch moody s&p jcr"]',
 1),

('LAWSUIT', 'NEGATIVE',
 'Significant new lawsuit or legal proceeding. Negative: uncertain liability, '
 'legal costs, management distraction.',
 '["amount_tl"]',
 '["dava açıldı", "hukuki süreç başlatıldı", "tazminat davası"]',
 1),

-- Financial results
('FINANCIAL_RESULTS', 'NEUTRAL',
 'Periodic financial statement publication. Neutral: results can beat or miss '
 'expectations. Actual direction requires comparison against consensus estimates.',
 '[]',
 '["finansal tablo", "bilanço açıklandı", "dönem sonu finansal"]',
 1),

('GUIDANCE', 'NEUTRAL',
 'Forward guidance or earnings revision. Neutral prior: direction of revision '
 '(upward vs downward) determines actual polarity.',
 '[]',
 '["beklenti revizyonu", "öngörü revizyonu", "kar uyarısı"]',
 1),

-- Distress
('DEFAULT', 'VERY_NEGATIVE',
 'Bankruptcy, concordat, or payment default. Very negative: existential threat '
 'to equity value. Turkish concordat (konkordato) often precedes significant '
 'equity haircuts.',
 '[]',
 '["konkordato", "iflas kararı", "kredi temerrüt"]',
 1)

ON CONFLICT (event_type) DO NOTHING;


-- =========================================================================
-- 2. kap_classifications
--    One row per classified disclosure.
--    disclosure_id FK references kap_disclosures.id (migration 062).
--    Use INSERT ON CONFLICT UPDATE to allow re-classification (e.g. after
--    a taxonomy update triggers a re-run of the rule stage).
-- =========================================================================
CREATE TABLE IF NOT EXISTS kap_classifications (
    id                  BIGSERIAL     PRIMARY KEY,
    disclosure_id       BIGINT        NOT NULL,   -- FK to kap_disclosures.id (062)
    event_type          VARCHAR(64)   NOT NULL REFERENCES kap_event_taxonomy(event_type),
    base_polarity       VARCHAR(32)   NOT NULL,
    params              JSONB         NOT NULL DEFAULT '{}',
    classifier_stage    VARCHAR(16)   NOT NULL DEFAULT 'unclassified',
                        -- 'rule' | 'llm' | 'unclassified'
    confidence          NUMERIC(4,3)  NOT NULL DEFAULT 0,
                        -- [0.000, 1.000] — classifier certainty only.
                        -- NOT a market-impact score.
    raw_subject         TEXT          NOT NULL DEFAULT '',
    extra               JSONB         NOT NULL DEFAULT '{}',
    classified_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (disclosure_id)              -- one active classification per disclosure
);

COMMENT ON TABLE kap_classifications IS
    'KAP disclosure event classifications produced by the two-stage rule+LLM '
    'classifier (modules/advisor/core/kap/classifier.py). '
    'One row per disclosure_id. Re-classification overwrites via ON CONFLICT UPDATE. '
    'classifier_stage: rule (deterministic regex) | llm (LLM fallback) | unclassified. '
    'confidence: classifier certainty [0,1] — NOT a market-impact score. '
    'base_polarity: qualitative prior from taxonomy — NOT a quantitative prediction.';

COMMENT ON COLUMN kap_classifications.confidence IS
    'Classifier certainty [0.000, 1.000]. rule stage = 0.900, llm stage = 0.650. '
    'This is NOT a market-impact score. Impact estimation requires historical '
    'return statistics from kap_returns (accumulated over months).';

COMMENT ON COLUMN kap_classifications.base_polarity IS
    'Qualitative market-impact prior from kap_event_taxonomy. '
    'STRONG_POSITIVE | POSITIVE | NEUTRAL | NEGATIVE | VERY_NEGATIVE. '
    'This is a documented prior, NOT a quantitative return prediction.';

COMMENT ON COLUMN kap_classifications.params IS
    'Structured params extracted from disclosure text. '
    'Possible keys: bonus_ratio, dividend_per_share, contract_value, amount_tl, '
    'amount_usd, pct_stake, rating_from, rating_to, exec_name, exec_role. '
    'Empty object if no params could be extracted.';

CREATE INDEX IF NOT EXISTS idx_kap_clf_disclosure
    ON kap_classifications (disclosure_id);

CREATE INDEX IF NOT EXISTS idx_kap_clf_event_type
    ON kap_classifications (event_type, classified_at DESC);

CREATE INDEX IF NOT EXISTS idx_kap_clf_polarity
    ON kap_classifications (base_polarity, classified_at DESC);

CREATE INDEX IF NOT EXISTS idx_kap_clf_stage
    ON kap_classifications (classifier_stage, classified_at DESC);


-- =========================================================================
-- 3. Config seeds — KAP classifier settings (advisor_config)
-- =========================================================================

-- Enable/disable KAP classifier (default off until kap_disclosures populated)
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES (
    'advisor_config',
    'kap_classifier_enabled',
    'false',
    'boolean',
    'Enable KAP disclosure event classification loop. '
    'Requires migration 062 (kap_disclosures) and a populated kap_disclosures table. '
    'Set true after the kap_listener (sibling agent) is running. Default false.'
)
ON CONFLICT (config_type, key) DO NOTHING;

-- Classify-on-ingest vs batch-only
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES (
    'advisor_config',
    'kap_classifier_live_mode',
    'true',
    'boolean',
    'When true, classify each disclosure as it is ingested (live path). '
    'When false, classify only on batch_classify_unclassified() schedule. Default true.'
)
ON CONFLICT (config_type, key) DO NOTHING;

-- LLM fallback toggle for classifier (separate from rationale LLM)
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES (
    'advisor_config',
    'kap_classifier_llm_fallback',
    'true',
    'boolean',
    'Enable LLM fallback (Stage 2) when rule stage produces no match. '
    'Requires advisor_anthropic_api_key or advisor_openai_api_key. Default true.'
)
ON CONFLICT (config_type, key) DO NOTHING;

-- Batch backfill interval
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES (
    'advisor_config',
    'kap_classifier_batch_interval_minutes',
    '60',
    'integer',
    'How often (minutes) to run the batch classify-unclassified backfill loop. '
    'Default 60. Reduce if kap_listener ingests at high frequency.'
)
ON CONFLICT (config_type, key) DO NOTHING;


COMMIT;

-- DOWN (reversible rollback):
-- BEGIN;
-- DROP TABLE IF EXISTS kap_classifications;
-- DROP TABLE IF EXISTS kap_event_taxonomy;
-- DELETE FROM config_settings
--   WHERE config_type = 'advisor_config'
--     AND key IN (
--       'kap_classifier_enabled',
--       'kap_classifier_live_mode',
--       'kap_classifier_llm_fallback',
--       'kap_classifier_batch_interval_minutes'
--     );
-- COMMIT;
