-- Migration 064: KAP Phase-1 wiring config seeds — Wave-23
--
-- Seeds config_settings (config_type='advisor_config') for the KAP
-- classification worker, Telegram polarity-prior alerts, and the BIST
-- advice-engine KAP context overlay wired in this phase:
--
--   advisor_kap_classify_interval_s   default '60'   (classify loop cadence)
--   advisor_kap_alert_min_confidence  default '0.5'  (alert confidence floor)
--   advisor_kap_alert_max_per_cycle   default '10'   (per-cycle alert cap)
--   advisor_kap_lookback_days         default '7'    (advice overlay lookback)
--
-- All KAP behavior remains gated by advisor_kap_enabled (default false,
-- seeded in migration 062). These keys are opt-in tunables only.
--
-- HONESTY NOTE: KAP emits a documented base_polarity PRIOR only. There is NO
-- market-impact score. Forward-return impact statistics accumulate over months
-- in kap_returns and are not produced by this wiring.
--
-- Idempotent: INSERT ... ON CONFLICT (config_type, key) DO NOTHING.
-- Reversible: DOWN block at bottom.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('advisor_config', 'advisor_kap_classify_interval_s', '60', 'int',
     'Seconds between KAP classification worker cycles. Each cycle pulls '
     'unclassified disclosures, runs the two-stage rule+LLM classifier, and '
     'persists results. Default 60. Requires advisor_kap_enabled=true.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_kap_alert_min_confidence', '0.5', 'float',
     'Minimum classifier confidence to fire a Telegram polarity-prior alert '
     'for a non-NEUTRAL disclosure. rule stage = 0.90, llm stage = 0.65. '
     'Default 0.5. This is classifier certainty, NOT a market-impact score.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_kap_alert_max_per_cycle', '10', 'int',
     'Maximum KAP polarity-prior Telegram alerts sent per classification '
     'cycle. Prevents a backfill burst from spamming the chat; excess alerts '
     'are logged, not sent. Default 10.',
     NOW(), NOW()),

    ('advisor_config', 'advisor_kap_lookback_days', '7', 'int',
     'Lookback window (days) for the BIST advice-engine KAP context overlay. '
     'Recent classified disclosures within this window are surfaced as '
     'operator CONTEXT on BIST advice; they do NOT modify the advice action '
     'or numeric confidence. Default 7.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;


-- =========================================================================
-- DOWN (reversible) -- run manually to undo
-- =========================================================================
-- BEGIN;
-- DELETE FROM config_settings
--   WHERE config_type='advisor_config'
--     AND key IN (
--       'advisor_kap_classify_interval_s',
--       'advisor_kap_alert_min_confidence',
--       'advisor_kap_alert_max_per_cycle',
--       'advisor_kap_lookback_days'
--     );
-- COMMIT;
