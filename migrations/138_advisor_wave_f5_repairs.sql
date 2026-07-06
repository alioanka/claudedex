-- Migration 138: Wave-F5 advisor repairs (docs/agents/wave-f5/06_advisor_bist.md)
--
-- 1. sim_horizon_days_long 365 -> 90: LONG-horizon sims squatted a channel
--    slot for a YEAR, saturating the per-channel caps (75/75 observed) and —
--    before the Wave-F5 code fix — silencing all advice. CONDITIONAL update:
--    only rows still at the mig-060 default '365' are touched; any operator
--    override is preserved.
--
-- Single-quoted SQL literals only ('' escapes apostrophes). Idempotent.

UPDATE config_settings
SET value = '90', updated_at = NOW()
WHERE config_type = 'advisor_config'
  AND key = 'sim_horizon_days_long'
  AND value = '365';

-- =========================================================================
-- 2. One-off NaN cleanup (Wave-F5 fix 3). Legacy rows persisted before the
--    non-finite guards (levels.horizon_levels / portfolio_engine) hold
--    'NaN'::numeric values; json.dumps then emitted bare NaN tokens and the
--    advice-history + simulations pages died on res.json(). The endpoints are
--    now NaN-safe; this cleans the stored rows. Idempotent (Postgres treats
--    NaN = NaN as TRUE for numeric comparisons).
-- =========================================================================

UPDATE advisor_advice SET entry_low    = NULL WHERE entry_low    = 'NaN'::numeric;
UPDATE advisor_advice SET entry_high   = NULL WHERE entry_high   = 'NaN'::numeric;
UPDATE advisor_advice SET target_price = NULL WHERE target_price = 'NaN'::numeric;
UPDATE advisor_advice SET stop_price   = NULL WHERE stop_price   = 'NaN'::numeric;

UPDATE advisor_sim_positions SET current_price = NULL WHERE current_price = 'NaN'::numeric;
UPDATE advisor_sim_positions SET target_price  = NULL WHERE target_price  = 'NaN'::numeric;
UPDATE advisor_sim_positions SET stop_price    = NULL WHERE stop_price    = 'NaN'::numeric;
UPDATE advisor_sim_positions SET exit_price    = NULL WHERE exit_price    = 'NaN'::numeric;
UPDATE advisor_sim_positions SET pnl_pct       = NULL WHERE pnl_pct       = 'NaN'::numeric;
UPDATE advisor_sim_positions SET pnl_usd       = NULL WHERE pnl_usd       = 'NaN'::numeric;

-- entry_price is NOT NULL, so it cannot be nulled: a sim with a NaN entry can
-- never produce a valid PnL — close it out explicitly (bookkeeping only; the
-- advisor is ADVICE-ONLY, no orders exist). Idempotent via the status guard.
UPDATE advisor_sim_positions
SET status = 'closed', close_reason = 'nan_entry_cleanup',
    pnl_pct = NULL, pnl_usd = NULL,
    closed_at = NOW(), updated_at = NOW()
WHERE status = 'open'
  AND entry_price = 'NaN'::numeric;

-- =========================================================================
-- 3. BIST universe default -> curated bist50 (Wave-F5 fix 6). The mig-083
--    'auto' default resolves to the Fonoloji live /stocks/list, which is
--    ~88% garbage (Midas-US symbols + funds/options/ISINs): 53 of the 60
--    scan slots failed every source, every cycle (~2,200 error lines/day),
--    while real BIST names beyond the watchlist were never attempted. The
--    curated BIST-50 snapshot in universes.py is the honest default.
--    CONDITIONAL: only the mig-078/083 seeded defaults ('watchlist'->'auto'
--    lineage and 'fonoloji') are updated; an explicit operator choice of
--    'watchlist' / 'bist30' / 'custom' is preserved. The live list remains
--    available by setting advisor_bist_universe='fonoloji' — the code-side
--    Midas-US row filter now cleans it.
-- =========================================================================

UPDATE config_settings
SET value = 'bist50',
    description = 'BIST scan universe: bist50 (DEFAULT — curated BIST-50 '
        'snapshot in universes.py) | bist30 | watchlist (watchlist-only) | '
        'fonoloji (live /stocks/list, Midas-US rows filtered) | auto (live '
        'list when a Fonoloji key is present) | custom. Watchlist tickers '
        'are always included; advisor_universe_max caps the total.',
    updated_at = NOW()
WHERE config_type = 'advisor_config'
  AND key = 'advisor_bist_universe'
  AND value IN ('auto', 'fonoloji', '');

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('advisor_config', 'advisor_bist_universe', 'bist50', 'string',
     'BIST scan universe: bist50 (DEFAULT — curated BIST-50 snapshot in '
     'universes.py) | bist30 | watchlist (watchlist-only) | fonoloji (live '
     '/stocks/list, Midas-US rows filtered) | auto (live list when a '
     'Fonoloji key is present) | custom. Watchlist tickers are always '
     'included; advisor_universe_max caps the total.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- =========================================================================
-- 4. KAP bounded re-queue knob (Wave-F5 fix 7). Budget-starved disclosures
--    are stamped classifier_stage='unclassified' and were NEVER retried
--    (the dedupe join treated any stamp as done) — weeks of ingestion left
--    thousands of permanent UNCLASSIFIED rows. kap_store.get_unclassified
--    now re-surfaces stamps older than this many hours (never-classified
--    rows always sort first; every attempt re-stamps classified_at, so a
--    row retries at most once per window). 0 disables the re-queue.
-- =========================================================================

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('advisor_config', 'advisor_kap_reclassify_after_hours', '24', 'float',
     'Hours after which a KAP disclosure stamped classifier_stage='
     '''unclassified'' is re-queued for classification (budget-starved rows '
     'get retried once LLM budget exists). New disclosures always take '
     'priority; each attempt re-stamps classified_at so a row retries at '
     'most once per window. 0 disables the re-queue. Default 24.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;
