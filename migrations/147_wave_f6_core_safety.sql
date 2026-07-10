-- Migration 147: Wave-F6 core safety seeds
--
-- 1) trading.block_unverified_contracts (default TRUE = fail-closed).
--    Adjudication finding #15: core/engine.py _final_safety_checks step 6
--    let an unverified/unknown contract proceed with only a WARNING. The
--    claimed "binding gate elsewhere" (RiskScore.contract_risk in the
--    scorer) is only a weighted score component, not a hard gate, so an
--    unverifiable contract could enter on volume/liquidity alone. The
--    engine now hard-rejects entries whose contract verification signal is
--    absent/negative unless this knob is 'false' (warn-only escape hatch).
--    NOTE (honesty): the engine-level verifier is an always-unverified
--    stub and RiskScore.verified_contract is only populated when the chain
--    collector's contract analysis succeeds — with today's collectors this
--    gate blocks ALL new DEX entries. That is the intended fail-closed
--    posture per the F5 "unverifiable safety signal must not pass"
--    principle; flip to 'false' to restore warn-only DRY_RUN data
--    collection until a real source-verification check is wired.
--
-- 2) DEX-idle diagnosis (Wave-F6 03_trading_sweep HIGH, NO seed changed):
--    the "0 entries since 07-09" state is NOT a chain_weights misconfig —
--    mig 095 already weights ethereum/base/bsc/pulsechain at 1.0 and only
--    zeroes solana/monad on measured negative edge (deliberate). The root
--    cause is collector-side: data/collectors/dexscreener.py discovery
--    strategies 1-2 scan GLOBAL boost/profile lists (Solana-dominated),
--    and strategies 3-4 mostly return established pools that fail the
--    24h max-age filter, so EVM chains genuinely discover zero new pairs.
--    Restoring a non-zero Solana weight would loosen a measured-negative-
--    edge gate and is NOT done here. Operator action: fix EVM pair
--    discovery in the collector (chain-scoped source), then EVM entries
--    resume through the existing weights.
--
-- DRY_RUN-safe: no live flag is touched. Idempotent: ON CONFLICT DO NOTHING
-- (conditional seed — an operator who already set the key keeps their value).

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('trading', 'block_unverified_contracts', 'true', 'bool',
     'Fail-closed contract-verification gate in DEX final safety checks '
     '(Wave-F6, adjudication #15). true = REJECT new entries whose contract '
     'verification signal is absent or negative (an unverifiable safety '
     'signal must not pass). false = legacy warn-only behavior. Entry-only; '
     'exits are never gated by this knob.',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;
