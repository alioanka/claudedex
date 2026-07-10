-- Migration 149: COPY_TRADING Wave-F6 — discovery/shadow unconditional enable
-- + Helius cadence cuts.
--
-- Root causes addressed (docs/agents/wave-f6/01_rate_limiting.md and
-- 04_advisory_sweep.md):
--   (a) The Wave-F5 discovery revival NEVER RAN. Mig 141 flipped
--       copy_auto_discovery_enabled / copy_shadow_sim_enabled 'false'->'true'
--       only WHERE value = 'false'; the live rows did not match the expected
--       default, so the conditional UPDATE was a no-op and
--       _maybe_run_discovery_v3 / _maybe_tick_shadow_sim returned immediately
--       for three weeks (zero [discovery-v3] / CopyShadowSim log lines;
--       operator symptom: "discovery finds only my wallet", 0 copies).
--   (b) Helius saturation: 90,365 pool-side rate-limit events in ~2 days.
--       copy_helius_rps=8 nearly consumes the ENTIRE ~10 rps Helius free-tier
--       ceiling by itself (the "5 keys" in the pool are five NAMES over ONE
--       account — see the pool_engine dedup shipped with this wave), leaving
--       nothing for sniper/solana. copy_poll_interval_s=15 doubles the daily
--       call count for no DRY-mode benefit with a 4-wallet watchlist.
--
-- SAFETY / NON-TRADING STATEMENT (why the enable is intentionally
-- UNCONDITIONAL this time):
--   * copy_auto_discovery_enabled gates a READ-ONLY sweep that writes wallet
--     candidates to copy_discovered_wallets / copy_leader_candidates for
--     OPERATOR APPROVAL. It cannot place an order.
--   * copy_shadow_sim_enabled gates the PAPER shadow simulator; every row it
--     writes is is_simulated=true. It cannot place an order.
--   * Neither key is a live-execution flag. A discovered wallet is still only
--     traded after operator approval (copy_leader_candidates flow) or the
--     pre-existing double-gated auto-promote (default OFF, max_leaders=0),
--     and every live gate (should_skip_live, RiskManager, DRY_RUN) is
--     unchanged. Because the conditional flip already failed once and left
--     the headline F5 deliverable dead for weeks, this migration sets both
--     flags to 'true' UNCONDITIONALLY (operator can still turn them off
--     afterwards; the engine re-reads the flag every cycle).
--
-- Cadence seeds are DRY_RUN-safe load reducers only: conditional UPDATE from
-- the old default (operator overrides preserved) + INSERT ON CONFLICT DO
-- NOTHING for fresh installs.
--
-- Idempotent; single-quoted literals only.

BEGIN;

-- ---------------------------------------------------------------------
-- 1. UNCONDITIONAL discovery + shadow-sim enable (non-trading; see header).
-- ---------------------------------------------------------------------
INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('copytrading_config', 'copy_auto_discovery_enabled', 'true', 'boolean',
     'COPY v3 wallet-discovery sweep gate (read-only: writes candidates for '
     'operator approval, never trades). Wave-F6: set true UNCONDITIONALLY - '
     'the mig 141 conditional flip never matched the stored value and '
     'discovery never ran.'),
    ('copytrading_config', 'copy_shadow_sim_enabled', 'true', 'boolean',
     'COPY v3 paper shadow-copy simulator gate (writes is_simulated=true '
     'rows only, never trades). Wave-F6: set true UNCONDITIONALLY - same '
     'mig 141 conditional-flip failure as discovery.')
ON CONFLICT (config_type, key) DO UPDATE
    SET value = 'true', updated_at = NOW();

-- ---------------------------------------------------------------------
-- 2. Helius cadence cuts (~3-4x demand reduction; DRY-safe knobs).
--    Conditional UPDATE from the OLD default so operator overrides survive;
--    INSERT ON CONFLICT DO NOTHING covers fresh installs.
-- ---------------------------------------------------------------------

-- copy_helius_rps 8 -> 2: with ONE real Helius account shared by copy +
-- sniper + solana, an 8 rps copy ceiling alone nearly saturates the ~10 rps
-- free tier. 2 rps leaves headroom for the sibling modules.
UPDATE config_settings SET value = '2', updated_at = NOW()
 WHERE config_type = 'copytrading_config'
   AND key = 'copy_helius_rps'
   AND value IN ('8', '8.0');

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('copytrading_config', 'copy_helius_rps', '2', 'float',
     'Outbound requests/second ceiling for the copy-trading Solana/Helius '
     'wallet fan-out (shared pool_engine token bucket). Wave-F6 default 2 '
     '(was 8): the Helius account is shared with sniper+solana and 8 rps '
     'alone nearly consumed the ~10 rps free ceiling. Clamped 1..50 in '
     'copy_engine. Raise only with DISTINCT extra Helius accounts or a paid '
     'plan.')
ON CONFLICT (config_type, key) DO NOTHING;

-- copy_poll_interval_s 15 -> 30: halves per-day Helius calls; a 4-wallet
-- DRY watchlist does not need 15 s cadence. The signal-staleness gate
-- self-adjusts (_effective_signal_age_s = poll + buffer), so no signal is
-- spuriously rejected by the slower cycle.
UPDATE config_settings SET value = '30', updated_at = NOW()
 WHERE config_type = 'copytrading_config'
   AND key = 'copy_poll_interval_s'
   AND value IN ('15', '15.0');

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
    ('copytrading_config', 'copy_poll_interval_s', '30', 'float',
     'Full monitor-cycle cadence in seconds for the copy-trading Solana/EVM '
     'poll. Wave-F6 default 30 (was 15): halves Helius daily consumption; '
     'the staleness gate self-adjusts to the poll cadence. Clamped 5..300 '
     'in copy_engine. Lower toward 15 only with real spare Helius quota.')
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;
