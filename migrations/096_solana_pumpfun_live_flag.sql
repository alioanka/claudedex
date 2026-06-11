-- Migration 096: pump.fun LIVE-broadcast kill knob (Solana module)
-- Context: operator-reported pump.fun LIVE-flag bug — the JupiterHelper
-- killswitch/pause gate returned a simulated sentinel that the engine
-- recorded as a LIVE fill. The engine fix gates every LIVE broadcast on
-- core.dry_run.should_skip_live; this knob adds a per-strategy DB switch
-- so the operator can disable pump.fun LIVE entries alone (DRY_RUN
-- pump.fun entries are unaffected) without pausing the whole module.
--
-- Seeded 'true' to preserve current behavior; engine read is fail-soft
-- (missing row / DB outage -> default True).

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
  ('solana_pumpfun', 'pumpfun_live_enabled', 'true', 'bool',
   'Per-strategy LIVE kill knob: when false, pump.fun entries never broadcast on-chain (DRY_RUN entries unaffected). Engine refuses the entry outright; nothing is recorded. Fail-soft default true.')
ON CONFLICT (config_type, key) DO NOTHING;
