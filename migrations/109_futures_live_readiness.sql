-- Migration 109: futures LIVE-readiness — close-path safety knob
--
-- Companion to the futures_engine close-path honesty fix: when live orders
-- are blocked (killswitch / pause / DRY_RUN flipped back on) the engine now
-- REFUSES to paper-close a LIVE position (previously it recorded a simulated
-- close and dropped the position from monitoring, orphaning real exchange
-- exposure with no exchange-side stop). This knob opts in to sending REAL
-- reduce-only closes in that blocked state (risk-reducing orders only —
-- standard desk policy: a kill switch flattens, it does not freeze).
--
-- Seeded 'false' to match the code default exactly: no orders of any kind
-- while blocked. Applying this migration changes NO behavior on its own.
--
-- Idempotent: ON CONFLICT (config_type, key) DO NOTHING.
-- NOTE: all SQL string literals are single-quoted; '' escapes an apostrophe.

INSERT INTO config_settings (config_type, key, value, value_type, description)
VALUES
(
    'futures_risk',
    'reduce_only_close_when_paused',
    'false',
    'bool',
    'LIVE-flip safety: when live orders are blocked (killswitch/pause/DRY_RUN flip) the engine refuses to paper-close LIVE positions and keeps monitoring them. Set true to instead allow REAL reduce-only closes in that state (risk-reducing orders only). Default false = no orders while blocked; the operator''s manual exchange close remains the fallback.'
)
ON CONFLICT (config_type, key) DO NOTHING;
