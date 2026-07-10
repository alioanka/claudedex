-- Migration 152: Wave-F7 — sentinel anomaly lifecycle (resolve / acknowledge)
--
-- Closes the external audit's sentinel finding: "Current cycles see no
-- anomalies, despite a retained critical record of sniper 100% rejection.
-- Alert state needs lifecycle/acknowledgement." sentinel_anomalies rows had
-- no terminal state — a critical that stopped firing weeks ago was
-- indistinguishable from an active condition.
--
-- What this adds (all additive, zero behavior change until the engine runs):
--   * resolved_at      — stamped by the engine's auto-resolve pass when an
--                        OPEN row has not re-fired for
--                        anomaly_auto_resolve_minutes (seeded below, 240).
--                        Rows are KEPT for audit, never deleted.
--   * acknowledged_at / acknowledged_by — operator-ack columns. Schema-ready;
--                        the dashboard ack route is a documented follow-up
--                        (RBAC operator-gated when built). NULL until used.
--   * partial index on open rows so the per-tick auto-resolve UPDATE and any
--                        "active anomalies" dashboard query stay cheap.
--
-- ADVISORY-safety unchanged: no live/paid flag, no autopilot flip, no seed
-- touches anything but the new lifecycle knob. Idempotent: ADD COLUMN IF NOT
-- EXISTS / CREATE INDEX IF NOT EXISTS / INSERT ON CONFLICT DO NOTHING.

BEGIN;

ALTER TABLE sentinel_anomalies
    ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ;

ALTER TABLE sentinel_anomalies
    ADD COLUMN IF NOT EXISTS acknowledged_at TIMESTAMPTZ;

ALTER TABLE sentinel_anomalies
    ADD COLUMN IF NOT EXISTS acknowledged_by TEXT;

CREATE INDEX IF NOT EXISTS idx_sentinel_anomalies_open
    ON sentinel_anomalies (last_seen_at DESC)
    WHERE resolved_at IS NULL;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('sentinel', 'anomaly_auto_resolve_minutes', '240', 'int',
     'Wave-F7 anomaly lifecycle: an OPEN anomaly row (resolved_at IS NULL) '
     'that has not re-fired for this many minutes is auto-stamped '
     'resolved_at by the sentinel tick, so stale records are never mistaken '
     'for active conditions. Rows are kept for audit. 0 disables '
     'auto-resolve.', NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;

-- down:
-- BEGIN;
-- DELETE FROM config_settings WHERE config_type='sentinel' AND key='anomaly_auto_resolve_minutes';
-- DROP INDEX IF EXISTS idx_sentinel_anomalies_open;
-- ALTER TABLE sentinel_anomalies DROP COLUMN IF EXISTS acknowledged_by;
-- ALTER TABLE sentinel_anomalies DROP COLUMN IF EXISTS acknowledged_at;
-- ALTER TABLE sentinel_anomalies DROP COLUMN IF EXISTS resolved_at;
-- COMMIT;
