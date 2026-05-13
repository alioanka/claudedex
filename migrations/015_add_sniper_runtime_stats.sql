-- Sniper runtime stats snapshot
-- Single row (id=1) updated periodically by the sniper subprocess so
-- the standalone dashboard can read per-loop counters that otherwise
-- live only in the sniper engine's in-process _stats dicts.
CREATE TABLE IF NOT EXISTS sniper_runtime_stats (
    id INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    stats JSONB NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO sniper_runtime_stats (id, stats)
VALUES (1, '{}'::jsonb)
ON CONFLICT (id) DO NOTHING;
