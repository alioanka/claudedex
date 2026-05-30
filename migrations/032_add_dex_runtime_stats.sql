-- Migration: dex_runtime_stats — single-row heartbeat snapshot written by the
-- DEX subprocess (modules/dex_trading/main_dex.py::_status_reporter) every ~60s
-- so the standalone dashboard can tell a LIVE-but-idle DEX engine apart from a
-- dead one.
--
-- Before this table the dashboard's /api/modules DEX branch inferred health
-- purely from RECENT `trades` rows, so a running engine that simply had no
-- trade in the last 2h reported "ENABLED (no health)". This mirrors the
-- sniper_runtime_stats (migration 015) single-row pattern; the only REQUIRED
-- field for liveness is a fresh updated_at. `stats` carries optional context
-- (wallet_address + a few engine counters) for diagnostics.
--
-- Single row (id=1). UPSERT keyed by id. Fail-soft on the writer side — a DB
-- hiccup must never crash the DEX loop.
--
-- Date: 2026-05-26

CREATE TABLE IF NOT EXISTS dex_runtime_stats (
    id          INT         PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stats       JSONB       NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO dex_runtime_stats (id, stats)
VALUES (1, '{}'::jsonb)
ON CONFLICT (id) DO NOTHING;

-- down:
-- DROP TABLE IF EXISTS dex_runtime_stats;
