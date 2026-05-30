-- Migration: ai_runtime_stats — single-row heartbeat snapshot written by the
-- AI subprocess (modules/ai_analysis/core/sentiment_engine.py::run) every
-- cycle (~15 min) so the standalone dashboard can tell a LIVE-but-idle AI
-- engine apart from a dead one.
--
-- Before this table the dashboard's /api/modules AI branch inferred health
-- purely from `sentiment_logs` freshness, which only got a row when news was
-- BOTH retrieved AND analyzed. The "No news data retrieved, skipping
-- analysis" branch (frequent in low-news windows) wrote nothing, so the AI
-- card showed "ENABLED (no health)" for hours even though the cycle loop
-- was firing on schedule. Mirrors sniper_runtime_stats (migration 015) and
-- dex_runtime_stats (migration 032) single-row UPSERT pattern; the only
-- REQUIRED field for liveness is a fresh updated_at. `stats` carries
-- optional context (provider counts, last skip reason, delegates_to=futures
-- per Wave-7 AI->Futures routing, cycle/positions counters) for diagnostics.
--
-- Single row (id=1). UPSERT keyed by id. Fail-soft on the writer side — a
-- DB hiccup must never crash the AI loop.
--
-- Idempotent: CREATE TABLE IF NOT EXISTS + INSERT ... ON CONFLICT DO
-- NOTHING. Re-applying is safe.
--
-- Date: 2026-05-28

CREATE TABLE IF NOT EXISTS ai_runtime_stats (
    id          INT         PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stats       JSONB       NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO ai_runtime_stats (id, stats)
VALUES (1, '{}'::jsonb)
ON CONFLICT (id) DO NOTHING;

-- down:
-- DROP TABLE IF EXISTS ai_runtime_stats;
