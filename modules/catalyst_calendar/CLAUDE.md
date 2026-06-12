# CATALYST_CALENDAR Module

## What it does
**Advisory forward-catalyst feed.** Aggregates known, scheduled forward
catalysts — token unlock/vesting cliffs, exchange listing/delisting
announcements, and macro event dates — from FREE sources into the `catalysts`
table. Other modules and the operator read it as a *risk input* (e.g. futures
declines new carry entries into a large unlock; the dashboard shows an
upcoming-catalyst panel). **PURE ADVISORY** — it never places a trade, never
writes `logs/.pause_*` or `logs/.killswitch`, requires no keys, and spends no
LLM/API budget.

Internal sales pitch is honest: unlock effects are heavily studied and mostly
priced in. The surviving value is **risk avoidance** (don't be long into a 5%
supply cliff), not alpha. Treat rows as constraints, not signals.

## Entry point
`modules/catalyst_calendar/main_catalyst_calendar.py` — launched by `main.py`
when `CATALYST_CALENDAR_MODULE_ENABLED=true` (default **false**). Health server
on port 8096 (`CATALYST_CALENDAR_HEALTH_PORT`); `/health` liveness, `/status`
last-tick summary with per-source fetch/error state.

Parsing/normalization is PURE (no network/DB) in `core/normalizer.py`,
self-tested offline against a static fixture:
`python -m modules.catalyst_calendar.core.normalizer`.
HTTP clients: `core/sources.py` (shared client-side rate cap, fail-soft).
Loop/persistence: `core/calendar_engine.py`.

## Source matrix — coverage honesty (read this before trusting the feed)
| Source | Events | Free? | Key | Confidence | Fragility |
|---|---|---|---|---|---|
| DefiLlama emissions (`api.llama.fi/emissions`) | `token_unlock` | YES | none | medium | UNOFFICIAL, undocumented schema; aggregated vesting data is best-effort and incomplete (long-tail tokens missing). A schema change silently degrades to zero events. |
| Binance CMS announcements (catalogId 48) | `exchange_listing` / `exchange_delisting` | YES | none | low | Scrape-class title-regex over an undocumented endpoint; can change or geo-block any time. Only BINANCE listings; the announcement timestamp is used as event_time (go-live time in titles is not reliably parseable). |
| Static macro schedule (in code) | `macro_fomc` | YES | none | high | Official FOMC 2026 decision dates only (19:00 UTC statements). **EXPIRES 2026-12-31** — past expiry it emits NOTHING and logs a warning; refresh `STATIC_MACRO_2026` or set `macro_static_override`. No CPI dates shipped (exact BLS dates were not verifiable at build time — honesty over fake coverage). |
| FMP economic calendar | `macro_cpi`/`macro_fomc`/`macro_other` (High-impact US only) | free tier | `FMP_API_KEY` (secrets_manager → env) | high | **Default OFF.** Without a key the source returns zero events; the loop never blocks on it. |

**Known coverage gaps:** no CoinGecko/CryptoRank/TokenUnlocks unlock data (paid
or keyed), no non-Binance exchange announcements, no CPI/NFP dates without an
FMP key. Consumers MUST treat absence of a row as "no known constraint", never
as "no event exists" (fail-open feed; their own risk gates still apply), and
MUST check `last_seen_at` freshness before trusting a row — a stale calendar is
worse than none.

## Key config (DB-backed, config_type='catalyst_calendar'; migration 124)
| Key | Default | What it does |
|---|---|---|
| `refresh_interval_seconds` | 3600 | Refresh cadence (slow by design) |
| `lookahead_days` | 30 | Forward window written to the table |
| `listing_recent_window_hours` | 72 | Keep exchange announcements this recent |
| `purge_after_days` | 90 | Delete events older than this |
| `max_requests_per_minute` | 10 | TOTAL outbound HTTP cap across all sources |
| `source_defillama_unlocks_enabled` | true | Unlock feed on/off |
| `source_binance_listings_enabled` | true | Listings feed on/off |
| `source_macro_static_enabled` | true | Built-in macro schedule on/off |
| `source_fmp_macro_enabled` | false | Key-gated FMP macro (off; degrades to empty) |
| `macro_static_override` | (empty) | JSON list replacing the built-in macro schedule |

## Kill switch
- Global: `logs/.killswitch` — tick skipped (poller started at boot so the
  dashboard emergency stop also terminates this subprocess).
- Per-module: `logs/.pause_catalyst_calendar` — tick skipped.

## Logs
`logs/catalyst_calendar/` — `catalyst_calendar.log`, `catalyst_calendar_errors.log`.

## DB tables (migration 124)
- `catalysts` — forward calendar; UNIQUE `(source, symbol, event_type,
  event_time)`; re-seen events bump `last_seen_at` (per-row freshness);
  `symbol='*'` = market-wide (macro); `magnitude` = unlock fraction of max
  supply (0..1) or NULL; `confidence` in ('high','medium','low').

Consumer query shape (e.g. futures avoiding entries into a large unlock):
```sql
SELECT * FROM catalysts
WHERE (symbol = $1 OR symbol = '*')
  AND event_time BETWEEN NOW() AND NOW() + INTERVAL '48 hours'
  AND last_seen_at > NOW() - INTERVAL '24 hours'   -- freshness gate
ORDER BY event_time;
```

## Isolation / safety
No order paths, no keys required, no flags written. Reuses read-only:
`core/dry_run.py` (killswitch poller only), `security/secrets_manager`
(optional FMP key, fail-soft). Writes ONLY the `catalysts` table. Never imports
a trading executor, never signs anything. Failure mode of the whole module is
"missing data", never lost money.
