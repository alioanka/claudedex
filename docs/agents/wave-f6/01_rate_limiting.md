# Wave-F6 / 01 — "Why are we still rate-limited?" (Helius 429 root-cause)

**Scope:** read-only forensic analysis of `logs/` (3-day window, 2026-07-07 → 2026-07-09,
NOT a month — logs rotated at 10 MB). Deploy boundary: Wave-F5 multi-key rotation shipped
~2026-07-06 22:30 (verified RPC import) / 2026-07-07 08:18 process start. mig 146 credential
cards deployed 2026-07-07 07:58.

## TL;DR (the one sentence)
Rotation is working perfectly and it doesn't matter, because **there is only ONE real Helius
account key in the runtime pool.** All 5 "HELIUS_API" endpoints resolve to the same account
(`a78df2b2…`), so round-robin spreads the load across five names that share one free-tier quota.
Five names, one bucket. The fix Wave-F5 shipped cannot help until real key #2..#5 are actually
populated — right now the numbered slots are empty placeholders that mig 146 draws cards for.

---

## (a) Per-module / per-provider 429 timeline (split at the 07-07 deploy)

All logs are POST-deploy (process started 2026-07-07 08:18, after the Wave-F5 import). There is
**no pre-deploy tail in these files** — the operator's "still rate-limited" is entirely the
post-deploy regime. That is itself the headline: the fix is deployed and 429s continue.

### Pool-side (authoritative — `logs/pool_engine/pool_engine_rate_limits.log*`)
`Rate limited: HELIUS_API` events:
| Day | events |
|---|---|
| 2026-07-08 | 42,712 |
| 2026-07-09 | 77,777 |
| Total (HELIUS only) | **90,365** |

Perfectly even split across the 5 endpoint NAMES (this is rotation doing its job):
```
18,132  HELIUS_API - SOL #3
18,078  HELIUS_API - Helius API #5
18,065  HELIUS_API - SOL #1
18,053  HELIUS_API - HELIUS_API_KEY
18,037  HELIUS_API - SOL #2
```
Non-Helius providers: **zero** SOLANA public-RPC / Ethereum / Base / Arbitrum rate-limits of note.
The other 90 pool endpoints are healthy (`SUCCESS: ETHEREUM_RPC 1111`, `SOLANA_RPC 739`, etc.).

### Module-side (genuine 429 log lines, timestamp-noise filtered)
| Module | 07-07 | 07-08 | 07-09 | Provider | Notes |
|---|---|---|---|---|---|
| copy_trading | 5,547 | 29,799 | 25,637 | Helius | biggest emitter; "All HELIUS endpoints rate-limited" storm in `stderr.log` |
| sniper | 620 | 734 | 1,502 | Helius (SolanaListener WSS + getTransaction) | `⚠️ Rate limited - backing off` every ~75 s on 07-09; **listener went OFFLINE** (screenshot) |
| advisor | 48 | 48 | 54 | non-Helius (Fonoloji/BIST) | steady, unrelated, not a concern |
| solana_trading | 0 explicit | | | Helius via pool | absorbs 429s silently through pool STARVED fallback |
| dex / futures / arbitrage / polymarket / ai | ~0 | | | — | not Helius-bound; clean |

> **Note on false positives:** grepping `429` naively matches millisecond timestamps
> (`08:18:53,429`) and the `count=…` counters inside Solana stats blobs. Every count above is
> filtered to real `Rate limited` / `HTTP 429` / `too many requests` phrases.

---

## (b) Root causes, ranked with evidence

### RC-1 (PRIMARY) — Only one real Helius key exists in the runtime pool
The five "HELIUS_API" endpoints are five NAMES over one ACCOUNT:
- Startup: `pool_engine.log:2` `Loaded 95 endpoints from database`; line 3
  `Registered 1 API-key endpoint(s) from secrets manager` — **exactly one** key came from
  encrypted secrets.
- The DB-seeded Solana RPC rows (`SOL #1/#2/#3`, `Helius API #5`) are Helius URLs that embed the
  **same** account key. Across **every log file in the repo**, the only Helius key fingerprint
  ever emitted is `a78df2b2` (660 occurrences; dashboard `_fallback_api_modules` line 2062 shows
  the Solana module's `rpc_url = …helius-rpc.com/?api-key=a78df2b2…`). Zero other fingerprints.
- The numbered slots `HELIUS_API_KEY_2 / _3 / _4` were **never loaded**: grep of all logs finds
  no `HELIUS_API_KEY_2..9` and no `Helius API #2..#4` endpoint names — only `#5` (a DB row label,
  not a 5th key). mig 146 renders the credential cards, but the underlying `secure_credentials`
  rows are empty/placeholder — the pool would have logged `Registered N` with N>1 otherwise.
- Consequence: `get_next_endpoint` round-robins across 5 names → Helius sees the SAME account key
  on all 5 → one 1 M-credit/mo + 10 rps bucket is hit 5× faster in appearance but is really just
  one saturated account. `pool_engine.log` tail: `All HELIUS_API endpoints rate-limited; using
  least-penalized 'SOL #3' (recovers in 45s)` — the STARVED path (`config/pool_engine.py`
  starved-fallback) fires because every "endpoint" is the same exhausted account.

### RC-2 — Three modules share the one account concurrently
copy_trading + sniper + solana_trading all pull HELIUS_API from the same pool. Even if pacing were
perfect per-module, their sum exceeds 10 rps on a single free account:
- sniper: `Monitor status` 6,000 → 2,136,000 scans over ~60 h ≈ **9.9 scans/s**; the WSS listener
  does a `getTransaction` per detected pool (177 k pools, 495 k `NoResult` rejections in
  `WSS Rejections` counter) — enhanced/parsed tx on Helius = higher credit weight.
- solana_trading: continuous price-monitor on ~7 open positions + Jupiter quotes (`daily_trades`
  ~233 sim) polling every ~40 s.
- copy_trading: 4 wallets polled every 15 s, each doing `getSignaturesForAddress` + per-sig
  `getTransaction` (`copy_engine.py:2469, 2640`).
The token bucket (`copy_engine.py:2901` `Helius outbound pacing set to 8 req/s`) caps copy alone at
8 rps — which **by itself** nearly consumes the entire 10 rps free ceiling, leaving nothing for
sniper/solana. Evidence: `copy_trading/stderr.log` `All HELIUS_API endpoints rate-limited` storm.

### RC-3 — copy_trading pacing is per-process, not per-account-quota-aware
`configure_rate_limiter('HELIUS_API', 8.0)` (`copy_engine.py:2903`) sets an 8 rps ceiling assuming
"< Helius free ~10". With one shared account and two other consumers, 8 rps is already over budget.
The per-wallet key rotation added in Wave-F5 (`copy_engine.py:2917-2925`) is correct code aimed at
"spread the 33-wallet burst across 4 keys" — but with 4 wallets and 1 key it's a no-op.

### RC-4 (minor) — sniper listener has no independent Helius account
`SolanaListener` shares the pool; when the account is saturated it logs
`⚠️ Rate limited - backing off` and eventually the listener zombies (screenshot 07-09 23:57 shows
**Sniper OFFLINE**, Copy Trading OFFLINE). This is the Wave-F5 "silent zombie for 20 days" failure
mode recurring under quota exhaustion rather than auth failure.

---

## (c) Concrete fix list

### Operator actions (these are the actual fix — do these first)
1. **Create 3–4 more Helius free accounts** and store the keys in encrypted `secure_credentials`
   as `HELIUS_API_KEY_2`, `HELIUS_API_KEY_3`, `HELIUS_API_KEY_4` (the slots mig 146 already shows
   cards for). Verify with `python scripts/verify_and_import_rpcs.py` then confirm the next boot
   logs `Registered 4 API-key endpoint(s) from secrets manager` (currently `1`). **Until this line
   says >1, nothing else matters.**
2. Confirm the 3 DB-seeded `SOL #1/#2/#3` Helius rows aren't all the same key — either point each
   at a distinct account or demote them so they don't masquerade as separate quotas.
3. If sniper is to run continuously, give it a **dedicated** paid Helius/Triton endpoint — WSS
   new-pool detection at ~10 scans/s cannot live on a shared free account (see capacity verdict).

### Config / cadence changes (reduce demand to fit real capacity) — key → current → proposed
DB `config_settings.copytrading_config` (via ConfigManager):
| key | current | proposed | why |
|---|---|---|---|
| `copy_helius_rps` | 8.0 (`copy_engine.py:1237`) | **2.0** | leave headroom for sniper+solana on a shared account |
| `copy_poll_interval_s` | 15.0 (`copy_engine.py:1273`) | **30.0** | 4 wallets don't need 15 s cadence in DRY |
| `copy_max_concurrent_wallets` | 5 (`copy_engine.py:1227`) | 2 | 4 wallets, no need for 5 |

Solana module (position monitor / price poll cadence — raise the price-refresh interval from ~40 s
toward 90–120 s while DRY). Sniper: if kept on free tier, cap `getTransaction` enrichment to
promoted candidates only (it already filters — verify the `NoResult` 495 k count isn't spending
credits on dead pools).

### Code changes (defensive, secondary to the operator key work)
- `config/pool_engine.py` `_load_keys_from_secrets` (line 764): log the **distinct account key
  count** per provider, not just endpoint count, so "5 endpoints / 1 account" is visible at
  startup instead of hidden. Dedup HELIUS_API endpoints by embedded api-key so one account can't
  register as 5 rotating "endpoints" (the current `_has_api_key` check at line 660 dedups the
  api_mappings path but NOT the DB-seeded `SOL #x` URL rows whose key is in the URL).
- `modules/copy_trading/copy_engine.py:2903`: make the token-bucket rate a function of the number
  of DISTINCT Helius accounts in the pool (`rps = base_per_key * n_accounts`) rather than a flat 8.
- Sniper listener: on sustained `Rate limited - backing off`, surface a loud health-degraded state
  (it currently zombies to OFFLINE silently — same class as the Wave-F5 auth-zombie).

---

## (d) Honest capacity verdict — can free tiers carry this workload?

**No — not at the current cadence, and not for sniper at all.**

- Helius free tier ≈ 1 M credits/month + 10 rps. Rough demand on ONE account across the three
  Solana consumers: copy (capped 8 rps) + solana price-monitor (~0.5 rps) + sniper listener
  (~10 scans/s, `getTransaction` per pool). Sniper alone at ~10 rps of parsed-tx lookups
  (higher credit weight) would burn ~1 M credits in **days**, not a month. The pool logged
  **90,365 Helius rate-limit events in ~2 days** — this is a workload running continuously into a
  brick wall, not an occasional spike.
- With cadence cuts (copy 2 rps / 30 s, solana 90–120 s price poll) and **4 real Helius keys**
  (≈40 rps aggregate, 4 M credits/mo), copy_trading + solana_trading DRY monitoring becomes
  comfortably carryable on free tiers. That combination is the realistic target.
- **Sniper is the exception:** continuous WSS new-pool detection + per-pool tx enrichment at
  ~10 scans/s is a paid-tier workload regardless of key count. Either (a) give sniper one dedicated
  paid Helius/Triton/Chainstack endpoint, or (b) accept it will oscillate between backing-off and
  OFFLINE on free tier (as the 07-09 screenshot shows).

**Bottom line for the operator:** rotation isn't broken — the pool has one Helius account wearing
five name tags. Add real keys #2–#5 into `secure_credentials` (verify boot logs `Registered >1`),
cut copy/solana cadence, and put sniper on a dedicated paid endpoint. Free tiers can carry
copy+solana DRY monitoring after that; they cannot carry the sniper listener at any key count.
