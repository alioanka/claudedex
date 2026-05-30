# Wave-13 Agent 9 — Infra: Pool Engine + Secrets Manager

**Date:** 2026-05-30  
**Scope:** `config/pool_engine.py`, `security/secrets_manager.py`, `migrations/`  
**Branch:** `claude/friendly-ramanujan-nMWNv`

---

## DB-QUERY BLOCK

The PostgreSQL instance was not reachable from this worktree environment
(`docker.sock` absent; direct TCP to `localhost:5432` also refused).  All
analysis is therefore code-only (static).  The migration 015 is written with
`ON CONFLICT (provider_type, url) DO NOTHING` guards so it is idempotent
against whatever live state exists.  Before applying migration 015 in
production, run the following read-only discovery queries to confirm:

```sql
-- 1. Schema of rpc_api_pool
\d rpc_api_pool;

-- 2. Live endpoints per provider_type (masked URL suffix)
SELECT
    provider_type,
    name,
    LEFT(url, 40) || '...' AS url_prefix,
    priority,
    status,
    is_enabled,
    rate_limit_count,
    health_score
FROM rpc_api_pool
ORDER BY provider_type, priority;

-- 3. Provider types catalogue
SELECT provider_type, endpoint_type, chain, is_required
FROM rpc_api_provider_types ORDER BY provider_type;

-- 4. secure_credentials count (DO NOT SELECT VALUES)
SELECT category, COUNT(*) AS total,
       COUNT(*) FILTER (WHERE encrypted_value != 'PLACEHOLDER') AS configured
FROM secure_credentials
WHERE is_active = TRUE
GROUP BY category ORDER BY category;
```

---

## Architectural Gaps Found

### 1. SOLANA_RPC Starvation (P0 — operational)

**Root cause:** `ProviderEndpoints.get_next_endpoint()` calls
`get_available_endpoints()` which filters by `Endpoint.is_available`. When all
Helius endpoints are in RATE_LIMITED state, the list is empty and `None` is
returned. The caller falls through to `_get_env_fallback()` which reads the
single `SOLANA_RPC_URL` env var — potentially the same rate-limited Helius URL.

**Secondary cause:** `report_rate_limit()` applied a flat 300 s back-off
regardless of how many times an endpoint had been rate-limited. An endpoint
that has been 429'd 5 times in a row received the same 5-minute cooldown as
the first offence, causing the rotation to spread load across 3 equally-penalised
Helius endpoints in lock-step, cycling back to the same one as soon as the
first timer expires.

**Fixed by commit 4307e52:**
- `Endpoint.is_soft_usable` property: true if the endpoint is enabled and not
  hard-blocked; ignores the rate-limit timer.
- `Endpoint.seconds_until_available` property: 0 if available, else remaining
  rate-limit seconds (or ∞ if hard-blocked).
- `ProviderEndpoints._get_least_penalized_fallback()`: picks the soft-usable
  endpoint with the smallest `seconds_until_available`.
- `get_next_endpoint()` falls through to this fallback instead of returning
  None.
- `get_endpoint()` logs a clear WARNING when returning a still-rate-limited
  endpoint including the endpoint name and seconds-to-recovery.
- `report_rate_limit()` now applies exponential back-off:
  `base_seconds × 2^(rate_limit_count)`, capped at 1800 s (30 min).
  Caller-supplied `Retry-After` durations are still honoured verbatim.

### 2. BASE_RPC and FANTOM_RPC Empty Pools (P1 — operational)

**Root cause:** Migration 011 registers these provider types in the catalogue
but seeds no actual URLs. Unless the operator had populated `BASE_RPC_URL` /
`FANTOM_RPC_URL` in `.env`, the pool boots with zero endpoints for those types.
Callers receive `None`, honeypot_checker fails on Fantom, and any Base-chain
DEX activity fails silently.

**Fixed by migration 015:** Three public BASE endpoints (Coinbase mainnet.base.org
priority 100, Ankr 120, Blast 130) and two Fantom endpoints (FTM Tools 100,
Ankr 110) seeded with `ON CONFLICT DO NOTHING`.

### 3. Secrets Manager Re-Init Churn (P1 — operational noise)

**Root cause:** `SecureSecretsManager.initialize(db_pool)` unconditionally
cleared `_cache` and `_source_map` and logged an INFO message whenever `db_pool`
was non-None and `_initialized` was True. Since each module subprocess shares the
singleton in the same process, every module that called `initialize(db_pool)`
during startup triggered this path, even with the identical pool object.

**Fixed by commit 8c0f111:**
- `_db_pool_id: Optional[int]` field stores `id()` of the last accepted pool.
- If the same pool object is re-passed, the method is a no-op (DEBUG log only).
- A genuinely new pool object (reconnect after crash) still triggers a cache
  clear, logged at INFO.
- Downgrade path (initialized-with-pool, called-with-None) is now a no-op
  rather than silently falling into bootstrap mode.

---

## Ops / Security Risks (Remaining)

### R1 — Helius API keys stored in `rpc_api_pool.api_key` column (plaintext)

The `api_key` column in `rpc_api_pool` is `TEXT` with no encryption.  Helius
keys and any other per-endpoint API keys are visible to anyone with `SELECT`
access to the DB.  Recommend: encrypt at application layer using
`security/encryption.py` before persisting, and decrypt on load in
`_load_from_database()`.  This is a medium-severity finding for a hosted
environment; low-severity if DB access is tightly controlled.

### R2 — Priority drift is permanent until success recovery

`report_rate_limit()` increases `priority` by +50 per event (up to 1000) and
`report_success()` decreases by -5 per success.  After a Helius endpoint
accumulates 18 consecutive rate-limit events its priority becomes ~1000; it
needs 180 successful calls to return to priority 100.  During that recovery
period the public fallback endpoints (priority 200) would be preferred over
the high-quality Helius endpoint.  Consider: reset priority to base (100) when
`rate_limit_until` expires, rather than relying on the success-recovery ratchet.

### R3 — Health check interval is 3600 s (1 hour)

`_health_check_interval = 3600`.  Rate-limited endpoints that recover within
minutes are not re-probed until the next scheduled health check.  The
`report_success()` path does reset their status on the next actual call, so
this is low-urgency in normal operation.  But a Helius endpoint that is
rate-limited at 23:00 and recovers at 23:05 may not be re-promoted until
01:00.  Recommend: add a targeted recovery probe (async) when `rate_limit_until`
passes rather than waiting for the full 1-hour cycle.

### R4 — `_get_env_fallback()` is called in STARVED branch

Even after the fix in commit 4307e52, if `get_next_endpoint()` returns a
soft-usable (rate-limited) endpoint, `get_endpoint()` uses it.  But if the
pool has zero endpoints (`provider` is None), the code still falls through to
`_get_env_fallback()`.  This is correct behaviour for unconfigured provider
types, but it means freshly-deployed environments without a seeded DB will use
raw `.env` URLs without any health tracking.

---

## Ranked Fixes Shipped (This Wave)

| # | Commit | File | Change |
|---|--------|------|--------|
| 1 | 4307e52 | `config/pool_engine.py` | Anti-starvation fallback + exponential back-off |
| 2 | 8c0f111 | `security/secrets_manager.py` | Idempotent re-init; eliminate cache-clear churn |
| 3 | d5ab048 | `migrations/015_*` | Seed Solana/BASE/FANTOM public RPC fallbacks |

---

## Cross-Module Handoffs

- **SOLANA module** (`main_solana.py`, sniper WSS): after migration 015 is
  applied the Ankr and mainnet-beta Solana endpoints become available as a
  fallback.  However these public endpoints are rate-limited at ~10-40 RPS,
  far lower than Helius.  The SOLANA module should call
  `pool.report_rate_limit(provider_type, url, duration_seconds=60)` when it
  receives HTTP 429 rather than letting the pool use its default back-off, so
  the real server-side timer is honoured.

- **DEX module** (`main_dex.py`): BASE_RPC and FANTOM_RPC are now seeded.
  The honeypot checker (`risk/honeypot_checker.py`) should see FANTOM endpoints
  after migration 015.  Verify the checker reads from pool_engine rather than
  a raw env var.

- **Dashboard** (`main_dashboard.py`): the pool engine dashboard page
  (`/api/pool/endpoints`) will now show the seeded public endpoints.  Consider
  adding a visual indicator ("fallback tier") for endpoints with priority > 150.

- **Secrets manager consumers** (all modules): with the idempotency fix, log
  volume from `secrets_manager` will drop significantly on multi-module
  startups.  If a module still sees repeated re-init logs, it is likely
  creating a new db_pool object on every call — that pool should be cached and
  reused.
