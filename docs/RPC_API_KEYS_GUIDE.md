# RPC / API Multi-Key Setup Guide (Wave-F5)

Free-tier rate limits are the #1 operational bleed: runtime evidence showed
~67k Helius rate-limit events per endpoint, an Ankr ETH monthly-quota
exhaustion that blinded arbitrage for 11 days, and copy-trading discovery
starved by 429s. The fix is cheap: **several free accounts per provider**,
registered as numbered keys. `config/pool_engine.py` rotates every request
across them and cools any key that returns HTTP 429, so quota burn is spread
N ways and a single limited account no longer stalls a module.

## 1. Which accounts to create

| Provider | Accounts | Env var(s) | Used by |
|---|---|---|---|
| Helius | 3-4 | `HELIUS_API_KEY`, `HELIUS_API_KEY_2..4` | COPY (leader monitor + discovery), SOLANA, SNIPER |
| Etherscan | 2 | `ETHERSCAN_API_KEY`, `ETHERSCAN_API_KEY_2` | COPY EVM monitor, smart-money lookups |
| Alchemy | 2 | `ETHEREUM_RPC_URL` + `ETHEREUM_RPC_URL_2` (and `BASE_RPC_URL_2`, `ARBITRUM_RPC_URL_2` ... per chain key) | ARBITRAGE, DEX, EXECUTION paths |
| Infura | 1-2 | further `ETHEREUM_RPC_URL_3` (RPC URLs mix providers freely) | same EVM pool |
| Birdeye | 1 | `BIRDEYE_API_KEY` (`_2` supported) | COPY wallet discovery |

Notes:
- **Alchemy/Infura keys are RPC URLs**, not API-key vars — paste the full
  `https://.../v2/<key>` URL into the numbered `<CHAIN>_RPC_URL[_N]` vars.
  All numbered URLs join the same chain pool at equal priority.
- Any provider in the env key map accepts `_2.._9`; unused slots are simply
  skipped (zero extra keys = exactly the old single-key behaviour).

## 2. Where to put the keys

Pick ONE of (checked in this order):

1. **Encrypted DB (preferred for prod)** — store each key in the
   `secure_credentials` table under the same names (`HELIUS_API_KEY`,
   `HELIUS_API_KEY_2`, ...), e.g. via the dashboard credentials page or
   `setup_env_keys.py`. `pool_engine.initialize()` probes the secrets
   manager for all numbered variants and registers each as an endpoint.
2. **Dashboard `/settings/rpc-api`** — add rows directly to the pool
   (`rpc_api_pool` table). Multiple rows per provider_type already work;
   give siblings EQUAL priority so round-robin rotation applies. New
   provider type for Birdeye: `BIRDEYE_API`.
3. **`.env`** — numbered vars as documented in `.env.example`. On first
   start with a DB attached these are seeded into `rpc_api_pool`
   (idempotent; existing rows are never overwritten).

Endpoints for keys sharing one base URL (Etherscan/Birdeye slot 2+) are
stored with a harmless `?key_slot=N` tag — that is expected, the upstream
API ignores it; it only satisfies the `UNIQUE(provider_type, url)`
constraint.

## 3. How rotation + cooldown work

- `PoolEngine.get_api_key(provider_type)` returns the **current** healthy
  key plus its endpoint id, round-robining through all equal-priority
  siblings (`ProviderEndpoints.get_next_endpoint`).
- Consumers re-fetch the key per call batch (60s local TTL in the copy
  engine), never pin one at process start.
- On HTTP 429 the consumer calls `report_key_rate_limit(<id or key>)`:
  the key enters `rate_limited` state (default 5 min, exponential up to
  30 min on repeats, or the parsed Retry-After), its priority is penalized,
  and the *next* `get_api_key` returns a sibling account. Consumers retry
  once on the sibling before backing off.
- Successes (`report_key_success`) heal priority/health; a fully-starved
  pool falls back to the least-penalized key rather than returning nothing.
- COPY discovery's Helius daily budget (`helius_daily_call_budget`, seed
  500) is **per key** — the effective ceiling scales with the number of
  registered Helius accounts.

## 4. Verifying it works

- **Startup logs** (`logs/pool_engine/pool_engine.log`):
  `Loaded N endpoints from .env` / `Registered N API-key endpoint(s) from
  secrets manager` — N should reflect every numbered key.
  `logs/copy_trading/copy_trading.log`: `HELIUS_API_KEY: SET (3 key(s),
  pool-rotated)`.
- **Rotation**: `logs/pool_engine/pool_engine_full.log` shows
  `KEY SELECTED: HELIUS_API -> HELIUS_API_KEY_2 (id=...)` lines cycling
  through siblings.
- **Cooldown**: `logs/pool_engine/pool_engine_rate_limits.log` shows
  `Rate limited: HELIUS_API - HELIUS_API_KEY (until ..., count=...)`
  followed by selections of the other keys, then
  `Recovered from rate limit: ...`.
- **Dashboard**: `/settings/rpc-api` pool page lists per-endpoint status,
  health score, success/failure counts and rate-limit state per key.
- `rpc_api_usage_history` records per-endpoint success/429 rows for
  after-the-fact quota audits.

## 5. Quota math (why these account counts)

- **COPY leader monitor**: 33 wallets / 15s cycle ≈ 190k Helius calls/day
  — exceeds one free tier (~100k credits/day). 2 keys ≈ 95k/day each
  (borderline); 3-4 keys ≈ 48-63k/day each (comfortable). Alternative
  levers: raise `copy_poll_interval_s` to 30s (halves the burn) or trim the
  watchlist.
- **COPY discovery**: budget-capped at `helius_daily_call_budget` (500) per
  key per UTC day — a good tenant next to the monitor.
- **Arbitrage/DEX EVM reads**: monthly quotas (Ankr-class incident) are the
  risk, not req/s — two Alchemy + one Infura URL per chain keeps one
  account's exhaustion from blinding the module; pool_engine keep-alive
  pings prevent idle-disable on the fallbacks.
- **Etherscan free tier**: 5 req/s, 100k/day per key. The COPY EVM monitor
  polls 5 txs/wallet/15s; two keys halve per-account burn and survive one
  key's daily cap.
- **Birdeye free tier**: low rps, Solana-only — one key suffices; a second
  only if discovery sweeps 429 in the logs.

## 6. Failure semantics (fail-soft guarantees)

- 0 numbered keys → identical to pre-Wave-F5 single-key behaviour.
- Secrets manager unavailable / DB down → env keys still load; reporting
  degrades to in-memory state.
- All keys rate-limited → least-penalized key is still returned (starvation
  fallback) and a single throttled WARNING is logged per state transition.
