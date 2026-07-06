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

## 7. Bulk verify + import (`scripts/verify_and_import_rpcs.py`)

Have a spreadsheet full of endpoint/key combos (several Ankr / dRPC /
Alchemy accounts per chain, a stack of Helius/Etherscan/Birdeye keys) and
no idea which still work? This tool probes every candidate with one cheap
read-only call, prints a verdict table, and — only when you say so —
imports the working set into `rpc_api_pool` + the encrypted secrets slots.

### Operator flow (exact commands)

```bash
# 1. update + rebuild (the bot keeps running on your existing .env —
#    the env bootstrap is unchanged; NEVER `down -v`, that wipes the DB)
git pull
docker compose up -d --build

# 2. copy your spreadsheet into ./data on the HOST — docker-compose mounts
#    it at /app/data inside the container
cp ~/Downloads/rpc_keys.xlsx ./data/

# 3. DRY-RUN (default): probe everything, write logs/rpc_verify_report.json
docker compose exec trading-bot \
    python scripts/verify_and_import_rpcs.py --file /app/data/rpc_keys.xlsx

# 4. review the console table (and logs/rpc_verify_report.json)

# 5. import the verified working set
docker compose exec trading-bot \
    python scripts/verify_and_import_rpcs.py --file /app/data/rpc_keys.xlsx --apply

# 6. restart so every module reloads the pool
docker compose restart trading-bot
```

If `openpyxl` is unavailable the tool tells you to export the sheet as
CSV — CSV is always supported. Expected sheet layout (sheet name `RPCs`
or first sheet): column A = chain/provider block label on the first row
of each block, B = key, C = base URL (final URL = C+B), D = account
email, E = the C&B formula (ignored), F = optional wss URL.

### What it verifies

Candidates come from four places, deduped: your `--file`, every RPC/key
var in `.env` (numbered slots included), the existing `rpc_api_pool`
rows (re-verified, even disabled ones), and the encrypted secrets slots.
Probes are strictly read-only: EVM https `eth_chainId` (must match the
chain the row claims — the operator's `solana_DEVNET` Ankr rows and any
devnet/testnet URL classify `WRONG_NETWORK` and are never imported) +
`eth_blockNumber` + a 1-block `eth_getLogs` capability check (drives
smart_money / copy EVM discovery — shown in the `LOGS` column); EVM wss
`eth_chainId` over the socket; Solana https `getSlot` + `getGenesisHash`
(must equal mainnet-beta `5eykt4Us…`); Solana wss `slotSubscribe`;
Helius `getSlot` (~1 credit); Etherscan V2 `proxy.eth_blockNumber`
(free); Birdeye `defi/price` for SOL. Statuses: `OK`, `AUTH_FAIL`,
`RATE_LIMITED`, `QUOTA_EXHAUSTED`, `WRONG_NETWORK`, `TIMEOUT`, `DEAD`,
`UNTESTED`. GoPlus keys are always `UNTESTED`: GoPlus auth is a signed
app_key+app_secret token flow, so a bare key cannot be positively
verified — it is reported but never imported.

### What `--apply` does (and refuses to do)

- Verified-OK RPC/WSS URLs upsert into `rpc_api_pool` under the existing
  provider taxonomy (`ETHEREUM_RPC` … `SOLANA_RPC`, `SOLANA_WS`, …).
  Keyed/account URLs get priority 50, publics 100 (lower = preferred).
  Existing rows only get `status`/`last_health_check_at` refreshed —
  your priority/weight/name edits survive. wss URLs for chains without a
  `*_WS` provider type (only ETHEREUM/BSC/ARBITRUM/SOLANA exist) are
  skipped with a note rather than inventing a type nothing consumes.
- Verified-OK bare keys are stored **Fernet-encrypted** in the numbered
  secrets slots (`HELIUS_API_KEY`, `_2` …): keys already sitting in a
  slot keep it, new keys fill empty slots, and only slots holding a key
  that verified hard-failed are ever overwritten (loudly). Their pool
  endpoints are seeded exactly like the Wave-F5 bootstrap.
- Rows that failed `AUTH_FAIL` / `DEAD` / `WRONG_NETWORK` are set
  `is_enabled=false` — **never deleted**. `RATE_LIMITED` /
  `QUOTA_EXHAUSTED` rows stay enabled (transient; pool_engine cools them
  at runtime); such *new* candidates are skipped with a re-run note.
- It refuses `--apply` outright if the DB or the encryption key is
  unavailable (no partial writes), and it is idempotent — re-running is
  always safe.

Offline self-test of the classification + slot logic:
`python scripts/verify_and_import_rpcs.py --mock`.

**Plaintext honesty note:** `rpc_api_pool` URLs (which can embed provider
keys, e.g. Helius/Alchemy key-in-URL) are stored plaintext in Postgres —
same as every row the `/settings/rpc-api` page writes today. Only bare
API keys go through the encrypted `secure_credentials` store. Full
at-rest encryption of pool URLs is a candidate future hardening.

### Etherscan July-2026 free-tier note

Etherscan capped free-tier list endpoints (txlist etc.) at 1000 rows per
page in July 2026. Our only consumer (the COPY EVM monitor) requests
`offset=5`, so we are unaffected; a defensive
`ETHERSCAN_TXLIST_PAGE_SIZE` clamp (≤1000) was added in
`modules/copy_trading/copy_engine.py` so future edits cannot silently
break free-tier accounts.
