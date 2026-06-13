# ClaudeDex Deployment Guide

End-to-end instructions for standing up the full stack from a clean host.
Pairs with `docs/OPERATIONS_GUIDE.md` (day-to-day use), `docs/MODULE_CATALOG.md`
(one-screen module index), `docs/deployment.md` (topology deep-dive),
`docs/staging.md` (promotion workflow), and `docs/runbook.md` (incidents).

## 1. Prerequisites

- Linux host with Docker Engine + Docker Compose v2 (`docker compose`, not `docker-compose`).
- ~4 GB RAM headroom for the bot container (compose caps it at `mem_limit: 4g`, `cpus: 3.0`) plus Postgres/Redis.
- Outbound HTTPS (RPC providers, exchange APIs) — no inbound ports required except the dashboard.
- Git checkout of the repo; all commands below run from the repo root.

The stack is three containers (`docker-compose.yml`):

| Service | Image | Ports published |
|---|---|---|
| `postgres` | `timescale/timescaledb:latest-pg14` | internal only |
| `redis` | `redis:7-alpine` | internal only |
| `trading-bot` | built from `Dockerfile` | `8080:8080` (dashboard ONLY) |

Note: module health ports (8081-8105, section 6) are **not published** by
compose — probe them from inside the container.

## 2. Secrets and configuration model

Two layers, deliberately separate:

1. **`.env` (host file, mounted read-only into the container)** — module
   enable/disable flags (`<MODULE>_MODULE_ENABLED=true|false`), DB host/port,
   log level. Editing `.env` requires only `docker compose restart trading-bot`
   (it is a volume mount, NOT baked at build time). **No rebuild needed for
   `.env` changes.**
2. **DB `settings` table** — everything else: every strategy knob, risk limit,
   shadow/live flag, threshold. Seeded by migrations, edited live from the
   dashboard Settings tabs. Changes take effect without restarts (modules
   re-read config on their tick).

Sensitive credentials:

- Docker secrets files (read by compose `secrets:` section):
  ```bash
  mkdir -p secrets && chmod 700 secrets
  echo "bot_user"            > secrets/db_user
  echo "<strong-password>"   > secrets/db_password
  echo "<strong-password>"   > secrets/redis_password
  chmod 600 secrets/*
  ```
  or interactively: `./scripts/setup_secrets.sh`.
- `.encryption_key` — created/verified by `./scripts/setup_encryption_key.sh`,
  mounted read-only at `/app/.encryption_key`. Required by the secrets manager
  that encrypts API keys/private keys at rest in the `secure_credentials` table.
- Exchange/chain API keys and wallet private keys go into the DB via
  `scripts/setup_secure_credentials.py` (or `scripts/migrate_credentials_to_db.py`
  for an existing `.env`-based install). They do **not** live in `.env`.

## 3. Build

```bash
docker compose build trading-bot
```

Two things to know about the image (`Dockerfile`):

- **The pip dependency list is HARDCODED in the Dockerfile stages** —
  `requirements.txt` is COPIED but **not** pip-installed. If you add a Python
  dependency, you must add it to the appropriate Dockerfile stage or it will
  not be in the image. (Several stages document past breakage from forgetting
  this, e.g. the advisor's `yfinance<0.2.59` pin and tefas libraries in
  Stage 7c.)
- **Migrations are baked into the image** (`COPY . .` includes `migrations/`),
  and the entrypoint (`scripts/docker-entrypoint.sh`) runs them automatically
  before starting `main.py`. A code change to migrations therefore DOES
  require a rebuild; a `.env` change does not.

Heavy/optional stages (TensorFlow, PyTorch, Kronos deps, advisor data libs)
are fail-soft: a transient PyPI failure degrades a feature instead of
breaking the build. Check the build log for `⚠️` lines.

## 4. Database migrations

Migrations live in `migrations/NNN_*.sql`, currently numbered 001-135 with
**intentional gaps** (048, 059, 084-087, 090-091, 093-094, 098-099, 114-117,
134 are unused numbers — do not "fill" them). 120-133 are the module-expansion
wave (execution_quality through smart_money); 135 is the copy-trading v3
discovery/shadow-sim wave.

How they run:

- **Automatically**: `scripts/docker-entrypoint.sh` runs
  `python scripts/migrate_database.py` on every container start, before the
  app. If a migration fails, the container exits — the bot never starts on a
  half-migrated schema.
- **Manually** (e.g. against a stopped stack):
  ```bash
  docker compose run --rm trading-bot python scripts/migrate_database.py
  ```

Properties of the runner (`scripts/migrate_database.py`):

- **Idempotent** — applied versions are tracked in a `migrations` table
  (filename stem = version); already-applied files are skipped. Safe to run
  repeatedly.
- Each file is applied in a transaction; failure rolls back that file.
- Files are applied in **sorted filename order** — never renumber an applied
  migration.
- Canonical directory is top-level `migrations/`; the legacy
  `data/storage/migrations/` path is a fallback only.

## 5. First start

```bash
# 1. Secrets (section 2), then:
cp .env.example .env        # set module flags; start with everything except dashboard OFF
docker compose up -d
docker compose logs -f trading-bot   # watch migrations apply, then module launch
```

`main.py` starts the dashboard subprocess first (so it stays reachable even if
trading modules fail), then spawns each module whose `<MODULE>_MODULE_ENABLED=true`,
restarting crashed children with backoff. Compose `restart: "no"` on the bot
container is deliberate — a crashed orchestrator stays down for operator
inspection rather than flap-restarting against a bad state.

Verify:

```bash
curl -s http://localhost:8080/health        # dashboard liveness (also the container HEALTHCHECK)
```

Then log in at `http://<host>:8080/login` and check the Modules page.

## 6. Health-port map

`/health` = liveness, `/status` = stats (where bound). Ports marked `*` are
**probe-only defaults**: that module binds no health server; the dashboard
falls back to DB heartbeats for its status badge. `orchestrator_ai` and
`portfolio_allocator` bind no health server at all (DB heartbeats only).
Override pattern: `<MODULE>_HEALTH_PORT` env var.

| Port | Module | | Port | Module |
|---|---|---|---|---|
| 8080 | dashboard | | 8093 | treasury |
| 8081 | futures | | 8094 | sentinel |
| 8082 | solana | | 8095 | market_data_warehouse |
| 8083 | sniper* | | 8096 | catalyst_calendar |
| 8084 | arbitrage* | | 8097 | options_vol |
| 8085 | dex | | 8098 | yield_treasury |
| 8086 | advisor | | 8099 | execution_gateway |
| 8087 | ai* | | 8100 | clmm_lp |
| 8088 | copy_trading* | | 8101 | param_tuner |
| 8089 | polymarket | | 8102 | intent_solver |
| 8090 | meta_controller | | 8103 | basis_desk |
| 8091 | regime_allocator | | 8104 | stat_arb |
| 8092 | execution_quality | | 8105 | smart_money |

Only 8080 is published to the host. Probe the rest from inside:

```bash
docker compose exec trading-bot curl -s http://localhost:8094/health   # e.g. sentinel
```

## 7. Enabling a module

1. Edit `.env`: set `<MODULE>_MODULE_ENABLED=true`
   (full flag list is in `main.py`; one flag per module, default `false` for
   everything new — only DEX defaults `true` in compose).
2. `docker compose restart trading-bot` (no rebuild).
3. Verify it came up:
   - `docker compose exec trading-bot curl -s http://localhost:<port>/health`
   - dashboard Modules page badge goes green (or DB-heartbeat for probe-only modules)
   - `tail -f logs/<module>/<module>.log` on the host (logs are volume-mounted)
4. Confirm its DB config was seeded: dashboard Settings tab for that module
   should show its keys (seeded by the module's migration).

To enable from the dashboard instead, use the per-module enable/disable/pause
controls under `/modules` — see `docs/OPERATIONS_GUIDE.md`.

## 8. DRY_RUN-first doctrine

**Nothing goes LIVE on first enable.** Every module ships with safe defaults:

- Trading modules honor their `DRY_RUN` flag, the global kill switch
  (`logs/.killswitch`), and per-module pause files (`logs/.pause_<module>`)
  via `core/dry_run.should_skip_live(...)`.
- All module-expansion-wave modules (migs 120-133) are advisory/shadow by
  default; the ones with a live path at all (`options_vol`, `clmm_lp`,
  `basis_desk`-class strategies, `polymarket`) sit behind **dual flags**
  (`shadow_mode=true` AND `live_execution_enabled=false`) plus
  `should_skip_live` plus RiskManager. No migration seed flips any live flag
  to true.
- `yield_treasury`'s live deposit path is **not built** — even with every flag
  open it terminates in `live_deposit_path_not_built`. `intent_solver` is a
  parked scaffold.

Promotion path per module is **shadow → DRY_RUN → LIVE**, with observation
gates at each step — see `docs/staging.md` and `docs/OPERATIONS_GUIDE.md`.

## 9. Pre-LIVE punch-list

Before flipping any module's live flag, walk this list:

1. **Global**: kill switch clear (`logs/.killswitch` absent), dashboard
   emergency-stop tested once, RiskManager limits reviewed for the account size.
2. **Module ran ≥1 week in DRY_RUN/shadow** with its dashboard panel showing
   sane simulated P&L and no error-log growth (`logs/<module>/*_errors.log`).
3. **DEX**: set `trading.live_max_execute_retries=1` in DB (seeded default is
   3; the executor retry loop can double-spend a flaky broadcast — see
   `modules/dex_trading/CLAUDE.md`).
4. **SNIPER**: flip `safety_check_enabled=true` in DB (LIVE-mode safety-filter
   startup guard expects it).
5. **ARBITRAGE / POLYMARKET / OPTIONS_VOL / CLMM_LP**: these need BOTH
   `shadow_mode=false` AND `live_execution_enabled=true` — flipping one alone
   does nothing (by design).
6. **FUTURES**: confirm testnet vs mainnet API keys; review per-symbol tiering
   (mig 088) and leverage overrides.
7. Credentials present in `secure_credentials` for the venue the module
   trades on (a missing key fails closed, but verify before, not after).
8. Take a DB backup (`docker compose exec postgres pg_dump ...`) before any
   live flip so you can reconstruct state.

## 10. Updating and rollback

- **Config change** (DB settings): instant, no restart.
- **`.env` change**: `docker compose restart trading-bot`.
- **Code change**: `git pull && docker compose build trading-bot && docker compose up -d trading-bot`.
  New migrations apply automatically on start.
- **Rollback**: migrations are forward-only (no down scripts). Roll back code
  via git + rebuild; if a migration must be undone, restore the DB backup
  taken in step 9.8. Always engage the kill switch
  (`python scripts/emergency_stop.py` or dashboard emergency-stop) before a
  rollback of live-trading code.
