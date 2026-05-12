# ClaudeDex Production Deployment

How to deploy the bot stack. Pairs with `docs/staging.md` (promotion
workflow) and `docs/runbook.md` (operational procedures).

## 1. Topology

ClaudeDex runs as **one parent orchestrator process + N module
subprocesses**:

```
main.py (TradingBotOrchestrator)
├── dashboard subprocess         (modules/dashboard/main_dashboard.py)
├── dex subprocess               (modules/dex_trading/main_dex.py)
├── futures subprocess           (modules/futures_trading/main_futures.py)
├── solana subprocess            (modules/solana_trading/main_solana.py)
├── sniper subprocess            (modules/sniper/main_sniper.py)
├── ai_analysis subprocess       (modules/ai_analysis/main_ai.py)
├── arbitrage subprocess         (modules/arbitrage/main_arbitrage.py)
└── copy_trading subprocess      (modules/copy_trading/main_copy.py)
```

`main.py:460 TradingBotOrchestrator` starts the dashboard first (so
it stays reachable even if trading modules fail), then spawns each
enabled module as `subprocess.Popen` and restarts crashed children
with backoff (`psutil` for liveness). Each subprocess reads config
from the DB via `ConfigManager`, logs to `logs/<module>/`, and polls
`logs/.killswitch` + `logs/.pause_<module>` (BaseModule auto-wires).

Module-enable flags live in `.env` ONLY — every other knob is in DB:

```
DEX_MODULE_ENABLED=true
FUTURES_MODULE_ENABLED=true
SOLANA_MODULE_ENABLED=true
SNIPER_MODULE_ENABLED=true
AI_MODULE_ENABLED=true
ARBITRAGE_MODULE_ENABLED=true
COPY_TRADING_MODULE_ENABLED=true
DASHBOARD_MODULE_ENABLED=true
```

## 2. Docker / docker-compose

`docker-compose.yml` defines three services:

| Service | Image | Role |
|---|---|---|
| `postgres` | `timescale/timescaledb:latest-pg14` | Primary DB. TimescaleDB chosen for time-series trade hypertables. |
| `redis` | `redis:7-alpine` | Cache + pub/sub (used by `data/storage/cache.py`). |
| `trading-bot` | local `Dockerfile` build | The orchestrator + all modules in one container. |

Secrets bootstrap (docker swarm-style file secrets, mounted at
`/run/secrets/`):

```yaml
secrets:
  db_user:       { file: ./secrets/db_user }
  db_password:   { file: ./secrets/db_password }
  redis_password:{ file: ./secrets/redis_password }
```

The files have no extension; Postgres reads them via
`POSTGRES_USER_FILE` / `POSTGRES_PASSWORD_FILE`.

Pre-deploy:

1. `mkdir -p ./secrets && chmod 700 ./secrets`, create the three
   files above, then `chmod 600 ./secrets/*`.
2. `bash scripts/setup_encryption_key.sh` — generates `.encryption_key`
   (Fernet, 44 chars), mounted read-only at `/app/.encryption_key`.
3. `docker compose up -d --build`.

Do NOT run `docker compose down -v` — it wipes `postgres-data` and
`redis-data` (admin user, encrypted credentials, trade history all
vanish).

Variants: **`Dockerfile`** (full deps), **`Dockerfile.light`**
(minimal, for per-module containers), **`Dockerfile.redis`** (tuned
eviction). `docker-compose copy.yml.example` is reference only.

## 3. Kubernetes

Manifests in `kubernetes/`:

| File | Purpose |
|---|---|
| `configmap.yaml` | Non-secret config (chain IDs, RPC URLs, feature flags). |
| `deployment.yaml` | One `Deployment` running `your-registry/trading-bot:latest`. Namespace `trading`. Resources: 1 CPU / 2Gi mem (req) -> 2 CPU / 4Gi mem (lim). Liveness `/health`, readiness `/ready` on port 8080. |
| `service.yaml` | ClusterIP exposing the dashboard on port 8080. |
| `ingress.yaml` | External HTTPS termination for the dashboard. |

**Missing — operator must create:** `kubernetes/secret.yaml`. The
`Deployment` references a `Secret` named `trading-secrets` with keys
`database-url`, `redis-url`, `api-keys` (`deployment.yaml` lines
31-45). Intentionally NOT shipped — secrets do not belong in source
control. This is the **MB-BE-03 / DASH-BE-03 audit-flagged gap**.
Bootstrap:

```bash
kubectl create namespace trading
kubectl -n trading create secret generic trading-secrets \
  --from-literal=database-url='postgresql://user:pass@host:5432/tradingbot' \
  --from-literal=redis-url='redis://:pass@host:6379/0' \
  --from-literal=api-keys='{"etherscan":"...","alchemy":"..."}'
```

`ENCRYPTION_KEY` is NOT in the shipped Secret — it rides on a mounted
`.encryption_key` volume; `JWT_SECRET` loads from `config_sensitive`
after credentials migration. Extend the Secret if org policy requires.

Apply order: `configmap.yaml` -> create `trading-secrets` ->
`service.yaml` -> `deployment.yaml` -> `ingress.yaml` (after TLS).

## 4. Secrets bootstrap

Order matters:

1. **Encryption key** — `bash scripts/setup_encryption_key.sh`.
   Writes a Fernet key to `.encryption_key` (chmod 600). Idempotent.
2. **DB migrations** — `python scripts/migrate_database.py` (§5).
3. **Credentials migration** — `python scripts/migrate_credentials_to_db.py`.
   Reads remaining sensitive `.env` values and writes them encrypted
   into the `config_sensitive` table. After this, operators rotate
   any single secret via the dashboard's `/settings/credentials` UI
   (admin-gated, CSRF-protected — MB-26/27/28).
4. **Admin user** — `python scripts/init_auth.py`. Generates a
   `secrets.token_urlsafe(24)` admin password printed ONCE to stdout
   (NOT to logs, per MB-29b). Record it immediately.

For key-outside-project-tree setups use
`scripts/setup_secure_credentials.py --init`. After bootstrap, `.env`
holds only: module-enable flags, infra URLs, `LOG_LEVEL`.

## 5. DB migration order

Apply migrations from `migrations/` in numerical order:

```
001_add_auth_tables.sql               # users, sessions, audit_logs
002_add_config_tables.sql             # config_settings + config_sensitive
002_add_module_support.sql            # module-related tables (two files share 002 — apply both)
003_add_missing_chain_configs.sql
004_add_strategies_config.sql
005_add_futures_solana_configs.sql
006_add_futures_trades_table.sql
007_add_monad_pulsechain_chains.sql
008_add_solana_trades_table.sql
009_add_arbitrage_copytrading_tables.sql
010_add_sniper_ai_tables.sql
011_add_rpc_api_pool_table.sql
012_add_secure_credentials_table.sql
013_add_solana_positions_table.sql    # MB-09
```

Legacy `V0XX__*.sql` files (V001..V004) are predecessor versions
kept for reference. Do NOT apply them — the numbered series is
authoritative. `python scripts/migrate_database.py` runs the
numbered series in order.

## 6. Reverse-proxy / TLS termination

The dashboard MUST run behind HTTPS in production. Two supported
patterns:

**Pattern A — Kubernetes ingress** (`kubernetes/ingress.yaml`):
configure your ingress controller (nginx-ingress / Traefik) with a
TLS cert (Let's Encrypt + cert-manager, or external).

**Pattern B — Docker + nginx sidecar**: add an `nginx` service to
`docker-compose.yml` that proxies 443 -> `trading-bot:8080` with a
mounted TLS cert.

For either pattern, set `DASHBOARD_HTTPS=true` (controls session
cookie `secure` flag; default `true`, set `false` ONLY for local
dev) and `DASHBOARD_CORS_ORIGINS=<public-hostname>`. The dashboard
itself serves plain HTTP; TLS terminates upstream.

## 7. First-deploy smoke checklist

1. DB migrations applied (`scripts/migrate_database.py`).
2. `setup_encryption_key.sh` -> `migrate_credentials_to_db.py` ->
   `init_auth.py` run in order. Admin password recorded.
3. `DRY_RUN=true` in `.env` (per `docs/staging.md` stage 1).
4. Module-enable flags set per the operator's target topology.
5. `docker compose up -d --build` (or `kubectl apply`); orchestrator
   should show each subprocess starting in logs.
6. Reach `https://<dashboard-host>/login`; rotate the admin password
   if you have not already.
7. Verify the top-right badge reads `🔵 DRY-RUN` (MB-32).
8. Follow `docs/staging.md` for the DRY_RUN -> testnet -> canary ->
   prod promotion.

## See also

- `docs/staging.md` — promotion workflow
- `docs/runbook.md` — incident response
- `docs/engines.md` — engine API
- `docs/agents/MASTER_BACKLOG.md` — outstanding follow-ups
- Per-module `modules/<name>/CLAUDE.md`
