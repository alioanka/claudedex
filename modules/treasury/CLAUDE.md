# TREASURY Module (Phase 1 — OBSERVE-ONLY)

## What it does
Wallet/gas/inventory **observer**. Every tick it polls native + key token
balances of the bot's wallets (PUBLIC addresses only), reconciles them against
the trade ledgers' open LIVE positions, writes one `treasury_snapshots` row per
wallet, and logs alerts. It **NEVER signs, NEVER transfers, NEVER decrypts a
key, and NEVER touches `logs/.killswitch`**. Phase 2 (gas top-up / profit
sweep) is a separate, separately-gated build — none of it exists here.

## Entry point
`modules/treasury/main_treasury.py` — launched by `main.py` when
`TREASURY_MODULE_ENABLED=true` (default **false**). Health server on port 8093
(`TREASURY_HEALTH_PORT`): `/health` liveness, `/status` last-tick summary.
Engine: `core/treasury_engine.py` (self-tested:
`python -m modules.treasury.core.treasury_engine`). RPC readers:
`core/balance_reader.py` (self-tested:
`python -m modules.treasury.core.balance_reader`).

## Wallets observed (PUBLIC addresses DERIVED from the secrets-managed private keys)
Each address is computed from its private key (resolved via `security/secrets_manager`)
so the observed wallet can never drift from the actual signer. The key is read
ONLY to derive its public address — this module never signs, never transfers,
never persists the key. The env address column is a **fallback only** (fresh
install / key not yet loaded).
| Group | Derived from (secrets_manager) | Env fallback | Chains |
|---|---|---|---|
| `evm` | `PRIVATE_KEY` → `eth_account` address (shared EVM EOA: DEX/Arb/AI) | `WALLET_ADDRESS` | `evm_chains` config (default ethereum,arbitrum,base) |
| `dex_solana` | `SOLANA_PRIVATE_KEY` → solders pubkey | `SOLANA_WALLET` | solana |
| `solana_module` | `SOLANA_MODULE_PRIVATE_KEY` → solders pubkey | `SOLANA_MODULE_WALLET` | solana |
| custom | — | `extra_wallets` DB config (label, chain, address) | any |

RPC URLs come ONLY from `config/pool_engine.PoolEngine.get_endpoint('<CHAIN>_RPC')`
with `report_success` / `report_failure` / `report_rate_limit` after every read
(read-only use of the shared pool; this module never edits it).

## Alert conditions (logged to `logs/treasury/treasury_errors.log`; also embedded in the snapshot row)
| Alert | Condition | Severity |
|---|---|---|
| `gas_low` | native balance < `gas_floor_<chain>` (else `gas_floor_default`) | WARNING; **ERROR when the ledgers show open LIVE positions on that chain** (exit gas starvation) |
| `hot_wallet_high` | stable-token balance > `hot_wallet_ceiling_usd` | WARNING (unswept profit = hot-wallet risk) |
| `reconcile_drift` | abs(stable balance − previous snapshot) > `reconcile_drift_usd` | WARNING (unexplained flow — reconcile vs ledgers) |
| `ledger_unbacked` | ledgers imply open LIVE positions but wallet native ≤ `native_dust` and stables ≤ $1 | ERROR (ledger/on-chain mismatch) |

Ledger reconcile reads (fail-soft, LIVE rows only where the schema marks DRY):
`positions` (dex), `sniper_positions`, `arbitrage_positions`,
`copytrading_positions`, `ai_positions` (all `status='open'` + chain), and
`solana_positions` (`NOT is_simulated`; whole table = open solana positions).

## Key config (DB-backed, config_type='treasury'; migration 121)
| Key | Default | What it does |
|---|---|---|
| `poll_interval_seconds` | 300 | Tick cadence (slow loop) |
| `evm_chains` | ethereum,arbitrum,base | Chains on which the EVM EOA is observed |
| `gas_floor_default` | 0.005 | Fallback native low-water mark |
| `gas_floor_ethereum` | 0.02 | Per-chain floors (ETH) |
| `gas_floor_arbitrum` / `gas_floor_base` | 0.005 | Per-chain floors (ETH) |
| `gas_floor_solana` | 0.05 | SOL floor (fees + Jito tips + ATA rent) |
| `hot_wallet_ceiling_usd` | 1000 | Unswept-profit observation threshold |
| `reconcile_drift_usd` | 250 | Snapshot-to-snapshot stable drift alert |
| `native_dust` | 0.0005 | Empty-wallet cutoff for `ledger_unbacked` |
| `extra_wallets` | [] | Extra PUBLIC addresses to observe |
| `evm_tokens` / `solana_tokens` | canonical USDC per chain | Key tokens read per chain; `stable: true` counts toward the ceiling |

Env knobs: `TREASURY_MODULE_ENABLED` (gate, default false),
`TREASURY_HEALTH_PORT` (8093), `TREASURY_POLL_INTERVAL` (pre-migration fallback).

## Kill switch
- Global: `logs/.killswitch` — tick skipped (polled via `core.dry_run.start_killswitch_poller`; this module only READS the flag).
- Per-module: `logs/.pause_treasury` — tick skipped.

## Logs
`logs/treasury/` — `treasury.log` (INFO), `treasury_errors.log` (WARNING+, so
treasury ALERTS land there, not only failures).

## DB tables
- `treasury_snapshots` (migration 121) — one row per wallet per tick: balances,
  token map, stable USD, ledger reconcile detail, alerts JSON.

## Isolation / safety
Fail-soft everywhere: RPC error/429 → that chain is skipped and reported to
pool_engine; missing ledger table/column → that table is skipped; missing
config → observe-safe code defaults. No paid LLM. Writes ONLY
`treasury_snapshots` and its own logs. Never imports an executor. It resolves
`PRIVATE_KEY`/`SOLANA_PRIVATE_KEY`/`SOLANA_MODULE_PRIVATE_KEY` via
`security/secrets_manager` ONLY to DERIVE the public address (`eth_account` /
solders pubkey) — it never signs, never transfers, and never persists the key.

## Orchestrator wiring (DONE)
- `main.py`: registered — launches `modules/treasury/main_treasury.py` when `TREASURY_MODULE_ENABLED=true` (module_key `treasury`, in `needs_database`).
- `.env.example`: `TREASURY_MODULE_ENABLED=false`, `TREASURY_HEALTH_PORT=8093`.
- Root `CLAUDE.md` module table + health-port map (8093 = treasury) updated.
