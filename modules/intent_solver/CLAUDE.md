# INTENT_SOLVER Module — EXPERIMENTAL, SHADOW-ONLY SCAFFOLD

## VERDICT: PARK IT (do not build the full solver/filler)
`docs/agents/NEW_MODULE_IDEAS.md` #11 rejects this as a real P&L module:
top CoW solvers and UniswapX fillers run colocated infra, private
market-maker inventory, and sub-100ms quoting at single-digit-bps margins;
CoW additionally requires a six-figure staked bond and vetting. The deficit
is **structural, not a build-size problem**. This scaffold exists only to
prove the data path cheaply and answer one falsifiable question: *does
fillable edge net of costs even appear at a rate worth caring about?*

## Revisit conditions (both required, per the doc)
1. The execution_gateway + TCA stack demonstrates top-decile internal
   execution quality for **6+ months**, AND
2. a niche emerges where the bot has genuinely private edge (e.g. long-tail
   Solana routes via its Jupiter/pump.fun infrastructure).
Until then: keep this module disabled or running shadow-only for data.

## What it does (v1 = everything it will ever do without a new decision)
Polls open intents from **free public read-only APIs** — CoW Protocol
orderbook auction (`GET /{chain}/api/v1/auction`) and optionally UniswapX
open Dutch orders — and for each intent asks: could the bot have filled it
profitably? It compares the intent's limit to a reference DEX quote
(CoW `POST /quote`, free) minus gas and a safety buffer, and records
qualifying rows to `intent_fill_opportunities` with **`is_simulated=true`,
always**.

## Why there is NO live path (not gated off — absent)
- CoW settlement requires being a bonded, vetted solver winning batch
  auctions; UniswapX filling is a pure latency race on reactor contracts.
- The module contains **no keys, no signing, no order submission, no
  settlement, no bonding code**. `shadow_mode` / `live_execution_enabled`
  config rows exist only for fleet-wide config consistency and are
  documented NO-OPs. No RiskManager hook is wired because there is nothing
  to gate — adding a live path would require a new design review against
  the revisit conditions above, plus a `core/risk_manager.py` kill-switch
  hook as a hard precondition.

## Fill-evaluation math (pure: `modules/intent_solver/evaluator.py`)
```
sell intent: gross = quote_buy_amount / limit_buy_amount - 1     (decimals cancel)
buy  intent: gross = 1 - quote_sell_amount / limit_sell_amount   (decimals cancel)
net_bps   = gross*1e4 - gas_bps - safety_buffer_bps
gas_bps   = gas_cost_usd / notional_usd * 1e4   (or fallback_gas_bps if unpriced)
recordable iff net_bps >= min_edge_bps AND (edge_usd unknown OR >= min_edge_usd)
```
Costs accounted: DEX route fees (inside the quote), settlement gas
(config `settlement_gas_units` x `gas_price_gwei` x native USD), quote
staleness/slippage (`safety_buffer_bps`). Notional USD uses CoW
`native_price` (atom-space — no token-decimals lookup) x DexScreener
wrapped-native USD. UniswapX Dutch orders use linear decay interpolation at
evaluation time. Self-test (offline, fixture-backed):
`python -m modules.intent_solver.evaluator`.

## Entry point
`modules/intent_solver/main_intent_solver.py` — launched by `main.py` when
`INTENT_SOLVER_MODULE_ENABLED=true` (default **false**). Health server on
port 8102 (env override: `INTENT_SOLVER_HEALTH_PORT`).

## Key config (DB-backed, config_type='intent_solver'; migration 130)
| Key | Default | What it does |
|---|---|---|
| `poll_interval_s` | 60 in code; DB row raised to **3600** by mig 143 (conditional — only while still at the mig-130 default 60) | Cycle cadence. The scaffold is PARKED; hourly is plenty for its one falsifiable question |
| `cow_enabled` / `cow_chains` | true / mainnet | CoW polling (mainnet, xdai, arbitrum_one, base) |
| `uniswapx_enabled` / `uniswapx_chain_ids` | false / 1 | Optional UniswapX polling |
| `max_quotes_per_cycle` | 10 | Reference-quote budget (respect the free API) |
| `min_edge_bps` / `min_edge_usd` | 30 / 5 | Record floors |
| `safety_buffer_bps` | 20 | Staleness/slippage haircut |
| `settlement_gas_units` / `gas_price_gwei` | 350000 / 5 | Gas haircut model (no RPC calls in v1) |
| `fallback_gas_bps` | 50 | Gas haircut when notional unpriceable |
| `shadow_record_interval_s` | 300 | Per-order-uid throttle |
| `shadow_mode` / `live_execution_enabled` | true / false | Documented NO-OPs in v1 (no live path exists) |

## Kill switch
- Global: `logs/.killswitch` — poller honored (stops the subprocess pattern-consistently).
- Per-module: `logs/.pause_intent_solver` — cycle idles.

## Logs
`logs/intent_solver/` — `intent_solver.log`, `intent_solver_errors.log`.

## DB tables (migration 130)
`intent_fill_opportunities` — simulated fill ledger (`is_simulated` default TRUE).

## Wave-F5 ingest fix (2026-07): CoW 403-block + log spam
CoW's Cloudflare front 403-blocked the default aiohttp User-Agent on every
call since Jun 15 (28,589 log lines, `intent_fill_opportunities` 0 rows
ever). Fixes in `clients.py`: (a) browser-class `User-Agent` on all client
requests; (b) dead-source backoff — after 3 CONSECUTIVE 403s on a path the
client warns ONCE and re-probes only hourly (recoverable without restart).
`main_intent_solver.py` additionally suppresses repeated identical
source-fetch warnings to once/hour. Mig 143 conditionally raises the DB
`poll_interval_s` 60 → 3600 (only while still at the seeded default).

## Data-source matrix (all free, no keys, no RPC)
| Path | Source | Key |
|---|---|---|
| Open intents | CoW orderbook API / UniswapX order API | None |
| Reference quotes | CoW `POST /quote` | None |
| USD pricing | CoW `native_price` + DexScreener | None |

## Isolation
No imports from any trading module. Reuses read-only: `core/dry_run.py`
(killswitch/pause), `security/docker_secrets.get_database_url`. Writes only
`intent_fill_opportunities`. Needs only `aiohttp` + `asyncpg` (already in
the image). No `pool_engine` use because v1 makes zero RPC calls.
