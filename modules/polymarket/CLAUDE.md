# POLYMARKET Module

## What it does
Prediction-market module for **Polymarket** (Polygon PoS, chainId 137; USDC.e
collateral). Polls the **free read-only Gamma API** for active binary markets
(YES/NO outcome tokens priced 0–1) and runs two shadow strategies:
1. **Risk-free arb detector** — flags markets where best `YES_ask + NO_ask < 1`
   by more than `min_arb_edge_bps` *after* the `fee_gas_buffer_bps` haircut
   (buying both sides redeems to a guaranteed $1).
2. **Event-momentum / new-market signal** — transparent score on liquid markets
   with a large short-window YES-price move; ADVICE only.

Trading-capable, but **LIVE execution is GATED OFF by default**. Every detected
opportunity is recorded as a **simulated** `polymarket_trades` row
(`is_simulated=true`); signals go to `polymarket_signals`. Mirrors the
ARBITRAGE module's shadow-first safety shape.

## Entry point
`modules/polymarket/main_polymarket.py` — launched as a subprocess by `main.py`
when `POLYMARKET_MODULE_ENABLED=true` (default **false**). Health server on port
8089 (env override: `POLYMARKET_HEALTH_PORT`).

## LIVE-execution gate (all four required before any real CLOB order)
```
shadow_mode == false
AND live_execution_enabled == true
AND not core.dry_run.should_skip_live(module_dry_run, module='polymarket', account=...)
AND RiskManager.validate_trade(...) passes
```
Any one failing → the trade is recorded simulated (or skipped) with a
`skip_reason`; no order is placed. `py-clob-client` is imported lazily and
fail-soft (missing dep → live path unavailable, shadow continues).

## Key config (DB-backed, config_type='polymarket_config'; migration 101)
| Key | Default | What it does |
|---|---|---|
| `shadow_mode` | true | Record-only; never reach the live order path |
| `live_execution_enabled` | false | Explicit LIVE opt-in (fail-safe off) |
| `gamma_base_url` | https://gamma-api.polymarket.com | Read-only market data (no key) |
| `gamma_max_requests_per_minute` | 30 | Client-side Gamma rate cap |
| `clob_base_url` | https://clob.polymarket.com | Live order API (py-clob-client) |
| `chain_id` | 137 | Polygon PoS |
| `usdc_address` / `ctf_address` / `ctf_exchange_address` | (Polygon) | Operator-verifiable contracts |
| `poll_interval_s` | 60 | Cycle cadence |
| `max_markets_per_poll` | 200 | Markets pulled per cycle |
| `category_filter` | (empty) | Optional category allowlist |
| `max_position_size_usd` | 50 | Per-trade notional cap |
| `min_arb_edge_bps` | 100 | Risk-free arb floor (after buffer) |
| `fee_gas_buffer_bps` | 100 | Fee+gas haircut on gross edge |
| `momentum_min_liquidity_usd` | 10000 | Momentum liquidity floor |
| `momentum_min_move_frac` | 0.05 | Min YES-move to signal |
| `momentum_min_score` | 0.3 | Min momentum score to publish |
| `shadow_record_interval_s` | 300 | Per (signal,market) record throttle |

## Secrets (only needed for LIVE, which is off by default)
`POLYMARKET_PRIVATE_KEY` — resolved via `security/secrets_manager` (encrypted DB)
→ `os.getenv` fallback. Never required in shadow mode.

## Kill switch
- **Global**: `logs/.killswitch` — polled by BaseModule via `core.dry_run.start_killswitch_poller`.
- **Per-module**: `logs/.pause_polymarket` — written by dashboard pause/resume.
- **Effect**: cycle idles; no order path reached.

## Logs
`logs/polymarket/` — `polymarket.log` (all), `polymarket_errors.log` (ERROR+).

## DB tables (migration 101)
- `polymarket_trades` — simulated (and, when enabled, live) trade ledger; `is_simulated` default TRUE.
- `polymarket_signals` — advice/signal stream (always simulated).

## Data-source matrix
| Path | Source | Free? | Key |
|---|---|---|---|
| Market data (signals/arb) | Gamma API REST | YES (no key) | None |
| Live orders (gated off) | CLOB API via py-clob-client | order fees | `POLYMARKET_PRIVATE_KEY` |

## Dependencies
`py-clob-client` is added to the Docker image pip list (the image installs a
hardcoded list, NOT requirements.txt). Read-only Gamma needs only `aiohttp`
(already present). Self-test (offline, fixture-backed):
`python -m modules.polymarket.strategies`.

## Isolation
Does NOT import or modify any trading module's engine/executor. Reuses
read-only: `core/dry_run.py` (killswitch/pause/dry-run), `core/risk_manager.py`
(live gate), `security/secrets_manager.py` (own key). Config: `polymarket_config`
rows only.
