# POLYMARKET Module

## What it does
Prediction-market module for **Polymarket** (Polygon PoS, chainId 137; USDC.e
collateral). Polls the **free read-only Gamma API** for active binary markets
(YES/NO outcome tokens priced 0–1) and runs two shadow strategies:
1. **Risk-free arb detector** — flags markets where `YES + NO < 1` by more than
   `min_arb_edge_bps` *after* the `fee_gas_buffer_bps` haircut (buying both
   sides redeems to a guaranteed $1). Wave-F5 book-honesty gates: liquidity
   floor (`arb_min_liquidity_usd`), 24h volume > 0, live uncrossed
   best_bid/best_ask required, and a too-good-to-be-true stale-book cap
   (`arb_max_edge_bps`) — the only firing in the first 20 shadow days was a
   dead tennis book, and these gates exist to kill that class of signal.
2. **Event-momentum / new-market signal** — asymptotic score in [0,1) (never
   pins 1.0), near-resolution exclusion (yes outside [0.02, 0.98] never
   signals), category denylist (`momentum_exclude_categories`), per-market
   direction-flip cooldown (`momentum_flip_cooldown_minutes`), and `new_market`
   requires a verifiable Gamma `created_at` younger than
   `new_market_max_age_hours`. ADVICE only — never traded automatically.

Trading-capable, but **LIVE execution is GATED OFF by default**. Every detected
opportunity is recorded as a **simulated** `polymarket_trades` row
(`is_simulated=true`); signals go to `polymarket_signals`. Mirrors the
ARBITRAGE module's shadow-first safety shape.

**Edge proof (wave-F5):** each cycle writes `polymarket_price_snapshots` for
the top-N watched markets and LATE-marks `polymarket_signal_outcomes` at
1h/6h/24h horizons (smart_money pattern: a horizon is written only after it
fully elapses, from the price observed at mark time). `fwd_return_*` is
probability points ×100 signed by direction — positive means the signal
pointed the right way. **Wave-F6 heartbeat:** the pipeline used to log only
on error (unverifiable from logs); it now emits one INFO line per cycle —
`polymarket outcomes: N marked, M snapshots written, K signals
pending-horizon (cumulative marked=...)` — so the operator can confirm the
scorecard is accumulating without DB access. Verify with
`grep 'polymarket outcomes:' logs/polymarket/polymarket.log`.

## Entry point
`modules/polymarket/main_polymarket.py` — launched as a subprocess by `main.py`
when `POLYMARKET_MODULE_ENABLED=true` (default **false**). Health server on port
8089 (env override: `POLYMARKET_HEALTH_PORT`).

## Dashboard (wave-F5 dedicated page set)
`/polymarket/dashboard` (gate-chain card + tiles + signals/hour + sim-PnL),
`/polymarket/markets` (snapshot catalog + per-market YES/NO history chart),
`/polymarket/signals` (stream + outcome columns + edge-over-time),
`/polymarket/positions` (net live exposure, honest shadow banner),
`/polymarket/performance` (expected-PnL curve + outcome-horizon scorecard),
plus the existing `/config/polymarket_config` editor. JSON under
`/api/polymarket/*` in `monitoring/enhanced_dashboard.py` — all read-only and
fail-soft. Operator guide: `docs/POLYMARKET_GUIDE.md`.

## LIVE-execution gate (all four required before any real CLOB order)
```
shadow_mode == false
AND live_execution_enabled == true
AND not core.dry_run.should_skip_live(module_dry_run, module='polymarket')
AND the built-in Polymarket risk gate passes
```
The risk gate is polymarket-appropriate (wave-F5 BUG-3 replaced the old
`core.risk_manager.validate_trade` call, which ran EVM honeypot analysis on
CLOB token ids): per-market notional cap (`max_market_exposure_usd`), total
exposure cap (`max_total_exposure_usd`), max open markets
(`max_open_markets`) — all fail-closed. Any gate failing → the trade is
recorded simulated with a `skip_reason`; no order is placed.

## Live path shape (executor.py)
- **Two-leg atomic-ish arb**: N = size/pair_cost shares of YES then NO,
  FOK preferred, place → poll → cancel-on-timeout (`order_fill_timeout_s`).
  Leg-2 failure → CRITICAL log + honest `live_leg2_failed` exposure record +
  immediate leg-1 unwind attempt (`live_unwound` / `live_unwind_failed`).
- **CLOB client**: `py-clob-client` lazy import; missing dep or key →
  `skip_reason='clob_client_unavailable'`, never raises. Optional API creds
  (`POLYMARKET_API_KEY/SECRET/PASSPHRASE`) used when all three resolve, else
  derived from the key. Proxy wallets: `signature_type` (1/2) +
  `funder_address` config keys.
- **Startup reconciliation** (live gates open only): lists resting CLOB
  orders and rebuilds the in-memory exposure map from the non-simulated
  ledger (last 30 days, BUY−SELL net).
- Shadow arb records carry `price = pair_cost` (both legs' true cost per $1
  redemption).

## Key config (DB-backed, config_type='polymarket_config'; migs 101 + 137)
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
| `arb_min_liquidity_usd` | 1000 | Arb book-honesty liquidity floor (mig 137) |
| `arb_max_edge_bps` | 500 | Stale-book cap: gross edge above = rejected (mig 137) |
| `momentum_min_liquidity_usd` | 10000 | Momentum liquidity floor |
| `momentum_min_volume_24h_usd` | 5000 | Momentum 24h-volume floor (seeded mig 137) |
| `momentum_min_move_frac` | 0.05 | Min YES-move to signal |
| `momentum_min_score` | 0.3 | Min momentum score to publish |
| `momentum_exclude_categories` | (empty) | Category denylist, e.g. `Sports` (mig 137) |
| `momentum_flip_cooldown_minutes` | 30 | Opposite-direction suppression window (mig 137) |
| `new_market_max_age_hours` | 24 | Max Gamma age for a `new_market` label (mig 137) |
| `shadow_record_interval_s` | 300 | Per (signal,market) record throttle |
| `max_market_exposure_usd` | 100 | Live risk gate: per-market notional cap (mig 137) |
| `max_total_exposure_usd` | 500 | Live risk gate: total exposure cap (mig 137) |
| `max_open_markets` | 10 | Live risk gate: distinct markets cap (mig 137) |
| `order_fill_timeout_s` | 30 | Live order poll window before cancel (mig 137) |
| `signature_type` / `funder_address` | (empty) | Proxy-wallet support (mig 137) |
| `snapshot_top_n_markets` | 50 | Markets snapshotted per cycle (mig 137) |
| `snapshot_retention_days` | 14 | Snapshot prune horizon (mig 137) |

## Secrets (only needed for LIVE, which is off by default)
`POLYMARKET_PRIVATE_KEY` (required for live) and optional
`POLYMARKET_API_KEY` / `POLYMARKET_API_SECRET` / `POLYMARKET_API_PASSPHRASE` —
all resolved via `security/secrets_manager` (encrypted DB) → `os.getenv`
fallback, fail-closed when missing. Never required in shadow mode.

## Kill switch
- **Global**: `logs/.killswitch` — polled via `core.dry_run.start_killswitch_poller`.
- **Per-module**: `logs/.pause_polymarket` — written by dashboard pause/resume.
- **Effect**: cycle idles; no order path reached.

## Logs
`logs/polymarket/` — `polymarket.log` (all), `polymarket_errors.log` (ERROR+).
Sub-loggers propagate to the parent handlers (no duplicate lines — wave-F5
BUG-6 removed the double-attach).

## DB tables (migrations 101 + 137)
- `polymarket_trades` — simulated (and, when enabled, live) trade ledger; `is_simulated` default TRUE.
- `polymarket_signals` — advice/signal stream (always simulated).
- `polymarket_price_snapshots` — per-cycle top-N market snapshots (mig 137).
- `polymarket_signal_outcomes` — LATE 1h/6h/24h forward marks per signal (mig 137).

## Data-source matrix
| Path | Source | Free? | Key |
|---|---|---|---|
| Market data (signals/arb) | Gamma API REST | YES (no key) | None |
| Live orders (gated off) | CLOB API via py-clob-client | order fees | `POLYMARKET_PRIVATE_KEY` |

## Dependencies
`py-clob-client` is added to the Docker image pip list (the image installs a
hardcoded list, NOT requirements.txt). Read-only Gamma needs only `aiohttp`
(already present). Self-tests (offline, fixture-backed):
`python -m modules.polymarket.strategies` and
`python -m modules.polymarket.gamma_client`.

## Isolation
Does NOT import or modify any trading module's engine/executor. Reuses
read-only: `core/dry_run.py` (killswitch/pause/dry-run) and
`security/secrets_manager.py` (own keys). Risk gating is built into
`modules/polymarket/executor.py`. Config: `polymarket_config` rows only.

## Known limitations (honest)
- Gamma `outcomePrices` are mids, not executable asks — shadow edges are an
  upper bound; the live path re-quotes nothing beyond its own FOK order price
  yet (book re-quote before order is the next hardening step).
- No settlement/redemption accounting: dashboard "simulated PnL" is expected
  edge, not realized money; resolution watcher + CTF `redeemPositions` are
  not built.
- No USDC balance / allowance preflight — the going-live checklist in
  `docs/POLYMARKET_GUIDE.md` covers the manual one-time approvals.
