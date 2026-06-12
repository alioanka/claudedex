# BASIS_DESK Module

## What it does
Cross-venue funding/basis desk — **delta-neutral carry ADVISOR** (shadow-first,
advice-only). Polls free key-less public REST data (Bybit V5 linear+spot
tickers, Binance premiumIndex + spot ticker) for a configured symbol universe,
and for each (venue, symbol) costs out the **complete hedged structure**:

- funding > 0 (longs pay shorts) → **SHORT perp + LONG spot** (no borrow)
- funding < 0 (shorts pay longs) → **LONG perp + SHORT spot** (needs spot
  borrow; gated by `allow_short_spot`, default false — evaluated and recorded
  non-actionable so the operator can see the missed side)

This is the missing piece of futures funding-carry v2 (mig 102), which opens
the perp leg only and leaves the spot hedge as a manual operator action.
This module ADVISES both legs as one structure; it places **no orders**.

## Entry point
`modules/basis_desk/main_basis_desk.py` — launched as a subprocess by
`main.py` when `BASIS_DESK_MODULE_ENABLED=true` (default **false**; wiring is
owned by the orchestrator agent). Health server on port **8103**
(env override: `BASIS_DESK_HEALTH_PORT`); `GET /health` == `GET /status`.

## Carry / breakeven math (pure, self-tested)
`modules/basis_desk/carry_math.py` — no I/O, no engine imports.
Self-test: `python -m modules.basis_desk.carry_math`.
```
round_trip_cost_bps = 2*perp_taker_fee_bps + 2*spot_taker_fee_bps
                    + perp_slippage_bps + spot_slippage_bps
                    + liquidation_premium_bps           (defaults: 44 bps)
adverse_basis_cost_bps = max(0, -favorable_basis_bps)   (favorable basis is
                                                         NEVER credited)
total_cost_bps      = round_trip_cost_bps + adverse_basis_cost_bps
breakeven_intervals = total_cost_bps / |funding_bps|
net_carry_bps(H)    = |funding_bps| * H - total_cost_bps
```
Gates: `net_carry_bps(horizon_intervals) >= min_net_carry_bps` AND
`breakeven_intervals <= max_breakeven_intervals` AND borrow feasibility.
Entry quality: `ConfirmTracker` requires `confirm_polls` consecutive
actionable polls (the basis_desk analogue of funding-carry v2's stability
window). Hedge sizing: equal BASE quantity both legs
(`hedge_legs`), reconciliation predicate `legs_balanced` with
`hedge_epsilon_frac`.

## LIVE-execution gate (chain present, final hop intentionally unwired)
Evaluated in order in `modules/basis_desk/executor.py`; the recorded
`skip_reason` is always the FIRST failed gate:
```
shadow_mode == false                       (DB; default true)
AND live_execution_enabled == true         (DB; default false)
AND not core.dry_run.should_skip_live(module_dry_run, module='basis_desk')
AND RiskManager.validate_trade(symbol, notional) passes
AND <live order placement>                 -- NOT IMPLEMENTED BY DESIGN:
                                           -- records status='live_blocked',
                                           -- skip_reason='live_path_not_implemented'
```
Every recorded row is `is_simulated=true`. Wiring real two-leg execution
(perp via the futures module, spot via Bybit V5 spot) is a separate,
explicitly-reviewed change.

## Key config (DB-backed, config_type='basis_desk'; migration 131)
| Key | Default | What it does |
|---|---|---|
| `shadow_mode` | true | Record-only; never reach the (unwired) live path |
| `live_execution_enabled` | false | Explicit LIVE opt-in (fail-safe off) |
| `poll_interval_s` | 300 | Cycle cadence |
| `symbols` | BTC,ETH,SOL,XRP,DOGE,BNB (USDT) | Symbol universe |
| `venues` | bybit,binance | Venue allowlist |
| `funding_interval_hours` | 8 | Funding interval assumption |
| `min_net_carry_bps` | 10 | Net-carry-at-horizon suggestion floor |
| `horizon_intervals` | 6 | Holding horizon for the net gate (2 days) |
| `max_breakeven_intervals` | 3 | Max intervals to cover round-trip costs |
| `perp_taker_fee_bps` / `spot_taker_fee_bps` | 6 / 10 | Per-fill fees |
| `perp_slippage_bps` / `spot_slippage_bps` | 5 / 5 | Round-trip slippage model |
| `liquidation_premium_bps` | 2 | Margin-buffer reserve on every structure |
| `max_notional_usd` | 200 | Per-structure suggested notional cap |
| `allow_short_spot` | false | Enable negative-funding direction (needs borrow) |
| `confirm_polls` | 2 | Consecutive actionable polls before recording |
| `suggest_interval_s` | 1800 | Per (venue,symbol) record throttle |
| `hedge_epsilon_frac` | 0.02 | Live-book leg-balance tolerance (see below) |

## Risk notes (read before any future live wiring)
- **Funding decays exactly when crowded.** Funding is a risk premium paid by
  the crowded side; when everyone harvests the carry, the rate compresses
  toward (or through) the cost model. Expect long flat stretches; the
  confirm gate + breakeven cap reject one-print spikes, but a position
  opened on real persistence can still see its edge vanish — exit when the
  live rate stops paying, do not wait for the horizon.
- **Spot-hedge operational note.** The structure is delta-neutral ONLY while
  both legs exist at equal base quantity. Place BOTH legs or NEITHER
  (leg-out on entry = naked directional perp). On a live book, reconcile
  every tick: `|perp_qty - spot_qty| / perp_qty <= hedge_epsilon_frac`,
  else FLATTEN (never resize into a moving market). An exchange outage
  mid-position turns a neutral book directional — kill-to-flat on
  hedge-leg failure. The perp leg can still be liquidated while hedged
  (margin lives on the perp side only) — hence `liquidation_premium_bps`
  and ISOLATED margin if/when executed.
- **Short-spot direction needs borrow** the stack does not have; it stays
  advisory-only behind `allow_short_spot=false`.
- **Basis is not guaranteed to converge** on a perp (no expiry): the model
  charges adverse entry basis in full and never credits favorable basis.

## Kill switch
- Global: `logs/.killswitch` — polled via `core.dry_run.start_killswitch_poller`.
- Per-module: `logs/.pause_basis_desk` — cycle idles; nothing is recorded.

## Logs
`logs/basis_desk/` — `basis_desk.log` (all), `basis_desk_errors.log` (ERROR+).

## DB tables (migration 131)
- `basis_carry_suggestions` — advisory ledger, `is_simulated` default TRUE;
  one row per confirmed actionable structure (throttled), with the full cost
  decomposition so the shadow track is auditable.

## Data-source matrix
| Path | Source | Free? | Key |
|---|---|---|---|
| Perp funding + mark (Bybit) | `/v5/market/tickers?category=linear` | YES | None |
| Spot last (Bybit) | `/v5/market/tickers?category=spot` | YES | None |
| Perp funding + mark (Binance) | `/fapi/v1/premiumIndex` | YES | None |
| Spot last (Binance) | `/api/v3/ticker/price` | YES | None |

No paid APIs, no LLM spend. Dependencies: `aiohttp`, `asyncpg`, `python-dotenv`
(all already in the image — no Dockerfile change needed).

## Isolation
Does NOT import or modify any trading module's engine/executor (explicitly
not `modules/futures_trading/`). Reuses read-only: `core/dry_run.py`
(killswitch/pause/dry-run), `core/risk_manager.py` (gate in the unwired live
chain). Config: `config_type='basis_desk'` rows only. Writes only
`basis_carry_suggestions`.
