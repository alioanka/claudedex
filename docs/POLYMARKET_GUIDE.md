# Polymarket Module — Operator Guide

Shadow-first prediction-market engine (Polygon PoS, USDC.e). This guide covers
day-to-day shadow operation, every tunable knob, the evidence bar a signal
stream must clear before any live flag is touched, the going-live checklist,
and the incident playbook.

Module docs: `modules/polymarket/CLAUDE.md`. Analysis that drove the wave-F5
fixes: `docs/agents/wave-f5/05_polymarket.md`.

---

## 1. What this module is

- Polls the free read-only **Gamma API** every `poll_interval_s` (60 s) for the
  top-volume active binary markets.
- Runs two strategies:
  - **risk_free_arb** — YES+NO priced below $1 by more than the fee buffer,
    behind book-honesty gates (liquidity, volume, live uncrossed bid/ask,
    stale-book cap). Recording is throttled per (signal, market).
  - **momentum / new_market** — directional ADVICE on liquid markets with a
    large short-window YES move; never traded automatically.
- Records everything simulated: `polymarket_signals` + `polymarket_trades`
  (`is_simulated=true`), plus per-cycle `polymarket_price_snapshots` and LATE
  forward-outcome marks in `polymarket_signal_outcomes` (1h/6h/24h).
- **LIVE CLOB execution is OFF by default** behind the full gate chain:
  `shadow_mode=false` AND `live_execution_enabled=true` AND
  `should_skip_live` clear (no DRY_RUN, no killswitch, no pause) AND the
  built-in exposure risk gate.

"Simulated" means: the executor resolved the gate chain, at least one gate
blocked, and an honest ledger row was written with the blocking `skip_reason`
(`shadow_mode`, `live_execution_disabled`, `risk:market_exposure_cap`,
`clob_client_unavailable`, …). Nothing was sent anywhere.

## 2. Running in shadow

1. `.env`: `POLYMARKET_MODULE_ENABLED=true` (default false), then
   `python main.py` — or enable from the dashboard `/modules` page.
2. Apply migrations 101 + 137 (both idempotent).
3. Health: `curl http://localhost:8089/status` (port override
   `POLYMARKET_HEALTH_PORT`). Includes gate state, cycle stats,
   `live_exposure_usd`, `outcomes_marked`.
4. Logs: `logs/polymarket/polymarket.log` (rotating),
   `polymarket_errors.log` (ERROR+).
5. Pause: `logs/.pause_polymarket` (dashboard pause button writes it).
   Global stop: `logs/.killswitch`.
6. Watch it on the dedicated pages: `/polymarket/dashboard` (gate chain +
   tiles + charts), `/polymarket/markets`, `/polymarket/signals`,
   `/polymarket/positions`, `/polymarket/performance`, and Settings →
   `/config/polymarket_config`.

## 3. Knobs (config_type `polymarket_config`, editable in Settings)

### Signal quality
| Key | Default | Effect |
|---|---|---|
| `min_arb_edge_bps` | 100 | Net arb floor after the fee buffer |
| `fee_gas_buffer_bps` | 100 | Haircut applied to every gross arb edge |
| `arb_min_liquidity_usd` | 1000 | Markets below this liquidity never arb-signal |
| `arb_max_edge_bps` | 500 | Gross edge above this = stale/dead book, rejected |
| `momentum_min_liquidity_usd` | 10000 | Momentum liquidity floor |
| `momentum_min_volume_24h_usd` | 5000 | Momentum 24h-volume floor |
| `momentum_min_move_frac` | 0.05 | Min YES move per poll window to signal |
| `momentum_min_score` | 0.3 | Publication floor for the [0,1) score |
| `momentum_exclude_categories` | (empty) | Comma denylist — set `Sports` to drop in-play game noise |
| `momentum_flip_cooldown_minutes` | 30 | Suppress opposite-direction re-signals per market |
| `new_market_max_age_hours` | 24 | `new_market` requires Gamma created_at younger than this |
| `category_filter` | (empty) | Hard allowlist applied at fetch time (empty = all) |
| `max_markets_per_poll` | 200 | Universe size per cycle |
| `shadow_record_interval_s` | 300 | Per (signal, market) row throttle |

**Recommended shadow posture:** set `momentum_exclude_categories=Sports`
(in-play games dominate the top-volume window and flip direction within
minutes), or raise `momentum_min_move_frac` to 0.08–0.10 if the stream is
still noisy. Leave the arb gates at defaults — an arb signal should be rare;
~1/day that survives the gates is more informative than 990 momentum rows.

### Outcome tracking / dashboard data
| Key | Default | Effect |
|---|---|---|
| `snapshot_top_n_markets` | 50 | Markets snapshotted per cycle (feeds charts + marks) |
| `snapshot_retention_days` | 14 | Snapshot prune horizon (hourly prune) |

### Live-path safety (only consulted after the flag gates open)
| Key | Default | Effect |
|---|---|---|
| `max_position_size_usd` | 50 | Per-trade notional cap (both legs of a pair) |
| `max_market_exposure_usd` | 100 | Risk gate: max live notional per market |
| `max_total_exposure_usd` | 500 | Risk gate: max total live notional |
| `max_open_markets` | 10 | Risk gate: max distinct markets with exposure |
| `order_fill_timeout_s` | 30 | Poll window before an unfilled order is cancelled |
| `signature_type` | (empty) | Empty/0 = EOA wallet; 1/2 = Polymarket proxy wallet |
| `funder_address` | (empty) | Proxy (funder) address for signature_type 1/2 |
| `clob_base_url`, `chain_id` | clob.polymarket.com, 137 | Live endpoints |

Locale note: enter decimals with a dot (`0.05`), never a comma — a comma
value fails float parsing and stops signal cycles until corrected.

## 4. Proving edge before live (acceptance criteria)

The outcome tracker is the evidence. Read `/polymarket/performance` →
Outcome-Horizon Scorecard (`polymarket_signal_outcomes`; `fwd_return_*` is in
probability points, signed by direction — positive = the signal was right).

Do not touch any live flag until ALL of the following hold, out-of-sample
(i.e. measured on signals generated AFTER the last knob change):

- **Arb**: every arb detection over ≥30 days, manually re-quoted against the
  real CLOB book (Polymarket UI order book is enough), would still have been
  net-positive after fees at executable asks. Zero detections is an acceptable
  outcome; a false positive is not.
- **Momentum**: ≥200 fully-marked signals with `hit_24h > 0.55` AND
  `avg_24h > 0` — sustained, not a single lucky week.
- **new_market**: treat as WATCH-only forever unless it separately clears the
  same bar.
- The signals/hour chart shows the stream is not dominated by one category or
  a handful of market_ids (concentration = one game's noise, not edge).

If momentum never clears the bar, that is a fine result: keep the module as a
free data/advice layer and only ever consider the arb leg for live.

## 5. Going-live checklist (in order, no skipping)

1. **Dedicated Polygon wallet.** Fresh EOA used for nothing else. Decide
   EOA vs Polymarket proxy wallet: if the funds live in a Polymarket-UI
   account, set `signature_type` (1 = email/magic, 2 = browser wallet proxy)
   and `funder_address` to the proxy address; a bare EOA leaves both empty.
2. **Fund it**: POL for gas plus a SMALL amount of USDC.e
   (`0x2791Bca1...4174` — verify against `usdc_address` in config).
3. **One-time approvals** (EOA path; the proxy path already has them):
   approve USDC.e to the CTF Exchange (`ctf_exchange_address`) and
   `setApprovalForAll` on the CTF ERC-1155 (`ctf_address`). Without these the
   first order fails at the exchange.
4. **Store secrets** via the dashboard credentials page / secrets manager:
   `POLYMARKET_PRIVATE_KEY` (required). Optional API creds
   (`POLYMARKET_API_KEY/SECRET/PASSPHRASE`) — if absent they are derived from
   the key on first client init.
5. **Verify the client**: with flags still shadow, check
   `logs/polymarket/polymarket.log` has no
   `POLYMARKET_PRIVATE_KEY not configured` / `py-clob-client not installed`
   warnings, and recent trades do not show
   `skip_reason='clob_client_unavailable'` once flags open.
6. **Set risk caps LOW**: `max_position_size_usd=10`,
   `max_market_exposure_usd=20`, `max_total_exposure_usd=50`,
   `max_open_markets=2` for the first week.
7. **Flip the flags in this order**: `shadow_mode=false`, then
   `live_execution_enabled=true`. Ensure `POLYMARKET_DRY_RUN`/`DRY_RUN` do not
   force dry-run and no pause/killswitch file exists.
8. **Restart the module** and confirm the startup reconcile lines: it must
   report open CLOB orders (expected 0) and the rebuilt exposure map.
9. **Watch the first trade end-to-end** on `/polymarket/positions`: statuses
   `live_filled` (good), `live_failed` (leg 1 never filled — harmless),
   `live_leg2_failed` / `live_unwind_failed` (see playbook below).

## 6. Live operations

- **Order lifecycle**: place (FOK preferred) → poll up to
  `order_fill_timeout_s` → cancel on timeout. Ledger statuses:
  `live_filled`, `live_failed` (nothing filled), `live_leg2_failed`
  (unhedged YES recorded honestly), `live_unwound` (leg-1 sold back),
  `live_unwind_failed` (operator action required).
- **Restart reconciliation** runs automatically when live gates are open:
  resting CLOB orders are logged loudly; exposure is rebuilt from the ledger
  (BUY−SELL net, 30-day window).
- **Settlement**: NOT automated. Winning YES+NO pairs redeem $1/share via CTF
  `redeemPositions` — currently a manual step (Polymarket UI is fine).
  Dashboard PnL stays "expected" until redemption accounting exists.
- **Fees**: taker fees + Polygon gas are approximated by
  `fee_gas_buffer_bps`; verify against the first real fills and raise the
  buffer if reality is worse.

## 7. Incident playbook

| Situation | Action | Effect |
|---|---|---|
| Stop THIS module now | create `logs/.pause_polymarket` (dashboard pause) | Cycle idles; no orders; other modules unaffected |
| Stop new LIVE orders, keep shadow running | set `live_execution_enabled=false` (Settings) | Next executor call records simulated again |
| Stop EVERYTHING | `logs/.killswitch` via emergency-stop | All modules' live paths blocked |
| `live_leg2_failed` in the ledger | Check the paired `live_unwound` row. If instead `live_unwind_failed`: you hold unhedged YES shares | Sell the position manually (Polymarket UI) or hold to resolution; the CRITICAL log line has shares/size |
| Suspect resting orders after a crash/halt | restart the module (reconcile lists open orders) or check the account's open orders in the UI | Never assume flat after a halt — verify |
| Gamma timeouts / `No markets fetched` | transient; engine fail-softs and retries next cycle | Investigate only if `gamma_last_error` persists across many cycles |
| Config typo stops cycles (`could not convert string to float`) | fix the value in Settings (dot decimals) | Cycles resume next poll |

Order of preference when unsure: pause file (surgical, reversible) →
`live_execution_enabled=false` (kills live only) → killswitch (everything).

## 8. Known limitations

- **Gamma mids ≠ executable asks.** Shadow edges are upper bounds; the live
  order uses the detector's price with FOK (fills entirely at that price or
  not at all), but a full order-book re-quote before ordering is not built.
- **In-play sports noise** dominates the top-volume universe; use the
  category denylist.
- **UMA resolution risk**: a "resolved" market can be disputed; near-
  resolution filtering reduces but does not eliminate this.
- **Single venue**: all exposure is on one platform's CLOB + one L2.
- **No balance/allowance preflight**: an underfunded wallet fails at order
  time (recorded honestly), not before.
