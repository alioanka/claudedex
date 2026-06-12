# OPTIONS_VOL Module

## What it does
Crypto-options **hedging advisor** for **Deribit** (BTC/ETH only — alt option
books are uncrossable for a bot this size). Shadow/advisory-first. Each cycle:
1. Reads the fleet's **net directional exposure** (USD) from the same
   open-position tables `core/allocation_guard.py` uses (spot modules summed
   long, futures signed long-minus-short), scaled by `fleet_beta`.
2. Pulls the **free public Deribit chain** (no key), recomputes IV from mark
   prices with its own bisection (never trusts feed greeks), builds a
   per-expiry ATM surface, and computes trailing **realized vol** from hourly
   perp closes.
3. When fleet net long delta exceeds `hedge_delta_threshold_usd`, suggests a
   **protective put** (25-delta, 5–21d), or a **collar** (sell an OTM call to
   finance the put) when ATM IV / RV > `collar_min_ivrv`. Sized to cover
   `hedge_coverage_ratio` of the excess, hard-capped by
   `max_premium_per_hedge_usd`. Every leg is recorded **SIMULATED**
   (`options_vol_suggestions`, `is_simulated=true`).

## Honest note: why this may not be worth running
Per `docs/agents/NEW_MODULE_IDEAS.md` #6 — this module's economics are
genuinely skeptical territory:
- **Premium bleed**: systematic put-buying is negative-EV in calm regimes. If
  the monthly budget cap were not enforced, the hedge would eat more than the
  tail it insures. It buys *convexity*, not P&L.
- **Premium selling is the classic bot-killer** (pays daily, dies yearly) —
  the SELL live path is deliberately **not implemented** here; collar short
  calls are record-only.
- **Operational**: options margin + expiry/settlement is harder than perps; a
  missed expiry is an unhedged weekend. Live use needs an **isolated Deribit
  sub-account** so a margin error cannot touch spot/perp capital.
- With the current small fleet NAV, the hedgeable excess is tiny — run it in
  shadow to *measure* what hedging would have cost/saved before ever paying
  premium. If shadow shows the premium line persistently exceeding avoided
  drawdown, **turn it off**.

## Entry point
`modules/options_vol/main_options_vol.py` — subprocess; needs
`OPTIONS_VOL_MODULE_ENABLED=true` in `.env` (default **false**; main.py wiring
is owned by the orchestrator agent). Health server on port **8097**
(env override: `OPTIONS_VOL_HEALTH_PORT`).

## LIVE-execution gate (ALL required before any real Deribit order)
```
side == 'BUY'                       (SELL legs are ALWAYS record-only)
AND shadow_mode == false            (DB; default true)
AND live_execution_enabled == true  (DB; default false)
AND not core.dry_run.should_skip_live(module_dry_run, module='options_vol')
AND RiskManager.validate_trade(instrument, premium_usd) passes
AND month-to-date LIVE premium + this premium <= monthly_premium_budget_usd
      (counted from options_vol_suggestions; gate fails CLOSED on DB error)
AND ccxt deribit + DERIBIT_API_KEY/DERIBIT_API_SECRET resolvable
      (secrets_manager first, env fallback; missing -> simulated, never raises)
```
Any gate failing → simulated row with `skip_reason`; no order.

## Key config (DB-backed, config_type='options_vol'; migration 125)
| Key | Default | What it does |
|---|---|---|
| `shadow_mode` | true | Record-only; never reach the live order path |
| `live_execution_enabled` | false | Explicit LIVE opt-in (fail-safe off) |
| `premium_selling_enabled` | false | Reserved name only — selling live path not implemented |
| `poll_interval_s` | 300 | Cycle cadence |
| `max_requests_per_minute` | 20 | Deribit public API client-side cap |
| `currencies` | BTC,ETH | Surfaces built for these only |
| `hedge_currency` | BTC | Advisories issued in this currency only |
| `fleet_beta` | 1.0 | Maps alt-heavy book onto hedge currency |
| `rv_window_hours` | 720 | Realized-vol window (30d hourly) |
| `hedge_delta_threshold_usd` | 1000 | Net long delta above which to hedge the excess |
| `hedge_coverage_ratio` | 0.5 | Fraction of excess delta to offset |
| `put_target_delta` / `call_target_delta` | -0.25 / 0.25 | Leg selection targets |
| `tenor_min_days` / `tenor_max_days` | 5 / 21 | Short-dated window |
| `collar_min_ivrv` | 1.15 | IV/RV above this → collar instead of plain put |
| `min_open_interest` / `max_quote_spread_frac` | 10 / 0.25 | Liquidity filters |
| `max_premium_per_hedge_usd` | 25 | HARD per-hedge premium cap (scales size down) |
| `monthly_premium_budget_usd` | 50 | HARD monthly LIVE premium cap |
| `shadow_record_interval_s` | 3600 | Advisory record throttle |
| `surface_snapshot_interval_s` | 3600 | Surface snapshot throttle |
| `deribit_base_url` | https://www.deribit.com | Point at test.deribit.com to rehearse |

## Secrets (only needed for LIVE, which is off by default)
`DERIBIT_API_KEY` / `DERIBIT_API_SECRET` — `security/secrets_manager`
(encrypted DB) → `os.getenv` fallback. Never required in shadow mode.

## Kill switch
- **Global**: `logs/.killswitch` — polled via `core.dry_run.start_killswitch_poller`.
- **Per-module**: `logs/.pause_options_vol` — cycle idles, no order path reached.

## Logs
`logs/options_vol/` — `options_vol.log` (all), `options_vol_errors.log` (ERROR+).

## DB tables (migration 125)
- `options_vol_suggestions` — one row per hedge LEG (collars share
  `structure_id`); `is_simulated` default TRUE; signed `premium_usd`.
- `options_vol_surface` — per-expiry ATM IV vs RV history (signal audit trail).

## Self-tests (offline, deterministic, no network)
- `python -m modules.options_vol.vol_math` — Black-76 parity, IV round-trip,
  greeks signs, premium-cap sizing.
- `python -m modules.options_vol.hedge_advisor` — synthetic-chain IV recovery,
  put/collar regime switch, flat-book no-op, illiquid-chain refusal.

## Data-source matrix
| Path | Source | Free? | Key |
|---|---|---|---|
| Chain/index/RV (advisory) | Deribit public REST | YES (no key) | None |
| Live orders (gated off) | Deribit private via ccxt | trading fees | `DERIBIT_API_KEY`/`SECRET` |

## Isolation
Does NOT import or modify any trading module's engine/executor. Reuses
read-only: `core/dry_run.py`, `core/risk_manager.py`,
`core/allocation_guard.get_all_committed`, `security/secrets_manager.py`.
Config: `options_vol` rows only. No LLM spend anywhere in this module.
