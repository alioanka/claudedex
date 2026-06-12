# STAT_ARB Module

## What it does
Market-neutral **statistical pairs mean-reversion** on liquid USDT perps
(NEW_STRATEGY_BACKLOG.md item P1-2). Every cycle it:
1. fetches CLOSED-bar closes for the configured universe from FREE sources
   (market_data_warehouse reader if populated and fresh, else Bybit v5 /
   Binance USD-M public klines — no keys);
2. evaluates every candidate pair with pure math (`core/pair_math.py`):
   log-return correlation gate, OLS hedge ratio `ln(y)=alpha+beta*ln(x)`,
   spread z-score, AR(1) mean-reversion t-stat gate (Engle-Granger-style,
   threshold -2.9 ≈ 5% ADF critical — documented approximation, not a full
   ADF), and a half-life band filter;
3. records spread diagnostics to `stat_arb_spread_state` (throttled) and
   SHADOW-records paired suggestions (long the cheap leg / short the rich
   leg) to `stat_arb_trades` with `is_simulated=true`.

Edge source in one sentence: cointegrated/beta-stable alt pairs mean-revert
on hours-days horizons; this captures the band-reentry of the spread while
staying dollar-neutral to market direction. Fits the regime_allocator's
`range_compression` regime.

## Pair-break tail risk (the honesty section)
Crypto cointegration is **regime-fragile**: pairs break on idiosyncratic news
(unlock, hack, listing) and the loss on a broken pair is **fat-tailed** — the
spread does not have to come back, ever. The controls, all structural:
- **HARD z-stop** (`stop_z`, default 4.0): adverse |z| at/past it closes the
  pair immediately. Never widened. The favorable direction past the mean is a
  take-profit, not a stop.
- **NO averaging down by construction**: there is no code path that adds to
  or resizes an open pair — fixed per-leg notional in, one close out.
- **Entry refusal past the stop**: |z| ≥ stop_z while flat is treated as a
  pair break, not an opportunity.
- **Post-stop cooldown** (`pair_cooldown_hours`): a hard-stopped pair is
  barred from re-entry until a fresh window re-qualifies it.
- **Time stop** (`max_hold_hours`) + **retest-fail close**: stale or
  no-longer-evaluable pairs are cut, not nursed.
- Per-pair fixed notional + `max_concurrent_pairs` cap bound total exposure.
Simulated PnL is **pessimistic by design**: 4 fills per round trip, each
charged `taker_fee_bps + slippage_bps`. The shadow season (4+ weeks minimum)
IS the walk-forward/out-of-sample test — nothing is tuned in-sample on the
recorded ledger.

## Entry point
`modules/stat_arb/main_stat_arb.py` — launched as a subprocess by `main.py`
when `STAT_ARB_MODULE_ENABLED=true` (default **false**). Health server on port
8104 (`STAT_ARB_HEALTH_PORT`): `GET /health`, `GET /status` (cycle stats, open
pair keys, live/simulated counters). Pure math self-test (offline,
deterministic): `python -m modules.stat_arb.core.pair_math`.

## LIVE-execution gate (all four required before any real perp order)
```
shadow_mode == false
AND live_execution_enabled == true
AND not core.dry_run.should_skip_live(module_dry_run, module='stat_arb')
AND RiskManager.validate_trade(pair_key, 2 * notional_per_leg_usd) passes
```
Any one failing → the pair is recorded simulated with a `skip_reason`; no
order is placed. The live path (ccxt Bybit linear, lazy + fail-soft) is
two-leg atomic on entry: if the second leg fails, the first is flattened
reduce-only immediately — never run one-legged. Closes are reduce-only and
are never blocked by entry gates (close > hold), but a failed LIVE close
keeps the row open and logged rather than paper-closing it.

## Key config (DB-backed, config_type='stat_arb'; migration 132)
| Key | Default | What it does |
|---|---|---|
| `shadow_mode` | true | Record-only; never reach the live order path |
| `live_execution_enabled` | false | Explicit LIVE opt-in (fail-safe off) |
| `universe` | 10 liquid majors | CSV of perp symbols scanned pairwise |
| `pairs` | (empty) | Explicit `Y\|X` pair list; empty = scan all combos |
| `timeframe` / `lookback_bars` | 1h / 240 | Estimation window (10 days) |
| `min_correlation` | 0.6 | Gate 1: log-return Pearson floor |
| `adf_tstat_max` | -2.9 | Gate 2: AR(1) t-stat ceiling (mean reversion) |
| `min/max_half_life_bars` | 4 / 120 | Gate 3: tradable reversion-speed band |
| `entry_z` / `exit_z` / `stop_z` | 2.0 / 0.5 / 4.0 | Bands (stop_z > entry_z > exit_z enforced in code) |
| `max_hold_hours` | 96 | Time stop |
| `max_concurrent_pairs` | 3 | Open-pair cap |
| `notional_per_leg_usd` | 100 | Fixed per-leg size (never resized) |
| `pair_cooldown_hours` | 48 | Re-entry bar after a hard stop |
| `retest_fail_exits` | 6 | Cycles of failed evaluation before forced close |
| `taker_fee_bps` / `slippage_bps` | 5.5 / 3 | Cost model per fill (4 fills/round trip) |
| `poll_interval_s` | 300 | Cycle cadence |
| `spread_state_record_interval_s` | 900 | Diagnostics-row throttle per pair |
| `max_candidate_pairs` | 45 | Per-cycle evaluation cap |
| `request_spacing_s` | 0.25 | Free-API politeness spacing |

## Secrets (only needed for LIVE, which is off by default)
`BYBIT_API_KEY` / `BYBIT_API_SECRET` — resolved via `security/secrets_manager`
(encrypted DB) → `os.getenv` fallback. Never required in shadow mode.

## Kill switch
- **Global**: `logs/.killswitch` — polled via `core.dry_run.start_killswitch_poller`;
  also enforced inside the executor gate chain via `should_skip_live`.
- **Per-module**: `logs/.pause_stat_arb` — cycle idles; no order path reached.

## Logs
`logs/stat_arb/` — `stat_arb.log` (all), `stat_arb_errors.log` (ERROR+).

## DB tables (migration 132)
- `stat_arb_trades` — paired suggestions / (future) live ledger; one row per
  round trip; `is_simulated` default TRUE; restart reconcile reloads
  `status='open'` rows so a subprocess restart never orphans a pair.
- `stat_arb_spread_state` — per-pair diagnostics (alpha/beta/z/t-stat/
  half-life/tradable + first failed gate), throttled.

## Data-source matrix
| Path | Source | Free? | Key |
|---|---|---|---|
| Closes (signals) | warehouse reader → Bybit v5 → Binance USD-M public | YES | None |
| Live orders (gated off) | Bybit linear via ccxt | taker fees | `BYBIT_API_KEY/SECRET` |

## Isolation
Does NOT import or modify any trading module's engine/executor. Reuses
read-only: `core/dry_run.py`, `core/risk_manager.py` (live gate),
`security/secrets_manager.py`, `modules/market_data_warehouse/reader.py`
(fail-soft consumer API). Config: `stat_arb` rows only. Writes only
`stat_arb_trades` + `stat_arb_spread_state`. No LLM spend anywhere.

## Known scope cuts (documented follow-ups, not defects)
- Cross-sectional variant (short the day's biggest beta-adjusted outlier vs
  basket) is NOT built — pairs only for now; same math file would host it.
- Funding-rate drag is in the cost model only via the flat slippage haircut;
  explicit per-pair funding netting (warehouse `get_funding`) is a follow-up.
- Live sizing is dollar-neutral, not beta-neutral; beta is used for the
  spread, acceptable at the seeded notional, revisit before any size-up.

## Orchestrator wiring still needed (NOT done by this module — report items)
- `main.py`: launch `modules/stat_arb/main_stat_arb.py` when
  `STAT_ARB_MODULE_ENABLED=true` (mirror the meta_controller block).
- `.env.example`: `STAT_ARB_MODULE_ENABLED=false`, `STAT_ARB_DRY_RUN=true`,
  `STAT_ARB_HEALTH_PORT=8104`.
- Root `CLAUDE.md` module table + health-port map: 8104 = stat_arb.
- Dashboard (dashboard agent): surface the migration-132 knobs in Settings;
  read-only panels on `stat_arb_trades` / `stat_arb_spread_state` + the
  :8104 `/status` heartbeat. Fail-soft when tables absent.
- No new Dockerfile deps: aiohttp, asyncpg, ccxt already in the image.
