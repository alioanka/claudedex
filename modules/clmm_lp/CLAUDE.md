# CLMM_LP Module

## What it does
SHADOW/ADVISORY-FIRST concentrated-liquidity market-making (Uniswap v3 EVM /
Orca Solana). Polls **free public pool data** (DexScreener REST, no key; plus a
one-time on-chain Uniswap v3 `fee()` verification via `config/pool_engine`,
READ-ONLY) for a small configured candidate set, proposes symmetric range
positions, and records them **simulated** (`clmm_shadow_positions`,
`is_simulated=true`) with a transparent net-of-IL expected APR. Open shadow
positions are marked against the **HODL benchmark every cycle** — IL is booked
continuously, never accrued silently.

## Honesty disclaimer (read before trusting any number)
The source idea doc (`docs/agents/NEW_MODULE_IDEAS.md` §9) is explicitly
skeptical of this module and ranks it "Weak now — defer": an LP position is a
**short-gamma, short-vol position paid in fees**, fee APR looks great while IL
accrues unrealized, and informed flow picks off stale ranges. **This module may
not be worth running.** The shadow track exists precisely to prove or disprove
the edge out-of-sample before a single dollar is minted. If shadow `net_usd`
is not consistently positive over weeks, the honest action is to leave it off.

## Net-of-IL expected return model (pure, self-tested)
`modules/clmm_lp/fee_il_math.py` — offline self-test:
`python -m modules.clmm_lp.fee_il_math`.
```
M        = 2*sqrt(P) / (2*sqrt(P) - P/sqrt(Pb) - sqrt(Pa))   # concentration factor
fee_apr  = M * (volume_24h * fee_rate / TVL * 365) * fee_decay_factor
il_apr   = M * sigma^2 / 8                                   # LVR run-rate
rebal    = sigma^2/(ln(P/Pa)*ln(Pb/P)) * rebalance_cost_frac # expected exits/yr * cost
net_apr  = fee_apr - il_apr - rebal
```
Fees and IL scale with the SAME `M`, so the model cannot present a narrow range
as free yield. Realized vol (quadratic-variation estimator on poll prices) is
**floored at `default_annual_vol` while samples are sparse** — underestimating
vol overstates net APR. Mark-to-market: `il_usd = position_value(P) -
hodl_value(P)`; fees accrue only while in range at the current pool fee
run-rate; every simulated close books `rebalance_cost_bps`.

## Entry point
`modules/clmm_lp/main_clmm_lp.py` — launched by `main.py` when
`CLMM_LP_MODULE_ENABLED=true` (default **false**; wiring not yet in main.py —
see repo-level report). Health server on port **8100**
(env override: `CLMM_LP_HEALTH_PORT`).

## LIVE-execution gate (v1 cannot trade at all)
All four required, in order (`modules/clmm_lp/executor.py`):
```
shadow_mode == false
AND live_execution_enabled == true
AND not core.dry_run.should_skip_live(module_dry_run, module='clmm_lp')
AND RiskManager.validate_trade(pool_address, size_usd) passes
```
Even all-green, v1 records simulated with
`skip_reason='live_path_not_implemented'` — there is **no mint/rebalance/burn
transaction code** by design until the shadow track earns it.

## Key config (DB-backed, config_type='clmm_lp'; migration 128)
| Key | Default | What it does |
|---|---|---|
| `shadow_mode` | true | Record-only; never reach a live path |
| `live_execution_enabled` | false | Explicit LIVE opt-in (and v1 still refuses) |
| `poll_interval_s` | 300 | Cycle cadence |
| `candidate_pools` | 2 pools (JSON) | chain / pool_address / fee_rate_bps / label; quote must be USD-stable |
| `range_width_pct` | 10 | Symmetric half-width around spot |
| `min_net_apr_pct` | 10 | Proposal floor on net-of-IL APR |
| `fee_decay_factor` | 0.7 | Haircut on the spot fee run-rate |
| `default_annual_vol` | 0.8 | Vol bootstrap + sparse-sample floor |
| `min_vol_samples` | 12 | Returns needed before realized vol is used |
| `rebalance_cost_bps` | 30 | Cost per simulated re-range (charged at close) |
| `max_position_size_usd` | 100 | Per-position notional cap |
| `max_open_positions` | 2 | Doc guidance: max 2 pools in v1 |
| `shadow_record_interval_s` | 3600 | Per-pool proposal throttle |
| `max_position_age_hours` | 168 | Forced close (fee decay cannot hide) |
| `out_of_range_exit_buffer_pct` | 2 | Exit buffer beyond range edge |
| `dexscreener_base_url` | https://api.dexscreener.com | Free pool data |

## Kill switch
- Global: `logs/.killswitch` — polled via `core.dry_run.start_killswitch_poller`;
  also blocks the (unimplemented) live path via `should_skip_live`.
- Per-module: `logs/.pause_clmm_lp` — cycle idles.

## Logs
`logs/clmm_lp/` — `clmm_lp.log` (all), `clmm_lp_errors.log` (ERROR+).

## DB tables (migration 128)
`clmm_shadow_positions` — proposal + per-cycle mark ledger (`fees_usd`,
`il_usd`, `net_usd`, `close_reason`); `is_simulated` default TRUE.

## Data-source matrix
| Path | Source | Free? | Key |
|---|---|---|---|
| Price / volume / TVL | DexScreener REST | YES | None |
| EVM fee-tier verify | `eth_call fee()` via PoolEngine.get_endpoint | YES (read-only) | None |
| Live mint/burn | — none in v1 — | — | — |

## Isolation
Does NOT import or modify any other module. Reuses read-only:
`core/dry_run.py`, `core/risk_manager.py` (gate only), `config/pool_engine.py`
(endpoints + report_success/failure/rate_limit; never sends transactions).
No LLM spend, no private keys, no secrets needed in shadow mode.
Config: `clmm_lp` rows only. Dependencies: `aiohttp`, `asyncpg`,
`python-dotenv` (all already in the image).
