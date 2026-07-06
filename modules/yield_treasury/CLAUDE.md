# YIELD_TREASURY Module (ADVISORY — observe-first)

## What it does
Idle-capital carry **advisor**. Every tick it reads the bot's idle float from
fresh `treasury_snapshots` (the treasury module's observer output), reads
blue-chip yields via FREE sources (Aave v3 USDC supply APR by `eth_call`
`getReserveData`; jitoSOL/stETH via free public APIs), runs pure carry math,
and writes one `yield_treasury_advice` row per allowlisted venue. It **NEVER
deposits, NEVER signs, NEVER touches `logs/.killswitch`**.

## Honesty (from `docs/agents/NEW_MODULE_IDEAS.md` idea #7)
Upside is **LOW**: on a $50k idle float this is ~$2-3k/yr; on this bot's
current float it is far less, and the self-tests prove a $250 float on
Ethereum mainnet is an honest HOLD (gas never pays back). This module must
never be prioritized over a safety or measurement module — it exists as desk
discipline ("never hold unremunerated cash"), recorded transparently,
including all the ticks where the answer is "carry does not pay".

## Entry point
`modules/yield_treasury/main_yield_treasury.py` — launched by `main.py` when
`YIELD_TREASURY_MODULE_ENABLED=true` (default **false**). Health server on
port 8098 (`YIELD_TREASURY_HEALTH_PORT`): `/health` liveness, `/status`
last-tick summary. Engine: `core/yield_engine.py` (self-tested:
`python -m modules.yield_treasury.core.yield_engine`). APR readers:
`core/yield_sources.py`. Pure math: `core/yield_math.py` (both self-tested
the same way).

## Venue allowlist (HARD-CODED in `core/yield_sources.py` — code review to extend, never config)
| Venue | Chain(s) | Asset | Recall latency (default) |
|---|---|---|---|
| `aave_v3` | ethereum, arbitrum, base | USDC supply | 120 s (one withdraw tx) |
| `jito_jitosol` | solana | SOL -> jitoSOL | 120 s (Jupiter swap-out) |
| `lido_steth` | ethereum | ETH -> stETH | 259200 s (withdrawal queue) — always HOLD under the default 1-day cap, by design |

## The math (pure, `core/yield_math.py`)
```
touchable   = max(0, idle - undeployed_floor)            # gas + trading float untouchable
deployable  = min(touchable * max_deploy_frac,           # fleet cap
                  idle * venue_cap_frac)                 # per-venue SC-risk cap
cost        = fixed_gas + deployable * fee_bps/10000     # deposit + withdraw
gross_daily = deployable * apr / 365
breakeven   = cost / gross_daily                         # days; None if no yield
net_horizon = gross_daily * horizon_days - cost
DEPLOY_CANDIDATE iff deployable >= min AND apr > 0
    AND withdrawal_latency_s <= max_withdrawal_latency_s
    AND breakeven <= max_breakeven_days AND net_horizon > 0
```
**Latency caveat:** every advice row carries `latency_note` — recalled capital
is unavailable for `withdrawal_latency_s`; while parked it cannot fund trading
entries/exits or gas. Venues slower than `max_withdrawal_latency_s` are HOLD
regardless of APR.

## Live path (NOT the default — and not built)
`resolve_live_skip_reason` mirrors the polymarket gate chain, in order:
`shadow_mode=false` -> `live_execution_enabled=true` ->
`should_skip_live(module='yield_treasury')` passes (no DRY_RUN via
`YIELD_TREASURY_DRY_RUN`, no killswitch, no pause) -> `RiskManager.validate_trade`
approves. Even with every gate open it returns `live_deposit_path_not_built`:
actual deposit/withdraw wiring is a Phase-2 build, gated on treasury Phase 2
being proven, per the ideas doc. All rows are written `shadow=true`.

## Key config (DB-backed, config_type='yield_treasury'; migration 126)
| Key | Default | What it does |
|---|---|---|
| `poll_interval_seconds` | 900 | Tick cadence |
| `shadow_mode` / `live_execution_enabled` | true / false | Live gates (fail-safe) |
| `min_idle_usd` / `min_idle_sol` / `min_idle_eth` | 250 / 1.0 / 0.05 | HOLD below these deployables |
| `max_deploy_frac` / `venue_cap_frac` | 0.5 / 0.25 | Fleet + per-venue caps |
| `undeployed_floor_usd` / `_sol` / `_eth` | 100 / 0.2 / 0.02 | Never-touch float |
| `horizon_days` / `max_breakeven_days` | 30 / 10 | Net-benefit window + gate |
| `max_withdrawal_latency_s` | 86400 | Hard recall-latency cap |
| `roundtrip_cost_usd_<chain>` | 16 / 0.30 / 0.20 | Gas assumptions (eth/arb/base) |
| `lst_roundtrip_bps` / `sol_tx_cost_sol` / `eth_lst_roundtrip_cost_eth` | 10 / 0.001 / 0.004 | LST roundtrip costs |
| `withdrawal_latency_s_aave` / `_jitosol` / `_steth` | 120 / 120 / 259200 | Recall latencies |
| `lido_apr_url` / `jitosol_apy_url` | Lido / Sanctum free endpoints | APR feeds (failure = HOLD `apr_unavailable`) |

Env knobs: `YIELD_TREASURY_MODULE_ENABLED` (gate, default false),
`YIELD_TREASURY_HEALTH_PORT` (8098), `YIELD_TREASURY_POLL_INTERVAL`
(pre-migration fallback), `YIELD_TREASURY_DRY_RUN` (default true).

## Kill switch
- Global: `logs/.killswitch` — tick skipped (this module only READS the flag).
- Per-module: `logs/.pause_yield_treasury` — tick skipped.

## Logs
`logs/yield_treasury/` — `yield_treasury.log` (INFO),
`yield_treasury_errors.log` (WARNING+, so YIELD ADVICE lines land there).

## DB tables
- `yield_treasury_advice` (migration 126) — one row per venue per tick: APR,
  idle, deployable, daily/horizon carry, breakeven, latency note,
  recommendation + reason, `shadow` flag, details.
- Reads `treasury_snapshots` (mig 121) for idle balances — without a running
  treasury module (fresh rows < 2h) the tick is an idle no-op, fail-soft.
  Wave-F5 note: the Jun 15–Jul 5 "no fresh idle data" idle streak was PURELY
  upstream (treasury's missing `secrets.initialize` meant 0 snapshots ever,
  fixed in treasury); this module needs no change and self-heals as soon as
  treasury writes fresh snapshots.

## Isolation / safety
RPC URLs ONLY via `config/pool_engine.PoolEngine.get_endpoint('<CHAIN>_RPC')`
with `report_success`/`report_failure`/`report_rate_limit` (read-only use).
Fail-soft everywhere: feed/RPC failure -> HOLD `apr_unavailable`, never a
deploy on stale yield. No paid LLM. Writes ONLY `yield_treasury_advice` and
its own logs. Never imports an executor, never reads private keys, never
calls `security/encryption` decrypt paths.

## Orchestrator wiring still needed (NOT done here — file ownership)
- `main.py`: launch `modules/yield_treasury/main_yield_treasury.py` when
  `YIELD_TREASURY_MODULE_ENABLED=true`.
- `.env.example`: `YIELD_TREASURY_MODULE_ENABLED=false`,
  `YIELD_TREASURY_HEALTH_PORT=8098`, `YIELD_TREASURY_DRY_RUN=true`.
- Root `CLAUDE.md` module table + health-port map (8098 = yield_treasury);
  dashboard advice panel is an optional follow-up.
