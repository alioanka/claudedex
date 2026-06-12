# SMART_MONEY Module (ADVISORY — signal-only, never trades)

## What it does
On-chain smart-money flow follower — generalizes copy_trading beyond NAMED
leaders to CLUSTER-DETECTED accumulation. Every tick it:
1. **Ingests** large swaps on the top DexScreener pairs per EVM chain
   (`eth_getLogs` V2/V3 Swap decoding, buyer attributed via tx `from`,
   budget-capped) into `smart_money_wallet_events`.
2. **Marks** realized FORWARD returns on past events whose horizons have
   FULLY elapsed (late-marked from the then-current price — never early).
3. **Scores** each wallet by decay-weighted realized forward return + hit
   rate over a rolling window -> `smart_money_wallet_scores`.
4. **Emits** an ADVISORY `accumulation` row to `smart_money_signals` when a
   fresh cluster (>= N distinct wallets buying one token inside the window)
   contains enough historically-profitable wallets.

**It NEVER places a trade. There is no broadcast path in this module** — no
executor import, no key access, no `security/encryption` decrypt. The operator
(or a future copy/DEX consumer, behind that module's own risk gates) decides
whether a signal is acted on. Any live/execution concept is documented-only.

## No-look-ahead guarantee
Enforced in the PURE layer (`core/cluster_scorer.score_wallet`): an event
contributes to a wallet's score only when `event_ts + max(horizon)*60 <=
now`, regardless of what marks exist — a buggy/poisoned mark cannot leak the
future. Marks themselves are written by `flow_engine._mark` only AFTER a
horizon has fully elapsed, using the price observed at mark time (so marks
are slightly LATE by up to one poll interval — conservative, never early).
Self-test asserts this: `python -m modules.smart_money.core.cluster_scorer`.

## Crowding / decay honesty
- **Crowding:** Nansen/Arkham-style followers compress this edge. Signal
  strength is multiplied by `crowding_factor = soft_cap/n` above
  `crowding_soft_cap_wallets` participants — crowded accumulation is late
  accumulation, and a very crowded cluster approaches strength 0.
- **Decay:** wallet edges rotate and die. Event weight halves every
  `score_half_life_days`; a wallet that stops winning bleeds score even
  without new losses. Scores are recomputed from scratch every tick over
  `scoring_window_hours` — nothing is grandfathered.
- **False positives:** wallet attribution is tx `from`, so routers/bots/CEX
  hot wallets and MEV bundlers can score well for reasons that do not
  transfer to a follower. Treat `smart_money_signals` as a research feed,
  not an order queue.

## Data-source coverage limits (honest)
- **EVM only in v1** (`ethereum,base,arbitrum` by default). A `solana` entry
  in `chains` is ignored until a Solana swap parser ships (program-level
  parsing is a separate build; copy_trading's monitor is NOT reused here).
- Only the top `watch_pairs_per_chain` DexScreener pairs per chain are
  watched — accumulation in unwatched/long-tail pools is invisible.
- Only swaps >= `min_event_usd` with successful wallet attribution (capped at
  `max_wallet_lookups_per_tick`) become events; quiet sub-threshold
  accumulation is invisible by design (RPC budget discipline).
- Event time is ESTIMATED from head-block timestamp and chain block spacing;
  `price_usd_at_event` is the DexScreener price at observation, so logs older
  than `max_event_age_minutes` are dropped rather than mismarked.
- Forward marks depend on DexScreener still listing the token; delisted/rugged
  tokens may never get marked and then never score (which is the safe bias).

## Entry point
`modules/smart_money/main_smart_money.py` — launched by `main.py` when
`SMART_MONEY_MODULE_ENABLED=true` (default **false**). Health server on port
8105 (`SMART_MONEY_HEALTH_PORT`): `/health` liveness, `/status` last-tick
summary. Engine: `core/flow_engine.py`; ingestion: `core/chain_scanner.py`;
pure math (self-tested): `core/cluster_scorer.py`.

## Key config (DB-backed, config_type='smart_money'; migration 133)
| Key | Default | What it does |
|---|---|---|
| `poll_interval_seconds` | 300 | Tick cadence |
| `chains` | ethereum,base,arbitrum | Chains scanned (EVM only in v1) |
| `watch_pairs_per_chain` / `min_pair_liquidity_usd` / `min_pair_volume_24h_usd` | 12 / 100k / 250k | Pair watchlist filters |
| `min_event_usd` / `max_event_age_minutes` | 2000 / 30 | Event size + freshness gates |
| `max_blocks_per_tick` / `initial_lookback_blocks` / `max_wallet_lookups_per_tick` / `max_mark_tokens_per_tick` | 600 / 300 / 60 / 60 | RPC + API budgets |
| `forward_horizons_minutes` | 60,360,1440 | Forward-return horizons (no-look-ahead gate uses the longest) |
| `scoring_window_hours` / `min_events_per_wallet` / `score_half_life_days` / `n_target_events` | 336 / 3 / 14 / 12 | Wallet scoring window, floor, decay, confidence saturation |
| `min_wallet_score` / `min_forward_return_pct` | 0.55 / 3.0 | Smart-wallet + cluster quality gates |
| `cluster_window_minutes` / `cluster_min_wallets` / `min_smart_wallets` | 45 / 3 / 2 | Accumulation-cluster definition |
| `crowding_soft_cap_wallets` | 12 | Crowding decay knee |
| `signal_cooldown_minutes` / `event_retention_days` | 240 / 45 | Signal dedupe + disk discipline |

Env knobs: `SMART_MONEY_MODULE_ENABLED` (gate, default false),
`SMART_MONEY_HEALTH_PORT` (8105), `SMART_MONEY_POLL_INTERVAL`
(pre-migration fallback).

## Kill switch
- Global: `logs/.killswitch` — tick skipped (this module only READS the flag;
  killswitch poller started for dashboard stop support).
- Per-module: `logs/.pause_smart_money` — tick skipped.

## Logs
`logs/smart_money/` — `smart_money.log` (INFO), `smart_money_errors.log`
(WARNING+, so SMART-MONEY SIGNAL lines land there).

## DB tables (migration 133)
- `smart_money_wallet_events` — observed swaps + late forward-return marks
  (idempotent on `(chain, tx_hash, log_index)`; purged after retention).
- `smart_money_wallet_scores` — one current score row per (chain, wallet).
- `smart_money_signals` — advisory accumulation signals (`advisory=TRUE`
  always; full audit history, never purged by the module).

## Isolation / safety
RPC URLs ONLY via `config/pool_engine.PoolEngine.get_endpoint('<CHAIN>_RPC')`
with `report_success`/`report_failure`/`report_rate_limit` (read-only use:
`eth_blockNumber`, `eth_getBlockByNumber`, `eth_getLogs`, `eth_call
decimals()`, `eth_getTransactionByHash`). Free APIs only (DexScreener, no
key); no paid LLM (`core/llm_budget` untouched — nothing to gate). Fail-soft
everywhere: any source failure shrinks the tick to a no-op, never a crash and
never a fabricated mark. Writes ONLY its three tables and its own logs.

## Orchestrator wiring still needed (NOT done here — file ownership)
- `main.py`: launch `modules/smart_money/main_smart_money.py` when
  `SMART_MONEY_MODULE_ENABLED=true`.
- `.env.example`: `SMART_MONEY_MODULE_ENABLED=false`,
  `SMART_MONEY_HEALTH_PORT=8105`.
- Root `CLAUDE.md` module table + health-port map (8105 = smart_money).
- Optional follow-ups: dashboard signals panel (read-only, fail-soft);
  copy_trading may propose top-scored wallets into `copy_leader_candidates`
  (operator-approval flow from mig 092) — NOT wired here.
