# SNIPER Detection-Latency Reduction Plan

## Status
**Date**: 2026-05-13. **Author**: Multi-agent session (latency-investigation pass).
**Module verdict**: AMBER. **Blocker per `modules/sniper/CLAUDE.md`**: structural
detection latency 15-120s vs competitors' 50-400ms.

## Latency budget — current state (verified)
| Stage | Code path | Latency | Bottleneck |
|---|---|---|---|
| EVM listener poll | `modules/sniper/core/evm_listener.py:83-127` (`get_new_pairs`) — HTTP `eth.get_logs` over last 5 blocks per call | 5-15s (1 block on ETH = ~12s) | HTTP polling on finalized logs; depends on monitor-loop call cadence (0.1s, line 280) |
| Solana listener poll | `solana_listener.py:240-261` (`_run_polling_listener`) — sleeps `poll_interval` seconds between cycles | 15-30s (default 15s per `:95`) plus per-source 0.5s gap (`:252`) | Top-level `asyncio.sleep(self.poll_interval)` floor |
| Solana per-source HTTP | `solana_listener.py:263-327` (`_poll_source`) — `getSignaturesForAddress` then **N x `getTransaction`** per signature (up to 20, `:271`) | 1-6s per source per cycle; with N enabled AMMs = N x that | Two serial RPC calls per signature (`:329-378`, `:380-457`); no batching, no `jsonRpc batch` |
| Monitor loop tick | `sniper_engine.py:258-280` — `await asyncio.sleep(0.1)` per iter (`:280`) | adds ~100ms per tick | Pull-based: only drains queue when iter runs |
| Safety check | `token_safety.py:114-139` → up to 3 serial HTTP calls (GoPlus `:413`, Honeypot.is `:428`, RugCheck `:441`); 5-min cache `:99` | 500ms-3s on cold cache, serial | Three separate `session.get` calls done sequentially in `_check_evm_token` / `_check_solana_token` |
| Broadcast | `trade_executor.py` — Jupiter `/swap` (`:504-530`) then `_sign_and_send_solana_tx` (`:374`); EVM router send | 300-800ms | Network RTT, RPC submit |

**Detection-only floor today**: ~15-30s on Solana, ~5-15s on EVM. Competitor
floor: 50-400ms end-to-end. Gap: 30-150x slower on the detection stage.

Note: `solana_listener.py:96` reads `SNIPER_USE_WEBSOCKET` into `self.use_websocket`
but **nothing consumes it** — there is no WSS code path; the flag is dead.

## Root causes
1. **HTTP polling on both chains.** Neither listener subscribes to anything.
   Even with `poll_interval=0`, the RPC round-trip plus block-cadence floor
   sets the wall.
2. **EVM looks at finalized blocks only.** `eth.get_logs` (evm_listener
   `:102-106`) returns logs from already-mined blocks; the pair has existed
   for at least one block by the time we see it.
3. **Solana per-iteration sleep is 15s minimum.** A pool created 1s after a
   poll waits ~14s for the next one (`solana_listener.py:261`).
4. **Solana spends ~2 RPC calls per signature.** For each enabled AMM we
   pull 20 signatures, then **fetch every full transaction** to inspect
   logs (`solana_listener.py:271`, `:291` → `_check_pool_transaction:380`).
   With 2 AMMs enabled, that is ~42 RPC calls per cycle.
5. **Safety checks are serial and uncached on first encounter.** A fresh
   token hits 2-3 external APIs before snipe approval (`token_safety.py:157,
   240, 317`); the 5-min cache (`:99`) only helps re-encounters.
6. **Dead WSS flag.** `SNIPER_USE_WEBSOCKET` is read but unwired
   (`solana_listener.py:96`) — gives a false sense of readiness.

## Proposed reductions (ordered by ROI)

### 1. [HIGH ROI / MEDIUM EFFORT] Solana program-subscribe via WSS
Replace the 15-30s polling loop with `programSubscribe` to Raydium V4 /
Raydium CPMM / Orca / Meteora program IDs already defined at
`solana_listener.py:40-46`, or Helius enhanced WSS (`helius_api_key`
plumbing already exists at `:92, :133-151`). Expected detection latency:
15-30s -> 100-500ms.

**Files**:
- `modules/sniper/core/solana_listener.py` — add `_run_wss_listener`
  alongside `_run_polling_listener` (`:240`); branch in `initialize`
  (`:196`) on `SNIPER_LISTENER_MODE`. Reuse `_check_pool_transaction`
  (`:380`) for log inspection on subscription notifications.
- Wire the existing-but-dead `SNIPER_USE_WEBSOCKET` flag (`:96`) or
  rename to `SNIPER_LISTENER_MODE={polling|wss}` for clarity.

**Risks**:
- Public Solana RPC rarely supports `programSubscribe` at scale; Helius /
  Triton / Quicknode endpoint required. The RPC rotation helper at
  `:563-578` is HTTP-only and will not help here.
- Solana WSS frequently disconnects; reconnection + topic
  re-subscription is mandatory. `websockets==12.0` is already in
  `requirements.txt:60` so no dep work.
- `programSubscribe` returns account-state diffs, not parsed pool
  events. Initial filter to "is this an init tx" still needs the
  `getTransaction` call (`:380-457`); only the discovery hop becomes
  push-based.

**Phase rollout**:
- Phase 1: WSS support behind `SNIPER_LISTENER_MODE`, default `polling`.
- Phase 2: A/B compare WSS-detected pools vs polling-detected pools for a
  week; verify zero misses (WSS missing a pool is a silent failure mode).
- Phase 3: flip default to `wss`; keep polling as fallback when no WSS
  endpoint configured.

### 2. [HIGH ROI / MEDIUM EFFORT] EVM `eth_subscribe` for `PairCreated`
Replace `eth.get_logs` polling with `eth_subscribe('logs', {topics:
[PairCreated]})` over WSS. Expected detection latency: 5-15s -> 50-200ms
after block inclusion.

**Files**:
- `modules/sniper/core/evm_listener.py` — replace polling `get_new_pairs`
  (`:83-127`) with a WSS-driven background task that pushes parsed events
  to an `asyncio.Queue`, mirroring the Solana pattern at
  `solana_listener.py:105` (`new_pools_queue`).
- `sniper_engine.py:264` currently calls `get_new_pairs()` per tick; will
  become a non-blocking `queue.get_nowait()` drain.

**Risks**:
- Web3.py async WSS requires `WebsocketProviderV2`; verify import path
  for `web3==6.20.4` (`requirements.txt:58`) — V2 was added in 6.x but
  remains beta-tagged.
- WSS endpoint required — Alchemy/Infura paid tier or self-hosted.
- Reconnect on drop + topic re-subscription required.
- Reorg handling: a `PairCreated` log seen pre-finality may revert.
  Tolerable for sniping (false positives are caught by safety check)
  but must not corrupt `known_pairs` state at `evm_listener.py:48`.

**Phase rollout**: same `polling -> A/B -> wss-default` pattern as #1.

### 3. [VERY HIGH ROI / HIGH EFFORT / HIGH RISK] Mempool watching
`eth_subscribe('newPendingTransactions')` + decode tx data to find
`UniswapV2Factory.createPair` / `addLiquidity` calls **before**
confirmation. Sub-100ms detection but ugly tradeoffs.

**Files**:
- New `modules/sniper/core/mempool_listener.py` — pending-tx subscription
  + tx-data decoder for known router contracts (factory list already at
  `evm_listener.py:36-40`).
- `sniper_engine.py:_monitor_new_pairs` (`:252-283`) — add a third source.

**Risks**:
- Reverted txs waste safety-check API budget (GoPlus / Honeypot.is are
  rate-limited; `token_safety.py:413, 428`).
- Frontrunning by other bots on the same mempool view — once we react,
  so do they.
- Many RPC providers omit pending-tx data on public tier; only paid /
  self-hosted nodes deliver complete mempool.
- Token contract may not be queryable at decode time — safety check
  needs deferral until `PairCreated` confirms; partial readiness state
  is non-trivial to manage.
- Sniping unconfirmed liquidity opens new rug vectors (LP add can be
  followed by immediate remove in the same block).

**Phase rollout**: defer until #1 and #2 are stable in prod. Mempool is
a separate evolution, not a fast follow.

## Instrumentation prerequisite (must land before Phase 2 of any option)

Without per-stage timing markers, A/B comparison of WSS vs polling is
anecdotal. Stamp these on the opportunity dict in
`sniper_engine.py:_evaluate_target` (`:285-304`):
- `t_detect` — when listener queued the pool
- `t_safety_start` / `t_safety_done` — bracket the `token_safety.check_token`
  call at `:354`
- `t_broadcast_start` / `t_broadcast_done` — bracket `executor.execute_buy`
  at `:477`

Emit a single structured log line per snipe with the deltas. Surface
P50/P95 detection-to-broadcast latency on the dashboard (new SNIPER
panel in `modules/dashboard/`).

**Files**:
- `modules/sniper/core/sniper_engine.py:285, :462` — stamp markers.
- `monitoring/alerts.py` or new sniper metrics emitter — push to the
  dashboard's metric surface (coordinate with dashboard agent).

## Out of scope for this design doc
- Co-located infrastructure / staked endpoints.
- Custom mempool relays / private orderflow (Flashbots, bloXroute).
- Validator-direct submission paths (Jito bundles, etc.).
These are infrastructure decisions, not in-repo code changes.

## Effort estimates (developer-days, rough)
- Instrumentation (prerequisite): 1 day
- Phase 1 Solana WSS (`programSubscribe`): 2-3 days
- Phase 1 EVM WSS (`eth_subscribe('logs')`): 1-2 days
- Phase 2 A/B comparison + dashboard panel: 2 days
- Mempool watching: 5-10 days

## Acceptance — module verdict bump path
SNIPER stays AMBER until Phase 2 confirms median detection-to-broadcast
latency in the 50-400ms band over a week of production traffic, with
zero pools missed vs the polling-control side of the A/B. After that,
`modules/sniper/CLAUDE.md` verdict can move to GREEN candidate.
