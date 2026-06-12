# EXECUTION_GATEWAY Module (shared service)

## What it does
MEV-aware **EVM send-policy library** — the execution sibling of
`config/pool_engine.py`: pool_engine answers "which RPC do I READ from?",
this answers "how do I SEND safely?". Given a tx intent (pre-signed raw OR
unsigned tx + signer callback), it:
1. asserts `core.dry_run.should_skip_live` at the send boundary (DRY_RUN /
   killswitch / `logs/.pause_<module>` can never be bypassed by a caller bug);
2. selects a route — private order flow (Flashbots Protect / MEV-Blocker-style
   RPC) vs public mempool — via pure policy in `core/route_policy.py`;
3. fills EIP-1559 gas via pure policy in `core/gas_policy.py` and nonce via
   `core/nonce_manager.py` (per-(chain,sender) serialized, rollback on failed
   broadcast);
4. broadcasts; private failure falls back to the pool_engine public RPC
   (fail-soft, `public_fallback_enabled`), reporting success/failure/429 back
   to pool_engine;
5. audits every send (incl. simulated) to `execution_gateway_sends`, fail-soft.

It **never originates a trade** and holds **no private keys** — callers pass a
`sign_fn`; keys stay in the caller's existing `security/` handling.

## Library API (`modules/execution_gateway/gateway.py`)
```python
gw = ExecutionGateway(module='dex', db_pool=asyncpg_pool)   # db_pool optional
res = await gw.send(TxIntent(
    chain='ethereum',
    tx=unsigned_tx_dict,            # OR signed_raw='0x...'
    sign_fn=acct.sign_transaction,  # sync or async; returns raw/SignedTransaction
    sender='0x...',
    direction='entry',              # 'entry' | 'exit'
    notional_usd=1500.0,
    urgency='normal',               # 'normal' | 'fast' | 'rescue'
    prefer_private=None,            # False = hard opt-out (latency-critical)
), dry_run=module_dry_run)
# SendResult: ok, simulated, route, tx_hash, fallback_used, error, gas, nonce, reason
```
`module=` must be the caller's short engine key so its own pause flag gates the
send. Replacement of stuck txs: `core.gas_policy.bump_for_replacement`.
Import-safe: no network/DB/env side effects at import; aiohttp lazy; asyncpg
optional.

## Entry point (diagnostics only)
`modules/execution_gateway/main_execution_gateway.py` — health server on port
**8099** (`EXECUTION_GATEWAY_HEALTH_PORT`); `/health`, `/status` (config-key
count + per-route send counts last hour). Never broadcasts. Default DISABLED;
launch gated on `EXECUTION_GATEWAY_MODULE_ENABLED=true` (wiring below).
Logs: `logs/execution_gateway/`.

## Key config (DB, config_type='execution_gateway'; migration 127 — all default-safe)
| Key | Default | What it does |
|---|---|---|
| `private_send_enabled_<chain>` | false | Per-chain private routing (ethereum/bsc/base/arbitrum/polygon). ALL off = today's behavior |
| `private_rpc_url_ethereum` | rpc.flashbots.net/fast | Inert while flag false; empty = env `FLASHBOTS_RPC` |
| `private_rpc_url_<chain>` | '' | Operator-provisioned relays (bsc/base/arbitrum/polygon) |
| `public_fallback_enabled` | true | Private failure -> public RPC (fill rate over secrecy) |
| `private_send_timeout_seconds` | 30 | Relay submission timeout |
| `private_min_notional_usd` | 0 | Below this, skip private (dust not worth latency) |
| `priority_fee_gwei_default` / `_<chain>` | 1.5 | Priority-fee floor |
| `max_fee_multiplier` | 2.0 | maxFee = base*mult + priority |
| `gas_ceiling_gwei` | 150 | Hard clamp, never exceeded |
| `replacement_bump_pct` | 15 | Stuck-tx bump (floored at node min 12.5) |
| `audit_enabled` | true | Rows to `execution_gateway_sends` |

## Kill switch / safety
- `should_skip_live(dry_run, module=<caller>)` checked INSIDE `send()` before
  any broadcast — defense in depth, not a replacement for callers' gates.
- `logs/.killswitch` polled by the diagnostics subprocess; `logs/.pause_execution_gateway`
  pauses only diagnostics (the library is gated by each CALLER's pause flag).
- Sequencer-ordered chains (base, arbitrum) noted in route reasons — private
  routing there buys little; flags should usually stay off.

## Self-tests (offline, mocked sends — no broadcasts)
```
python -m modules.execution_gateway.core.route_policy
python -m modules.execution_gateway.core.gas_policy
python -m modules.execution_gateway.core.nonce_manager
python -m modules.execution_gateway.selftest
```

## Per-module opt-in integration plan (FOLLOW-UP — nothing rewired yet)
Adopt one module at a time behind a per-module DB flag
(`<module>_config.use_execution_gateway`, default false), legacy send path kept
until the flag has soaked. Order:
1. **DEX** (`modules/dex_trading/`) — lowest latency sensitivity; entries+exits.
2. **Arbitrage** (`modules/arbitrage/`) — replaces its bespoke public broadcast;
   the existing FlashbotsExecutor bundle path stays for atomic bundles (the
   gateway handles single-tx protect-RPC sends, not bundles).
3. **Copy EVM leg** (`modules/copy_trading/`) — BUY mirror + exit mirroring.
4. **Sniper EVM leg** (`modules/sniper/`) — EXITS ONLY first
   (`prefer_private=False` on entries: latency IS the edge there).
Acceptance per module: execution_quality/TCA before-vs-after comparison; revert
is flipping the flag back.

## main.py / .env wiring (FOLLOW-UP — owned by the orchestrator agent)
- `.env.example`:
  `EXECUTION_GATEWAY_MODULE_ENABLED=false` and
  `EXECUTION_GATEWAY_HEALTH_PORT=8099`.
- `main.py`: add to the module table exactly like meta_controller —
  `('EXECUTION_GATEWAY', 'EXECUTION_GATEWAY_MODULE_ENABLED', 'modules/execution_gateway/main_execution_gateway.py')`.
- Dockerfile: **no new deps** (aiohttp, asyncpg, python-dotenv already shipped);
  only ensure `modules/execution_gateway/` is inside the existing `COPY modules/`.
- Port map: 8099 = execution_gateway (extends the 8080-8090 map; gap left for
  other waves' services).

## DB tables
- `execution_gateway_sends` (migration 127) — full send audit incl. simulated
  DRY_RUN routing, so route policy is inspectable before any live flip.
