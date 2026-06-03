# RPC / API Free-Tier Provisioning Plan

Goal: provision enough RPC/API endpoints, per chain, that **no single free-tier
key gets starved** under the bot's real request patterns, with redundancy and
rate-limit headroom built into priority tiers (primary / secondary /
public-fallback). The operator creates the accounts below and registers each
URL in the DB-backed RPC/API pool (dashboard: Settings -> RPC/API, or seed via
migration). The PoolEngine then load-balances + health-rotates across them.

This plan is grounded in:
- The provider types the pool actually loads (`config/pool_engine.py`
  `_load_from_env` / `_get_env_fallback`): Solana, Ethereum, Base, Arbitrum,
  BSC, Polygon, Optimism, Avalanche, Fantom, Cronos, Monad, Pulsechain, plus
  the Helius / Jupiter / GoPlus / 1inch / Etherscan APIs.
- Which modules use each chain (DEX, SNIPER, ARBITRAGE, COPY, SOLANA).
- The observed live failure: COPY polling 33 Solana wallets stampeded a single
  Helius free key into instant 429.

## Request-pattern summary (per chain, per module)

| Chain | Modules | Dominant pattern | Rough sustained load |
|---|---|---|---|
| **Solana** | COPY, SNIPER, SOLANA | COPY: 33 wallet polls/cycle (Helius enhanced-tx REST); SNIPER: WSS + getTransaction; SOLANA: Jupiter quotes + getSlot | **Highest.** COPY alone ~2.2 req/s sustained @15s cadence, bursty; SNIPER adds per-event getTransaction; SOLANA periodic. Combined easily 5-10 req/s, spiky. |
| **Ethereum** | DEX, ARBITRAGE | block polling, eth_call (reserves/quotes), pair discovery | Moderate, steady. 1-3 req/s. |
| **Base** | DEX, ARBITRAGE, COPY(EVM) | same as ETH; Base is the busiest EVM after ETH | Moderate. 1-3 req/s. |
| **Arbitrum** | DEX, ARBITRAGE, COPY(EVM) | eth_call reserves, block polling | Low-moderate. <1-2 req/s. |
| **BSC** | DEX, ARBITRAGE | eth_call, block polling | Low. <1 req/s. |
| **Polygon** | DEX | eth_call, block polling | Low. <1 req/s. |
| **Optimism** | COPY(EVM) | leader-tx scan | Very low. |
| **Avalanche** | DEX (chain discovery) | block polling | Very low. |
| **Fantom / Cronos / Monad / Pulsechain** | DEX discovery (opportunistic) | block polling | Negligible / occasional. |

The PRIMARY scaling pressure is **Solana** (COPY's 33-wallet fan-out), then the
busy EVM trio **Ethereum / Base / Arbitrum**. Everything else is light enough
that a single keyed endpoint + one public fallback is sufficient.

---

## Per-chain provisioning

Notation per tier: **provider × count (provider_type)** — why.

### Solana — HIGHEST priority, most redundancy
The bot needs both a JSON-RPC endpoint AND the Helius enhanced-tx REST API for
COPY. With pacing now in place (`copy_helius_rps`, default 8 req/s) one good key
survives, but redundancy across keys lets the round-robin spread the burst.

- **Primary (tier priority ~50):**
  - **Helius × 2 keys** (`HELIUS_API`) — two separate free accounts. ~10 req/s
    each → ~20 req/s aggregate headroom; the new round-robin spreads the
    33-wallet burst across both before either starves. Helius free = 100k
    credits/day **per key**, so 2 keys ≈ 200k/day, comfortably above COPY's
    ~95-190k/day depending on cadence.
- **Secondary (priority ~100):**
  - **QuickNode × 1** (`SOLANA_RPC`) — free Solana endpoint, used for
    getSignatures/getSlot when Helius is busy.
  - **Alchemy Solana × 1** (`SOLANA_RPC`) — free tier, separate provider for
    independence.
- **Public fallback (priority ~150, keep-alive warm):**
  - `https://api.mainnet-beta.solana.com` (`SOLANA_RPC`)
  - `https://rpc.ankr.com/solana` (`SOLANA_RPC`) — Ankr public.
  - `https://solana-rpc.publicnode.com` (`SOLANA_RPC`)
- **WebSocket (SNIPER):** Helius WSS (`SOLANA_WS`, derived from a Helius key) +
  one public WSS fallback.

> **Operator note (the binding constraint):** COPY's daily Helius credit usage
> = `wallets × (86400 / copy_poll_interval_s)`. At 33 wallets / 15s that is
> ~190k/day → needs **2 Helius keys** OR raise `copy_poll_interval_s` to ~30s
> (~95k/day, fits 1 key) OR trim to ~17 wallets. A paid Helius Developer plan
> (10M credits/mo) removes the constraint entirely.

### Ethereum — high priority
- **Primary:** **Alchemy × 1** + **Infura × 1** (`ETHEREUM_RPC`). Two
  independent 300M-CU/mo (Alchemy) / 100k-req/day (Infura) free keys; the pool
  round-robins between them.
- **Secondary:** **Ankr × 1** keyed (`ETHEREUM_RPC`) or **dRPC × 1**.
- **Public fallback:** `https://eth.llamarpc.com`,
  `https://ethereum-rpc.publicnode.com`, `https://rpc.ankr.com/eth`.
- **Etherscan API × 1** (`ETHERSCAN_API`) for COPY EVM leader-tx scans (V2 API
  covers Base/Arbitrum/Optimism under one key).

### Base — high priority
- **Primary:** **Alchemy Base × 1** + **Infura Base × 1** (`BASE_RPC`).
- **Secondary:** **dRPC Base × 1** (`BASE_RPC`).
- **Public fallback:** `https://mainnet.base.org`,
  `https://base-rpc.publicnode.com`, `https://base.llamarpc.com`.

### Arbitrum — moderate
- **Primary:** **Alchemy Arbitrum × 1** (`ARBITRUM_RPC`).
- **Secondary:** **Infura Arbitrum × 1** or **dRPC × 1** (`ARBITRUM_RPC`).
- **Public fallback:** `https://arb1.arbitrum.io/rpc`,
  `https://arbitrum-one-rpc.publicnode.com`.

### BSC — low
- **Primary:** **QuickNode BSC × 1** or **Ankr BSC × 1 keyed** (`BSC_RPC`).
- **Public fallback:** `https://bsc-dataseed.binance.org`,
  `https://bsc-rpc.publicnode.com`, `https://rpc.ankr.com/bsc`.

### Polygon — low
- **Primary:** **Alchemy Polygon × 1** (`POLYGON_RPC`).
- **Public fallback:** `https://polygon-rpc.com`,
  `https://polygon-bor-rpc.publicnode.com`.

### Optimism — very low (COPY EVM only)
- **Primary:** **Alchemy Optimism × 1** (`OPTIMISM_RPC`) — optional; public is
  fine at this volume.
- **Public fallback:** `https://mainnet.optimism.io`,
  `https://optimism-rpc.publicnode.com`.

### Avalanche — very low
- **Public only:** `https://api.avax.network/ext/bc/C/rpc`,
  `https://avalanche-c-chain-rpc.publicnode.com` (`AVALANCHE_RPC`).

### Fantom / Cronos / Monad / Pulsechain — opportunistic discovery
- **Public only**, 1-2 each (`FANTOM_RPC` / `CRONOS_RPC` / `MONAD_RPC` /
  `PULSECHAIN_RPC`):
  - Fantom: `https://rpc.ftm.tools`, `https://fantom-rpc.publicnode.com`
  - Cronos: `https://evm.cronos.org`
  - Monad: current testnet/public endpoint
  - Pulsechain: `https://rpc.pulsechain.com`

### Supporting APIs
- **Jupiter** (`JUPITER_API`) — `https://lite-api.jup.ag` (free) primary; a paid
  `quote-api.jup.ag` key as secondary if SOLANA volume grows.
- **GoPlus** (`GOPLUS_API`) — 1 key for token-safety checks.
- **1inch** (`1INCH_API`) — 1 key (EVM aggregator quotes).

---

## Recommended minimum account list (operator checklist)

| Provider | Accounts to create | Covers (provider_type) |
|---|---|---|
| **Helius** | **2** free | `HELIUS_API` + `SOLANA_WS` (Solana — the critical one) |
| **Alchemy** | 1 (multi-chain key) | `ETHEREUM_RPC`, `BASE_RPC`, `ARBITRUM_RPC`, `POLYGON_RPC`, `OPTIMISM_RPC`, `SOLANA_RPC` |
| **Infura** | 1 (multi-chain key) | `ETHEREUM_RPC`, `BASE_RPC`, `ARBITRUM_RPC` |
| **QuickNode** | 1-2 | `SOLANA_RPC`, `BSC_RPC` |
| **dRPC / Ankr** | 1 each (optional) | extra `ETHEREUM_RPC` / `BASE_RPC` / `BSC_RPC` secondary |
| **Etherscan** | 1 (V2 multichain) | `ETHERSCAN_API` (COPY EVM scans) |
| **Jupiter / GoPlus / 1inch** | 1 each | `JUPITER_API`, `GOPLUS_API`, `1INCH_API` |
| **Public RPCs** | 0 (no signup) | fallback tier on every chain |

That is **~7-9 free signups** for full redundancy. The two Helius keys are the
single most impactful item; everything else is one keyed primary + public
fallbacks.

## Tiering / priority guidance when registering endpoints
- Keyed primaries: `priority = 50`.
- Keyed secondaries / second provider: `priority = 100`.
- Public fallbacks: `priority = 150`.
- The pool round-robins within an equal-priority tier and only drops to the next
  tier when the whole upper tier is rate-limited (anti-starvation). The
  keep-alive loop pings even the priority-150 public fallbacks a few times/week
  so they stay warm and providers like Ankr don't idle-disable a key.

## Headroom math (sanity check)
- **Solana / COPY:** 2 Helius keys ≈ 20 req/s + 200k credits/day vs COPY's
  ~2.2 req/s sustained / ~95-190k credits/day → adequate with the new
  `copy_helius_rps=8` pacing. Add SNIPER+SOLANA and you are still inside 2-key
  headroom at default cadence; if you run all three Solana modules hot, prefer a
  paid Helius Developer plan.
- **Ethereum/Base:** Alchemy 300M CU/mo ≈ 100+ req/s equivalent — DEX+ARB
  steady 1-3 req/s is comfortably inside one key; the second provider is purely
  for redundancy/failover.
- **Everything else:** public endpoints absorb the sub-1-req/s load; keep one
  keyed primary only where you want reliability (BSC/Polygon).

## What still needs LIVE verification (cannot test from here)
- Actual Helius free-tier credit cost per enhanced-tx call vs the 100k/day
  budget — run `scripts/test_rpc_pool.py` and watch the Helius dashboard credit
  meter for a day at your real wallet count + cadence.
- Whether 2 Helius keys are enough at your specific `copy_poll_interval_s` +
  wallet count, or whether to go paid.
- Public-RPC reliability varies day to day; the keep-alive + health-check loops
  will demote flaky ones automatically, but confirm at least one healthy
  fallback per chain after registering.
