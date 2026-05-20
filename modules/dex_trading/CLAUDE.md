# DEX Module
## What it does
Spot trading on EVM DEXes (Uniswap V2/V3, SushiSwap, PancakeSwap) across Ethereum, BSC, Polygon, Arbitrum, Base. Routes through `trading/executors/direct_dex.py` with optional Flashbots MEV protection.
## Entry point
`modules/dex_trading/main_dex.py` — launched as a subprocess by `main.py` when `DEX_MODULE_ENABLED=true`. Engine: `modules/dex_trading/dex_module.py` wraps the shared `trading/trading_engine.py`.
## Key config (DB-backed via `ConfigManager`)
- `max_slippage_bps` — per-trade slippage cap (default 50)
- `max_gas_price` — gwei ceiling for tx submission (default 50)
- `mev_protection` — toggle Flashbots private-bundle path (default True)
- `supported_dexs` — list of router names enabled for routing
- `jupiter_routing` — kept for future SOL-leg routing; currently EVM only
## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_dex` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Effect: `should_skip_live` returns `True` -> live-write gates in `direct_dex.py:64,276,1013` return their `_simulate_*` path.
## Logs
`logs/dex_trading/` — main, errors, trades (rotating handler).
## Primary risk-policy gate
`core.risk_manager.RiskManager.validate_trade(token, amount)` — wired through the shared `trading/trading_engine.py` order-execution path. Per-executor caps (`max_slippage_bps`, `max_gas_price`) enforced inline at `trading/executors/direct_dex.py:88,567`.
## Live-trade readiness
AMBER → GREEN candidate (pending production verification). MB-01 (decimals — both legs now via `core.units.to_raw_evm`), MB-02 (Flashbots EIP-191) and P1-04 (`pool_engine` RPC unification) all closed.
## Wave-2 hardening (campaign `claude/create-expert-agents-JFSF5`)
- P0: `DirectDEXExecutor.max_slippage` initialized (was AttributeError when `order.slippage` unset).
- P0: `MEVProtectionLayer.protect_transaction` `bundle_id` defaulted to `None` (was `UnboundLocalError` on ADVANCED+low-risk path).
- P0: MB-01 input-side — `amount_in_raw` now uses `to_raw_evm(chain, token_in, amount)` instead of `ether_to_wei` (USDC/USDT/WBTC fix).
- P1: web3 v6 API drift — `toChecksumAddress` → `to_checksum_address`, `isAddress` → `is_address`, `isConnected` → `is_connected`, PoA middleware import made version-safe.
- P1: per-chain gas-price ceiling (`_CHAIN_MAX_GWEI_DEFAULTS` + `chain_max_gas_gwei` config override). Single 50-gwei cap was wrong on Polygon/L2s.
- P1: `_get_optimal_gas_price` runs `eth_gasPrice` via `loop.run_in_executor` (was blocking the async loop).
- P1: `mev_protection._apply_gas_randomization` clamps post-randomization gasPrice to `max_gas_price` ceiling.
- P1: Flashbots ADVANCED-tier gate now requires `chain ∈ {ethereum, eth, mainnet}` — silently falls through to private-mempool routing on BSC/Polygon/Arb/Base (Flashbots relay does not service those chains).
- Enhancement: `get_best_quote` ranks by `_score_quote = amount_out * (1 - impact) - gas_cost_native`, not raw headline output. Routes to the DEX with the best NET fill.
- Enhancement: `tests/unit/test_dex_decimals.py` — 5 regression tests for MB-01 (USDC 6 dec, WBTC 8 dec, WETH 18 dec sanity) + scoring (prefers lower gas, prefers lower impact).
## Wave-3 hardening (campaign `claude/create-expert-agents-JFSF5`)
- `_quote_v3` real Uniswap V3 QuoterV2 binding (was `int(amount * 0.997)` placeholder). Iterates fee tiers {100, 500, 3000, 10000} for single-hop, falls back to `quoteExactInput(bytes,uint256)` for multi-hop. QuoterV2 addresses seeded per chain (ETH/POLY/ARB/BASE/OP/BSC), overridable via `config['v3_quoter_addresses']`. Returns 0 (not the placeholder) when no pool exists — so route ranking treats V3 as "no liquidity" instead of silently winning. Tests: `tests/unit/test_dex_quoter_v3.py`.

## Wave-4 hardening (campaign `claude/create-expert-agents-JFSF5`)
- `_estimate_price_impact` now does a real chunked QuoterV2 round-trip (was linear extrapolation from a 0.1% sample). Quotes the actual size and a 1%-of-size probe; impact = `(eff_tiny - eff_actual) / eff_tiny`. Both legs flow through `_simulate_swap`, which already routes V3 paths through the real QuoterV2. 1% is the smallest probe that still produces a non-degenerate quote on 6-dec USDC/USDT majors. Tests: `tests/unit/test_dex_price_impact.py`.
- New `max_price_impact_bps` config (default **200 bps** — matches the Uniswap-frontend "high impact" warning). `get_best_quote` drops any DEX candidate whose impact exceeds the cap; when nothing passes it returns None instead of signing a tx that would eat the entire slippage tolerance. Override via `config['max_price_impact_bps']`.
- `MEVProtectionLayer._attempt_bloxroute_bsc` — bloXroute BDN private-tx routing on BSC (Flashbots is Ethereum-only, so BSC swaps previously had no private path). Submits `blxr_private_tx` JSON-RPC to `https://api.blxrbdn.com` with `Authorization: <CLOUD_API_KEY>` header. Gated by `chain == 'bsc' and config.get('bloxroute_enabled', False)`. Auth header read from `config['bloxroute_auth_header']` (encrypted-secret friendly) or `BLOXROUTE_AUTH_HEADER` env. Returns None on missing-header / HTTP error / RPC error / timeout — `protect_transaction` then falls back to the public-mempool send path so the tx still ships. Endpoint overridable via `config['bloxroute_bsc_endpoint']`. Ethereum Flashbots branch is unchanged. Tests: `tests/unit/test_mev_bloxroute_bsc.py`.

## See also
- Phase 1 audit reports: `docs/agents/reports/DEX_*.md` (smartcontract / quant / analyst).
- Wave-2 campaign report: `docs/agents/reports/DEX_CAMPAIGN.md`.
- Wave-3 campaign report: `docs/agents/reports/DEX_WAVE3.md`.
- Wave-4 campaign report: `docs/agents/reports/DEX_WAVE4.md`.
- Canonical engine API: `docs/engines.md`.
