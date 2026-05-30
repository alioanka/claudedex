# DEX Campaign Report (Wave 2)

**Module:** DEX (`modules/dex_trading/`)
**Agent:** A1 — smartcontract-web3-expert
**Branch:** `claude/create-expert-agents-JFSF5`
**Base commit:** `0200aeb`

## 1. Scope re-verification

| Item | Status | Notes |
|---|---|---|
| MB-01 (decimals on min_amount_out) | PARTIAL | `_build_swap_transaction` fixed via `to_raw_evm` for output; **input** `ether_to_wei(order.amount)` still hardcoded on lines 594 & 601 |
| MB-02 (Flashbots EIP-191 signing) | OK | `_sign_flashbots_bundle` uses `keccak(text=body).hex()` then `encode_defunct` + `sign_message` |
| P1-04 (pool_engine sweep) | OK | `direct_dex.initialize` routes through `get_chain_rpc_url` → RPCProvider/pool_engine |

## 2. Residual issues (priority ordered)

### P0 — crash bugs in hot paths
1. **`direct_dex.py:568`** — references `self.max_slippage` which is **never initialized** on `DirectDEXExecutor`. The fallback path `float(order.slippage or self.max_slippage)` raises `AttributeError` whenever `order.slippage` is None or 0 (the docstring says "use default"). This silently bricks every order that didn't set slippage at the order layer.
2. **`mev_protection.py:232`** — `bundle_id=bundle_id if 'flashbots' in protection_methods else None`. `bundle_id` is only assigned inside an `if` branch (line 205); when risk_score ≤ 0.3 the else-branch runs and `bundle_id` was never defined → `UnboundLocalError`. Every "ADVANCED protection, low risk" call crashes.
3. **`direct_dex.py:594, 601`** — `ether_to_wei(order.amount)` for the swap **input** amount. Same class of bug as MB-01 but for input: USDC/USDT (6 dec), WBTC (8 dec) all overstate input by 10^12 / 10^10. Fix: route through `to_raw_evm(order.chain, order.token_in, order.amount)`.

### P1 — silent correctness / API drift
4. **`Web3.toChecksumAddress` (camelCase)** at lines 157, 454, 896–897, 905, 913 — removed in web3.py 6.x. Mixed with the modern `Web3.to_checksum_address` (lines 580, 592, 604) inside the same module. On any modern web3 this throws AttributeError at init.
5. **`geth_poa_middleware`** import (line 18) + injection (125) — removed in web3.py 6.x; replaced by `web3.middleware.ExtraDataToPOAMiddleware`. BSC + Polygon will fail to initialize.
6. **`Web3.isConnected()`** at `mev_protection.py:854` — camelCase removed in v6; should be `is_connected()`.
7. **`_quote_v3`** (line 783) is a placeholder: `return int(amount * 0.997)`. This ignores the V3 quoter contract entirely, then `get_best_quote` picks the largest `amount_out`, so V3 will always "win" by always returning the same ratio. Best-quote routing is broken for any pair where V2/V3 actually differ.
8. **`_get_optimal_gas_price`** (line 767) returns legacy `gasPrice` only. Ignores EIP-1559 (`maxFeePerGas`, `maxPriorityFeePerGas`). On post-London Ethereum, mempool inclusion is poor without priority fee. Also calls `w3.eth.gas_price` synchronously inside async context — blocks the event loop.
9. **L2 gas-oracle staleness** — Arbitrum/Base/Optimism `eth_gasPrice` does not include L1 data fee; the cap `max_gas_price` of 50 gwei is also nonsensical on Polygon (often 100+) and trivially absurd on ETH at 30+ gwei daily. Need per-chain gas cap (gwei) + EIP-1559 split.
10. **`_estimate_price_impact`** (line 614) — divides by 1000, assumes pool reserves curve is linear and uniform. For V3 concentrated-liquidity ranges this is wildly off; trades can clear a tick boundary and the linear extrapolation says "0%" impact.
11. **`_apply_time_delays`** (mev_protection.py:319) does `transaction['nonce'] = self.w3.eth.get_transaction_count(...)` (sync) without holding `nonce_lock` — race with concurrent DEX txs from the same wallet.
12. **`_apply_gas_randomization`** can raise gas above `max_gas_price` because randomization runs after the cap. Need to clamp after randomization.

### P2 — operational / observability
13. **MEV ↔ direct_dex protection-level toggle is global**; `mev_protection: True` config does not encode "Flashbots on Ethereum, bloXroute on BSC, Merkle on Polygon". On non-ETH chains Flashbots is meaningless — should silently downgrade.
14. **Route quality scoring missing.** `get_best_quote` picks `max(quotes, key=amount_out)` — ignores price impact, gas cost, slippage budget. A small DEX with thin liquidity but a marginally higher headline quote will win even when the net (after gas + impact) is worse.
15. **Gas estimate is hardcoded** per DEX in `_estimate_gas`. Real Uniswap V3 multi-hop costs vary 200k–450k. Use `w3.eth.estimate_gas` against the built tx.
16. **`_path_has_liquidity` returns hardcoded `True`** — no real liquidity gate. Cache populates whatever path was tried first.
17. **No decimals-correct unit-test coverage** for direct_dex's `_build_swap_transaction`. `tests/unit/test_units.py` only tests the helpers, not the call site.
18. **Constant `flashbots_relay`** does not differentiate Goerli/Sepolia/mainnet relays.

## 3. Fixes shipped (this campaign)

| # | Issue | Fix | Commit |
|---|---|---|---|
| 1 | `self.max_slippage` AttributeError in `_build_swap_transaction` | Init from `max_slippage` / `max_slippage_bps` / default 0.005 | `f7d7941` |
| 2 | `bundle_id` UnboundLocalError in MEV `protect_transaction` | Default `bundle_id = None` at function top + per-chain Flashbots gate | `e872121` |
| 3 | MB-01 input leg: `ether_to_wei(order.amount)` for V2/V3 swaps | Route through `core.units.to_raw_evm(chain, token_in, amount)` | `23d860d` |
| 4 | web3 v6 API drift (`toChecksumAddress`, `isAddress`, `isConnected`, PoA middleware) | Migrate to snake_case + try/except import fallback for PoA | `48d5f20` |
| 5 | Single 50-gwei gas cap broken on Polygon/L2s; sync RPC in async path | `_CHAIN_MAX_GWEI_DEFAULTS` + `loop.run_in_executor(eth_gasPrice)` | `162f711` |
| 6 | MEV gas randomization could lift `gasPrice` above ceiling | Clamp post-randomization to `max_gas_price` | `162f711` |

## 4. Enhancements shipped

| # | Enhancement | Commit |
|---|---|---|
| 1 | Per-chain MEV-protection toggle (Flashbots only on Ethereum mainnet, silent downgrade to private mempool on other chains) | `e872121` |
| 2 | Multi-DEX route quality scoring — `_score_quote(q, gas_price_wei) = amount_out * (1 - impact) - gas_cost_native`. `get_best_quote` now ranks by net fill, not raw headline. | `a40f69a` |
| 3 | Decimals-correct test coverage: `tests/unit/test_dex_decimals.py` — USDC 6 dec, WBTC 8 dec, WETH 18 dec sanity + scoring regressions (prefers lower gas / prefers lower impact). | `869eed3` |

## 5. Commit log (oldest → newest)

| Hash | Message |
|---|---|
| `28c484e` | `[dex] audit report: residual P0/P1 issues + enhancement backlog` |
| `f7d7941` | `[dex] direct_dex: init self.max_slippage to fix AttributeError in build_swap` |
| `e872121` | `[dex] mev_protection: fix UnboundLocalError + per-chain Flashbots gate` |
| `23d860d` | `[dex] direct_dex: decimals-correct amount_in in _build_swap_transaction` |
| `48d5f20` | `[dex] direct_dex+mev: web3 v6 API drift — toChecksumAddress, isAddress, PoA, isConnected` |
| `162f711` | `[dex] gas: per-chain max-gwei + async eth_gasPrice + randomization clamp` |
| `a40f69a` | `[dex] direct_dex: multi-DEX route quality scoring (net of gas + impact)` |
| `869eed3` | `[dex] tests: MB-01 decimals regression + route-quality scoring coverage` |

## 6. Issues deferred (next wave)

These were called out in Section 2 but deferred to keep commits ≤200 LoC and avoid touching wider executor refactors:

- `_quote_v3` placeholder (line 783) — needs a real Uniswap V3 QuoterV2 binding. Today it returns `amount * 0.997`, so V3 routing is fundamentally broken; route-quality scoring (#2) still ranks correctly within whatever data we have, but the V3 input data is wrong.
- EIP-1559 (`maxFeePerGas` / `maxPriorityFeePerGas`) on Ethereum mainnet — `_get_optimal_gas_price` still returns legacy `gasPrice`. Works (Type-0 still accepted) but suboptimal inclusion.
- `_estimate_price_impact` linearity assumption breaks for V3 concentrated liquidity at tick boundaries.
- `_apply_time_delays` nonce read bypasses `nonce_lock` — race with concurrent DEX txs from the same wallet.
- `_path_has_liquidity` returns hardcoded `True`.
- Dashboard pages (`dashboard_dex.html`, `performance_dex.html`, `trades_dex.html`, `positions_dex.html`) listed under P1-48 are still absent; out of A1 scope per PM_PLAN (worktree restricted to `dashboard/templates/dex/*.html` which doesn't exist yet).

