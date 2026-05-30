# DEX Wave-3 Report

**Module:** DEX (`modules/dex_trading/`, `trading/executors/direct_dex.py`, `trading/executors/mev_protection.py`)
**Agent:** A1 — smartcontract-web3-expert
**Branch:** `claude/create-expert-agents-JFSF5`
**Base commit:** `9c349b3`

## Scope

Wave-3 closes the five DEX carry-overs called out in `PM_FINAL.md` § 9:

1. `_quote_v3` real Uniswap V3 QuoterV2 binding (was `int(amount * 0.997)` placeholder)
2. EIP-1559 gas on Ethereum / 1559-capable chains (was legacy `gasPrice` Type-0 only)
3. `_estimate_price_impact` V3 concentrated-liquidity via QuoterV2 round-trip (linear extrapolation was wrong at tick boundaries)
4. `_apply_time_delays` nonce race — wrap nonce read in `nonce_lock`
5. `_path_has_liquidity` real reserves / `slot0`+`liquidity` check (was hardcoded `True`)

## Design notes & commits

To be filled in as commits land.

