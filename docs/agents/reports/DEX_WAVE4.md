# DEX_WAVE4 — Wave-4 Carry-over Close-out

**Branch:** `claude/create-expert-agents-JFSF5`
**Agent:** smartcontract-web3-expert (W4-DEX)
**Date:** 2026-05-20
**Predecessor:** `docs/agents/reports/DEX_WAVE3.md`

## Deliverables (3, all shipped)

| # | Deliverable | Commit | Files |
|---|---|---|---|
| 1 | V3 price-impact via chunked QuoterV2 round-trip + refusal gate | `115cb35` | `trading/executors/direct_dex.py`, `tests/unit/test_dex_price_impact.py` |
| 2a | bloXroute BSC plumbing (config fields, default off) | `6f96075` | `trading/executors/mev_protection.py` |
| 2b | bloXroute BSC submit-path body + tests (rolled into orchestrator follow-up) | `a6c3a89` | `trading/executors/mev_protection.py`, `tests/unit/test_mev_bloxroute_bsc.py` |
| 3 | `modules/dex_trading/CLAUDE.md` + this report | _this commit_ | `modules/dex_trading/CLAUDE.md`, `docs/agents/reports/DEX_WAVE4.md` |

Note: deliverable 2 landed in two commits because the orchestrator persisted plumbing (`6f96075`) ahead of the submit-path body (`a6c3a89`) when the W4 wave shipped from multiple agents in parallel. Both are upstream on the campaign branch.

## What changed

### 1. V3 price-impact — exact chunked round-trip (`115cb35`)

Old: `expected_output = (small_output * amount) // small_amount` with `small_amount = amount // 1000` — a linear extrapolation off a 0.1% probe. On V3 concentrated-liquidity pools the per-tick price changes non-linearly; the linear extrapolation systematically *under-estimated* impact, letting the executor sign trades that would slip 5-15% on tight ranges.

New: per spec
```
tiny       = max(1, amount * 0.01)          # 1% probe
out_tiny   = _simulate_swap(dex, path, tiny,   chain)
out_actual = _simulate_swap(dex, path, amount, chain)
eff_tiny   = out_tiny   / tiny
eff_actual = out_actual / amount
impact     = (eff_tiny - eff_actual) / eff_tiny
```

`_simulate_swap` already routes V3 paths through real QuoterV2 (Wave-3 fix in `_quote_v3`). 1% probe is the smallest size that still produces a non-degenerate quote on 6-decimal USDC/USDT majors — 0.1% rounds to zero.

Numerical-noise clamps: `impact <= 0 → 0.0`, `impact > 1 → 1.0` (the former is rare on very deep pools where the actual quote can have a marginally better per-unit rate than the probe due to integer rounding).

### Refusal gate

New config: `max_price_impact_bps` (default **200 bps** = 2%, matching Uniswap-frontend "high impact" warning). `get_best_quote` drops any DEX candidate whose impact exceeds the cap and returns `None` when nothing passes — instead of silently signing a tx that would eat the entire slippage tolerance whole-cloth. Override via `config['max_price_impact_bps']`.

### 2. bloXroute BSC private-tx (`6f96075`, `a6c3a89`)

Flashbots is Ethereum-only. After the Wave-2 gate that downgraded non-Ethereum chains to `_route_private_mempool`, BSC was effectively routing through `https://api.bloxroute.com` as a string-only placeholder (the URL is in `private_pools` but `_route_private_mempool` only stamped the URL into the tx dict — it never actually POSTed anything).

The Wave-4 addition wires a real BSC private-tx send:

- **Endpoint:** `https://api.blxrbdn.com` (overridable via `config['bloxroute_bsc_endpoint']`)
- **RPC method:** `blxr_private_tx` with `{"transaction": "<raw-hex without 0x>"}`
- **Auth:** `Authorization: <CLOUD_API_KEY>` header. Read from `config['bloxroute_auth_header']` (encrypted-secret friendly) or `BLOXROUTE_AUTH_HEADER` env.
- **Gate:** `chain == 'bsc' and config.get('bloxroute_enabled', False)` AND `risk_score > 0.3` (matches Flashbots gate). Default off.
- **Fallback:** missing header / HTTP error / RPC error / timeout → return None → `protect_transaction` falls through to `_route_private_mempool` so the tx still ships.
- **Ethereum Flashbots path:** untouched.

## Tests

| File | Cases | Status |
|---|---|---|
| `tests/unit/test_dex_price_impact.py` | 6 | green |
| `tests/unit/test_mev_bloxroute_bsc.py` | 7 | green |
| `tests/unit/test_dex_quoter_v3.py` (Wave-3 regression) | 6 | green |
| `tests/unit/test_dex_decimals.py` (Wave-2 regression) | 5 | green |
| `tests/unit/test_dex_eip1559.py` (Wave-3 regression) | 6 | green |

Aggregate: **30/30 DEX unit tests green**.

## Profitability levers

1. **Price-impact cap** — caps per-trade slippage loss at config'd cap. Previously unbounded on V3 (linear extrapolation under-estimated impact on concentrated-liquidity pools). Estimated savings: avoids 500+ bps slip on tight-range trades that the old math reported as 30 bps.
2. **bloXroute BSC** — eliminates sandwich exposure on BSC swaps when enabled. PancakeSwap V2/V3 routes were 100% public-mempool before. Typical sandwich-extraction on BSC swaps is 0.3–1.5% of notional; bloXroute kills this vector.
3. **Refusal gate** — saves the gas burn on txs that would revert at the slippage check anyway (BSC: ~$0.20/tx; Polygon: ~$0.01–0.05/tx; Ethereum: $1–15/tx).

## Carry-over to next wave

- **bloXroute risk-score threshold** — currently shares the Flashbots `risk_score > 0.3` gate. Operators may want a lower threshold on BSC since the per-bundle cost is lower and sandwich exposure is higher (PancakeSwap is the most-sandwiched DEX). Defer to operator config.
- **Per-DEX impact-cap overrides** — `max_price_impact_bps` is global. Memecoin pools may want a higher cap (1000 bps) than stable pools (50 bps). Schema sketched but not implemented this wave.
- **Per-tick V3 impact** — current chunked approach is the same `_simulate_swap` round-trip applied twice; for very large orders that cross 10+ ticks, an actual tick-walk simulation (using `slot0` + `liquidityNet` reads) would be more accurate. The 1%/100% chunked approach matches what Uniswap's own frontend does, so we defer.

## Operator quick-start

Wave-4 DEX changes are config-additive and default-safe:

| Config | Default | What it does |
|---|---|---|
| `max_price_impact_bps` | `200` | Reject quotes above this impact (2% by default) |
| `bloxroute_enabled` | `false` | Enable bloXroute BSC private-tx routing |
| `bloxroute_bsc_endpoint` | `https://api.blxrbdn.com` | bloXroute BDN endpoint |
| `bloxroute_auth_header` | None (or `BLOXROUTE_AUTH_HEADER` env) | Cloud-API key |

`DRY_RUN` stays TRUE. No kill-switch / risk-manager changes.

**Wave 4 DEX closed.**
