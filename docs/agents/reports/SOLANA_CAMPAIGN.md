# SOLANA Campaign Report — Wave 2

**Agent:** A3 (smartcontract-web3-expert)
**Branch:** `claude/create-expert-agents-JFSF5`
**Date:** 2026-05-19
**Module dirs:** `modules/solana_trading/`, `modules/solana_strategies/`, `trading/chains/solana/`

## MB closure verification

| ID | Description | State | Evidence |
|---|---|---|---|
| MB-06 | Close-path decimals hardcoded | **Hardened** | Close path fix was at `solana_engine.py:3506` (commit `a627b7a`). This wave found four residual hardcodes (`_get_token_balance(token_mint, 6)` at lines 2197, 2217, 3339, 3761 + `int(actual_balance * (10**6))` at line 2229). All replaced (`09a5c85`); `_get_token_balance` now defaults `decimals=None` and resolves via on-chain RPC. |
| MB-07 | Jupiter co-signer slot preservation | **Closed** | `trading/chains/solana/jupiter_executor.py:712-742` walks `num_required_signatures`, preserves existing non-zero sigs, refuses if a co-signer slot is missing. |
| MB-08 | Jupiter swap omits priority/dynamic params | **Closed** | `jupiter_helper.py:498-521` always sends `prioritizationFeeLamports` + `dynamicComputeUnitLimit`, optional `dynamicSlippage`. Default is `priorityLevelWithMaxLamports{maxLamports=1_000_000, priorityLevel=high}`. |
| MB-09 | Restart reconcile | **Closed** | `solana_engine.py:1217 → _reconcile_positions_on_startup` (line 1829) walks `solana_positions`, restores live positions, drops phantoms. |
| MB-10 | DRY_RUN gate indentation | **Closed** | `jupiter_executor.py:213` — gate is at function top, runs on every call. Also defense-in-depth gate in `jupiter_helper.execute_swap` (line ~982). |
| MB-15 | Drift hardening | **Closed this wave** | `661cee6` — dry-run gate, leverage cap, oracle deviation + Pyth confidence, funding sanity, all fail-closed. |

## P1 closures

| ID | Description | State |
|---|---|---|
| P1-07 | ML stack unwired | **Partial — rug gate wired** (`b1b358f`). Pump predictor deferred: requires per-token price-history buffer that the engine doesn't carry today. |
| P1-08 | Look-ahead label bug | **Re-classified.** On read, X uses `scaled_data[i-lookback:i]`, y uses `price[i] vs price[i-1]` — that's a legitimate next-bar binary label, X never sees y. The actual leakage is `scaler.fit_transform(data)` over the entire dataset before splitting (standardization leakage). Not fixed in this wave to avoid touching `ml/` outside the agent scope; noted in CLAUDE.md follow-ups. |

## Commits this wave

| Hash | Title | Net LoC | Purpose |
|---|---|---|---|
| `09a5c85` | engine: kill remaining MB-06 decimals hardcodes | +90 | Stops 10x oversell / 1000x undersell on emergency close paths for non-6-decimal tokens (BONK, modern launches). |
| `661cee6` | drift_helper: MB-15 pre-trade guards | +183 | Refuses Drift entry unless dry_run + leverage + oracle + funding all pass. |
| `83df4ad` | jupiter_helper: adaptive priority fee + quote freshness TTL | +153 | Two opt-in profitability levers; quote TTL prevents paying fees on stale routes. |
| `8cf0143` (bundle) | config + engine: wire MB-15 / adaptive-fee / quote-TTL / ML keys | +197 (solana subset) | Adds DB-backed config keys, accessor properties, and engine wiring for JupiterHelper + DriftHelper constructors. |
| `b1b358f` | engine: P1-07 ML rug gate wired into _open_position | +94 | Opt-in `solana_ml_enabled` flag; lazy-loaded RugClassifier; refuses entry above configurable rug probability. |

## Behavior summary (operator-relevant)

All new behavior is **off by default**. To activate:

1. **Adaptive priority fee**: set `adaptive_priority_fee_enabled = true` in `config_settings` (config_type `solana_priority`). Tune `adaptive_priority_fee_percentile` (50..90) and the lamport clamp.
2. **Quote freshness TTL**: lower `jupiter_quote_max_age_s` from 10s to 5s on high-volatility memecoin sessions.
3. **ML rug gate**: set `solana_ml_enabled = true` (config_type `solana_ml`). Requires a trained `RugClassifier` artefact in the model store; otherwise the gate stays idle.
4. **Drift**: stays `drift_enabled = false`. When eventually flipped, the four MB-15 guards engage automatically with conservative defaults (3x leverage cap, 50%/yr funding cap, 1% oracle deviation cap, 500 bps Pyth confidence cap).

## Residual risks / follow-ups

- **Jito bundle path**: not wired in solana_trading. The fully-built `JitoClient` already exists at `modules/arbitrage/solana_engine.py:297` with regional endpoints, rate-limit awareness, and tip-account rotation. Recommended next commit: lift it into `trading/chains/solana/jito_bundle.py` and add a `use_jito` toggle to JupiterHelper.execute_swap.
- **Pump predictor wiring**: needs a per-token rolling price buffer (~60-bar window) maintained by the engine's monitor loop.
- **Pump predictor scaler leakage**: `ml/models/pump_predictor.py:201` fits StandardScaler on the full dataset; split first, fit on train only.
- **`_get_token_balance` fallback**: when `get_spl_decimals` fails, falls back to 6. This is acceptable because Method 2 of `_get_token_balance` reads on-chain decimals from the ATA response, but worth tightening if `get_spl_decimals` ever becomes the bottleneck.
- The `monitoring/` dir and dashboard templates were left alone — they already render the new config keys via the generic `config_settings` table reader.
