# SNIPER Wave-3 Campaign Report

**Agent:** A4 smartcontract-web3-expert
**Branch:** `claude/create-expert-agents-JFSF5`
**Date:** 2026-05-19
**Scope:** Single Wave-3 item from `PM_FINAL.md` carry-over — Pyth-feed wiring
for blue-chip mints in `_get_token_price`.

## Mandate (from PM_FINAL.md / brief)

1. Pyth-feed wiring for blue-chip mints. Extension point in
   `_get_token_price` (sniper_engine.py). Wave-2 added Birdeye fallback;
   Wave-3 prepends Pyth for tokens with a mapped feed-id.
2. **DO NOT** flip `safety_check_enabled=true` — operator-only step.
3. **DO NOT** touch `/api/sniper/timing` per-chain GROUP BY — client-side
   widget already covers it.

## Deliverables

| # | Commit | What | Why |
|---|---|---|---|
| 1 | d89b1c4 | `modules/sniper/core/pyth_feed.py` + `modules/sniper/core/pyth_feed_ids.py` | Helper + static map. Singleton client with 3s per-feed TTL cache, 100ms process-wide throttle, 2s HTTP timeout, fail-soft everywhere. |
| 2 | f2e95d3 (bundled) | `_get_token_price` resolution chain prepends Pyth for mapped mints; feature-flag `sniper_pyth_feeds_enabled` default TRUE; counter `pyth_fallback_hits` preserved across stats reset | Removes residual Jupiter single-point-of-failure for blue-chip SL/TP. |
| 3 | 5c00192 | 10 unit tests in `tests/unit/test_sniper_new_paths.py` | Feed-id map, hex format, parse_price branches, cache hit, helper integration, feature-flag gating, stats shape. |
| 4 | (this file) | `modules/sniper/CLAUDE.md` Wave-3 block + this report | Documentation. |

## Mints mapped (13 blue-chips)

SOL, USDC, USDT, ETH (Wormhole), WBTC, JUP, WIF, BONK, PYTH, RAY,
ORCA, JTO, JLP. Source: https://pyth.network/developers/price-feed-ids
(Solana mainnet-beta). Pump.fun and other freshly-launched memecoins
intentionally NOT mapped — `get_pyth_feed_id` returns None and the
resolution chain falls through to Jupiter unchanged. No extra HTTP on
the hot path for new launches.

## Resolution order on Solana (after this wave)

```
1) Pyth Hermes (if mint has feed-id AND sniper_pyth_feeds_enabled)
2) Jupiter Price API v2
3) Jupiter /quote derivation (existing fallback)
4) Birdeye /defi/price (Wave-2 R5)
5) return 0  -> synthetic close in DRY_RUN
```

Pyth caches at the singleton level (3s TTL, 100ms throttle), engine
caches at the per-mint level (15s TTL). Two layers protect Hermes
from a monitor-loop burst across many positions.

## Profitability levers

1. **Loss-surface reduction (primary).** A Jupiter brown-out was the
   one residual cascade where every active position would
   synthetic-close simultaneously. Pyth is operationally independent
   of Jupiter/Birdeye and sub-second fresh for blue-chips. Cost of
   adding it: $0 (free public Hermes endpoint), and the rate-limit
   gate caps us at 10 req/s process-wide.

2. **No new latency on the hot path for memecoins.** Pump.fun
   resolution is a single dict lookup that returns None — no HTTP,
   no awaitable that resolves before the existing Jupiter call. Cold
   path for the 13 blue-chips picks up one extra HTTP, gated behind
   the singleton 3s cache so under steady-state monitor traffic the
   amortized cost is 1 request per 3 seconds per blue-chip held.

3. **Operator-revertible.** `sniper_pyth_feeds_enabled=false` in
   `config_settings` instantly reverts to Wave-2 behavior.

## Out-of-scope (explicitly)

- `safety_check_enabled=true` DB flip — operator-only LIVE step.
- `/api/sniper/timing` per-chain GROUP BY — client-side widget
  already delivers the operator-visible split (per Wave-2 R6).
- Pyth wiring for EVM tokens — EVM path uses DexScreener; future
  enhancement if the sniper widens scope. The Pyth feed-ids for
  ETH/WBTC are noted in the file header for that future work.

## Verification

- `python -m py_compile` clean on all touched files:
  `pyth_feed.py`, `pyth_feed_ids.py`, `sniper_engine.py`,
  `test_sniper_new_paths.py`.
- `pytest tests/unit/test_sniper_new_paths.py -k "pyth and not via_pyth
  and not feature_flag" -v` -> 7 passed, 0 failed (pure-python
  subset that runs in the sandbox without aiohttp/aiofiles deps).
- The 3 engine-import tests fail in the local sandbox with
  `ModuleNotFoundError: aiofiles` — same as the pre-existing
  `test_live_mode_refuses_to_start_with_safety_off` etc. They run
  on CI where deps are installed.
- DRY_RUN stays TRUE for every change. No flips to LIVE-related
  flags. No touches to `core/risk_manager.py`, `core/dry_run.py`,
  `config/pool_engine.py`, or `security/encryption.py`.

## Concurrent-agent note

The Wave-3 multi-agent worktree was racy: between my `git status` and
`git commit`, two other agents (DEX Wave-3, ARB Wave-3, SOLANA Wave-3,
FUTURES Wave-3) staged files into the shared index. Commit 2's
intended `[sniper] _get_token_price: prepend Pyth Hermes` message
ended up rolled into commit `f2e95d3` (ARB realized-slippage) because
the ARB agent's commit captured my `sniper_engine.py` staged changes
before my own `git commit` could fire. The code is on the branch and
on remote — only the commit message attribution is mis-routed. No
operational impact. Commits 1, 3, 4 (this report) are correctly
labelled.
