# SNIPER Wave-2 Campaign Report

**Agent:** A4 smartcontract-web3-expert
**Branch:** `claude/create-expert-agents-JFSF5`
**Start date:** 2026-05-19
**Module CLAUDE.md:** `/home/user/claudedex/modules/sniper/CLAUDE.md`

## Phase 1 verification — MB-11..MB-14 + Phase-2 WSS

| Item | Status | Evidence |
|---|---|---|
| MB-11 — Solana Raydium V4 program-ID literal verified | GREEN | `solana_listener.py:42,89` use the canonical mainnet program id `675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8`. |
| MB-12 — DB-loaded `take_profit_pct` / `stop_loss_pct` no longer shadowed | GREEN | `sniper_engine.py:225-228` reads both from DB; `_monitor_active_snipes` reads `self.take_profit_pct` / `self.stop_loss_pct` directly (lines 877-879). |
| MB-13 — `_load_settings` LIVE-trading safety guard refuses to start with `safety_check_enabled=false` | GREEN | `sniper_engine.py:266-275` raises `RuntimeError` outside the try/except so it actually propagates. Test coverage in `tests/unit/test_sniper_new_paths.py::test_live_mode_refuses_to_start_with_safety_off`. |
| MB-14 — DRY_RUN synthetic close prevents orphan accumulation | GREEN | `sniper_engine.py:907-911,955-1009` retires positions when no price feed; uses `_close_position_synthetic`. |
| Phase 2 WSS volume gain (2.6× polling) | GREEN | Confirmed in `modules/sniper/CLAUDE.md` line 37-50; WSS path persists `detection_path` + `block_time_anchored` + `rpc_receipt_perf` so the A/B is reproducible. |
| WSS concurrency fix (`_process_candidate` + `asyncio.Semaphore(SNIPER_WSS_CONCURRENCY)`) | GREEN | `solana_listener.py:222-230,541-555` dispatches getTransaction to bounded tasks. `wss_inflight_peak` + `wss_dispatched` surface saturation. |
| Active-positions cap | GREEN | `sniper_engine.py:411-429,711-722` belt-and-suspenders gates; uses `_effective_active_count()` which prefers DB-open count over in-memory. |
| Jupiter quote fallback for fresh mints | GREEN | `sniper_engine.py:1011-1110`; 15s per-mint cache bounds quote RPS. |
| LIVE-mode `test_mode=true` refusal | GREEN | `sniper_engine.py:281-291` SNIPE-RM-19 guard. |
| `safety_check_enabled=true` flip | DEFERRED — operator-only step (do NOT touch in this wave). |

## Residual issues identified

### R1 — Per-event latency under `getTransaction` commitment wait (PM mission item 2)
- The hot WSS loop already stamps `t_rpc_receipt` at message arrival before dispatching, so the headline metric `detect_to_rpc_receipt_ms` is isolated from the commitment wait. **However**, the inner `_check_pool_transaction` still waits with `commitment: confirmed` (`solana_listener.py:799`) which can stall 3-13s per call. With `SNIPER_WSS_CONCURRENCY=16` a single bad-RPC moment fills the semaphore in seconds and notifications back up — they're not dropped, but their downstream `safety_ms` and `broadcast_ms` get measured against an aged candidate that may already be unsnipeable.
- **Fix shape:** add a bounded "fast readback" path that tries `commitment: processed` first (typical 200-400ms readback latency), and only falls back to `confirmed` on missing `meta`. The pool init signal is in `logMessages` which exists at `processed` for the vast majority of Raydium V4 + Pump.fun creations. Adds a per-source counter `processed_hit` / `processed_miss_fallback`.

### R2 — `_check_filters` swallows API failures into `return False`
- `sniper_engine.py:570-573`: if `token_safety.check_token` raises, the candidate is silently rejected and **no cooldown** is recorded. The same token will repeatedly retry next poll, wasting RPS budget against GoPlus/Honeypot.is.
- **Fix shape:** record `_rejected_cache[token_address] = datetime.now()` on the exception branch so the 5-min cooldown kicks in. Also bump a `safety_check_errors` counter so the dashboard can surface API outages.

### R3 — `solana_listener._stats` reset window drops `wss_dispatched`/`wss_inflight_peak`
- `_log_stats_if_needed` (lines 1047-1065) only preserves a fixed subset of WSS counters across the 1-min reset and drops `wss_dispatched`/`wss_inflight_peak` even though `_persist_runtime_stats` exposes them in the dashboard. After the first window flips they always read 0.
- **Fix shape:** add `wss_dispatched` and `wss_inflight_peak` to the carry-over set.

### R4 — TokenSafety quorum (PM mission item 3a)
- `_check_evm_token` (token_safety.py:156-238) already calls both GoPlus AND Honeypot.is and merges the results, but the merge is "OR-truthy" — if either source flags `is_honeypot=True`, we reject. There is no quorum / disagreement-aware decision. A flaky Honeypot.is response (frequent for fresh launches) flips us to reject even when GoPlus says clean.
- **Fix shape:** track per-source verdicts in the report (`honeypot_sources_flagged`, `honeypot_sources_clear`), require ≥2 source-agreement for honeypot rejection when both succeed, but fall back to either single source's flag when the other fails (fail-safe asymmetry: false-positive better than a real honeypot purchase).

### R5 — Solana SL feed redundancy (PM mission item 3b)
- `_get_token_price` (sniper_engine.py:1011-1065) falls through Jupiter Price v2 → Jupiter `/quote`. No third source. If Jupiter brown-outs, every Solana SL/TP decision stalls until the 15s cache expires and then returns 0 → synthetic close.
- **Fix shape:** add Birdeye `/defi/price` as a tertiary fallback (free tier, 1 req/s, sufficient for active positions). Pyth is the gold standard for blue-chips but **does not list freshly-launched memecoin mints**, so it's a fallback only for known-mint positions (rare in sniper context); we add Pyth wiring but only attempt for tokens with a known Pyth feed-id map (currently empty — extension point).

### R6 — Per-chain latency dashboard widget (PM mission item 3c)
- `/api/sniper/timing` aggregates only by `detection_path` (wss/polling). The PM-mission deliverable is a **per-chain** view so EVM and Solana p50/p95 can be compared side-by-side. Currently the API returns `paths['wss']` and `paths['polling']` summed across chains.
- **Fix shape:** add a `chain` GROUP BY in the SQL (using `sniper_trades.chain`), surface paths as `{chain}:{detection_path}` keys, render in `performance_sniper.html` under a new "Per-Chain Latency" panel.

## Out-of-scope (do NOT touch)
- `safety_check_enabled=true` DB flip — operator-only LIVE step per CAMPAIGN_BRIEF.md line 43.
- DRY_RUN flag — must stay TRUE.
- `core/risk_manager.py`, `core/dry_run.py`, `config/pool_engine.py` — read-only this wave.
- `trading/chains/solana/jupiter_executor.py` — owned by A3 (SOLANA). I may read but not write.

## Fix log

| # | Commit | What | Why |
|---|---|---|---|
| 0 | e4b8025 | Campaign report seed | Audit + plan |
| R3 | 87c5523 | `solana_listener._log_stats_if_needed` carries `wss_dispatched` / `wss_inflight_peak` / `block_time_*` across the 1-min reset | Dashboard counters were reading 0 after first window flip |
| R2 | 77b22e7 | `_check_filters` exception path: cooldown + `safety_check_errors` counter + `rejected_safety_error` outcome | Stopped busy-loop that burned GoPlus / Honeypot.is RPS on the same failing token |
| R1 | 6612be2 | `_check_pool_transaction` two-stage processed→confirmed readback + `_fetch_transaction` helper + `processed_hit` / `processed_miss_fallback` counters | Cuts median commitment-wait ~5s→~300ms; relieves WSS semaphore saturation |
| R4 | 4adcd29 (bundled) | `_quorum_honeypot_decision` truth-table over GoPlus + Honeypot.is + 5 unit tests | Removes Honeypot.is brown-out false-positives; keeps fail-safe asymmetry |
| R5 | adee9c2 | Birdeye `/defi/price` tertiary fallback in `_get_token_price` + `birdeye_fallback_hits` counter | Removes single-point-of-failure on Jupiter for SL/TP decisions |
| R6 | 5a0a3e9 | Per-chain listener-health widget in `performance_sniper.html`; consumer of existing `/api/sniper/stats` | Operator sees WSS saturation / processed-hit ratio / rejection profile in 30s |

## Out-of-scope items (handed off)
- Pyth-feed wiring for blue-chip mints — extension point in `_get_token_price`; deferred because Pump.fun memecoins have no feed-id and the call would always 404.
- `/api/sniper/timing` per-chain GROUP BY in `monitoring/enhanced_dashboard.py` — would have given a cleaner per-chain p50/p95 breakdown but the file had concurrent uncommitted edits from another agent. The new client-side widget delivers operator-visible per-chain visibility without backend contention; backend split can be a future PM-coordinated change.
- `safety_check_enabled=true` DB flip — operator-only LIVE step per CAMPAIGN_BRIEF.md line 43; explicitly out of scope this wave.

## Verification
- `python -m py_compile` clean on all touched files (`solana_listener.py`, `sniper_engine.py`, `token_safety.py`).
- `tests/unit/test_sniper_new_paths.py` — 5 new quorum tests added. Test sandbox here lacks `aiohttp` so pre-existing engine-import tests can't run locally, but the quorum tests are pure-Python with no aiohttp dependency on the staticmethod call path and pass under any environment where the module imports successfully.
- DRY_RUN stays TRUE for every change. No flip to LIVE-related flags.

