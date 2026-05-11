# SOLANA_MODULE - Trading-Desk Risk & Live-Readiness Audit

**Auditor:** market-trading-analyst (20+ yrs spot, perps, on-chain)
**Date:** 2026-05-11
**Scope:** `modules/solana_trading/`, `modules/solana_strategies/`, `trading/chains/solana/`, plus shared core (`core/`, `trading/orders/`, `monitoring/`, `main.py`).
**Note:** `PLAN.md` was not found at repo root (only the agent definition at `.claude/agents/market-trading-analyst.md`). Reports `scripts/emergency_stop.py` and `scripts/close_all_positions.py` are **absent** from the repo (verified by directory listing of `scripts/`).

---

## 1. Executive verdict: **AMBER, leaning RED for Drift**

The Solana spot/Jupiter path has had real engineering effort poured into safety: SafetyEngine with sell-route verification, escalating slippage retries, circuit breakers, scam-name blacklist, Pump.fun trailing-stop tiers, stuck-position emergency-close pipeline, encrypted private keys via secrets manager. The **Jupiter spot leg is roughly ready for a small-capital live trial** *after* the four P0 fixes below.

The **Drift perpetuals leg is NOT live-ready** — it lacks any DRY_RUN gate inside `drift_helper.open_position` / `close_position`, has no leverage cap enforcement, no liquidation-buffer policy, no funding-decay sizing, and no perp-position reconciliation. If `SOLANA_STRATEGIES=drift` is set with `DRY_RUN=false`, the helper will hit `place_perp_order` directly with no guard rails. That's a one-keystroke way to blow up the account during a funding flip on SOL-PERP.

Top P0 live-readiness gap: **open positions are NOT reconciled on restart** (`_load_historical_stats` at `modules/solana_trading/core/solana_engine.py:1720` only loads aggregate stats — `active_positions` is reset to `{}` at line 1113). Any SPL tokens sitting in the wallet from before a crash are orphaned; the bot won't try to exit them, and a Drift perp will keep accruing funding with no monitor watching it.

Top profit-leak: **Jupiter `_get_quote` always passes `slippageBps=200`+ as a static value** with no fresh sanity-check against actual price-impact (`modules/solana_trading/core/solana_engine.py:1110`, `trading/chains/solana/jupiter_executor.py:75`). On low-liquidity pump.fun tokens the 12% close slippage is fine on average but is paid in full every trade — a bot doing 50 trades/day burns 6% of capital per round-trip just to slippage cap.

---

## 2. DRY_RUN propagation audit

| Code path | File:line | Honors DRY_RUN? | Notes |
|---|---|---|---|
| Solana orchestrator entrypoint | `modules/solana_trading/main_solana.py:694-700` | YES | Reads `DRY_RUN` env, allows `--dry-run` flag override, re-exports to env. |
| `SolanaTradingEngine.__init__` | `modules/solana_trading/core/solana_engine.py:1073-1075` | YES | `self.dry_run` set from env, logged. |
| Engine BUY path | `solana_engine.py:3035` | YES | `if self.dry_run:` -> simulated swap with fake signature. |
| Engine CLOSE path | `solana_engine.py:3295` | YES | Same pattern. |
| `JupiterExecutor.execute_trade` | `trading/chains/solana/jupiter_executor.py:207-214` | **NO — BUG** | `if self.dry_run:` is **indented inside** `if not self.session:` (line 207). On the **first** call (session is None) the gate works; on every subsequent call, session exists, the dry_run check is skipped, and code falls through to live `_get_quote`/`_execute_swap`. Mitigated by a second `if self.dry_run:` at line 642 inside `_execute_swap`, so a real signing call is avoided — but this is fragile, depends on flow never being refactored, and burns Jupiter quote API calls unnecessarily. **P0 fix.** |
| `JupiterExecutor._execute_swap` | `jupiter_executor.py:642` | YES | Secondary guard catches the bug above. |
| `JupiterHelper.execute_swap` | `modules/solana_strategies/jupiter_helper.py:906-1009` | **NO** | No `dry_run` parameter; relies on caller (engine) to gate. Safe only because solana_engine guards at 3035, but any future caller will sign and broadcast for real. **P1 fix.** |
| `DriftHelper.open_position` | `modules/solana_strategies/drift_helper.py:236-291` | **NO — P0** | Directly calls `drift_client.place_perp_order` (line 278). No `dry_run` branch anywhere in the file. |
| `DriftHelper.close_position` | `modules/solana_strategies/drift_helper.py:293-344` | **NO — P0** | Same story. |
| `solana_config_manager.is_dry_run` | `modules/solana_strategies/solana_config_manager.py:286-289` | YES | Property reads env. Used? Only indirectly. |
| `engine.py` (shared root engine) BUY DRY_RUN | `core/engine.py:1023` | YES | Standard pattern. |
| `engine.py` BUY balance-check skip | `core/engine.py:987` | YES | Skipped in DRY_RUN. |
| `engine.py` LIVE final safety check | `core/engine.py:1212-1213` | YES | `is_dry_run` defined, gates balance/position checks. |
| `OrderManager` | `trading/orders/order_manager.py:217-221, 520-538` | YES | `self.dry_run` flag, logged on init in CRITICAL. |
| `main.py` orchestrator | `main.py:266-273` | INDIRECT | Each child process reads `DRY_RUN` from its own env at boot; orchestrator forwards env via `subprocess.Popen` (not shown). |

**Verdict:** ~90% covered, but the **Drift helper is wide open** and the JupiterExecutor primary gate has the wrong indentation. These two items would let a misconfigured operator put real money on the line believing they're paper trading.

---

## 3. Risk-policy coverage matrix

| Policy | Required | Implemented? | Where | Gap |
|---|---|---|---|---|
| Per-trade max position USD | Yes | YES | `core/risk_manager.py:220` `max_position_size_usd=10.0`; cap at `risk_manager.py:971-972` | Hard-coded default; Solana engine has separate `position_size_sol` knob at `solana_engine.py:1106`. Two sources of truth. |
| Per-trade % of capital cap | Yes | PARTIAL | `risk_manager.py:219` `max_position_size_pct=10` | Solana engine bypasses this, uses raw SOL float. |
| Per-hour max trades | Yes | NO | — | No hourly rate limiter in Solana module. Pump.fun launches can spam dozens per hour. |
| Per-day max number of new bets | Yes | NO | `risk_metrics.daily_trades` is **counted** at `solana_engine.py:189` but never **enforced** as a cap. |
| Per-day $ burn / loss cap | Yes | YES | `daily_loss_limit_sol`, `solana_engine.py:1103/1109`, gate at `risk_metrics.can_trade` (`solana_engine.py:204-220`) | Default 5 SOL (~$1000) — way too loose for an unproven live config. |
| Max concurrent positions | Yes | YES | `max_positions` config, default 3 (`solana_engine.py:1067, 407`) | Reasonable. |
| Consecutive loss block (time-based) | Yes | YES | `RiskMetrics.trigger_loss_block`, progressive 2h→4h→6h (`solana_engine.py:247-275`) | Good. |
| Circuit breaker on error rate | Yes | YES | `risk_manager.py:261-328` shared; SafetyEngine local (`safety_engine.py:280-303`) | Two parallel circuit breakers, not unified — risk of one tripping and the other not knowing. |
| Slippage cap on Jupiter swap | Yes | YES | `jupiter_executor.py:75` (50 bps default), `solana_engine.py:1110` (200 bps default), SafetyEngine per-strategy (`safety_engine.py:124-169`) | Three sources of truth, each can override. Pump.fun close uses 1200bps. Defaults inconsistent. |
| Stale-quote rejection | Yes | YES | `jupiter_executor.py:526-556` rejects quotes > 10 s old | Excellent. Not implemented in `jupiter_helper.execute_swap` though — fresh quote each time, no re-quote-before-sign check. |
| Honeypot pre-buy check (sell route) | Yes | YES | `safety_engine.py:171-278`, called at `solana_engine.py:3083-3101` | Good. Note: returns True (allow trade) on rate-limit error (line 224) — pragmatic but conditionally unsafe. |
| Scam token name blacklist | Yes | YES | `modules/solana_trading/core/scam_blacklist.py:40-95` | Aggressive name regex; persisted to DB. Good. |
| Per-bet cap as % of capital (memecoin) | Yes | PARTIAL | Position size is fixed in SOL — does not scale to capital dynamically. | If capital grows 10x and `position_size_sol` is not adjusted, exposure ratio is wrong. |
| ATA / token-account sprawl cleanup | Yes | NO | No automated rent-recovery / ATA closure routine | Over weeks of trading hundreds of dust ATAs accumulate; ~0.002 SOL each in locked rent. |
| Wallet hot-key plaintext check | Critical | OK (encrypted) | Fernet-encrypted in DB; decrypted via `_get_decrypted_private_key` at `solana_engine.py:1254-1320`; **fallback to plain env at line 1282** | Acceptable, but plaintext env fallback is a footgun if anyone sets `SOLANA_MODULE_PRIVATE_KEY` directly. |
| Failed-tx SOL-burn budget per hour | Yes | NO | Failures recorded in SafetyEngine for circuit-break, but **no SOL-spent-on-failed-tx** running counter. | Pump.fun launches with `skipPreflight=false` still cost ~0.0005 SOL each to fail. 100 failures/hr = 0.05 SOL = $10 wasted. |
| RPC outage → forced-exit/freeze policy | Yes | PARTIAL | RPC rotation at `solana_engine.py:288-346`; helius failover via Pool Engine. **No "freeze new entries / flatten existing" policy** if all RPCs are red. | Bot will keep trying to monitor positions it can't price — risk of unbounded drawdown during a Helius outage. |
| Position reconciliation on restart (SPL) | Yes | **NO — P0** | `_load_historical_stats` (`solana_engine.py:1720`) loads aggregate counters only. `self.active_positions = {}` initialized empty at `solana_engine.py:1113`. SPL tokens in wallet from a crash are not picked up — except indirectly via the SafetyEngine "found in wallet but not tracked" path at `solana_engine.py:2027-2028` which only triggers if a stuck-position retry happens to fire. |
| Position reconciliation on restart (Drift) | Yes | **NO — P0** | `DriftHelper.get_positions` exists (`drift_helper.py:163-203`) but is never called in startup. |
| Idempotent order IDs | Yes | YES | `uuid.uuid4()` per trade. | No idempotency key sent to Jupiter, but Jupiter is stateless so OK. |
| Heartbeat → flatten/freeze | Yes | PARTIAL | Health endpoint at `:8082/healthz` (main_solana.py:128-143). No "if dashboard hasn't polled me in N seconds → freeze". |
| Emergency stop script reachable | Yes | **NO — P0** | `scripts/emergency_stop.py` does not exist. `scripts/close_all_positions.py` does not exist. Manual close is via HTTP POST `/close-all-positions` (main_solana.py:337-361) — works but requires the engine to be alive. |
| Drift: liquidation oracle lag check | Yes | **NO — P0** | No oracle-staleness gate before opening perp. |
| Drift: max leverage per market | Yes | **NO — P0** | `place_perp_order` at `drift_helper.py:278-284` takes raw `base_asset_amount`; no leverage clamp. |
| Drift: funding-rate decay sizing | Yes | NO | `get_funding_rate` exists (`drift_helper.py:346-375`) but is read-only, never feeds back into position sizing. |

---

## 4. Order / Position lifecycle review

**Entry (Jupiter spot):**
1. Engine main loop → `_check_trading_signals` → `_open_position` (`solana_engine.py:~2900`)
2. Pre-buy: scam-blacklist check, price-change reality check, real-time buy/sell pressure (`solana_engine.py:2950-2958`)
3. Pre-buy: balance check (`solana_engine.py:3043-3049`)
4. Pre-buy: SafetyEngine circuit-breaker (`solana_engine.py:3056-3060`)
5. Pre-buy: pump.fun graduation test quote (`solana_engine.py:3066-3079`)
6. Pre-buy: sell-route honeypot verification (`solana_engine.py:3083-3101`)
7. Strategy-specific slippage selection (`solana_engine.py:3104-3115`)
8. Execute via `JupiterHelper.execute_swap` (LIVE) or fake signature (DRY_RUN)
9. Post-fill: 4-attempt retry to read actual balance with delays 3/5/8/12s (`solana_engine.py:3139-3146`)
10. Post-fill: recalc entry_price from actual SOL spent / tokens received (`solana_engine.py:3157-3162`) — **excellent, this is a senior-trader detail**
11. Append to `self.active_positions[token_mint]`, increment exposure

**Issue 1:** Step 9 is correct but takes up to 28 s to confirm the balance. If a price collapse begins inside that window, the position is unmonitored. No "tight watch" interim using estimated amount. (`solana_engine.py:3139`)

**Issue 2:** No price-impact cap before sign. SafetyEngine's `verify_sell_route` checks SELL route, not buy slippage. A pump.fun token with 50% price impact will still execute because the slippage limit was set to 1200 bps but actual impact was 5000 bps — Jupiter will simulate-fail (`0x1771`), but the bot will retry with even higher slippage (per `retry_slippage_increment_bps=400`).

**Exit (Jupiter spot):**
- Tiered trailing stops + partial exits implemented (`solana_engine.py:2256-2570`) — sophisticated, tier0 → moon level. **Good.**
- Strategy-specific TP/SL: `jupiter_take_profit_pct`, `pumpfun_take_profit_pct` (`solana_engine.py:3006-3013`)
- Emergency exits on rapid decline (`emergency_rapid_decline`, `emergency_peak_decline` at lines 2355, 2392)

**Issue 3:** No max-hold-time cap on the position itself. If a pump.fun token sits flat for 6 hours, capital is locked. The shared `core/engine.py` has `max_hold_time: 60` (`engine.py:1058`) but that's for the root engine, not Solana module.

**Entry/Exit (Drift):**
- `open_position` (`drift_helper.py:236-291`): direction LONG/SHORT, market index, base_amount. **No leverage check, no margin-buffer check, no liquidation-price calculation.**
- `close_position` (`drift_helper.py:293-344`): opposite-direction market order.
- **No DRY_RUN.** **No position reconciliation.** **No funding-cost meter.**

**Issue 4:** Drift positions are not surfaced in `active_positions`, which is keyed by `token_mint` — Drift perps don't have a token mint. They'd need a parallel `active_perps` map. None exists.

---

## 5. Profit / loss-leak inventory

| # | Leak | Severity | File:line | Daily $ impact (est, $400 book) |
|---|---|---|---|---|
| L1 | Stale 200 bps default Jupiter slippage applied even on liquid majors (BONK, JTO) | HIGH | `solana_engine.py:1110`, `jupiter_executor.py:75` | $0.50-2 / trade × 20 trades = $10-40/day |
| L2 | Pump.fun close uses 1200 bps base + 400 bps retry escalation, capped 3500 bps | HIGH | `safety_engine.py:75-80` | On a $5 position, worst case $1.75 paid in slippage on close = 35% gone |
| L3 | `amount_out_min` math is "slippage_bps_to_minimum" — Jupiter handles, but stale quote can still pass | MED | `jupiter_executor.py:_get_quote` | Small, but real on volatile launches |
| L4 | ATA rent (~0.002 SOL each) locked in dust accounts forever | LOW | None — no cleanup | $0.40 / week per 10 dead tokens |
| L5 | Failed-tx SOL burn not budgeted; 0x1771 retries can run away | MED | `solana_engine.py:3104-3115` (no failed-tx-per-hour gate) | $5-20/day in failure storms |
| L6 | No live re-quote before sign in `jupiter_helper.execute_swap` (1-2s between quote and sign) | MED | `jupiter_helper.py:955-979` | 5-15 bps of unexpected slippage |
| L7 | Drift funding paid silently — no `funding * notional` decay accounting | HIGH (Drift only) | `drift_helper.py:346-375` (read-only) | Up to 0.1%/8h on a high-funding SOL-PERP = $1/day on $1000 notional |
| L8 | RPC failover doesn't have warm pool — first call after rotation can take 2-3s extra | LOW | `solana_engine.py:309-329` | Indirect — missed entries on fast moves |
| L9 | Scam-name regex `r'^[A-Z]{1,2}\d+$'` will reject legit tokens like "A1" or "X3"; tradeable inventory unnecessarily shrunk | LOW | `scam_blacklist.py:70` | Opportunity cost only |
| L10 | `safety_engine.verify_sell_route` returns `True` on rate-limit (`safety_engine.py:222-224`) — opens door to honeypot during Helius 429 storm | HIGH | Same | One bad fill = full position loss = $5-10 |
| L11 | `sol_price_usd` hardcoded to 200.0 initially (`solana_engine.py:1140`); USD-denominated stats wrong until first refresh | LOW | Same | Reporting/risk-budget noise |
| L12 | Telegram alert spam under loss-storms can mask the actually-critical alerts | MED | `solana_engine.py:1231-1238` | Indirect, but real ops cost |

---

## 6. Live-readiness checklist (pass/fail per item, evidence)

| Item | Status | Evidence |
|---|---|---|
| `DRY_RUN` honored on every send/order/sign path | **FAIL** | `jupiter_executor.py:210-214` indentation bug; `drift_helper.py:236-344` has no DRY_RUN gate at all |
| Per-trade max loss enforced | PASS (Jupiter) / FAIL (Drift) | Engine has `stop_loss_pct`; Drift has none |
| Per-hour max loss enforced | FAIL | No hourly accumulator; only daily |
| Per-day max loss enforced | PASS | `RiskMetrics.daily_loss_limit_sol` + `can_trade` gate |
| Per-day max new trades enforced | FAIL | `daily_trades` counted but not capped |
| Position reconciliation on startup (SPL) | **FAIL — P0** | `_load_historical_stats` only loads aggregate counts |
| Position reconciliation on startup (Drift) | **FAIL — P0** | `drift_helper.get_positions` exists but never invoked at startup |
| Idempotent order IDs / nonce management | PASS | UUID per trade |
| Heartbeat to dashboard | PARTIAL | `/healthz` exists, no auto-freeze on stale |
| Emergency stop wired and reachable | **FAIL — P0** | `scripts/emergency_stop.py` missing; only HTTP endpoint exists |
| Telegram alerts on circuit-break / loss-block | PASS | `solana_engine.py:1231-1238` |
| Slippage cap enforced on Jupiter swap | PASS (with caveats) | `safety_engine.get_slippage_for_strategy` |
| Stale-quote rejection (Jupiter) | PASS (executor) / FAIL (helper) | `jupiter_executor.py:526-556` good; `jupiter_helper.execute_swap` has no expiry check |
| Honeypot pre-check enforced | PASS (Jupiter) | `solana_engine.py:3083-3101` |
| Drift leverage cap enforced | **FAIL — P0** | `drift_helper.py:278-284` |
| Drift liquidation-buffer enforced | **FAIL — P0** | None |
| Drift funding-aware sizing | FAIL | None |
| Drift oracle staleness gate | FAIL | None |
| ATA cleanup / rent recovery | FAIL | None |
| RPC blackout policy (freeze on all-RPC-fail) | FAIL | RPC rotates but bot keeps trying |
| Encrypted private key handling | PASS | Fernet via secrets manager, env fallback present but acceptable |
| Failed-tx SOL-burn budget | FAIL | None |
| Position size scales with capital | FAIL | Hardcoded `position_size_sol` not % of `capital_sol` |

**Verdict per leg:**
- Jupiter spot path: **AMBER**. Tradeable in live with $50–100 capital cap after P0 fixes 1–4 (see backlog).
- Pump.fun: **AMBER**. Sophisticated trailing-stop ladder, but every Pump.fun-specific failure mode (rapid rug, dev sell, LP unlock) is mitigated only by *price action*, not by *event detection*. OK for a paper-trial.
- Drift perp: **RED**. Do not enable `SOLANA_STRATEGIES=drift` in live until backlog SOL-RM-09 / -10 / -11 are done.

---

## 7. Proposed action backlog

| ID | Priority | Title | Effort | Owner | Where |
|---|---|---|---|---|---|
| SOL-RM-01 | P0 | Fix `JupiterExecutor.execute_trade` DRY_RUN indentation bug | trivial | sm-contract / infra | `trading/chains/solana/jupiter_executor.py:207-214` — move the `if self.dry_run:` block out of the `if not self.session:` scope |
| SOL-RM-02 | P0 | Add DRY_RUN gate to `DriftHelper.open_position` and `close_position` | small | analyst | `modules/solana_strategies/drift_helper.py:236, 293` — return a mocked tx_sig on DRY_RUN |
| SOL-RM-03 | P0 | Reconcile SPL positions on startup | medium | analyst | New `SolanaTradingEngine._reconcile_positions()` that scans wallet ATAs via `getTokenAccountsByOwner`, cross-references DB `solana_trades` with `status='open'`, rebuilds `active_positions` |
| SOL-RM-04 | P0 | Reconcile Drift perp positions on startup | medium | analyst | Call `drift_helper.get_positions()` in `_init_drift` and rebuild active_perps map |
| SOL-RM-05 | P0 | Create `scripts/emergency_stop.py` + `scripts/close_all_positions.py` | medium | infra | Shell-callable, sets a kill-flag the engine polls; calls `/close-all-positions` endpoint with retry |
| SOL-RM-06 | P0 | Drift leverage hard-cap per market | small | analyst | `drift_helper.open_position` — clamp `base_amount` to `leverage_cap * collateral / mark_price`; reject if breach |
| SOL-RM-07 | P0 | Drift liquidation-buffer gate | medium | analyst | Reject `open_position` if implied liquidation price within X% (configurable, default 15%) of mark |
| SOL-RM-08 | P0 | Drift oracle-staleness gate | small | analyst | Reject if `market.amm.last_oracle_slot` > N slots behind current slot |
| SOL-RM-09 | P1 | Funding-aware sizing for Drift | medium | analyst | `expected_pnl_per_hour = notional * (price_edge - funding_rate)`; if negative, skip |
| SOL-RM-10 | P1 | Per-hour and per-day **new-trade-count** caps | small | analyst | Add counter to `RiskMetrics`, gate in `_check_trading_signals` |
| SOL-RM-11 | P1 | Failed-tx SOL-burn budget (per hour) | small | analyst | Counter in SafetyEngine, trip circuit-break if > 0.05 SOL/hr |
| SOL-RM-12 | P1 | RPC-blackout freeze policy | small | infra | If `rpc_manager.failed_rpcs == set(rpc_urls)` for > 30s → set `is_running=False`, alert critical |
| SOL-RM-13 | P1 | Position size as % of `capital_sol` not absolute SOL | small | analyst | `position_size_sol = capital_sol * position_size_pct`; clamp to min/max |
| SOL-RM-14 | P1 | Add `dry_run` parameter to `JupiterHelper.execute_swap` defensively | trivial | analyst | Defense in depth |
| SOL-RM-15 | P1 | Re-quote-before-sign in `jupiter_helper.execute_swap` | small | analyst | Refresh quote if > 5s old between quote and sign |
| SOL-RM-16 | P1 | Tighten `safety_engine.verify_sell_route` rate-limit fallback to "block" not "allow" for first-time-seen tokens | small | analyst | `safety_engine.py:218-224` |
| SOL-RM-17 | P2 | ATA cleanup job (rent recovery) | medium | analyst | Cron-style task that closes 0-balance ATAs |
| SOL-RM-18 | P2 | Unify circuit breakers (RiskManager + SafetyEngine into single state machine) | medium | analyst | Both currently independent; can trip differently |
| SOL-RM-19 | P2 | Dashboard heartbeat → if stale > 60s, freeze new entries | small | infra | `_status_reporter` in main_solana already runs every 120s; flip the dependency |
| SOL-RM-20 | P2 | Loosen scam-blacklist regex `^[A-Z]{1,2}\d+$` (rejects legit short names) | trivial | analyst | `scam_blacklist.py:70` |
| SOL-RM-21 | P2 | Max-hold-time cap on every Solana spot position | small | analyst | E.g., 240 min, exit at market regardless of P&L |
| SOL-RM-22 | P3 | Replace plaintext env fallback for `SOLANA_MODULE_PRIVATE_KEY` with mandatory secrets-manager-only | small | infra | `solana_engine.py:1281-1282` |
| SOL-RM-23 | P3 | Capital-allocation knob: % of total bot capital routed to Solana | small | analyst | Currently fixed per-module |

---

## 8. Recommended go-live sequence

1. Land SOL-RM-01, -02, -05, -14 (the four trivial-to-small DRY_RUN/safety fixes).
2. Land SOL-RM-03 (SPL position reconciliation). Test by killing the bot mid-position and restarting — verify position re-appears in `active_positions`.
3. Land SOL-RM-10, -11, -13. These are the desk-level guardrails for live-but-small trading.
4. Run Jupiter-only DRY_RUN for **48 hours continuous** with realistic config; verify circuit-breaker trips on synthetic loss storms; verify scam-blacklist hits real garbage.
5. Cut Jupiter-only LIVE with **`max_position_size_usd=5`, `max_positions=2`, `daily_loss_limit_sol=0.25` (~$50)**. Run for 72 hours, monitor `/healthz` + Telegram.
6. **Drift stays disabled** until SOL-RM-02, -04, -06, -07, -08, -09 are merged and DRY_RUN-tested with synthetic funding flips.
7. Drift cut-over: first market = SOL-PERP only, leverage cap 2x, single-position, 5-SOL collateral.
8. Pump.fun stays disabled in live until at least 1 week of clean Jupiter-only live data.

---

## 9. Open questions

1. What is the **total capital** intended for Solana module? `capital_sol=10.0` default is loose; needs to be set per-deployment.
2. Where does `sol_price_usd` get its first non-stub value? `_update_sol_price` is mentioned at `solana_engine.py:1197` — if that fails on start, all USD-denominated risk math is wrong.
3. Should the Solana module honor the **shared `RiskManager.consecutive_losses`** state, or only its own `RiskMetrics.consecutive_losses`? Currently they don't talk to each other.
4. Is Helius the only paid RPC, or are we paying for Triton / QuickNode too? If Helius is sole, an outage is a sole point of failure.
5. Who owns the `.encryption_key` file in production? File-on-disk with a Fernet key next to encrypted DB blobs is acceptable IF the disk is encrypted at rest — confirm with infra.
6. Are Drift markets currently traded LIVE by another bot from the same wallet? If yes, our reconciliation must not interpret another bot's positions as ours.
7. Pump.fun "graduated to Raydium" detection (`solana_engine.py:3067-3079`) relies on a Jupiter quote succeeding — is there a more reliable on-chain check we can add?
8. What is the dollar value of a "$10/day burn cap" target? Current `daily_loss_limit_sol=5.0` is far higher.

---

## 10. Bottom line

The Jupiter spot leg of the Solana module shows trader-grade thought (entry-price post-fill correction, tiered trailing stops, sell-route verification, scam blacklist, escalating-slippage retries). It is **almost** live-ready — four P0 fixes (DRY_RUN indentation, position reconciliation, emergency stop script, capital-relative sizing) stand between it and a small-capital live deployment. The **Drift leg should be treated as untested code** and disabled until SOL-RM-02 through SOL-RM-08 are merged. Pump.fun is a coin flip whether the trailing-stop logic preserves edge in live — paper-trade it for at least a week before flipping.

Overall verdict: **AMBER, leaning RED until Drift is fenced off or hardened.**
