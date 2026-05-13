# SNIPER_MODULE - Trading-Desk Risk & Live-Readiness Audit

**Auditor:** market-trading-analyst (20+ yrs spot, perps, on-chain incl. memecoin launches)
**Date:** 2026-05-11
**Scope:** `modules/sniper/main_sniper.py`, `modules/sniper/core/*.py`, plus shared core (`core/`, `trading/orders/`, `monitoring/`, `main.py`).
**Note:** `PLAN.md` was not found at repo root. `scripts/emergency_stop.py` and `scripts/close_all_positions.py` are **absent** from `scripts/` (verified by directory listing).

---

## 1. Executive verdict: **RED**

The sniper module is a fast-and-loose launch-grabber wrapped around a competent token-safety pre-check (GoPlus / Honeypot.is / RugCheck.xyz). The detection plumbing on both EVM (PairCreated polling) and Solana (multi-AMM polling) works. The token-safety report is genuinely useful. But the **trade-execution path has dealbreaker risk-policy holes** that mark it as not-yet-live-ready for anything beyond test wallets:

1. `_execute_evm_buy` and `_execute_evm_sell` set **`amount_out_min = 0`** (`modules/sniper/core/trade_executor.py:621, 709`). That is the literal "accept any output" setting — a 99.9% slippage MEV sandwich is fully permitted. The comment `"Accept any amount (risky, but for speed)"` confirms intent. **No live bot should ship with this.**
2. **No DRY_RUN simulated-sell test before commit.** Token safety APIs (GoPlus / Honeypot.is) are external opinions; the gold-standard pre-snipe check is "simulate buy + simulate sell on a fork" — the sniper does neither.
3. **No per-snipe capital cap as % of book.** `self.trade_amount` is a raw float loaded from DB (`sniper_engine.py:188`, default `0.1` SOL/ETH); does not scale with capital.
4. **No per-hour or per-day max-snipes rate limit.** During a launch farm event, the bot will fire on every passing pair (only blocked by `_rejected_cache` 5-min cooldown per token).
5. **No exit on dev-wallet sell, no exit on LP unlock, no time-stop, no trailing stop, no TP ladder.** Exit is a fixed 50% TP / -20% SL hardcoded in `_monitor_active_snipes` (`sniper_engine.py:580-581`), checked every 1 s. The values set in DB at `sniper_engine.py:207-210` (`take_profit_pct`, `stop_loss_pct`) are read into instance attributes but **never used** in the monitor loop — the monitor uses local-variable defaults. **P0 bug.**
6. **No anti-front-run / no anti-dev-snipe check.** If a dev wallet is the very first buyer of its own pool, our sniper is entering on a setup designed to dump on us.
7. **`amount_out_min=0` makes any honeypot-tax token a 100% loss** — the buy will execute, the sell won't, and the bot will only realize it on the next monitoring tick when price feeds return zero.
8. **No multi-wallet rotation.** A sniper that hits hundreds of launches from one address is trivially identifiable and front-runnable.

Top P0 live-readiness gap: **`amount_out_min=0` in EVM swap construction**, combined with no simulated-sell pre-check. One bad honeypot launch = full `trade_amount` lost.

Top profit-leak: **No partial profit-taking / no trailing stop.** A 100x runner is exited at 50% TP for a 1.5x — leaving 98.5x of upside on the table.

---

## 2. DRY_RUN propagation audit

| Code path | File:line | Honors DRY_RUN? | Notes |
|---|---|---|---|
| `SniperEngine.__init__` | `modules/sniper/core/sniper_engine.py:114` | YES | Default `self.dry_run = True` |
| `SniperEngine._load_settings` | `sniper_engine.py:212-220` | YES | Reads `DRY_RUN` env; logs mode |
| `SniperEngine._execute_snipe` | `sniper_engine.py:461-516` | INDIRECT | Calls `self.executor.execute_buy` — gating done inside `TradeExecutor` |
| `TradeExecutor.__init__` / `initialize` | `modules/sniper/core/trade_executor.py:109, 187` | YES | Reads `DRY_RUN` env |
| `TradeExecutor.execute_buy` | `trade_executor.py:242-243` | YES | `if self.dry_run: return _simulate_buy` |
| `TradeExecutor.execute_sell` | `trade_executor.py:274-275` | YES | Same |
| `TradeExecutor._execute_solana_buy` | `trade_executor.py:284-358` | NO (relies on caller gate) | If invoked directly, will sign and send |
| `TradeExecutor._execute_evm_buy` | `trade_executor.py:585-669` | NO (relies on caller gate) | If invoked directly, will sign and send |
| `TradeExecutor._sign_and_send_solana_tx` | `trade_executor.py:495-581` | NO | Defensive guard absent |
| `main_sniper.py` entrypoint | `modules/sniper/main_sniper.py:127-263` | Forwarded from env | Does not validate/log DRY_RUN at the orchestrator layer |

**Verdict:** Caller-gated DRY_RUN works correctly **for the current code paths**. But this is fragile — any future call site that hits `_execute_evm_buy` / `_execute_solana_buy` directly will sign real txs in DRY_RUN mode. The pattern in `jupiter_executor.py` (DRY_RUN check inside `_execute_swap` at line 642) is safer.

---

## 3. Risk-policy coverage matrix

| Policy | Required | Implemented? | Where | Gap |
|---|---|---|---|---|
| Per-snipe capital cap (≤ 0.X% of capital) | Yes | **NO — P0** | `sniper_engine.py:188` `self.trade_amount` is a raw float (default 0.1) | Does not scale with capital. A `trade_amount=0.1 SOL` on a $50 wallet = 40% per snipe. |
| Per-hour max snipes | Yes | **NO — P0** | None | Only per-token 5-min cooldown at `sniper_engine.py:316-322`. No global rate limit. |
| Per-day max snipes | Yes | **NO — P0** | None | Same. |
| Per-day max $ burn / loss | Yes | **NO** | None | Sniper bypasses shared `RiskManager` entirely. |
| Honeypot pre-check (API) | Yes | YES | `token_safety.py:114-300` (GoPlus, Honeypot.is, RugCheck) | Good. |
| Honeypot pre-check (simulated buy+sell) | Yes | **NO — P0** | None | Critical for new launches not yet in API DBs. |
| Auto-exit on dev-wallet sell | Yes | **NO** | None | A signature of every rug. |
| Auto-exit on LP unlock | Yes | **NO** | None | |
| Time-stop (max hold) | Yes | **NO** | None | A pump that doesn't pump in 30 min is dead capital. |
| Trailing stop / TP ladder | Yes | **NO** | `sniper_engine.py:578-625` only fixed TP/SL | TP=50%, SL=-20% hardcoded; a 100x runner exited at 1.5x. |
| Re-entry rules | Yes | NO | None | Once exited, no logic for re-entry on dip. |
| Multi-wallet rotation | Yes | **NO** | Single `evm_private_key` / `solana_private_key` (`trade_executor.py:174-175`) | Easily fingerprintable. |
| Race-loss timeout (abort if no fill in N seconds) | Yes | **NO** | `_execute_evm_buy` uses `wait_for_transaction_receipt(timeout=60)` (`trade_executor.py:649`) — that's confirmation, not abort | A 60s wait on a launch is forever. |
| Front-run prevention / dev-snipe avoidance | Yes | **NO** | None | No check on dev_wallet == first_buyer |
| Slippage cap enforcement (EVM) | Yes | **NO — P0** | `trade_executor.py:621` `amount_out_min = 0` | "Accept any amount (risky, but for speed)" — see comment. |
| Slippage cap enforcement (Solana) | Yes | YES (Jupiter side) | `slippage_bps` passed to Jupiter quote (`trade_executor.py:308`); user-configured `self.slippage` | OK on Solana. |
| Gas / priority-fee cap | Yes | PARTIAL | `priority_fee` is configurable per-call but no global hourly cap | A storm of failed snipes will burn ETH |
| Honeypot detection on Solana | Yes | YES | RugCheck.xyz (`token_safety.py:316-372`) + freeze-authority check | Good. |
| Tax checks (buy/sell tax) | Yes | YES | `token_safety.py:170-180`, gate at `sniper_engine.py:387-391` | Good. |
| Min liquidity threshold | Yes | YES | `min_liquidity` config; gate at `sniper_engine.py:394-398` | Good (default $1000). |
| Top-holder concentration check | Yes | PARTIAL | `top_holder_percentage > 50` → reject (`token_safety.py:62`) but this is INSIDE `is_safe_to_snipe` which is **never called from `_check_filters`** | The check exists but doesn't gate. |
| Position reconciliation on restart | Yes | **NO — P0** | `self.active_snipes = {}` initialized empty at `sniper_engine.py:105`. DB has snipe records (`_log_snipe_to_db`) but no startup reload. |
| Idempotent order IDs | Yes | YES | `uuid.uuid4()` in `_log_snipe_to_db` |
| Heartbeat / dashboard freeze | Yes | NO | No `/healthz` endpoint in sniper module (unlike Solana module) |
| Emergency stop wired | Yes | **NO — P0** | `scripts/emergency_stop.py` missing; sniper has no HTTP endpoint either |
| Encrypted private key handling | Yes | YES | Fernet via secrets manager (`trade_executor.py:140-159`) |
| EVM tx confirmation timeout | Yes | YES (60s) | `trade_executor.py:649` |
| EVM tx revert handling | Yes | YES | `receipt['status'] == 1` check |
| Failed-tx gas-burn budget | Yes | **NO** | No counter |
| Telegram alerts on snipe / exit | Yes | INDIRECT | Via `monitoring.telegram_bot` controller (`main_sniper.py:236-249`), not granular trade alerts |
| Test mode lower-bound clamp | Yes | YES | `test_mode_min_liquidity=10.0` default (`sniper_engine.py:121`) |

---

## 4. Order / Position lifecycle review

**Detection:**
- EVM: `EVMListener.get_new_pairs` polls `PairCreated` event logs across last 5 blocks (`evm_listener.py:96-129`). Polling-based; misses sub-block launches.
- Solana: `SolanaListener.get_new_pools` polls Raydium V4, Raydium CPMM, Pump.fun, Orca, Meteora program IDs (`solana_listener.py:38-44`).

**Filter (`_check_filters` `sniper_engine.py:305-411`):**
1. Cooldown check (5 min per token)
2. `token_safety.check_token` → SafetyReport
3. Reject if `is_honeypot` (line 367)
4. Reject if `DANGER` rating (line 373)
5. Reject if buy/sell tax over cap (line 387)
6. Reject if liquidity below cap (line 394)

**Issue 1:** `SafetyReport.is_safe_to_snipe` (`token_safety.py:54-64`) has additional checks — `top_holder_percentage > 50` → reject — but this method is **never called**. The top-holder gate is dead code.

**Issue 2:** Test mode (`sniper_engine.py:336-346`) raises max_tax to 50% and allows DANGER tokens. This is intentional for testing but if `test_mode=true` leaks into production via DB config drift, the bot will snipe garbage. There is no `DRY_RUN`-implies-test_mode safety crosswire.

**Execution (`_execute_snipe` `sniper_engine.py:461-516`):**
1. Call `executor.execute_buy(token_address, chain, amount_in=self.trade_amount, slippage, priority_fee)`
2. On success, store entry data in `data` dict, move to `active_snipes`
3. Log to DB (`_log_snipe_to_db` at line 518)

**Issue 3:** `data['entry_price'] = self.trade_amount / result.amount_out` (`sniper_engine.py:494`) — this gives entry price as **native-per-token** (e.g., SOL per token), but `_get_token_price` returns **USD-per-token via DexScreener** or **SOL-per-token via Jupiter v4** (`sniper_engine.py:627-655`). The units do not match. P&L math is **systematically wrong**.

**Issue 4:** `_execute_evm_buy` and `_execute_evm_sell` set `amount_out_min = 0` (lines 621, 709). This is a MEV-honeypot's dream. Anyone running a sandwich bot can extract 99% of the trade value. **P0.**

**Issue 5:** No deadline-aware abort. Once the buy tx is dispatched, the bot waits up to 60 s for confirmation. On a launch, a 60-s old buy is already too late.

**Monitoring (`_monitor_active_snipes` `sniper_engine.py:577-625`):**
- Loop every 1 second
- Calls `_get_token_price` per token (DexScreener call per tick → rate-limit hazard at scale)
- Fixed `take_profit_pct = 50.0`, `stop_loss_pct = -20.0` (lines 580-581) — **local variables, ignoring the DB-loaded values at sniper_engine.py:207-210**

**Issue 6 (P0):** The DB-configured `take_profit_pct` / `stop_loss_pct` are loaded into `self.take_profit_pct` / `self.stop_loss_pct` (lines 207-210) but the monitor uses LOCAL variables of the same name (lines 580-581), so the DB config is silently ignored. **One-line fix.**

**Issue 7:** No partial exits. A token that pumps 5x and then crashes still hits TP=50% (1.5x) on the way up, sells everything, misses the rest of the run.

**Exit (`_exit_position` `sniper_engine.py:657-698`):**
- `executor.execute_sell(token_address, chain, amount_in=amount, slippage=self.slippage, priority_fee=self.priority_fee)`
- On success, log exit to DB

**Issue 8:** Same `amount_out_min = 0` problem on EVM sells (`trade_executor.py:709`). The sell can also be sandwich-attacked.

---

## 5. Profit / loss-leak inventory

| # | Leak | Severity | File:line | Daily $ impact (est, $400 book) |
|---|---|---|---|---|
| S1 | `amount_out_min = 0` on EVM buy/sell — 100% slippage permitted | **CATASTROPHIC** | `trade_executor.py:621, 709` | A single sandwich = -50% of `trade_amount` |
| S2 | TP/SL config from DB **never used**; hardcoded 50/-20 in monitor | **HIGH** | `sniper_engine.py:580-581 vs 207-210` | Capping all winners at 1.5x kills the heavy-tail edge — every 100x runner = -98.5x opportunity loss |
| S3 | No partial profit ladder | HIGH | None | Same as S2 — heavy-tail truncation |
| S4 | No multi-wallet rotation; sandwich bots will profile the sniper wallet | HIGH | `trade_executor.py:174-175` | 10-30 bps degradation per trade once profiled |
| S5 | No dev-wallet-sell auto-exit | HIGH | None | Avg rug detected 30-90s after dev dump; bot exits on -20% SL = late |
| S6 | Entry price units mismatch with monitor price units | HIGH | `sniper_engine.py:494 vs 627-655` | Random false TP / SL triggers |
| S7 | `is_safe_to_snipe` top-holder gate is dead code | MED | `token_safety.py:54-64` not invoked | Lets concentrated-supply tokens through |
| S8 | Failed-tx gas burn not budgeted | MED | None | $1-5/day at $1 gwei × misses |
| S9 | Per-launch DexScreener call every 1s for every active snipe — rate limit hazard | MED | `sniper_engine.py:600` | Pricing fails → no exit → loss |
| S10 | No race-loss abort: if buy not filled in N seconds, no auto-cancel | MED | `_execute_snipe` | Locked nonce, missed next opportunity |
| S11 | EVM `priority_fee` set as gwei but combined with `priority_fee + 50` for max_fee — could overpay during low base fee | LOW | `trade_executor.py:636-637` | Small over-tip |
| S12 | Token amount decimal assumption: `amount_tokens = int(amount_in * 1e18)` on sell — assumes 18 decimals always | MED | `trade_executor.py:706` | Tokens with non-18 decimals → wrong amount → tx revert or partial sell |
| S13 | Jupiter quote on Solana sell assumes 6 decimals: `amount_tokens = int(amount_in * 1e6)` | MED | `trade_executor.py:377` | Same problem for SPL tokens with non-6 decimals |
| S14 | No idempotent guard on snipe entry — if `_evaluate_target` fires twice for the same target during the same scan, two buys could fire | LOW | `sniper_engine.py:284-303` | Double exposure on a single launch |
| S15 | `slippage` on Solana converted as `slippage_bps = int(slippage * 100)` — for 10% you get 1000 bps which is fine, but no upper sanity check | LOW | `trade_executor.py:302` | Operator typo "100" → 10000 bps (100%) accepted |

---

## 6. Live-readiness checklist (pass/fail per item, evidence)

| Item | Status | Evidence |
|---|---|---|
| `DRY_RUN` honored on every send/order/sign path | PASS (current call sites) | `trade_executor.py:242-243, 274-275` |
| Per-trade max loss enforced | FAIL | No SL enforcement on EVM (`amount_out_min=0`); SL on monitor is hardcoded |
| Per-hour max trades / max loss enforced | FAIL | None |
| Per-day max trades / max loss enforced | FAIL | None |
| Position reconciliation on startup | FAIL | `active_snipes = {}` at startup |
| Idempotent order IDs | PASS | UUID |
| Heartbeat to dashboard | FAIL | No `/healthz` |
| Emergency stop wired | FAIL | Missing script |
| Honeypot pre-check via API | PASS | GoPlus + Honeypot.is + RugCheck |
| Simulated buy+sell pre-check | FAIL | None |
| Slippage cap on EVM swap | **FAIL — P0** | `amount_out_min=0` |
| Slippage cap on Solana swap | PASS | Jupiter slippage_bps |
| Dev-wallet sell exit | FAIL | None |
| LP unlock exit | FAIL | None |
| Time-stop / max-hold | FAIL | None |
| Trailing stop / TP ladder | FAIL | Fixed 50/-20 |
| Re-entry rules | FAIL | None |
| Multi-wallet rotation | FAIL | Single wallet |
| Race-loss abort | FAIL | None |
| Dev-snipe avoidance | FAIL | None |
| Top-holder concentration gate | FAIL | Dead code |
| Test-mode safety crosswire | FAIL | Can be on in production |
| TP/SL config from DB respected | **FAIL — P0** | Local vars shadow DB config |
| Encrypted private key handling | PASS | Fernet via secrets manager |
| Failed-tx gas-burn budget | FAIL | None |
| EVM tx revert handling | PASS | `receipt['status']` check |
| Decimal-aware token amounts | FAIL | Hardcoded 6 / 18 decimals |

---

## 7. Proposed action backlog

| ID | Priority | Title | Effort | Owner | Where |
|---|---|---|---|---|---|
| SNIPE-RM-01 | P0 | Fix `amount_out_min = 0` on EVM buy and sell | medium | analyst+sm-contract | `trade_executor.py:621, 709` — compute `amount_out_min = expected_out * (1 - slippage_pct)` from a fresh `router.getAmountsOut` call |
| SNIPE-RM-02 | P0 | Wire DB-loaded `take_profit_pct` / `stop_loss_pct` into the monitor loop | trivial | analyst | `sniper_engine.py:580-581` — remove local var, use `self.take_profit_pct` / `self.stop_loss_pct` |
| SNIPE-RM-03 | P0 | Position reconciliation on startup | medium | analyst | At `SniperEngine.initialize`, query `sniper_trades WHERE status='open'`, rebuild `active_snipes` |
| SNIPE-RM-04 | P0 | Create `scripts/emergency_stop.py` and `scripts/close_all_positions.py` | medium | infra | Shell-callable; for sniper, sends `is_running=False` + close-all RPC |
| SNIPE-RM-05 | P0 | Add `/healthz` HTTP endpoint to sniper module | small | infra | Mirror what `modules/solana_trading/main_solana.py:115-188` does |
| SNIPE-RM-06 | P0 | Simulated buy+sell pre-check on EVM (Tenderly / local fork) | large | sm-contract | New `TokenSafetyChecker.simulate_round_trip()` |
| SNIPE-RM-07 | P0 | Per-hour and per-day snipe-count caps | small | analyst | Counters in `SniperEngine`, gate in `_evaluate_target` |
| SNIPE-RM-08 | P0 | Per-snipe capital cap as % of book (not raw float) | small | analyst | `trade_amount = balance * self.position_size_pct_of_balance` |
| SNIPE-RM-09 | P1 | Trailing stop + TP ladder | medium | analyst | Mirror Solana module's tier system (`solana_engine.py:2495-2570`) |
| SNIPE-RM-10 | P1 | Dev-wallet sell detection → auto-exit | medium | analyst | Watch top-3 holder addresses for `transfer` events; if dev sends > 10% to a DEX router → exit |
| SNIPE-RM-11 | P1 | LP unlock detection → auto-exit | medium | analyst | Cron-poll lock contracts (UnicryptV2, TeamFinance) for unlock events |
| SNIPE-RM-12 | P1 | Time-stop (max hold) | trivial | analyst | If `(now - entry_time) > max_hold_minutes` → close |
| SNIPE-RM-13 | P1 | Race-loss abort: if buy not confirmed in N s, cancel & refund gas | small | analyst | `wait_for_transaction_receipt(timeout=N)` with N=15s |
| SNIPE-RM-14 | P1 | Decimal-aware token amount math | small | analyst | `trade_executor.py:377, 706` — fetch `decimals()` first |
| SNIPE-RM-15 | P1 | Wire `is_safe_to_snipe` top-holder gate into `_check_filters` | trivial | analyst | Or inline the check |
| SNIPE-RM-16 | P1 | Front-run prevention: skip if dev wallet is one of the first 3 buyers | medium | analyst | Read pair `Swap` events for last 30s before our buy |
| SNIPE-RM-17 | P2 | Multi-wallet rotation | large | sm-contract | Wallet pool with round-robin + balance tracker |
| SNIPE-RM-18 | P2 | Idempotency guard in `_evaluate_target` | trivial | analyst | Set-based de-dup before adding to `pending_targets` |
| SNIPE-RM-19 | P2 | Test-mode hard-block in LIVE: if `dry_run=false` AND `test_mode=true` → refuse to start | trivial | analyst | `sniper_engine.py:_load_settings` |
| SNIPE-RM-20 | P2 | Defensive DRY_RUN guard inside `_execute_evm_buy` / `_execute_solana_buy` | small | analyst | Belt-and-suspenders |
| SNIPE-RM-21 | P2 | Failed-tx gas-burn counter + hourly cap | small | analyst | Mirror SafetyEngine pattern |
| SNIPE-RM-22 | P3 | Replace per-tick DexScreener call with WebSocket subscription | medium | infra | Reduces rate-limit hazard at scale |
| SNIPE-RM-23 | P3 | Re-entry logic on dip after exit | small | analyst | If exited on TP and price drops 30% within 10 min → re-enter at smaller size |

---

## 8. Recommended go-live sequence

1. **Stop.** Do not run sniper LIVE in any config until SNIPE-RM-01 (slippage protection) is fixed. The `amount_out_min=0` is a hand-grenade.
2. Land SNIPE-RM-01, -02, -03, -04, -05 in one PR sweep — these are the bare-minimum.
3. Land SNIPE-RM-06 (simulated round-trip) — this is the single biggest edge in sniper safety. Without it, you're trusting GoPlus.
4. Land SNIPE-RM-07, -08, -12, -13 — the per-trade / per-hour / time-stop guardrails.
5. Run **Solana-only sniper** in DRY_RUN for 24 hours, verify hit-rate and slippage stats on real launches.
6. Run **Solana-only sniper LIVE** with `trade_amount=0.02 SOL` (~$4) and `max_positions=1`, for 72 hours.
7. Only then enable EVM after SNIPE-RM-06 has had 1 full week of paper-trading on EVM.
8. **EVM sniper stays disabled** until SNIPE-RM-01 + SNIPE-RM-06 + SNIPE-RM-14 are all merged.
9. Trailing stops / dev-wallet exits (SNIPE-RM-09 / -10 / -11) are the next phase — these turn the bot from "lose less" into "actually capture upside."

---

## 9. Open questions

1. What is the intended `trade_amount` in live? `0.1` default is too rich for a $400 book.
2. Are honeypot APIs (GoPlus, Honeypot.is, RugCheck) on paid tiers? Free-tier rate limits collapse during launch storms.
3. Is the sniper expected to run only on Solana, only on EVM, or both? The dual-chain approach doubles attack surface for half the focus.
4. Pump.fun graduations vs. new launches — should sniper target both, or only graduations? Pre-graduation tokens are an order of magnitude riskier.
5. What is the policy if `solana_listener` and `evm_listener` both fire on the same heartbeat — does priority go to fastest detect, or highest safety score?
6. Are there any known launch farms (Banana Gun, Maestro, etc.) we should explicitly NOT compete with? The latency arms race is already lost to them — our edge has to be safety, not speed.
7. Should the sniper share `RiskManager` (shared core) state with DEX/Solana modules to enforce a global daily-loss cap across all modules? Currently it doesn't, so a sniper blowup doesn't trip the shared circuit breaker.
8. Is there a budget for a Helius / Alchemy paid plan dedicated to the sniper module? The polling approach uses heavy RPC credits.

---

## 10. Bottom line

The sniper module's **detection** and **safety-API integration** are solid. The **execution path is a disaster waiting to happen**: zero slippage protection on EVM, no simulated-sell pre-check, DB-configured TP/SL silently ignored, no rate-limiting, no startup reconciliation, no emergency stop, no exit ladder, no dev-sell detection. Each of these alone is enough to fail a live-readiness review; together they are an unmitigated red verdict.

This module is best characterized as a **prototype**, not a production system. Solana-only paper-trading is the only configuration I would currently endorse. Live deployment — even at $5 trade size — requires SNIPE-RM-01 through SNIPE-RM-08 as a minimum.

**Verdict: RED.**
