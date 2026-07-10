# Wave-F6 Arbitrage Audit — Profit Honesty + "Why did it stop?"

**Date:** 2026-07-09 · **Auditor:** market-trading-analyst (read-only) · **Scope:** `modules/arbitrage/`, `logs/arbitrage/`, `logs/orchestrator.log`, control-center screenshot `screenshots/screencapture-38-242-251-156-8080-control-center-2026-07-09-23_57_25.png`

## Verdict (one paragraph)

**The recorded arbitrage profit is FABRICATED.** Every dollar of the +1085 USD (7D) / +3250 USD (all-time) the operator sees on the control center was booked by the **TriangularArbitrageEngine's DRY_RUN branch**, which (a) quotes cycles at **dust size** (1 CRV ≈ $0.75) but books USD notional as if the amount were **1 ETH (~$1,778)** — a ~2,400× notional inflation from a unit bug; (b) books **gross quote-time spread with zero gas subtraction** (estimated gas ≈ $27/trade vs booked profit ≈ $32); (c) marks every trade `closed` instantly with `entry_timestamp == exit_timestamp`, so a **100% win rate and Sharpe 3.00 are structural artifacts** — a loss is unrepresentable; and (d) simulates fills on a path that **cannot execute live at all** (MB-05 atomic-receiver guard hard-returns `None` before any broadcast). Meanwhile the **spatial** engines — the only path with a real (gated) live route — are behaving **honestly**: post-Wave-F5 they report every spread as negative (ETH hourly median −96 bps, best −68 bps) and have recorded **zero** trades. The module did not "stop"; the phantom-profit generator throttled itself.

---

## (a) Profit-honesty audit — exact PnL computation path

### Where the money is "made"

The scan loop round-robins `TRIANGULAR_CYCLES` (`triangular_engine.py:104-133`; the offending cycle is `('CRV','WETH','USDC')` at line 132) and calls `_check_triangular_opportunity` (`triangular_engine.py:612`).

**Step 1 — dust-size quoting (the phantom-spread source), `triangular_engine.py:631-649`:**

```python
flash_loan_eth = self.config.get('flash_loan_amount', 10)
use_flash = self.config.get('use_flash_loans', False)
effective_eth = float(flash_loan_eth) if use_flash else self.trade_amount_eth
...
else:
    # For other tokens (CRV, UNI, LINK, etc.), convert ETH amount to token amount
    # Default to ETH-equivalent value to avoid thin liquidity phantom spreads
    amount_in = int(effective_eth * 1e18)
```

For token_a = CRV, `amount_in = 1e18 = 1 CRV ≈ $0.75` — **not** 1 ETH-equivalent of CRV. The comment says "convert ETH amount to token amount" but no conversion happens. At dust size, `getAmountsOut` (lines 661-716, which correctly embeds the 0.3%/hop V2 LP fee and price impact **at the quoted size**) returns a near-impactless mid-price differential between the CRV/WETH, WETH/USDC and USDC/CRV V2 pools. That differential is stable for hours (log shows the spread pinned at exactly `1.788%` from 09:46 to 09:54 and `1.400%` for 40+ minutes earlier) — the classic static-reserve phantom the code's own comment at lines 628-630 warns about ("Using tiny amounts (1 token) on thin liquidity pools gives misleading spreads").

**Step 2 — spread math, `triangular_engine.py:733-734`:**

```python
profit_pct = (best_hop3_out - amount_in) / amount_in
```

Honest *at 1-CRV size*, meaningless at booked size. The `MAX_REALISTIC_SPREAD = 0.10` sanity gate (line 740) does not catch 1.8%.

**Step 3 — DRY_RUN booking bypasses every gate, `triangular_engine.py:828-836`:**

```python
if self.dry_run:
    await asyncio.sleep(0.5)
    logger.info(f"✅ Triangular Arb Executed (DRY RUN): {cycle_key}")
    await self._log_trade(...)
    return
```

This branch runs **before** the P1-06 risk-manager gate (line 839) and **before** the MB-05 atomic-receiver guard (`_execute_triangular_swap`, lines 900-911), which unconditionally returns `None` on the live path ("Triangular arbitrage execution path is disabled (MB-05)"). So the module CLAUDE.md claim "triangular gated — scope cut" is only true for LIVE; **DRY_RUN books simulated fills on a strategy that has no execution path.**

**Step 4 — the fabricated write, `triangular_engine.py:1104-1141`:**

```python
amount_eth = amount_in / 1e18            # 1 CRV mislabeled as 1.0 "ETH"
entry_usd  = amount_eth * self._eth_price # 1.0 × $1778 = $1778 notional  ← ~2,400x inflation
profit_usd = entry_usd * profit_pct       # $1778 × 1.788% = $31.79      ← gross, NO gas subtracted
...
INSERT INTO arbitrage_trades (... profit_loss, ... status, is_simulated, entry_timestamp, exit_timestamp ...)
VALUES (... profit_usd, ... 'closed', self.dry_run, datetime.now(), datetime.now() ...)
```

- `profit_loss` = notional-inflated gross spread. **No gas** (the engine's own threshold math prices gas at ~$26.7: 500k gas × 30 gwei × $1778 ETH, `triangular_engine.py:568-610` — that is why the logged threshold is 1.730% = 0.08% base + 1.5% gas × 1.1 buffer). **No flash fee, no latency/MEV/revert risk.**
- `status='closed'` instantly → dashboard counts it as a realized win the same second.
- `is_simulated = True` **is** written (line 1139) — but no dashboard aggregate filters on it (see below).

### Sample trades traced end-to-end (all 5 post-restart fills)

| Log lines (`logs/arbitrage/arbitrage.log`) | Time (07-07) | Route | Quoted spread | Booked notional | Booked profit | Honest value |
|---|---|---|---|---|---|---|
| 361-362 | 09:46:38 | CRV→WETH→USDC (all uniswap_v2) | 1.788% | $1,778.09 | **+$31.79** | real notional 1 CRV ≈ $0.75 → gross ≈ **$0.013**; net of $26.7 gas ≈ **−$27**; unexecutable (MB-05) |
| 384-385 | 09:52:18 | same | 1.788% | $1,776.96 | +$31.77 | same |
| 403-404 | 09:58:01 | same | 1.788% | $1,776.96 | +$31.78 | same |
| 422-423 | 10:03:41 | same | 1.9% | $1,776.71 | +$33.77 | same |
| 444-445 | 10:09:17 | same | 1.9% | $1,781.26 | +$33.84 | same |

Total booked in the current log: **+$162.95 across 24 minutes, 100% wins, same cycle, spread frozen at 1.788-1.9%.** If that spread were real and executable at $1,778 size on Ethereum mainnet, MEV searchers would atomically extract it within one block; its persistence for hours is itself proof it is not.

### Cross-check vs SPREAD DISTRIBUTION (spatial = honest)

Post-F5 hourly lines confirm the spatial engines find **nothing**, correctly:

- `ETHEREUM SPREAD DISTRIBUTION (last hour, 346 samples): median -96.4bps | p25 -137.7bps | best -68.3bps` (log line 10979, 07-09 20:23) — every sample negative, consistent with the F5 verdict (V2-only legs vs ~65 bps cost floor).
- ARBITRUM and BASE: `no samples — check rpc_health / pair filters` every hour; ARB shows `Pairs w/Liquidity: 0/0`, BASE shows liquid pairs but zero spread samples — the F5 `arb_max_price_impact_bps` pre-filter is rejecting everything before quoting (correct at 1-ETH size on those V2 pools, though the BASE "22/22 pairs w/liquidity yet 0 samples" combination deserves one diagnostic pass by the dashboard/backend owner).
- All spreads negative + zero spatial trades **while the PnL tile climbs** = fabrication confirmed; the climb is 100% triangular.

### What the operator sees (screenshot, 07-09 23:57)

Arbitrage tile: TODAY 0.00 / 7D **+1085 USD** / ALL **+3250 USD**, WIN RATE **100.0%**, CLOSED **79**. Cross-module 7D row: 17 trades, expectancy +63.84 USD, max DD 0.0000, **Sharpe/trade 3.00**. The cumulative chart shows the arbitrage line jumping vertically ~0→1000 exactly at 07.07.2026. Only 5 of the 17 7D trades appear in the current log (the log is truncated at each restart; the module restarted 07-07 08:19 with the F5 image). The other ~12 (≈ +$920) were booked in the 07-05→07-07 pre-restart window by the same mechanism, and the remaining ~62 all-time rows (≈ +$2,165) date back further. **DB verification query** (not runnable from this analysis host — no DB creds in the repo checkout):

```sql
SELECT DATE(entry_timestamp), COUNT(*), SUM(profit_loss), BOOL_AND(is_simulated)
FROM arbitrage_trades WHERE metadata::text LIKE '%triangular%'
GROUP BY 1 ORDER BY 1;
-- expectation: 100% of positive PnL rows are triangular + is_simulated=true
```

### Why the dashboard swallows it

`monitoring/enhanced_dashboard.py:13457-13473` (arbitrage stats), `:18461` (portfolio aggregate) and the control-center rollups sum `profit_loss` over `arbitrage_trades` with **no `is_simulated` handling beyond the DRY column label and no exclusion mechanism**. The tile honestly says "DRY", but the DRY number itself is garbage-in. Ironically the dashboard already computes a triangular-vs-direct split (`:13493-13519`) — it just doesn't flag it.

---

## (b) Stop diagnosis — timeline

**The module never crashed.** Orchestrator: `Arbitrage - RUNNING (PID: 17, ..., Restarts: 0)` continuously through 07-09 20:38. Zero `Traceback` in `logs/arbitrage/stderr.log`; `arbitrage_errors.log` empty; no `logs/.killswitch`, `.pause_arbitrage`, or `.restart_arbitrage` flags. F5 RPC rotation works (regular `Rotated RPC endpoint via pool_engine` lines; PoolEngine health checks 83/96 healthy; zero 401/403 infra events post-F5).

| When | Event |
|---|---|
| 2026-07-07 08:19 | Module restarts on Wave-F5 code (log truncated; pre-restart trades survive only in DB) |
| 07-07 09:46 – 10:09 | Triangular books **5 DRY fills** on `CRV_WETH_USDC` (+$162.95) — cooldown 300s apart, until `_max_executions_per_cycle_per_day = 5` (**hardcoded**, `triangular_engine.py:326`) is hit. **These are the last trades the module ever recorded.** |
| 07-07 10:09 – 18:49 | ~145 more `✅ OPPORTUNITY` lines on the same cycle (spread 1.79-2.16%), all swallowed by the daily cap / cooldown (`:773-784`) |
| 07-08 → 07-09 | **Zero** `✅ OPPORTUNITY` lines. The phantom CRV spread drifted to 1.58-1.63%, below the gas-buffered 1.730% threshold. Triangular STATS: `Opportunities: 0 \| Executed: 0` |
| whole period | Spatial (ETH/ARB/BASE): `Opportunities: 0 | Executed: 0` every 5-min window; ETH engine in SLOW-SCAN mode (30min+ stale negative spreads); all-negative spread distributions |

So "why did it stop" has two honest answers: **(1) the spatial module never started producing** — post-F5 it correctly measures zero net-positive spreads (this is the F5 AMBER verdict working as designed, "stopped" = "honest"); **(2) the fabricated-profit stream stopped** on 07-07 10:09 when the triangular engine's own 5-per-cycle-per-day cap latched and the dust-size phantom spread subsequently fell below threshold. Nothing broke on 07-08/09; the bug simply went quiet.

---

## (c) Fixes (profit IS fabricated → both data repair and code fix required)

### Data repair — migration 145 sketch (follow the Solana mig-140A "tag, don't delete" pattern)

```sql
BEGIN;
-- Tag every simulated triangular fill as excluded from PnL (kept for audit)
UPDATE arbitrage_trades
SET metadata = jsonb_set(COALESCE(metadata::jsonb, '{}'::jsonb), '{excluded}', 'true'::jsonb)
             || '{"excluded_reason": "wave_f6_triangular_phantom_pnl_unit_bug"}'::jsonb
WHERE is_simulated = TRUE
  AND metadata::text LIKE '%triangular%'
  AND COALESCE((metadata::jsonb->>'excluded')::boolean, FALSE) = FALSE;  -- idempotent
COMMIT;
```

(Check `metadata` column type first — cast shape above assumes TEXT/JSONB compatible, same as mig 140A.) Then every dashboard aggregate over `arbitrage_trades` (enhanced_dashboard.py:2502, 4255, 4438, 6283, 7024, 13457-13531, 13649, 13787, 13964, 14035, 18461 plus `modules/orchestrator_ai`, `modules/execution_quality`, `modules/backtest_replay` consumers) must add `AND COALESCE((metadata::jsonb->>'excluded')::boolean, FALSE) = FALSE` — this is the dashboard/backend agent's work item. Expected post-repair tile: ALL ≈ $0, which is the truth.

### Code fixes (in priority order)

1. **Stop booking DRY fills on an unexecutable path.** Move the MB-05 guard *ahead* of the DRY_RUN branch in `_execute_triangular` (`triangular_engine.py:828`): if `_execute_triangular_swap` is structurally disabled, DRY mode must not write `arbitrage_trades` rows either — record to a shadow/opportunity surface (e.g. `arbitrage_runtime_stats.near_misses` with reason `tri_unexecutable_mb05`) instead. A simulation of a trade that cannot exist is not DRY_RUN; it is fiction.
2. **Fix the notional unit bug** (`:645-649`): for non-WETH/stable/WBTC token_a, size `amount_in` to the ETH-equivalent via a hop-1 pre-quote (two-pass: quote 1 token → derive token/ETH price → set `amount_in = effective_eth / price`), or simply restrict `token_a` to WETH/stables/WBTC. Never book `entry_usd = amount_in/1e18 * eth_price` for a non-ETH asset (`:1105-1106`).
3. **Net the PnL** (`:1107`): `profit_usd = entry_usd * profit_pct - gas_cost_usd` using the same `GasOracle.calculate_gas_cost_usd(500_000, eth_price)` the threshold already uses; store gross, gas, and net separately in `metadata`.
4. **Structural honesty**: any future sim fill should carry win/loss capability (adverse re-quote at booking time + revert probability haircut). A table whose win rate cannot mathematically drop below 100% must never feed a PnL tile, a Sharpe, or the meta_controller/portfolio_allocator scoring loops.
5. **Config hygiene**: `_max_executions_per_cycle_per_day = 5` is hardcoded (`:326`) — move to ConfigManager with the other `tri_*` keys if the engine survives.

---

## (d) Recommendation: PARK the module

- **Spatial**: F5's honest verdict stands and is now empirically confirmed — three months of scanning, zero positive samples, structural cost floor ~65 bps vs 1-30 bps real divergence on V2-only executable legs. Expected edge ≈ 0 until a V3 receiver leg + event-driven quoting ship. Running it costs RPC quota (it already self-demoted to SLOW-SCAN) and operator attention, and produces nothing.
- **Triangular**: has **no live execution path at all** (MB-05) and its only output for months has been fabricated PnL that corrupted the control center, the cross-module performance table (a fake Sharpe 3.00 leader), and potentially the advisory allocators that read `arbitrage_trades`.
- **Action**: set `ARBITRAGE_MODULE_ENABLED=false` (or at minimum disable the triangular scan loop), ship migration 145 + the dashboard exclusion filter so the historical record stops lying, and only revisit after the atomic-receiver/V3 contract work is actually funded. Do **not** treat the +3250 USD as evidence of edge in any allocation decision — it is the single worst data-integrity artifact currently on the dashboard.

## Live-readiness checklist status (for the record)

DRY_RUN honored on send paths: YES for spatial and (trivially) triangular — no live order was ever at risk; the failure here is **accounting honesty in DRY mode**, not a live-fund leak. Kill-switch/pause flags polled: verified present in run loop. Per-trade/hourly/daily loss caps: moot while parked. Verdict stays **AMBER at best; recommended PARKED**.
