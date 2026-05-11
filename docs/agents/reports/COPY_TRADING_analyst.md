# COPY_TRADING_MODULE — Trading-Desk / Risk / Live-Readiness Audit

Owner: market-trading-analyst (PRIMARY per PLAN.md row "COPY_TRADING_MODULE"). Secondary: quant-algo-expert.
Scope: `modules/copy_trading/{main_copy,copy_engine}.py`. Cross-module: `core/{risk_manager,portfolio_manager,decision_maker}.py`, `monitoring/{alerts,telegram_bot}.py`, `main.py`. Dashboard surface: `dashboard/templates/{settings_copytrading,positions_copytrading,dashboard_copytrading,trades_copytrading,performance_copytrading,wallets_copytrading,discovery_copytrading}.html`.

## Executive verdict: RED

The copy-trading module is **not a copy-trading product**. It is a wallet-mirror cron with a 15s poll loop (`copy_engine.py:591`) that copies any wallet address you paste into the `target_wallets` config field. There is no leader pool, no leader scoring, no survivorship-bias-free universe, no fractional Kelly, no correlation cap, no exit logic, no SL/TP, no integration with `core/risk_manager.py`, and the live executor performs un-quoted Solana swaps with a 1% slippage default while leaving EVM swaps at a configurable but ignored 10% slippage. The dashboard exposes config fields (`Max Copy Amount`, `Copy Ratio`, `Max Slippage`) that the engine **never reads** (`copy_engine.py:616-657` only consumes `target_wallets`). Operating this module with `DRY_RUN=false` against any non-trivial capital is reckless. Specific RED items:

1. **No risk-manager hook.** `grep -n "risk_manager\|RiskManager" modules/copy_trading/*.py` returns zero hits. `CopyTradingEngine` (`copy_engine.py:498`) never instantiates or consults `core/risk_manager.py`. Circuit breakers, daily-loss limits, drawdown caps, per-token correlation limits, and position-count caps in the cross-module risk manager (`core/risk_manager.py:261-328, 1226-1390`) **do not apply to copy trades**. Every other persona-owned module at least imports `RiskManager`; this one does not.
2. **Settings-form ↔ engine disconnect.** `settings_copytrading.html:22-49` exposes `enabled`, `dry_run`, `max_copy_amount`, `copy_ratio`, `slippage` — they are saved to `config_settings` under `config_type='copytrading_config'`. `copy_engine.py:_load_settings` (line 616) reads that exact table but only branches on `key == 'target_wallets'` (line 630). Every other knob is silently ignored. `self.max_copy_amount=100.0` and `self.copy_ratio=10` are hard-coded in `__init__` (lines 530-531). The dashboard is theater.
3. **`dry_run` is environment-only and frozen at startup.** `copy_engine.py:527` reads `os.getenv('DRY_RUN', 'true')` once. The dashboard toggle for `dry_run` writes to DB but the engine never re-reads it. Flipping live/paper from the UI is silently ignored.
4. **No exit logic. None.** The engine fires entry orders on leader-swap detection (line 871-879 for EVM, 1010-1017 for Solana) and then walks away. There is no SL, no TP, no time-stop, no leader-exit mirror that closes our position when the leader sells. For SELL events the code explicitly **does not place a closing order** — it stamps a fake `tx_hash='SELL_TRACKED_...'` (line 1102), updates the row to `closed`, and books P&L using the *current* price as if we exited at that price (lines 1148-1182). We hold the position indefinitely while the database lies that it is closed.
5. **EVM copy path is wide-open to sandwich attacks at any non-trivial size.** `copy_evm_swap` (line 262) uses Uniswap V2 router with `slippage: float = 10.0` (line 267) — 10% default tolerance is a *very* generous gift to MEV searchers. Worse, the on-chain quote (`getAmountsOut` line 324) is read from the *same block's mempool-visible state*, which a sandwich bot can manipulate between quote and execution. No flashbots, no MEV-protect, no private bundle. Persona-required: "leader fills cheap, we fill expensive". This is the textbook expensive-fill profile.

If `DRY_RUN=true` and the user only wants leader visibility / paper attribution, the module is roughly fine as a monitor. The moment `DRY_RUN=false` is flipped, the module is a wealth destroyer. Until items 1-5 are resolved, the module must remain `DRY_RUN=true`.

## DRY_RUN propagation audit

| Path | File:Line | DRY_RUN-gated? | Notes |
|---|---|---|---|
| `CopyTradingEngine.__init__` env read | `copy_engine.py:527` | SOURCE | `os.getenv('DRY_RUN', 'true').lower() in ('true','1','yes')`. Default safe. **Read once**; not re-read from DB. |
| `CopyTradeExecutor.__init__` | `copy_engine.py:103` | INHERITED | `dry_run: bool = True` arg passed by engine (line 566). |
| `copy_solana_swap` | `copy_engine.py:228` | YES | Early return to `_simulate_solana_swap` (line 471). Good. |
| `copy_evm_swap` | `copy_engine.py:270` | YES | Early return to `_simulate_evm_swap` (line 484). Good. |
| `_log_copy_trade` | `copy_engine.py:1216, 1218` | NO (informational) | Writes `is_simulated=self.dry_run` to row. Good for attribution. |
| Settings reload | `copy_engine.py:_load_settings` | MISSING | DB has `dry_run` key but engine never reads it. Cannot flip live/paper at runtime. **P1.** |
| Telegram `notify` on startup | `main_copy.py:150` | NO | Notifies "started", does not state mode. Operator cannot tell from notification whether bot is live. **P2.** |
| Dashboard `mode-badge` for Copy Trading | `dashboard_copytrading.html:429-430` | PARTIAL | Shows `DRY RUN` badge but `style="display: none"` by default — only shown if at least one row in `trades` payload has `dry_run/is_simulated=true` (line 643-644). If no trades yet, operator cannot see the mode. **P1.** |
| Main `base.html` dashboard DRY_RUN indicator | `dashboard/templates/base.html` (no match) | MISSING | The always-visible nav has no per-module DRY/LIVE badge. **P0.** |

The DRY_RUN source-of-truth is the env var, captured into `self.dry_run` once at startup, propagated correctly through the executor. The leak is the dashboard: users can toggle `dry_run` in settings and observe no behaviour change.

## Risk-policy coverage matrix

| Control | Status | Evidence |
|---|---|---|
| Per-trade $ cap | PRESENT (broken) | `self.max_copy_amount=100.0` (`copy_engine.py:530`). Hard-coded. EVM path applies `min(original_value * copy_ratio // 100, max_copy_amount * 1e18)` (line 904-907). **Bug**: `max_copy_amount * 1e18` is treated as wei, but `max_copy_amount` is meant as USD — at ETH=$2000 the cap effectively allows `$100 × 1e18 wei = 100 ETH = $200K`. The cap is 2000x too generous. Solana path uses `min(self.max_copy_amount / sol_price, 0.1) * 1e9` (line 1088) which is correct. |
| Per-hour $ cap | MISSING | No hourly bucket anywhere. A single leader pumping 20 buys in an hour will fire 20 of our copies. |
| Per-day $ cap | MISSING | No daily counter. `core/risk_manager.py` is not consulted. |
| Per-leader $ cap | MISSING | No per-leader notional budget. One leader can drain capital. |
| Per-leader trade-count rate-limit | PRESENT (weak) | Per-wallet 5-min cooldown (`_wallet_cooldown_seconds=300`, line 542). Avoids spam but lets a leader still hit 12/hour. |
| Aggregate cap across leaders | MISSING | Five correlated leaders all buy the same memecoin → we 5x our exposure. No correlation-aware sizing. |
| Per-token concentration cap | MISSING | A leader who frequently rotates into BONK results in unbounded BONK accumulation. |
| Fractional-Kelly sizing | MISSING | Sizing is a flat ratio of leader's notional (line 905). Persona explicitly demands "**fractional Kelly capped**, NOT 1:1 notional". |
| Leader scoring (Sharpe / max-DD / concentration / sample-size) | MISSING | Zero code. `grep -i "sharpe\|sortino\|drawdown\|kelly\|fractional"` returns no hits in `copy_engine.py`. |
| Survivorship-bias-free leader pool | MISSING | "Leader pool" = whatever address strings the user pastes into `target_wallets`. No on-chain leaderboard ingestion, no dead-wallet pruning. |
| Stop-following triggers | MISSING | No leader-DD-watchdog, no sample-size watchdog, no leader-silence detection. |
| Slippage budget | PRESENT (loose) | Solana 100 bps default (line 225), EVM 10% default (line 267). EVM 10% is a sandwich invitation. **P0.** |
| MEV protection | MISSING | Public Uniswap V2 router (`copy_engine.py:284`), public mempool. No flashbots / mev-blocker. |
| Stop-loss | MISSING | Never set. |
| Take-profit | MISSING | Never set. |
| Time-stop | MISSING | Position held until leader sells *and* the SELL is detected within the 5-min cooldown window and the post-balance diff matches the BUY token *and* the wallet is still being polled. Many failure modes. |
| Leader-exit mirror | BROKEN | SELL detection (`_execute_solana_copy_trade` line 1097-1106) explicitly **does not place a sell order** — fakes a tx hash and marks the row closed. We never actually exit the position. **P0.** |
| Drawdown freeze | MISSING | No drawdown calc, no freeze. |
| Circuit breaker integration | MISSING | `RiskManager.check_circuit_breakers` (`core/risk_manager.py:261`) is never called from copy-trading code. |
| Position reconciliation on startup | MISSING | `_known_tx_hashes`, `_known_solana_sigs` are in-memory `set()` (lines 537-538). On restart the engine forgets every previously seen TX and may re-copy. The cooldown table is also in-memory. |
| Idempotent order IDs | MISSING | `trade_id = f"copy_{uuid.uuid4().hex[:12]}"` (line 1143). Random per call — if the engine retries it creates duplicate rows. No `ON CONFLICT (source_tx, side)` clause. |
| Heartbeat | PRESENT (silent) | `_log_stats_if_needed` (line 596) logs every 5 min. Not pushed to dashboard. No "if stale >N seconds flatten" gate. |
| Emergency stop reachable | PARTIAL | Telegram `/stop_all` and `/emergency` (`telegram_bot.py:124-137`) and dashboard `/api/bot/emergency-exit` (`module_routes.py:76`) exist. **But** `bot_emergency_exit` (`module_routes.py:852-883`) iterates `module.get_positions()` and `module.close_position(position)` — copy_trading's engine **does not implement** `get_positions` or `close_position` at the module-base-class level. Emergency exit will skip copy_trading silently. **P0.** |
| Per-leader DRY_RUN (shadow new leader for N days) | MISSING | Persona explicitly demands. Not even a schema column for `leader_status='shadow'`. |
| KYC / consent disclosure | N/A on-chain | EVM/Solana wallet copying is consent-irrelevant — public-mempool data. No CEX copy-leader paths (Bybit/OKX/GMX/Hyperliquid leaderboards) — see "Leader pool" section. |

Verdict on policy coverage: 4 of 24 controls present, 1 of those 4 broken (max_copy_amount unit bug). Lethal gap.

## Order / Position lifecycle review

**Entry (EVM, swap into token, BUY).** Etherscan V2 `txlist` for each wallet, last 5 txs, filter to swap method-ids (line 863-869). Extract token from last 40 hex chars of input data (line 920) — **fragile**; this is offset-arithmetic on `swapExactETHForTokens` only. `swapExactTokensForTokens` and the SupportingFeeOnTransferTokens variants put the token at a different position in the calldata. We will mis-decode every non-vanilla swap. Then `copy_evm_swap` builds a `swapExactETHForTokens` even if the leader did `swapExactTokensForTokens` — we always pay in ETH (line 333), never in the input token the leader actually used. Wrong-side execution.

**Entry (Solana, swap into token).** `getSignaturesForAddress` poll, then `getTransaction` to inspect program IDs against the Jupiter/Raydium allow-list (lines 992-997). Token mint inferred from `postTokenBalances` minus `preTokenBalances` (line 1037-1078). For BUY we route `WSOL → token_mint` via Jupiter at 100 bps slippage. Reasonable, but no leader-side leverage check — if the leader used a perp DEX (Drift, Mango) we mis-classify as a spot swap.

**Exit.** None on EVM. On Solana, SELL detection writes `tx_hash=SELL_TRACKED_...` (line 1102) and **does not call Jupiter**. We never close. The P&L row is updated as if we exited at current price using `usd_value` computed from our copy lamports (line 1135-1140) — **the P&L number is fiction**. The `profit_loss` and `profit_loss_pct` fields (lines 1164-1166) reflect the comparison of `entry_usd` versus `usd_value` at the *moment we detected the leader's SELL*, not the price we got. Dashboard performance numbers are not real.

**Cooldowns and idempotency.** 5-min per-wallet cooldown (line 542). In-memory `_known_tx_hashes`, `_known_solana_sigs` sets (lines 537-538). After restart, the engine re-polls the last 5 txs (lines 731, 798) and could re-copy if the TX is still in the recent window. The cooldown dict is also wiped. Idempotency must be DB-backed.

**Sizing math.** EVM: `min(value * copy_ratio/100, max_copy_amount*1e18)` — the `*1e18` bug means the cap is 1e18× too high in USD terms. If a leader sends 50 ETH ($150K) we copy `min(5 ETH, 100*1e18 wei) = 5 ETH ($15K)` despite the user setting "$100 max". **Critical bug.** Solana: `min(max_copy_amount/sol_price, 0.1) * 1e9` — caps at 0.1 SOL (~$20) regardless of `max_copy_amount`. Solana cap is too tight (you cannot lift it via UI), EVM cap is non-existent.

**Concurrency.** Wallets are iterated sequentially in `_monitor_evm_wallets` (line 719) and `_monitor_solana_wallets` (line 791) inside one 15s loop. Latency: with N wallets and ~1s/Etherscan call, our entry lag versus leader is at minimum N seconds. A leader's $100K alt entry will move price 2-10% before our copy lands. Practical entry slippage from latency alone, before MEV.

## Leader pool / scoring / discovery

**Sources used.** None. `target_wallets` is operator-pasted strings. No `data/collectors/whale_tracker.py` call, no DEX leaderboard ingestion (DexScreener top-gainers wallets, Nansen smart-money, Lookonchain top traders, GMX/Hyperliquid leaderboards, Bybit/OKX copy-trading APIs). `discovery_copytrading.html` exists as a page — engine does not call it.

**Scoring.** None. The persona-required inputs (rolling Sharpe 30/90/180d, max-DD, % P&L from top trade, drawdown-recovery time, sample size, win-rate consistency, time-of-day skew) are completely absent. `_update_wallet_stats` (line 1229) only stores trade *count*. No win-rate, no Sharpe, no concentration.

**Survivorship-bias-free pool.** Cannot exist — there is no pool. If we built one, we would need (a) entry-cohort snapshots ("top 100 wallets ranked Jan-1") to avoid picking only survivors today, (b) leader-mortality tracking, (c) drop-out attribution.

**Stop-following triggers.** None. A leader can rug us into a single bad trade and we never stop following them.

## Profit-leak / loss-leak inventory

1. **Latency-induced adverse selection.** 15s poll + sequential wallet iteration → average entry lag of dozens of seconds. For micro-caps that move 5-50% in 30s, we systematically buy the top of leader's pump. Expected: 100-500 bps of negative skew per entry vs leader's fill, before any MEV. *Loss-leak.*
2. **Sandwich-attack exposure on EVM.** 10% default slippage tolerance (line 267) + public mempool + no MEV protection. Searchers will detect our `swapExactETHForTokens` and sandwich for up to 9.5%. *Loss-leak — easily 200-500 bps per EVM copy.*
3. **No exits — Solana positions held indefinitely.** SELL is stamped, not executed (line 1100-1106). We accumulate dust positions and never lock gains. Any winner is unrealized; any loser is permanent. *Massive loss-leak.*
4. **Max-amount bug allows 2000× intended size on EVM.** `max_copy_amount * 1e18` units mismatch (line 906) — operator believes `$100` cap is in effect; engine enforces `100 ETH ≈ $200K` cap. *Single-trade ruin risk.*
5. **Wrong-side decoding for non-`ExactETHForTokens` swaps.** Leader sells token → method-id matches `swapExactTokensForETH` (line 866) → we treat as SELL → route to `_execute_evm_copy_trade` (line 886) → it calls `copy_evm_swap` (line 931) which **only supports BUY via WETH path** (line 302). On a leader-sell we either error or buy WETH→token (wrong direction). *Loss-leak + reputational.*
6. **EVM token-address extraction is fragile.** Last-40-hex slice (line 920) breaks on multi-hop paths and `Tokens→Tokens` swaps that have a different layout. We will paste random middleman addresses into `copy_evm_swap`. *Catastrophic mis-execution.*
7. **In-memory dedup/cooldown.** After restart we re-copy the leader's last 5 txs. *Loss-leak on restarts.*
8. **No funding-cost or gas-cost attribution.** `_log_copy_trade` (line 1120) does not subtract gas paid, priority fees, or Jupiter fees. P&L overstated by ~30-150 bps on Solana, ~50-500 bps on EVM. *Reported edge is too high.*
9. **Price snapshot uses CoinGecko free tier with 60s cache (line 54-87).** Pricing for P&L is stale and only available for SOL/ETH/BTC; token-level price for the copied alt is *never* fetched. The `entry_usd` written to DB equals `amount_native × native_price` — i.e. we record the ETH value we spent, not the USD value of the token we received. P&L row is in ETH terms unintentionally.

## Live-readiness checklist

- [ ] `DRY_RUN` honored on every send path — **YES on entries; broken at runtime toggling**.
- [ ] Per-trade max loss — **NO** (no SL).
- [ ] Per-hour max loss — **NO**.
- [ ] Per-day max loss — **NO**.
- [ ] Position reconciliation on startup — **NO** (in-memory dedup).
- [ ] Idempotent order IDs — **NO**.
- [ ] Heartbeat to dashboard — **NO** (logs only).
- [ ] Emergency stop reachable from dashboard — **NO** (endpoint exists, copy_trading not wired to base-class `close_position`).
- [ ] Leader scoring + survivorship-bias-free pool — **NO**.
- [ ] Fractional-Kelly sizing — **NO**.
- [ ] Per-leader cap, aggregate cap, correlation cap — **NO**.
- [ ] Exit mirror on leader exit — **NO**.
- [ ] SL/TP — **NO**.
- [ ] MEV-protected EVM execution — **NO**.

**Zero of 14 green.** Module is not live-ready. Keep `DRY_RUN=true`.

## Action backlog (ranked, ID prefix `CT-RM-`)

P0 (blockers for any live $):
- **CT-RM-01** — Fix EVM `max_copy_amount` unit bug. Treat field as USD; convert via `eth_price` to wei. Add explicit assertion `copy_amount_wei < int(max_usd * 1e18 / eth_price)`. (~10 lines, `copy_engine.py:903-907`.)
- **CT-RM-02** — Implement leader-exit mirror. On detected SELL: actually call `copy_solana_swap(input_mint=token_mint, output_mint=WSOL_MINT, amount=our_position_lamports)`. Persist open positions in a `copytrading_open_positions` DB table so exits can find them across restarts. (~80 lines.)
- **CT-RM-03** — Wire copy_trading into `core/risk_manager.py`. Before every copy: `RiskManager.check_circuit_breakers(...)` and `RiskManager.validate_trade(token, usd_amount)`. Update trade metrics after fill. (~40 lines.)
- **CT-RM-04** — Make `_load_settings` honor `dry_run`, `enabled`, `max_copy_amount`, `copy_ratio`, `slippage`. Re-read every loop. Document that toggling these takes effect on next 15s tick. (~30 lines.)
- **CT-RM-05** — Implement `close_position(position)` and `get_positions()` on the engine so dashboard emergency-exit (`module_routes.py:852`) and Telegram `/stop_all` actually flatten copy_trading. (~50 lines.)
- **CT-RM-06** — Implement EVM-side `getAmountsOut`-aware sizing with **2% max slippage** default (not 10%). Cap at the configured value. Use private RPC/flashbots-protect endpoint when configured. (~60 lines.)
- **CT-RM-07** — Persist `_known_tx_hashes` and `_known_solana_sigs` to DB (`copytrading_processed_tx`) so restarts do not re-copy. UNIQUE INDEX. (~30 lines.)

P1 (live trading gate):
- **CT-RM-08** — Leader scoring service. Daily job that computes per-wallet 30/60/180-day Sharpe, max-DD, hit-rate, % P&L from top trade, sample size, last-trade-recency. Store in `copytrading_leader_scores`. Engine filters `target_wallets` against `score > threshold`.
- **CT-RM-09** — Per-leader / aggregate / correlation-aware caps. `per_leader_usd_cap`, `aggregate_usd_cap`, `per_token_usd_cap`, correlation-cluster cap (group leaders by 30-day P&L correlation > 0.7).
- **CT-RM-10** — Stop-following triggers: leader DD > X%, leader silent > N days, leader sample-size shrinks, leader Sharpe drops below threshold. Auto-demote to `shadow` mode.
- **CT-RM-11** — Per-leader DRY_RUN flag (`leader_status='shadow'` vs `'live'`). Shadow leaders compute attribution paper-trades for 14 days before promotion is allowed.
- **CT-RM-12** — Fractional-Kelly sizing. `kelly = max(0, (p*b - q)/b)` with `b = avg_win / avg_loss` and `p = win_rate` from the leader score. Apply `kelly_fraction = 0.25 × kelly × per_leader_capital`. Reuse pattern from `core/risk_manager.py:898-978`.
- **CT-RM-13** — Robust EVM calldata decoding. Use proper ABI decoding (eth_abi) for all 5 swap method-ids, including SupportingFeeOnTransferTokens variants. Reject if unknown layout.
- **CT-RM-14** — Real P&L: subtract gas, priority fees, Jupiter platform fees, slippage realized. Compare to a `paper_leader_pnl` that simulates the leader's fill at the leader's block.

P2 (polish):
- **CT-RM-15** — Replace 15s sequential poll with Helius webhook (Solana) and Alchemy notify-API or Etherscan websocket (EVM) for sub-second leader detection.
- **CT-RM-16** — Surface in dashboard: per-leader live P&L attribution, leader scores, stop-following alerts, shadow→live promotion timer.
- **CT-RM-17** — Add Telegram `/copy_leaders`, `/copy_stop_leader <wallet>`, `/copy_shadow <wallet>` commands.

## Go-live sequence

1. CT-RM-01..07 (P0 fixes) shipped, tested in DRY_RUN against 3 well-known on-chain wallets for 7 days.
2. CT-RM-08, CT-RM-12 (scoring + Kelly) shipped, run scoring against historical 90-day on-chain history; verify Sharpe numbers are non-degenerate.
3. CT-RM-09, CT-RM-10, CT-RM-11 (caps + stop-follow + shadow) shipped, all leaders default to `shadow` for first 14 days post-deploy.
4. Single-leader live trial with $50 per-trade cap, $200 daily cap, $500 aggregate cap, 1 wallet, 30 days.
5. If 30-day attribution beats paper-leader by < 50 bps total cost-of-execution, expand to 3 leaders, $100/$500/$2000 caps.
6. Persona-required live-readiness checklist must be 100% green before raising caps above $2k aggregate.

## Open questions

1. Are we copying *informational* (read leader, decide independently) or *execution-mirror* (replay leader's trade)? Current code is execution-mirror without any informational filter — every leader buy is our buy. Persona implies the former is preferable.
2. Capital allocation: should copy_trading share the global `RiskManager.max_position_size_usd=$10` cap? At $10/trade and a 5-min cooldown × 10 leaders that is roughly $1200/day max exposure — too small to matter, too large to bypass. Need explicit allocation tier.
3. Why CEX copy-leader sources (Bybit/OKX copy-trading APIs, GMX/Hyperliquid leaderboards) are absent — module name implies CEX support, code is on-chain-only. Either rename or extend.
4. How do we treat leaders running flash-loans, MEV searcher patterns, or PnL inflation via wash trades? No filter today; we will copy whatever a labeled "smart money" wallet does.
5. Should leader-pubkey be encrypted in DB? Today they are plaintext rows in `config_settings` (line 623). For competitive reasons operators may want this opaque.
