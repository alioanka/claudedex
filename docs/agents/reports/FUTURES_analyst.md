# FUTURES_MODULE — Trading-Desk / Risk / Live-Readiness Audit

Owner: market-trading-analyst (PRIMARY per PLAN.md row "FUTURES_MODULE"). Secondaries: quant-algo-expert, backend-devops-expert.
Scope: `modules/futures_trading/{futures_module,main_futures,futures_risk_manager}.py`, `modules/futures_trading/core/{futures_engine,futures_alerts}.py`, `modules/futures_trading/exchanges/{binance_futures,bybit_futures}.py`, `modules/futures_trading/strategies/{funding_arbitrage,hedge_strategy,trend_following}.py`, `modules/futures_trading/config/{futures_config,futures_config_manager}.py`. Cross-module: `core/risk_manager.py`, `core/portfolio_manager.py`, `core/decision_maker.py`, `core/engine.py`, `monitoring/alerts.py`, `trading/orders/{order_manager,position_tracker}.py`, `main.py`.

## Executive verdict: RED

The futures module ships two **completely separate** trading engines that look like they belong to different products: the "module-style" `FuturesTradingModule` (`futures_module.py`) and the engine-style `FuturesTradingEngine` (`core/futures_engine.py`). Only the engine-style is actually launched by `main_futures.py:592-599`. The module-style is loaded nowhere except `__init__.py` re-exports and is full of placeholder TODOs (`futures_module.py:678 return None`, `futures_module.py:861 pass`, `futures_module.py:425 'direction': 'LONG'  # Placeholder`). Anyone touching the module-style code thinking it is live will be wrong. P0 cleanup before anything else.

The live path (`FuturesTradingEngine`) has three RED issues that make it unfit for live trading at any size:

1. **`AttributeError` during init on the live path.** `core/futures_engine.py:262` reads `not self.dry_run` inside the `__init__` testnet-resolution block, but `self.dry_run` is not set until line 361 (after the `if config_manager:` block exits). Python evaluates `elif not self.dry_run:` only when the two `FUTURES_TESTNET` env branches fail. Under the documented operating mode (`FUTURES_TESTNET` unset in `.env`, `config_manager` present), this line is hit at every startup and raises `AttributeError`. Either nobody has actually run this in the documented config or the env var is always set in production — both possibilities are scary. This bug also means the "DRY_RUN=false → MAINNET safety default" comment at lines 263-266 is a lie: the safety check never executes successfully.

2. **`marginType=ISOLATED` is never set on the live engine path.** `BinanceFuturesExecutor.open_long/open_short` (`exchanges/binance_futures.py:310, 366`) calls `set_margin_type(symbol, 'ISOLATED')`, but that executor is only wired by the dead `FuturesTradingModule`. The live engine (`core/futures_engine.py`) calls `self.exchange_client.create_market_order(...)` via ccxt directly (lines 1689, 1774, 1519) without ever touching `marginType`. Whatever the user/account default is on Binance (CROSS for most users) is what trades will use. The persona file explicitly demands `marginType=ISOLATED unless cross is the explicit edge`. On a cross-margin account, one liquidation drains the entire futures wallet, not just the position margin. With `default_leverage=10` (`futures_config_manager.py:63`) and `max_leverage=20` (line 64), a 5% adverse move at 20x will eat the whole sub-wallet.

3. **No bounded leverage validation against `max_leverage`, no per-symbol cap, and the cap that exists is read from the database (trustable but not pinned to exchange-tier limits).** The engine pulls `self.leverage = leverage_config.default_leverage` (line 288, `default_leverage=10`) and applies that same value to every symbol (`_setup_symbols` at line 729 calls `set_leverage(self.leverage, symbol)` once). There is **no per-symbol leverage cap** — BTC and a small-cap alt with ~3x available margin are treated identically. `futures_risk_manager.FuturesRiskManager.validate_new_position` (lines 50-122) checks leverage against `self.max_leverage` (default 3, line 40) — but this validator is **only consulted from the dead `FuturesTradingModule._execute_futures_trade`** (`futures_module.py:542-549`). The live engine never calls `validate_new_position` before `create_market_order`. Practical result: every leverage cap policy in this module is symbolic; the only effective cap is the database value, which the dashboard can change without restart.

Until items 1-3 are resolved, the module must not run with `DRY_RUN=false`. Issue 1 may make it impossible to run at all in the documented config, which is the only reason this hasn't already caused losses.

## DRY_RUN propagation audit

Per-callsite. The engine has a single source of truth (`self.dry_run` set from env at `core/futures_engine.py:360-361`) but at least three gates downstream.

| Path | File:Line | DRY_RUN-gated? | Notes |
|---|---|---|---|
| `FuturesTradingEngine.__init__` | core/futures_engine.py:360-361 | SOURCE | `dry_run_env = os.getenv('DRY_RUN', 'true')`. Default `'true'` — safe-by-default. Good. |
| `_setup_symbols` (set_leverage call) | core/futures_engine.py:727 | YES (inverted) | `if not self.dry_run or self.testnet` — calls Binance `set_leverage` even in dry-run if testnet=True. Fine for testnet but writes to exchange state. |
| `_sync_positions` | core/futures_engine.py:740 | YES | Early return when `self.dry_run`. **Problem**: skipping sync means dry-run never reconciles against real positions; if user flips DRY_RUN at runtime the engine is unaware of any live positions. |
| `_open_position` (entry) | core/futures_engine.py:1681 | YES | Branches on `self.dry_run`: dry path is print + state-tracking; live path is `create_market_order` (line 1689) |
| `_close_position` (exit) | core/futures_engine.py:1768 | YES | Same pattern. Sends `reduceOnly=True` (line 1778). Good — every exit is reduceOnly. |
| `_partial_close_position` (TP hit) | core/futures_engine.py:1512 | YES | Same pattern (line 1515-1528). `reduceOnly=True` set (line 1523). Good. |
| `AITradeExecutor.execute_trade` (cross-module call) | modules/ai_analysis/core/sentiment_engine.py:70 | YES | Branches on `self.dry_run`. **But** this is a parallel executor with its own DRY_RUN read (line 237) — see AI_analyst.md. |
| `main_futures.py` arg propagation | modules/futures_trading/main_futures.py:790-796 | PARTIAL | Reads `DRY_RUN` from env, sets `os.environ['DRY_RUN']`. `--dry-run` CLI forces dry. **But** there is no `--live` flag, so flipping live requires editing `.env` — minor safety. |
| `FuturesTradingModule._execute_futures_trade` | modules/futures_trading/futures_module.py:557, 563 | NO | The dead module-style path has **no dry-run gate** — calls `executor.open_long` directly. If this code is ever wired up without adding the gate, it will fire live trades. P0 if module-style is ever revived. |
| `HealthServer.unblock_trading_handler` | modules/futures_trading/main_futures.py:448-485 | N/A but dangerous | Allows operator to zero `daily_pnl` and `consecutive_losses` from an HTTP POST. No auth. If exposed past localhost, anyone can unblock the bot mid-drawdown. P0 ops issue. |

The single canonical source is OK but the cross-engine duplication (AI module reads its own `os.getenv('DRY_RUN')` and instantiates a parallel `AITradeExecutor` that signs Binance Futures orders independently — `sentiment_engine.py:113-187`) means the futures engine and the AI engine can be in different dry/live states. This is identical to the "three-engine arbitrage" anti-pattern flagged in `ARBITRAGE_analyst.md`. P1 hazard.

## Risk-policy coverage matrix

| Control | Status | Evidence |
|---|---|---|
| Per-trade max loss | PARTIAL | `stop_loss_pct=1.2%` (`futures_config_manager.py:85`) gives ~1.2% max price-loss; with `default_leverage=10` that is **12% of margin per trade**. No per-trade $-cap. |
| Per-hour max loss | MISSING | `risk_metrics.daily_pnl` exists (`core/futures_engine.py:167`) but no hourly bucket. The cross-module `RiskManager` has no hourly logic either. |
| Per-day max loss | PRESENT but unenforced cross-module | `RiskMetrics.daily_loss_limit` (line 169) and `can_trade` property (line 178) work locally — but `core/risk_manager.py` never sees futures P&L. Two daily limits running in parallel, neither informs the other. |
| Per-module capital cap | PRESENT | `capital_allocation=300.0` default (`futures_config_manager.py:48`). Used in `_calculate_position_size` (`core/futures_engine.py:1338`). Good. |
| Per-symbol concentration cap | MISSING | `max_positions=5` (`futures_config_manager.py:57`) is total, not per-symbol. The engine refuses re-entry on a symbol that already has a position (line 996-1000) but does not cap notional per symbol. |
| Per-symbol leverage cap | MISSING | One global `default_leverage` for all symbols (line 288). BTC at 10x is reasonable; small-cap-alt at 10x is reckless. No per-symbol override. |
| Per-exchange max position vs internal max | PARTIAL | `validate_new_position` (`futures_risk_manager.py:50`) checks `max_total_exposure=500.0` (line 42). Not consulted by live engine. |
| Leverage hard-cap configurable from DB | PRESENT | `max_leverage=20` (`futures_config_manager.py:64`). But the validator that enforces it is dead code. |
| Trust-exchange-max for leverage? | YES (DANGEROUS) | `binance_futures.py:196` clamps to `self.max_leverage` (default 3 in that file) — but that path is dead. The live engine sends whatever `self.leverage` (DB-default 10) without exchange-tier validation. If a user sets 50 in DB and the symbol allows it, 50 it is. |
| `marginType=ISOLATED` enforced | MISSING (live path) | See verdict item 2. Only `binance_futures.py:310, 366` calls `set_margin_type('ISOLATED')`. Live engine never does. |
| `reduceOnly` on every exit | PRESENT | `_partial_close_position` (line 1523), `_close_position` (line 1778), `BinanceFuturesExecutor.close_position` (line 428). Good — but `set_stop_loss` (line 511, dead code) uses `closePosition: 'true'`, which is exchange-side, also fine. |
| Mark-price vs last-price for SL/TP/liq | WRONG | `_check_exit_conditions` (line 898) uses `position.current_price` which comes from `_get_ticker` → `last`. Exchange-side liquidation uses **mark price**. Stop-losses computed from `last` will trip *before* mark crosses — usually fine in trending markets; in low-liquidity wicks, **trips on noise that mark-price never reaches**. P1 profitability leak. |
| Funding payment accounting | MISSING | `BinanceFuturesExecutor.get_funding_rate` (line 443) reads the rate; no code path **credits funding to P&L** when paid. Strategy `funding_arbitrage.py:185-219` calculates expected funding but the live engine never books it. P&L will silently understate funding-arb profit (or overstate the loss in negative-funding periods you're paying). |
| Liquidation buffer formula | DUMB | `liquidation_price = current_price * (1 - 0.9 / self.leverage)` (line 1640) — assumes 90% maintenance margin coverage with no per-symbol tier awareness. Binance maintenance margin tiers vary by notional and symbol. The `0.9` is also misleading — actual liquidation is at maintenance margin (~0.4-0.5% on BTC at low notional, higher tiers add tier-margin). At 10x leverage Binance's actual liq for a $1000 BTC position is roughly at -9.5%, the code computes -9.0%. Close, but not for higher leverage or higher notional. |
| Per-hour API rate-limit handling | NAIVE | `binance_futures.py:124-125` sleeps `min_request_interval=0.1` (100ms) — that is unrelated to Binance Futures weight system. No 429-retry-after handling, no exponential backoff. On a burst (multi-TP close on multiple positions) the bot will hit weight bans. |
| API key permission scoping | OPERATIONAL | Cannot enforce in-code. Must be set on the exchange: **trade-only, no-withdraw, IP-allowlisted**. Document in runbook. The bot stores both keys in encrypted DB / Docker secrets (`futures_engine.py:570-583`) — good. |
| Drawdown freeze ladder (-X day → halve, -Y day → flat, -Z week → halt) | MISSING | Single threshold `risk_metrics.can_trade` flips true→false at `daily_loss_limit`. No progressive deleveraging. Persona explicitly demands a ladder. |
| Cross-strategy correlation cap | MISSING | Trend + hedge + funding-arb strategies (`strategies/*.py`) can all suggest long BTCUSDT simultaneously; nothing prevents the engine from opening all three. The live engine doesn't even consult `HedgeStrategy` or `FundingArbitrageStrategy` — they are referenced from the dead `FuturesTradingModule` only (`futures_module.py:165-175`). |
| Cross-module correlation with DEX | MISSING | If DEX module is long ETH/USDT and futures opens ETH/USDT long at 10x, total directional exposure is double-counted. `core/risk_manager.check_correlation_limit` (`risk_manager.py:1328`) exists but futures never calls it. |
| Stale-fill / phantom-position guard | MISSING | If `create_market_order` returns 200 but the websocket fill notification is lost, the in-memory `active_positions[symbol]` is updated with the optimistic fill price. No fill-confirmation read-back from `fetch_my_trades`. P1. |
| Position reconciliation on startup | PARTIAL | `_sync_positions` (line 738) calls `fetch_positions` — **but only if `not self.dry_run`** (line 740). In any DRY_RUN→LIVE transition the engine starts blind to existing live positions. Should always reconcile and only refuse to **place** new orders in dry run. |
| Heartbeat / freeze on stale data | WEAK | `_status_reporter` (`main_futures.py:667`) logs every 2 min; nothing freezes trading if exchange `fetch_time` fails. `get_health` (line 1977) reports `exchange_connected` boolean but no consumer auto-flattens. |
| Emergency stop wired | PARTIAL | `close_all_positions_handler` (`main_futures.py:398`) exists at HTTP `POST /positions/close-all`; `Telegram /flatten` may be wired via `telegram_controller`. No project-wide `scripts/emergency_stop.py`. Persona demands one. |
| Idempotent order IDs | MISSING | `ccxt create_market_order` is called without `clientOrderId` (line 1689). If the bot retries on network timeout, duplicate orders can fire. Binance requires explicit `newClientOrderId` for idempotency. P0 for live trading. |
| Funding-arb sizing formula per persona | MISSING | Persona demands: `size = funding * notional - taker_fees * 2 - expected_slippage - liquidation_premium`. `funding_arbitrage.py:202-219` returns gross funding APR with no subtraction for fees, slippage, or liquidation buffer. The strategy will green-light trades that the desk math marks unprofitable. P0 if funding arb is ever enabled (`funding_arbitrage_enabled=False` default — safe for now). |

## Order/Position lifecycle review

The live path (`FuturesTradingEngine`) runs a 30s scan loop (`scan_interval_seconds` default, `core/futures_engine.py:329`) doing:

1. **Daily reset check** (`_check_daily_reset`, line 799) — UTC date rollover zeros `daily_pnl`, `daily_trades`, `consecutive_losses`. Issue: zeros `consecutive_losses` at midnight even if the bot just had 4 losers in the last hour — losing streak metric resets across day boundary. Minor.
2. **Risk gate** (`risk_metrics.can_trade`, line 178) — checks daily loss + consecutive losses (>=5 → block). No drawdown gate. No hourly gate.
3. **Monitor positions** (`_monitor_positions`, line 829) — fetches `last` price per position, updates highest/lowest, computes leveraged P&L, checks TPs then SL.
4. **Scan opportunities** (`_scan_opportunities`, line 984) — runs 5-indicator scoring (RSI/MACD/Volume/BB/EMA), applies trend+volume+momentum+alignment filters, opens at most ONE new position per scan cycle (line 1123 `break`).
5. **Open position** (`_open_position`, line 1608) — calculates notional, fees, SL, TP1-4 cascade, builds `Position` dataclass, calls `create_market_order` if live, adds to `active_positions`.

Desync points:

- **D1: No fill confirmation.** `create_market_order` returns immediately when ccxt receives the 200; partial fills are not re-queried. `position.entry_price` is set to `order['average']` (line 1701) but if `average` is `None` or `0` (happens on partial-fill or fast markets), entry price stays at `current_price` from ticker — i.e. **the bot books P&L against the pre-trade ticker, not the realised fill**.
- **D2: Bot crash between order send and `active_positions[symbol] = position`.** Order lands on exchange, bot restarts, `_sync_positions` (line 738) reads it back — but only in live mode and only if no exception. The `Position` dataclass loses all metadata (TP levels, original_size, signal score). Restarted bot will manage the position with default 5%/10% TPs instead of the configured cascade.
- **D3: TP1 → breakeven SL pegged at exact entry price.** `_check_tp_levels` line 1459 sets `position.trailing_stop_price = position.entry_price` after TP1. If price oscillates around entry by 1 tick, the breakeven SL trips on the first wick, closing the rest of the position for ~zero net (plus fees = small loss). Should be entry + half a fee buffer at minimum.
- **D4: TP4 size_pct=10% with `original_size` precision.** `_partial_close_position` reduces `position.size` by `original_size * 0.1` (line 1440). Floating point drift over 4 partials means the 4th close might be 9.99% or 10.01%; the `if position.size <= 1e-8:` tolerance check (line 1597) usually catches this, but exchange-side `reduceOnly` can reject a tiny residual order — leaving a dust position open until next cycle.
- **D5: ccxt symbol vs config symbol.** Config uses `"BTC/USDT"` (line 31 of `futures_config.py`); ccxt's binance future market is `"BTC/USDT:USDT"` (linear perp). `_setup_symbols` (line 720-737) loops `self.symbols` and checks `if symbol not in self.exchange_client.markets:` — this will silently skip every symbol whose form doesn't match exactly. Needs a normalisation step.
- **D6: `_check_daily_reset` uses `datetime.utcnow()` (line 801) but most other timestamps use `datetime.now()`.** Mixing tz-naive with local-time naive across the codebase — if the host is not UTC, the daily-reset will misalign with the daily-loss windows logged by trade timestamps. Real money risk if host runs on US/Pacific: daily reset at 17:00 local, but tracker rolls over at 00:00 local in logs.
- **D7: Failed close on the SL path silently leaves the position open.** `_close_position` (line 1782-1783) catches the exception, returns. The next monitor cycle (10s later, line 805) re-evaluates and tries again — but `consecutive_losses` is incremented inside this same function only on success. If the close fails for 60s and SL was at -1.2% of price, actual realised loss may be much larger when the close eventually goes through. No retry-with-tighter-slippage, no emergency-market-IOC. P1.

## Profit-leak / loss-leak inventory

| ID | Category | File:Line | Estimated impact | Fix sketch |
|---|---|---|---|---|
| FL-01 | `AttributeError` at init blocks live path | core/futures_engine.py:262 | Bot fails to start in documented config | Move DRY_RUN read (line 360) above the testnet block; reference `dry_run_env` directly in the elif at line 262 |
| FL-02 | Missing `marginType=ISOLATED` on live engine | core/futures_engine.py:728-733 | One liquidation drains entire futures wallet on cross-margin accounts | Add `set_margin_type(symbol, 'ISOLATED')` inside `_setup_symbols` before `set_leverage` |
| FL-03 | Validator dead-code; live path skips `validate_new_position` | core/futures_engine.py:1608-1740 | Leverage cap, exposure cap, capital cap all unenforced at order time | Instantiate `FuturesRiskManager` in `FuturesTradingEngine.__init__` and call `validate_new_position` before every `create_market_order` |
| FL-04 | Last-price (not mark) for SL/TP | core/futures_engine.py:841, 898 | Stops trip on wicks that don't move mark; ~3-8% extra exits/month in choppy markets | Use `fetch_premium_index['markPrice']` in `_get_ticker` for SL/TP comparisons; keep last for entry |
| FL-05 | Funding payments not credited to P&L | (missing) | Funding-arb strategy reports gross APR but realised P&L misses funding cashflow | Hourly job: `fetch_funding_history` per active symbol, add to `total_pnl` and `daily_pnl` |
| FL-06 | No `clientOrderId` (idempotency) on orders | core/futures_engine.py:1689 | Network retry can double-fill | Generate `clientOrderId = f"cd_{position_id}_{retry_count}"`; ccxt accepts `params={'newClientOrderId': ...}` |
| FL-07 | Cross-engine DRY_RUN duplication | core/futures_engine.py:360-361 + modules/ai_analysis/core/sentiment_engine.py:237 | Two independent flag reads; one can be live while other dry | Centralise: ConfigManager (DB) emits a single `dry_run` value; both engines subscribe |
| FL-08 | Breakeven SL pegged exactly at entry | core/futures_engine.py:1459 | Wicks at entry close runners for ~0 net | Set breakeven SL at entry + 2*taker_fee on LONG (or entry - 2*taker_fee on SHORT) |
| FL-09 | Daily reset zeros consecutive losses across midnight | core/futures_engine.py:805-807 | Losing streak metric resets; bot resumes 6th-loss-in-a-row trade at 00:01 UTC | Persist consecutive_losses across days; only zero on a winning trade |
| FL-10 | No per-hour gas/loss budget | (missing) | Bot can lose `daily_loss_limit` in 30 minutes and stop only after | Add `hourly_pnl` bucket; freeze at -3% in any 60-min window |
| FL-11 | Liquidation buffer 90% is symbol-blind | core/futures_engine.py:1640, 1644 | Wrong liq estimate on high-tier notional (BTC > $1M) or alt-coins | Read `fetch_position` `liquidationPrice` for actual on-exchange liq, not estimate |
| FL-12 | `unblock_trading_handler` no-auth | main_futures.py:448 | Anyone with access to `:8081` can unblock the bot post-drawdown | Require Bearer token from secrets manager, or restrict to `127.0.0.1` |
| FL-13 | ccxt symbol form mismatch silent-skip | core/futures_engine.py:722 | Configured pair `"BTC/USDT"` may not match ccxt-internal `"BTC/USDT:USDT"`; bot trades nothing | Normalise: prefer `BTC/USDT:USDT` (linear perp) in config; warn loudly on skip |
| FL-14 | `unrealized_pnl_pct` is "leveraged %" but compared to non-leveraged `stop_loss_pct` | core/futures_engine.py:874, 900-904, 935 | The un-leverage step at 904 is correct but fragile to refactor; one accidental remove and SLs trigger 10x earlier | Add unit test that asserts SL trips at price-move of `stop_loss_pct` regardless of leverage |
| FL-15 | `_sync_positions` skipped in dry run | core/futures_engine.py:740 | Cannot detect orphaned live positions during a paper run | Always sync; only block new orders in dry run |
| FL-16 | Telegram alerts route through MarkdownV2 with no fallback | core/futures_alerts.py:191 | One malformed character → silent alert failure | Catch on send_message → retry with plain text |
| FL-17 | Funding arb strategy ignores liquidation premium | strategies/funding_arbitrage.py:202-219 | Strategy will recommend trades that lose money on the next funding window if a spike happens | Subtract `notional * 2 * taker_fee + slippage + liq_premium` from expected funding |
| FL-18 | Hedge strategy uses `cost`/`pnl` from DEX positions assuming USD | strategies/hedge_strategy.py:67-78 | If DEX positions use native-token denominations, the drawdown ratio is wrong | Normalise to USD before ratio math |
| FL-19 | Bybit executor is a stub | exchanges/bybit_futures.py:84-111 | If a user configures Bybit, the bot logs "placeholder" and returns None — no warning to operator | Raise NotImplementedError so the bot refuses to start on `exchange=bybit` until implemented |
| FL-20 | `tp4_size_pct=10` with `tp1+tp2+tp3 = 90` allows residual dust | futures_config_manager.py:96-99 | 10% final close may leave sub-min-notional residual | Adjust 4th TP to "close all remaining" not a fixed pct |

## Live-readiness checklist (per persona)

- [ ] DRY_RUN honored on every send/order/sign path — **PARTIAL**. Live engine path is gated. Module-style path is ungated. AI module's parallel Binance executor is gated independently — risk of skew.
- [ ] Per-trade max loss enforced — **FAIL**. SL exists as % but no dollar cap; risk-mgr validator is dead code.
- [ ] Per-hour max loss enforced — **FAIL**. No hourly bucket anywhere.
- [ ] Per-day max loss enforced — **PARTIAL**. Local `risk_metrics.can_trade` works (`futures_engine.py:178`); cross-module risk manager not consulted.
- [ ] Position reconciliation on startup — **FAIL** in dry-run mode (skipped). **PARTIAL** in live: reads exchange but loses TP cascade state.
- [ ] Idempotent order IDs — **FAIL**. No `clientOrderId` on `create_market_order`.
- [ ] Heartbeat to dashboard; freeze on stale > N seconds — **FAIL**. Logs only; no freeze.
- [ ] Emergency stop reachable from dashboard — **PARTIAL**. HTTP endpoint exists, no auth. No `scripts/emergency_stop.py`.
- [ ] `marginType=ISOLATED` per symbol — **FAIL** on live engine.
- [ ] Per-symbol leverage cap, configurable — **FAIL**. Global cap only.
- [ ] `reduceOnly` on every exit — **PASS** (this is the one bright spot).
- [ ] Mark-price for SL/TP/liquidation — **FAIL**. Uses `last`.
- [ ] Funding accounting — **FAIL**. Read-only; never credited to P&L.
- [ ] Drawdown freeze ladder — **FAIL**. Binary toggle only.
- [ ] Cross-strategy correlation cap — **FAIL**. No coordinator.
- [ ] Stale-fill / phantom-position guard — **FAIL**. No fill confirmation.
- [ ] API key permissions trade-only / no withdraw / IP-allowlist — **OPERATIONAL** (cannot enforce in code; must document in runbook).
- [ ] Per-exchange rate-limit handling — **FAIL**. Naive sleep, no 429 backoff.
- [ ] Tail funding event (>50 bps/hr) handling — **FAIL**. Funding-arb has no shock-out trigger.
- [ ] AttributeError-free init — **FAIL** (verdict item 1).

Overall: 1 PASS, 4 PARTIAL, 14 FAIL, 1 OPERATIONAL. **Not live-ready.**

## Proposed action backlog (ranked by P&L-protection per dev-hour)

P0 (must land before any live capital):
- **FUT-RM-01** Fix init order so `self.dry_run` is set before testnet logic (core/futures_engine.py:262, 360-361). Owner: analyst+backend. ~30 min. Risk: bot does not start in live config.
- **FUT-RM-02** Add `set_margin_type(symbol, 'ISOLATED')` call in `_setup_symbols` for every symbol; refuse to start if any returns failure (core/futures_engine.py:718). Owner: analyst+backend. ~1 hr. Risk: wallet-draining liquidation.
- **FUT-RM-03** Wire `FuturesRiskManager.validate_new_position` into `_open_position` before `create_market_order` (core/futures_engine.py:1608). Touch `futures_risk_manager.py` to also enforce per-symbol leverage. Owner: analyst. ~3 hr.
- **FUT-RM-04** Generate and pass `clientOrderId` to every order (core/futures_engine.py:1689, 1519, 1774). Owner: analyst+backend. ~1 hr. Risk: duplicate fills on retry.
- **FUT-RM-05** Auth the `unblock_trading_handler` HTTP endpoint (main_futures.py:448). Bind health server to `127.0.0.1` by default; require Bearer token to expose externally. Owner: backend. ~1 hr.

P1 (before scaling above $1k capital):
- **FUT-RM-06** Use mark price (`fetch_premium_index`) for SL/TP/liquidation checks (core/futures_engine.py:841, 898). Owner: analyst. ~2 hr.
- **FUT-RM-07** Credit funding payments to P&L: hourly `fetch_funding_history` job (cross-module). Owner: analyst+quant. ~3 hr.
- **FUT-RM-08** Always run `_sync_positions` (live + dry); only block new orders in dry. Persist TP cascade state to DB so it survives restart (core/futures_engine.py:738, 1700). Owner: analyst+backend. ~4 hr.
- **FUT-RM-09** Add hourly P&L bucket + freeze ladder: -3% in 60 min → halve sizing; -5% in 60 min → flat all (futures_engine.py:813 area). Owner: analyst. ~2 hr.
- **FUT-RM-10** Add fill-confirmation read-back: after `create_market_order`, call `fetch_order` until status=closed, then update entry price from realised fill. Owner: analyst. ~2 hr.
- **FUT-RM-11** Cross-module risk integration: futures engine pushes trades to `core/risk_manager.update_trade_metrics` so circuit breakers see futures P&L. Owner: analyst+backend. ~2 hr.

P2 (post-launch hardening):
- **FUT-RM-12** Per-symbol leverage caps from DB. Owner: analyst+backend. ~3 hr.
- **FUT-RM-13** Funding-arb strategy: subtract fees+slippage+liq-premium per persona formula (funding_arbitrage.py:202). Add 50bps/hr funding shock-out. Owner: analyst+quant. ~3 hr.
- **FUT-RM-14** Breakeven SL with fee buffer (futures_engine.py:1459). ~30 min.
- **FUT-RM-15** Cross-strategy correlation cap: refuse new long if total long exposure on symbol already > X% of capital. Owner: analyst. ~2 hr.
- **FUT-RM-16** Bybit executor: implement or raise `NotImplementedError` at init (bybit_futures.py:84). ~2 hr to stub-fail; ~2 days to implement.
- **FUT-RM-17** Per-exchange weight-based rate limiter (binance_futures.py:120-125). Owner: backend. ~3 hr.
- **FUT-RM-18** Symbol normalisation (`BTC/USDT` → `BTC/USDT:USDT`) with explicit warning on skip (futures_engine.py:722). ~1 hr.
- **FUT-RM-19** UTC consistency: standardise `datetime.utcnow()` everywhere or migrate to tz-aware. ~2 hr.
- **FUT-RM-20** Delete or quarantine `FuturesTradingModule` (module-style dead code) under `.deprecated/`. PM-Architect agent to schedule. ~1 hr.

## Recommended go-live sequence

1. **Stage 0 (do now)**: Land FUT-RM-01 through FUT-RM-05. Without these the bot is unfit to even paper-trade reliably.
2. **Stage 1 (paper, full coverage, 1 week)**: `DRY_RUN=true`, FUT-RM-06 through FUT-RM-11 landed. Verify: daily P&L matches DB sum, circuit breakers visible on dashboard, simulated orders carry `clientOrderId`, restart preserves TP cascade.
3. **Stage 2 (testnet, 72h)**: `DRY_RUN=false`, `FUTURES_TESTNET=true`. Same exchange API surface, fake money. Verify: `ISOLATED` margin set per symbol, every fill returns a `clientOrderId`, reduceOnly on every close.
4. **Stage 3 (live, micro-cap, 72h)**: `DRY_RUN=false`, mainnet, `capital_allocation=50.0`, `max_positions=2`, `max_leverage=3`, ONLY BTC and ETH. Run 72h. Verify: every fill in DB, every position in tracker, no zombie positions on restart, hourly freeze ladder triggers on injected fault.
5. **Stage 4 (live, scaled)**: only after Stage 3 clean and P2 backlog landed. Step `capital_allocation` up by 50% per week if Sharpe > 1.0 rolling 14d.

## Open questions

- Why does PLAN.md not mention `FuturesTradingModule` (`futures_module.py`) — is it intentional dead code or work-in-progress to replace the engine path? PM Agent decision required.
- Funding arbitrage strategy is config-disabled by default (`funding_arbitrage_enabled=False`). What is the intended go-live story for it? Currently it has no spot leg execution at all, only a perp leg signal — see hedge_strategy.py for similar half-implementation.
- AI module's parallel `AITradeExecutor` sends Binance Futures orders independently of the futures engine (`sentiment_engine.py:113-187`). Should AI-driven futures trades go *through* `FuturesTradingEngine.process_opportunity` (currently only on the dead module-style path) so they get the same risk gates? Recommend yes; see AI_analyst.md cross-ref.
- `CLOSE_POSITIONS_ON_SHUTDOWN=false` default (`main_futures.py:721`) — is this correct? Persona expects flat-on-shutdown for production. If the bot is restarted during a flash crash with positions open and stale risk state, the restart-side `_sync_positions` may leave it managing positions it doesn't fully understand.
- Per-strategy capital allocation: trend-following + funding-arb + hedge — what is the intended capital split? Currently they all draw from the same `capital_allocation` with no per-strategy cap.

