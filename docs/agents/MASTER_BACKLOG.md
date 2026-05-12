# ClaudeDex — Master Backlog
*Synthesized from 24 Phase 1 audit reports*

## Snapshot

- **Modules audited:** 8 (DEX, ARBITRAGE, SOLANA, SNIPER, FUTURES, AI, COPY_TRADING, DASHBOARD)
- **Reports synthesized:** 24 (smartcontract / quant / analyst / backend per module, where applicable)
- **Verdicts:** 0 GREEN · 5 AMBER · 19 RED (or quant-grade D/F)
- **Authoring agents:** `smartcontract-web3-expert`, `quant-algo-expert`, `market-trading-analyst`, `backend-devops-expert`
- **Branch:** `claude/create-expert-agents-JFSF5`
- **Date:** 2026-05-11

> **Live-trade gate:** **0 of 8 modules are GREEN for `DRY_RUN=false`.** Every module has at least one P0 live-blocker. Do not flip live until the Phase 2 prerequisites + Phase 3 per-module P0s for that module are merged.

---

## Verdict matrix

| Module | smartcontract | quant | analyst | backend |
|---|:---:|:---:|:---:|:---:|
| **DEX** | RED | RED | RED | (covered cross-cutting) |
| **ARBITRAGE** | AMBER (RED: triangular + Solana) | RED-ish | RED | (covered cross-cutting) |
| **SOLANA** | RED | survival-filter only | AMBER (RED: Drift) | (covered cross-cutting) |
| **SNIPER** | RED | RED | RED | (covered cross-cutting) |
| **FUTURES** | — | RED | RED | RED |
| **AI** | — | RED | RED | AMBER (shadow only) |
| **COPY_TRADING** | — | C-/D | RED | AMBER (cleanest secrets; misrouting P0) |
| **DASHBOARD** | — | D risk-math / F backtest | AMBER → RED for live | non-compliant |

**One-sentence justification per module:**

- **DEX** — Three divergent executors share supported-chain lists but disagree on slippage/gas/DRY_RUN; `min_amount_out` hardcodes `*10**18` regardless of decimals; Flashbots signing path broken; no `pool_engine` integration.
- **ARBITRAGE** — Spatial EVM arb is the strongest part of the codebase, but a DAI mainnet address typo zeroes out DAI; triangular fallback is non-atomic (3 sequential txs = stuck wallet); one-legged BUY-only broadcast guarantees inventory leak.
- **SOLANA** — Strong rules-based trailing-stop ladder, but Jupiter signing path discards co-signers, decimals hardcoded asymmetrically (6 in close, 9 in default), no Jito bundle path despite dashboard config exposure, ML stack completely unwired, Drift perp leg has no DRY_RUN gate.
- **SNIPER** — Structurally too slow to compete (15-120s detection latency vs competitors' 50-400ms), `amount_out_min=0` on EVM buy *and* sell, hardcoded TP/SL shadows DB config, Solana price feed uses deprecated `price.jup.ag/v4` (returns 0 → no SL ever fires), `is_simulated=True` hardcoded.
- **FUTURES** — Bot fails to start (`AttributeError` on `self.dry_run` before set), live path never sets `marginType=ISOLATED`, SL uses `last` not mark price, `BybitFutures` is a placeholder stub, 8 plaintext `os.getenv` API-key fallbacks, validator is dead code.
- **AI** — `StandardScaler.fit_transform` on a single live row poisons every feature; `train_test_split(stratify=y)` on time series leaks; sentiment from unsanitized news headlines fed to LLM (prompt injection); `AITradeExecutor` signs Binance orders bypassing risk-manager.
- **COPY_TRADING** — Solana SELL mirror is fake (stamps `SELL_TRACKED_*` without calling Jupiter); EVM `* 1e18` unit-bug lets a single trade be 2000× operator cap; zero leader scoring, zero Kelly; emergency-stop skips this module entirely.
- **DASHBOARD** — WebSocket unauthenticated; no CSRF; `secure=False` cookie; default `admin/admin123` printed publicly; per-module "pause" is cosmetic; "PANIC SELL ALL" returns 404; Sharpe implemented 4 times disagreeing 17%; backtest fee-/slippage-/gas-/funding-free.

---

## Top 20 P0s — live-blockers, must-fix before `DRY_RUN=false`

Ranked by combined severity × ROI. Owner is the primary specialist. Effort: **S** ≤ 200 LoC single PR, **M** ≤ 1000 LoC over 2-4 PRs, **L** ≥ 3 PRs.

| Rank | ID | Module | Title | File:Line | Owner | Effort | Source(s) |
|:---:|---|---|---|---|---|:---:|---|
| 1 | MB-01 / DEX-01 | DEX | `min_amount_out` hardcodes `*10**18` regardless of token decimals — ~60% of pairs (USDC/USDT/WBTC) broken | `trading/executors/direct_dex.py:560-563` | smartcontract | S | DEX_smartcontract |
| 2 | MB-02 / DEX-02 | DEX | Flashbots signature uses wrong hashing (`sha256+signHash` vs EIP-191 `keccak+sign_message`) — every "protected" tx silently falls back to public mempool | `trading/executors/mev_protection.py:388-393` | smartcontract | S | DEX_smartcontract |
| 3 | MB-03 / ARB-01 | ARBITRAGE | DAI mainnet address typo in 3 places — DAI silently non-tradable across all arbitrage paths | `modules/arbitrage/arbitrage_engine.py:169,1632`, `triangular_engine.py:74` | smartcontract | S | ARBITRAGE_smartcontract |
| 4 | MB-04 / ARB-02 | ARBITRAGE | One-legged BUY-only broadcast on Flashbots fallback — guaranteed inventory leak | `modules/arbitrage/arbitrage_engine.py:1856`, `triangular_engine.py:1052-1075` | smartcontract | S | ARBITRAGE_smartcontract, ARBITRAGE_analyst |
| 5 | MB-05 / ARB-03 | ARBITRAGE | Triangular bundle path uses placeholder `amountIn=1` on tx2/tx3 — intermediate tokens stranded on partial fills | `modules/arbitrage/triangular_engine.py:921-1075` | smartcontract | M | ARBITRAGE_smartcontract |
| 6 | MB-06 / SOL-01 | SOLANA | Token decimals hardcoded to 6 in close path while buy path reads correctly — BONK (5), modern launches (9) close at wrong size | `modules/solana_trading/core/solana_engine.py:3308,3335` (buy ok at `:3086`) | smartcontract | S | SOLANA_smartcontract |
| 7 | MB-07 / SOL-02 | SOLANA | Jupiter signing discards co-signers: `sign_message(bytes(message))` + `VersionedTransaction.populate([signature])` corrupts multi-signer routes | `trading/chains/solana/jupiter_executor.py:712-713`, `modules/sniper/core/trade_executor.py:561-562` | smartcontract | S | SOLANA_smartcontract |
| 8 | MB-08 / SOL-03 | SOLANA | Jupiter swap request omits `prioritizationFeeLamports`, `dynamicComputeUnitLimit`, `dynamicSlippage` — ~30-50% of fills drop under congestion | `modules/solana_strategies/jupiter_helper.py:478-484` | smartcontract | S | SOLANA_smartcontract |
| 9 | MB-09 / SOL-04 | SOLANA | Open positions never reconciled on restart — `active_positions={}` on init; `_load_historical_stats` only counters | `modules/solana_trading/core/solana_engine.py:1113,:1720` | smartcontract | M | SOLANA_analyst |
| 10 | MB-10 / SOL-05 | SOLANA | `JupiterExecutor.execute_trade` DRY_RUN check indented inside `if not self.session:` — skipped on every call after first | `trading/chains/solana/jupiter_executor.py:211` | smartcontract | S | SOLANA_analyst |
| 11 | MB-11 / SNIPE-01 | SNIPER | `amount_out_min=0` on EVM buy AND sell (comment: "Accept any amount (risky, but for speed)") — guaranteed sandwich | `modules/sniper/core/trade_executor.py:621,709` | smartcontract | S | SNIPER_smartcontract |
| 12 | MB-12 / SNIPE-02 | SNIPER | Hardcoded `take_profit_pct=50` / `stop_loss_pct=-20` shadows DB config — kills heavy-tail edge; every winner capped at 1.5x | `modules/sniper/core/sniper_engine.py:580-581` (shadows `:207-210`) | smartcontract | S | SNIPER_analyst, SNIPER_quant |
| 13 | MB-13 / SNIPE-03 | SNIPER | `is_simulated=True` hardcoded — every live snipe mis-labeled as paper (PnL stats corruption) | `modules/sniper/core/sniper_engine.py:564` | analyst | S | SNIPER_analyst |
| 14 | MB-14 / SNIPE-04 | SNIPER | Solana price uses deprecated `price.jup.ag/v4` (returns 0) — SL never fires on Solana snipes | `modules/sniper/core/sniper_engine.py` (price-fetch path) | smartcontract | S | SNIPER_smartcontract, SNIPER_analyst |
| 15 | MB-15 / SOL-06 | SOLANA | Drift perp helper has no DRY_RUN gate, no leverage cap, no oracle/liq/funding guards — RED leg of AMBER module | `modules/solana_strategies/drift_helper.py` | analyst | M | SOLANA_analyst |
| 16 | MB-16 / FUT-01 | FUTURES | Bot fails to start in documented config: `not self.dry_run` evaluated before `self.dry_run` is set | `modules/futures_trading/core/futures_engine.py:262` vs `:361` | backend | S | FUTURES_analyst |
| 17 | MB-17 / FUT-02 | FUTURES | Live engine never sets `marginType=ISOLATED`, never calls `FuturesRiskManager.validate_new_position` (validator is dead code) — cross-margin liq can drain entire wallet | `modules/futures_trading/core/futures_engine.py` live path | analyst | M | FUTURES_analyst, FUTURES_quant |
| 18 | MB-18 / FUT-03 | FUTURES | SL/TP/liquidation use ticker `last` price not exchange mark price — stops trip on wicks | `modules/futures_trading/core/futures_engine.py:841,898` | analyst | S | FUTURES_analyst |
| 19 | MB-19 / AI-01 | AI | `StandardScaler.fit_transform` on single live row — every feature becomes 0/NaN; ML signal is noise | `trading/strategies/ai_strategy.py:289-295` | quant | S | DEX_quant, AI_quant |
| 20 | MB-20 / AI-02 | AI | `AITradeExecutor` signs Binance orders inline bypassing `risk_manager.validate_trade`, no `reduceOnly`, no `ISOLATED`. Saved only by `direct_trading=False` default | `modules/ai_analysis/core/sentiment_engine.py:113-187,:232,:943` | analyst | M | AI_analyst, AI_backend |
| 21 | MB-21 / AI-03 | AI | Prompt-injection vector: unsanitized CryptoCompare headlines into LLM prompt — one adversarial headline flips sentiment to arbitrary value | `modules/ai_analysis/core/sentiment_engine.py:488-503,:511` | backend | S | AI_analyst |
| 22 | MB-22 / CT-01 | COPY_TRADING | Solana SELL mirror is FAKE — stamps `SELL_TRACKED_*` and marks closed without calling Jupiter. Mirror positions never exit | `modules/copy_trading/copy_engine.py:1097-1106` | smartcontract | S | COPY_TRADING_analyst |
| 23 | MB-23 / CT-02 | COPY_TRADING | EVM `max_copy_amount * 1e18` unit-bug — single trade can be 2000× operator-intended cap | `modules/copy_trading/copy_engine.py:906` | smartcontract | S | COPY_TRADING_analyst, COPY_TRADING_quant |
| 24 | MB-24 / CT-03 | COPY_TRADING | EVM executor hardcoded to Uniswap V2 on Ethereum while monitoring is multi-chain → wrong-chain swaps | `modules/copy_trading/copy_engine.py` (EVM swap path) | smartcontract | M | COPY_TRADING_backend |
| 25 | MB-25 / CT-04 | COPY_TRADING | Engine has no `get_positions()` / `close_position()` — emergency-stop silently skips COPY positions | `modules/copy_trading/copy_engine.py` | analyst | S | COPY_TRADING_analyst |
| 26 | MB-26 / DASH-SEC-01 | DASHBOARD | WebSocket is unauthenticated — anyone reaching the port reads/writes events | `monitoring/enhanced_dashboard.py:132,1446` | backend | S | DASHBOARD_backend |
| 27 | MB-27 / DASH-SEC-02 | DASHBOARD | No CSRF on any POST endpoint — including `/api/credentials/*` and `/api/bot/*` | dashboard route handlers | backend | M | DASHBOARD_backend |
| 28 | MB-28 / DASH-SEC-03 | DASHBOARD | Session cookie `secure=False` hardcoded; `/api/credentials/*` and `/api/bot/*` not admin-gated | `monitoring/auth_routes.py:109` + routes | backend | S | DASHBOARD_backend |
| 29 | MB-29 / DASH-SEC-04 | DASHBOARD | Default `admin/admin123` printed in login page HTML and startup logs | `dashboard/templates/login.html:92-99`, startup logger | backend | S | DASHBOARD_backend |
| 30 | MB-30 / DASH-OPS-01 | DASHBOARD | Per-module "Pause" is cosmetic — only flips flag; engine loops never check it | `modules/base_module.py:286-291` + engine loops | analyst | M | DASHBOARD_analyst |
| 31 | MB-31 / DASH-OPS-02 | DASHBOARD | "PANIC SELL ALL" posts to `/api/bot/emergency_exit` while route is `/api/bot/emergency-exit` (underscore vs dash) → 404 | `dashboard/templates/pro_controls.html` | backend | S | DASHBOARD_analyst |
| 32 | MB-32 / DASH-OPS-03 | DASHBOARD | No always-visible DRY_RUN/LIVE indicator — operator can flip live and forget | `dashboard/templates/base.html` | backend | S | DASHBOARD_analyst |
| 33 | MB-33 / DASH-OPS-04 | DASHBOARD | `_update_env_file` writes don't reach already-spawned subprocesses — operator sees "enabled" while module is off | `monitoring/enhanced_dashboard.py` `_update_env_file` | backend | M | DASHBOARD_backend |

> Items 21-33 are still P0s (live-blockers). The "Top 20" framing is shorthand; **the full P0 list is 33 items.**

---

## P1 backlog — high-impact, not-immediate-blockers

| Rank | ID | Module | Title | Owner | Effort | Source(s) |
|:---:|---|---|---|---|:---:|---|
| 34 | P1-01 / DEX-Q-01 | DEX | Hardcoded flat 10% stop-loss ignores existing `RiskManager.calculate_stop_loss` | `core/engine.py:1056,1080,1308` | analyst | S | DEX_analyst |
| 35 | P1-02 / DEX-P-01 | DEX | LIVE branch updates `active_positions` but not `position_tracker.positions` — live positions miss SL/TP monitoring | `core/engine.py:1316` vs `:3034` | smartcontract | S | DEX_analyst |
| 36 | P1-03 / DEX-P-02 | DEX | `_load_state` is `pass` stub — no position reconciliation on restart | `core/engine.py:2715` | smartcontract | S | DEX_analyst |
| 37 | P1-04 / DEX-MEV | DEX | No `pool_engine` integration; every `Web3(HTTPProvider(...))` direct read — bypasses RPC failover, rate-limit reporting | all executors | backend | M | DEX_smartcontract |
| 38 | P1-05 / ARB-Q-01 | ARBITRAGE | Hardcoded $15 gas + 0.6% slippage in P&L logging — massively misreports L2 economics | `modules/arbitrage/arbitrage_engine.py` PnL log path | quant | S | ARBITRAGE_smartcontract |
| 39 | P1-06 / ARB-RISK | ARBITRAGE | Bypasses cross-module `risk_manager` entirely — no breakers, no hourly gas budget | `modules/arbitrage/arbitrage_engine.py` | analyst | M | ARBITRAGE_analyst |
| 40 | P1-07 / SOL-Q-01 | SOLANA | ML stack (`pump_predictor`, `rug_classifier`, `volume_validator`, `token_scorer`) completely unwired from `solana_engine`; wire into `_open_position` at `:2823` with Kelly-fraction sizing | quant | M | SOLANA_quant |
| 41 | P1-08 / SOL-Q-02 | SOLANA | `ml/models/pump_predictor.py:184` labels built with future-price look-ahead | quant | S | SOLANA_quant |
| 42 | P1-09 / SOL-Q-03 | SOLANA | In-sample tuning — engine has tokens hard-named in comments (`solana_engine.py:2527,:2713`) | quant | S | SOLANA_quant |
| 43 | P1-10 / SOL-Q-04 | SOLANA | Emergency-close PnL hardcoded to −50% (`solana_engine.py:2076`) | analyst | S | SOLANA_quant |
| 44 | P1-11 / SOL-Q-05 | SOLANA | Static 200 bps default Jupiter slippage and 1200 bps pump.fun close slippage — $10-40/day burn on $400 book | quant | S | SOLANA_analyst |
| 45 | P1-12 / SNIPE-05 | SNIPER | No WebSocket / Geyser subscription — `SNIPER_USE_WEBSOCKET` flag is read and ignored. Detection latency 15-120s vs competitors' 50-400ms | `modules/sniper/core/solana_listener.py:94` + EVM listener | smartcontract | L | SNIPER_smartcontract |
| 46 | P1-13 / SNIPE-06 | SNIPER | `UNISWAP_V2_ROUTER` mainnet constant used for ALL EVM chains — non-mainnet snipes revert | `modules/sniper/core/trade_executor.py` | smartcontract | S | SNIPER_smartcontract |
| 47 | P1-14 / SNIPE-07 | SNIPER | `FACTORIES` dict has only Uniswap V2 + SushiSwap mainnet — no Base/Arb/BSC, no V3/V4, no Maverick/Curve | `modules/sniper/core/evm_listener.py:33-37` | smartcontract | M | SNIPER_smartcontract |
| 48 | P1-15 / SNIPE-08 | SNIPER | `_parse_log` returns `token0` as interesting token without checking which side is non-WETH/USDC | `modules/sniper/core/evm_listener.py:131,:117` | smartcontract | S | SNIPER_smartcontract |
| 49 | P1-16 / SNIPE-Q-01 | SNIPER | Port Solana multi-tier trailing-stop ladder to SNIPER exit; fix deprecated price feed first | quant | M | SNIPER_quant |
| 50 | P1-17 / SNIPE-Q-02 | SNIPER | `token_safety._calculate_score` hand-tuned with discrete 70/40 thresholds, never calibrated to realized outcomes | `modules/sniper/core/token_safety.py:477-542` | quant | M | SNIPER_quant |
| 51 | P1-18 / FUT-04 | FUTURES | No `clientOrderId` on `create_market_order` — duplicate-fill risk on retry | `modules/futures_trading/exchanges/binance_futures.py` | backend | S | FUTURES_analyst, FUTURES_backend |
| 52 | P1-19 / FUT-05 | FUTURES | Funding payments never credited to P&L | `modules/futures_trading/core/futures_engine.py` | analyst | M | FUTURES_analyst |
| 53 | P1-20 / FUT-06 | FUTURES | `BybitFutures` is pure placeholder stub — but the module advertises Bybit support | `modules/futures_trading/exchanges/bybit_futures.py` | backend | L | FUTURES_backend |
| 54 | P1-21 / FUT-07 | FUTURES | 8 `os.getenv()` API-key plaintext fallbacks — must move to encrypted DB via `setup_env_keys.py` | `modules/futures_trading/core/futures_engine.py:572-685` | backend | M | FUTURES_backend |
| 55 | P1-22 / FUT-Q-01 | FUTURES | `combined_score` attribute is set but never used — dynamic sizing silently disabled, every trade is mid-range fixed size | `modules/futures_trading/futures_module.py` `_combine_signals` | quant | S | FUTURES_quant |
| 56 | P1-23 / FUT-Q-02 | FUTURES | Funding-arb math omits fees — wrong sign on tight spreads | `modules/futures_trading/strategies/funding_arbitrage.py` | quant | S | FUTURES_quant |
| 57 | P1-24 / FUT-Q-03 | FUTURES | MACD signal-line broken | `modules/futures_trading/core/futures_engine.py:1308` | quant | S | FUTURES_quant |
| 58 | P1-25 / FUT-08 | FUTURES | 30-second liquidation polling instead of user-data WebSocket — fast liq invisible until drained | `modules/futures_trading/futures_module.py:820` | backend | M | FUTURES_backend |
| 59 | P1-26 / AI-04 | AI | `train_test_split(stratify=y)` on time-series — look-ahead leak in trainer | `ml/training/auto_trainer.py:259-263` | quant | S | AI_quant |
| 60 | P1-27 / AI-05 | AI | LSTM inference `seq_len=1` vs trained `seq_len=20` — shape mismatch silently degrades model | quant | S | AI_quant |
| 61 | P1-28 / AI-06 | AI | `_retrain_models` is a stub; synthetic-data class-balancing inflates fake 95% accuracy | `ml/training/auto_trainer.py` | quant | M | AI_quant |
| 62 | P1-29 / AI-07 | AI | `confidence_threshold` compared against sentiment magnitude, not LLM `confidence` field | `modules/ai_analysis/core/sentiment_engine.py` | quant | S | AI_analyst |
| 63 | P1-30 / AI-08 | AI | `symbol="ETH"` hardcoded for every sentiment-driven trade | `modules/ai_analysis/core/sentiment_engine.py:745` | backend | S | AI_backend |
| 64 | P1-31 / AI-09 | AI | Legacy LLM-call path triggers on legitimate neutral sentiment — double-pays for same headlines | `modules/ai_analysis/core/sentiment_engine.py` | backend | S | AI_backend |
| 65 | P1-32 / CT-Q-01 | COPY_TRADING | Add leader-score table: rolling 30/90/180d Sharpe + max-DD + concentration + sample-size, survivorship-bias-free | quant | M | COPY_TRADING_quant |
| 66 | P1-33 / CT-Q-02 | COPY_TRADING | Replace 1:1 mirror with fractional-Kelly (Kelly machinery exists unreachable at `core/decision_maker.py:698`, `core/portfolio_manager.py:399-484`) | quant | M | COPY_TRADING_quant |
| 67 | P1-34 / CT-05 | COPY_TRADING | `_load_settings` (`copy_engine.py:616-657`) only reads `target_wallets`; every other config knob is theater (dry_run, max_amount, slippage, copy_ratio) | backend | S | COPY_TRADING_analyst |
| 68 | P1-35 / CT-06 | COPY_TRADING | 15s polling + 60s recency window = ≥75s latency on memecoins → 30-90% entry penalty | backend | M | COPY_TRADING_backend |
| 69 | P1-36 / CT-07 | COPY_TRADING | No health-port endpoint — dashboard always shows COPY as offline | backend | S | COPY_TRADING_backend |
| 70 | P1-37 / CT-08 | COPY_TRADING | `min_out=0` quote fallback — sandwich exposure | `copy_engine.py` quote path | smartcontract | S | COPY_TRADING_backend |
| 71 | P1-38 / CT-09 | COPY_TRADING | `dry_run` snapshotted at `__init__` — can't be toggled live without restart | smartcontract | S | COPY_TRADING_backend |
| 72 | P1-39 / DASH-Q-01 | DASHBOARD | Unify Sharpe (single implementation, single annualization 252 vs 365, single RF treatment); ship as one PR with DASH-Q-02 + DASH-Q-03 | quant | M | DASHBOARD_quant |
| 73 | P1-40 / DASH-Q-02 | DASHBOARD | MTM-aware drawdown (not closed-trade-only) | quant | S | DASHBOARD_quant |
| 74 | P1-41 / DASH-Q-03 | DASHBOARD | Friction-aware backtest — current backtest at `enhanced_dashboard.py:7259-7322` replays trades with NO fees/slippage/funding/gas; hardcodes `sharpe:0, sortino:0` placeholders | quant | M | DASHBOARD_quant |
| 75 | P1-42 / DASH-Q-04 | DASHBOARD | Annual-vol unit bug (`* 100` on $-units) | quant | S | DASHBOARD_quant |
| 76 | P1-43 / DASH-Q-05 | DASHBOARD | Calmar unit-inconsistent | quant | S | DASHBOARD_quant |
| 77 | P1-44 / DASH-Q-06 | DASHBOARD | Backtest hardcodes `sharpe_ratio:0, sortino_ratio:0` placeholders — UI shows them as real | quant | S | DASHBOARD_quant |
| 78 | P1-45 / DASH-BE-01 | DASHBOARD | 101 `os.getenv` reads in `enhanced_dashboard.py` — must move to ConfigManager | backend | L | DASHBOARD_backend |
| 79 | P1-46 / DASH-BE-02 | DASHBOARD | K8s `/health` endpoint doesn't exist — liveness probes loop | backend | S | DASHBOARD_backend |
| 80 | P1-47 / DASH-BE-03 | DASHBOARD | `kubernetes/secret.yaml` referenced but not shipped | backend | S | DASHBOARD_backend |
| 81 | P1-48 / DASH-MISSING | DASHBOARD | DEX missing all 4 per-module pages (`dashboard_dex.html`, `performance_dex.html`, `trades_dex.html`, `positions_dex.html`); AI missing `trades_ai.html`, `positions_ai.html` | backend | M | DASHBOARD_backend |
| 82 | P1-49 / DASH-GUIDE | DASHBOARD | Settings Guide tab missing on DEX, Futures, Solana settings pages | backend | S | DASHBOARD_backend |
| 83 | P1-50 / OPS-EM | CROSS | `scripts/emergency_stop.py` and `scripts/close_all_positions.py` are MISSING from repo | analyst | M | SOLANA_analyst, SNIPER_analyst, DEX_analyst |

---

## P2 backlog — improvements, refactors, nice-to-haves

Grouped by module. Source: P2 + non-blocker items from the 24 reports.

- **DEX** — Unify three executors into one; consolidate divergent slippage caps, gas ceilings, approval semantics, native-token addresses; remove `ToxiSolAPIExecutor` if unused; document chain-ID → router map.
- **ARBITRAGE** — Bridge-time and reorg-risk pricing for cross-chain arb; oracle-deviation kill-switch; per-leg sizing optimization.
- **SOLANA** — Add Jito bundle path; ATA rent-recovery on exits; multi-RPC routing through `pool_engine`; Helius staked-connection prioritization.
- **SNIPER** — Multi-wallet rotation; race-loss abort with gas refund; auto-exit on dev-wallet sell / LP unlock / time-stop; honeypot pre-sim mandatory.
- **FUTURES** — Bybit V5 unified-margin endpoints; symbol-precision via `exchangeInfo`; rate-limit-header parsing; user-data WS for fills (replaces polling).
- **AI** — Provider fallback chain Anthropic → OpenAI → Cohere; per-day $-budget cap on LLM spend; prompt template versioning in DB; embeddings cache.
- **COPY_TRADING** — On-chain whale-tracking sources; CEX leaderboard ingestion (Bybit/OKX); per-leader paper sandbox.
- **DASHBOARD** — Mobile-responsive layout; logs page live stream with filter; latency widget; trade-rejection feed; alert acknowledgement workflow; multi-operator locking.
- **CROSS** — Prometheus counters per module (`{module}_signals_total`, `{module}_trades_total`, `{module}_pnl_usd`); Grafana dashboards; alert routing.

---

## Cross-cutting themes

1. **Secrets in plaintext.** Every module reads private keys / API keys from `.env` or `os.getenv` directly. Worst offenders: FUTURES (`futures_engine.py:572-685`, 8 fallbacks), DASHBOARD (`enhanced_dashboard.py`, 101 reads). COPY_TRADING is the clean example using `RPCProvider`/`secrets.get` first. **Fix:** route all sensitive reads through `security/encryption.py` + `config_sensitive` table via `setup_env_keys.py`.

2. **RPC pool bypass.** Every chain client builds `Web3(HTTPProvider(...))` directly — `config/pool_engine.py` is bypassed across DEX, ARBITRAGE, SOLANA, SNIPER. Failover, rate-limit reporting, Helius/Alchemy weighting all dead. **Fix:** force all RPC reads through `await pool_engine.get_endpoint(provider_type)` with success/failure reporting.

3. **DRY_RUN propagation gaps.** Solana `JupiterExecutor.execute_trade` skips DRY_RUN after first call due to wrong indentation; Drift helper has no DRY_RUN at all; AI's `AITradeExecutor` only saved by `direct_trading=False` default; FUTURES bot fails to start because `not self.dry_run` is evaluated before the attribute is set. **Fix:** lint pass — every `send/sign/order/transfer` path must call a single `should_skip_live(module, account)` helper.

4. **Decimals / amount-out / unit bugs.** Same class of bug in every module: DEX hardcodes `*10**18` ignoring token decimals; SOLANA hardcodes `6` in close while buy reads correctly; SNIPER hardcodes `6` in output conversion; COPY hardcodes `* 1e18` in EVM cap (2000× overflow). **Fix:** central `unit_convert(token, amount, kind)` helper using `pool_engine`-cached `decimals()` reads. Delete every literal exponent in trading code.

5. **Risk-manager bypass.** ARBITRAGE engine bypasses `core/risk_manager` entirely; FUTURES live path never calls `FuturesRiskManager.validate_new_position` (dead code); AI executor signs orders bypassing `risk_manager.validate_trade`. COPY_TRADING has zero integration. **Fix:** make every execution path go through one validator that emits a Prometheus `{module}_risk_rejection_reason` counter on denial.

6. **Position reconciliation on restart.** DEX (`_load_state` is `pass`), SOLANA (`active_positions={}`), COPY_TRADING (no `get_positions()`), FUTURES (no exchange-truth reconcile). **Fix:** every module must implement `async def reconcile_on_startup(self) -> list[Position]` and the orchestrator must call it before `start()`.

7. **Emergency-stop coverage gaps.** `scripts/emergency_stop.py` and `scripts/close_all_positions.py` are entirely MISSING from the repo; "PANIC SELL ALL" button returns 404; per-module "pause" is cosmetic; COPY_TRADING skipped silently. **Fix:** scripts + per-module `flatten()` + dashboard one-click button wired to a single `/api/bot/emergency-exit` endpoint that iterates every module's `flatten()`.

8. **Dashboard math truth.** Sharpe implemented 4× disagreeing 17%+; Calmar unit-inconsistent; backtest fee-/slippage-/gas-/funding-free; closed-trade-only max-DD; placeholder zeros displayed as real values. **Fix:** single `analytics/metrics.py` module imported everywhere; backtest replays through real execution-cost model.

9. **Missing per-module dashboard pages.** DEX: all 4 missing (`dashboard_dex`, `performance_dex`, `trades_dex`, `positions_dex`). AI: `trades_ai`, `positions_ai` missing. Settings Guide tab missing on DEX, Futures, Solana. **Fix:** scaffold all missing pages from a single Jinja macro template that takes `module_key`.

10. **Look-ahead / leakage / in-sample tuning.** SOLANA pump-predictor labels with future-price look-ahead (`ml/models/pump_predictor.py:184`); AI trainer uses `train_test_split(stratify=y)` on time series; engine has tokens hand-named in comments; SNIPER token-safety thresholds are hand-tuned never calibrated; FUTURES thresholds picked with no walk-forward. **Fix:** walk-forward CV harness in `ml/training/cv.py`; trainer refuses to fit without it.

---

## Recommended execution sequence

### Phase 2 — infra prerequisites (must precede module fixes)

These unblock the per-module work. Each item is a small commit ladder, ≤ 200 LoC per commit.

1. **Secrets migration utility hardening** — make `setup_env_keys.py` idempotent; produce a deterministic mapping of every `os.getenv` read to a `config_sensitive` key. **Owner:** backend. **Effort:** M. **Unblocks:** secrets remediation in all modules.
2. **`pool_engine` enforcement** — add a CI grep that fails on `Web3(Web3.HTTPProvider(...))` outside `config/pool_engine.py`; add `pool_engine.get_solana_endpoint()` helper. **Owner:** backend. **Effort:** S. **Unblocks:** RPC failover across DEX/ARB/SOLANA/SNIPER.
3. **Unit-conversion helper** — `core/units.py` with `to_wei(token, amount)`, `from_wei(token, amount)`, `to_lamports(token, amount)`, `from_lamports(token, amount)`; all use cached on-chain decimals via `pool_engine`. **Owner:** backend. **Effort:** S. **Unblocks:** MB-01, MB-06, MB-23.
4. **`should_skip_live` helper** — single source of DRY_RUN truth; replaces 30+ scattered checks. **Owner:** backend. **Effort:** S. **Unblocks:** every DRY_RUN P0.
5. **Risk-manager mandatory hook** — add `core/risk_manager.gate(trade) -> Allowed | Rejected(reason)`; refactor every execution path to call it. **Owner:** analyst. **Effort:** M. **Unblocks:** MB-17, MB-20, MB-25, ARB-RISK.
6. **Position reconciliation contract** — `BaseModule.reconcile_on_startup()` abstract method; orchestrator gating. **Owner:** backend. **Effort:** S. **Unblocks:** every restart-reconcile P0.
7. **Emergency-stop scripts + endpoint** — ship `scripts/emergency_stop.py`, `scripts/close_all_positions.py`; fix `/api/bot/emergency-exit` dash/underscore; wire `flatten()` on every module. **Owner:** backend + analyst. **Effort:** M. **Unblocks:** MB-25, MB-31, MB-32, OPS-EM.
8. **Single-source analytics module** — `analysis/metrics.py` with one Sharpe / Sortino / Calmar / MaxDD / friction-aware backtest; delete the 4 disagreeing copies. **Owner:** quant. **Effort:** M. **Unblocks:** every DASH-Q-NN.

### Phase 3 — module live-readiness (parallelizable across modules; serial within)

For each module, the order is: critical-bug fixes → reconciliation → risk-gating → observability → live-readiness checklist. Once all P0s for a module are merged, that module can flip `DRY_RUN=false` after a paper-mode dry-run.

**DEX** — MB-01 → MB-02 → P1-02 → P1-03 → P1-04 → P1-48 (dashboard pages) → checklist sign-off.
**ARBITRAGE** — MB-03 → MB-04 → MB-05 → P1-05 → P1-06 → checklist sign-off.
**SOLANA** — MB-06 → MB-07 → MB-08 → MB-09 → MB-10 → MB-15 (or remove Drift) → P1-07 → P1-11 → checklist.
**SNIPER** — MB-11 → MB-12 → MB-13 → MB-14 → P1-12 → P1-13 → P1-14 → P1-15 → P1-16 → checklist.
**FUTURES** — MB-16 → MB-17 → MB-18 → P1-18 → P1-21 (FUT-07 secrets) → P1-22 → checklist.
**AI** — MB-19 → MB-20 → MB-21 → P1-26 → P1-27 → P1-29 → P1-30 → checklist (shadow-mode only until calibrated).
**COPY_TRADING** — MB-22 → MB-23 → MB-24 → MB-25 → P1-32 → P1-33 → P1-34 → P1-35 → checklist.
**DASHBOARD** — MB-26 → MB-27 → MB-28 → MB-29 → MB-30 → MB-31 → MB-32 → MB-33 → P1-39 → P1-48 → P1-49 → checklist.

### Phase 4 — new strategies / modules (only after Phase 3)

Defer detailed planning. Candidates from the audits:
- Funding-rate arb cross-exchange (after FUT-08 user-data WS is in)
- Jito bundle on Solana (after MB-07, MB-08)
- Multi-wallet sniper rotation (after MB-11..MB-14)
- LLM-confidence-weighted sentiment trading (after MB-19..MB-21)
- AI-gated DEX strategy using the unwired ML stack (after P1-07)

### Phase 5 — docs + dead-code cleanup

**Dead-code candidates** (delete only after final specialist confirms zero importers):
- `core/engine.py.backup`, `core/engine_preSOL.py.backup`
- `data/collectors/dexscreener.py.backup`, `data/collectors/honeypot_checker.py.backup`, `data/collectors/honeypot_checker_beforeRugcheck.py.backup`
- `trading/chains/solana/jupiter_executor.py.backup`
- `trading/strategies/scalping copy.py`
- `dashboard/templates/index_new.html` (if `index.html` is canonical)
- `docker-compose copy.yml.example`
- `scripts/verify_claudedex_plus*.py` (3 versions — keep latest only)
- `FUTURES_MODULE`'s unreachable `FuturesTradingModule` class and `validate_new_position` dead code (after wiring it back via Phase 2 risk-gating)

**CLAUDE.md skeleton ownership** (PM agent):
- `CLAUDE.md` (root) — project overview, module table, run/deploy/oncall pointers
- `modules/dex_trading/CLAUDE.md`
- `modules/arbitrage/CLAUDE.md`
- `modules/solana_trading/CLAUDE.md` and/or `modules/solana_strategies/CLAUDE.md`
- `modules/sniper/CLAUDE.md`
- `modules/futures_trading/CLAUDE.md`
- `modules/ai_analysis/CLAUDE.md`
- `modules/copy_trading/CLAUDE.md`
- `modules/dashboard/CLAUDE.md`

**Reference docs to create:**
- `docs/engines.md` — RPC pool, decision maker, risk manager, portfolio manager, order manager, executors
- `docs/logging.md` — per-module log layout, levels, rotation, error paths
- `docs/dashboards.md` — page map, settings tabs, WS events, auth
- `docs/staging.md` — paper-trade workflow (DRY_RUN=true), testnet workflow, canary deploy
- `docs/deployment.md` — Docker/K8s, secrets bootstrap via `setup_env_keys.py`, DB migration order, rollback
- `docs/issues.md` — known issues, triage labels, escalation
- `docs/runbook.md` — incident response (stuck nonce, RPC blackout, oracle deviation, drawdown breach)

---

## Profitability playbook

*Grounded in audit findings. Use only after the named gates close.*

### Capital tiering by current verdict

| Module | Verdict | Max % of capital before fixes | After P0s green |
|---|---|---|---|
| ARBITRAGE (spatial EVM) | AMBER | 0% (DAI typo, MEV-fallback leak) | 15-25% (highest-conviction strategy in repo) |
| ARBITRAGE (triangular) | RED | 0% | 0% until atomic path proven |
| ARBITRAGE (Solana) | RED | 0% | 5-10% (Jito-only, no fallback) |
| SOLANA (Jupiter spot) | AMBER | 0% (decimals + signing) | 10-15% |
| SOLANA (Drift perps) | RED | 0% | Pull or harden — operator decision |
| DEX | RED | 0% | 10-15% after MB-01, MB-02 |
| SNIPER | RED | 0% | 1-3% (structurally late — start small) |
| FUTURES | RED | 0% (won't start) | 10-15% after MB-16..MB-18, P1-18..P1-25 |
| AI | RED | 0% live (shadow OK) | Signal-only until calibrated |
| COPY_TRADING | RED | 0% (fake SELL, unit bug) | 5-10% with ≤3 leaders, fractional-Kelly |

### Daily routine
- **Pre-market (UTC 00:00):** check overnight error log; check drawdown vs daily cap; check RPC pool health (success / failure / rate-limit counters); check WS reconnect timestamps; confirm `DRY_RUN` status per module on dashboard.
- **Mid-day (UTC 12:00):** reconcile positions vs exchange/on-chain truth; review trade-rejection feed (any new rejection reasons?); P&L attribution per strategy.
- **End-of-day (UTC 23:00):** snapshot equity curve; if daily drawdown > -2% halve sizing for next day; if -5% halt new entries until manual review.

### Weekly routine
- Re-run walk-forward backtest of any strategy with parameter changes that week.
- COPY_TRADING: refresh leader scores (rolling 30/90/180d); cull leaders with sample-size < 30 or DD-recovery > 30 days.
- ML models: trigger retrain only if drift detector fires (PSI > 0.2 on any feature) — never on calendar alone.

### Monthly routine
- Backtest re-validation against last 30 days of live fills (paper vs realized gap).
- Rotate API keys (CEX) and audit `config_sensitive.last_rotated`.
- Review missing/orphaned positions; clear stale entries.

### Red flags (auto-action)
- Daily DD > -3% → freeze new entries module-wide, hold existing.
- Hourly failed-tx gas > realized profit → ARBITRAGE module auto-halts.
- RPC pool: all endpoints unhealthy for > 60s → all on-chain modules flatten or freeze (operator-selected policy).
- Oracle deviation > 50 bps from second-source → all liquidation-sensitive modules flatten.
- WebSocket gap > 30s → operator alert; > 60s → freeze module.

---

## Open questions for operator

These decisions belong to the operator, not the team:

- Per-module capital cap targets (the table above is a starting point — what's the actual book size?)
- Primary CEX (Binance vs Bybit) — Bybit connector is a stub; either invest L-effort to build it or drop Bybit from the module's advertised capabilities.
- Drift perp leg of SOLANA module — currently RED. Pull and re-launch later, or harden in-place (MB-15, M-effort)?
- COPY_TRADING leader seeding policy — operator-curated list or algorithmic discovery (`discovery_copytrading.html`)? If algorithmic: require approval workflow before promotion.
- LLM provider order — Anthropic / OpenAI / Cohere / Perplexity / Mistral? AI module currently references Cohere/Perplexity/Mistral but doesn't implement them (`ai_provider.py` roadmap commitment or drop them).
- Dashboard hosting — public domain with strict auth + 2FA + IP allowlist, or VPN-only?
- Daily LLM-spend budget — currently $10/day (`ai_provider.py:160`), gives 6700 sentiment calls. What's the production target?
- `target_wallets` — should leader pubkeys in COPY config be encrypted (currently plaintext in `config_settings:623`)?
- Multi-operator coordination — should "pause/resume" require operator-locking?

---

## Appendix A: Finding → Module → Owner map

*Every MB-NN ID for cross-reference. Full owner / effort / source columns. Ordered by ID.*

| ID | Module | Owner | Effort | Source report(s) | Description |
|---|---|---|:---:|---|---|
| MB-01 | DEX | smartcontract | S | DEX_smartcontract | `min_amount_out` hardcodes `*10**18` |
| MB-02 | DEX | smartcontract | S | DEX_smartcontract | Flashbots signing broken |
| MB-03 | ARBITRAGE | smartcontract | S | ARBITRAGE_smartcontract | DAI mainnet address typo |
| MB-04 | ARBITRAGE | smartcontract | S | ARBITRAGE_smartcontract, ARBITRAGE_analyst | One-legged BUY fallback |
| MB-05 | ARBITRAGE | smartcontract | M | ARBITRAGE_smartcontract | Triangular bundle `amountIn=1` |
| MB-06 | SOLANA | smartcontract | S | SOLANA_smartcontract | Close-path decimals hardcoded |
| MB-07 | SOLANA | smartcontract | S | SOLANA_smartcontract | Jupiter sign drops co-signers |
| MB-08 | SOLANA | smartcontract | S | SOLANA_smartcontract | Swap req omits priority fee + dynamic params |
| MB-09 | SOLANA | smartcontract | M | SOLANA_analyst | No position reconcile on restart |
| MB-10 | SOLANA | smartcontract | S | SOLANA_analyst | DRY_RUN check indentation bug |
| MB-11 | SNIPER | smartcontract | S | SNIPER_smartcontract | `amount_out_min=0` on EVM buy+sell |
| MB-12 | SNIPER | smartcontract | S | SNIPER_analyst, SNIPER_quant | Hardcoded TP/SL shadows DB config |
| MB-13 | SNIPER | analyst | S | SNIPER_analyst | `is_simulated=True` hardcoded |
| MB-14 | SNIPER | smartcontract | S | SNIPER_smartcontract, SNIPER_analyst | Deprecated `price.jup.ag/v4` |
| MB-15 | SOLANA | analyst | M | SOLANA_analyst | Drift helper unguarded |
| MB-16 | FUTURES | backend | S | FUTURES_analyst | `dry_run` AttributeError at init |
| MB-17 | FUTURES | analyst | M | FUTURES_analyst, FUTURES_quant | No `ISOLATED`, no risk-validator call |
| MB-18 | FUTURES | analyst | S | FUTURES_analyst | SL/TP use `last` not `mark` |
| MB-19 | AI | quant | S | DEX_quant, AI_quant | `StandardScaler.fit_transform` single-row |
| MB-20 | AI | analyst | M | AI_analyst, AI_backend | `AITradeExecutor` bypasses risk-manager |
| MB-21 | AI | backend | S | AI_analyst | Prompt injection via headlines |
| MB-22 | COPY_TRADING | smartcontract | S | COPY_TRADING_analyst | Solana SELL mirror is fake |
| MB-23 | COPY_TRADING | smartcontract | S | COPY_TRADING_analyst, COPY_TRADING_quant | EVM `*1e18` unit-bug → 2000× cap |
| MB-24 | COPY_TRADING | smartcontract | M | COPY_TRADING_backend | Uniswap V2 mainnet hardcoded multi-chain |
| MB-25 | COPY_TRADING | analyst | S | COPY_TRADING_analyst | No `get_positions`/`close_position` |
| MB-26 | DASHBOARD | backend | S | DASHBOARD_backend | WebSocket unauthenticated |
| MB-27 | DASHBOARD | backend | M | DASHBOARD_backend | No CSRF on POSTs |
| MB-28 | DASHBOARD | backend | S | DASHBOARD_backend | `secure=False` cookie + admin gating |
| MB-29 | DASHBOARD | backend | S | DASHBOARD_backend | Default `admin/admin123` printed |
| MB-30 | DASHBOARD | analyst | M | DASHBOARD_analyst | Pause is cosmetic |
| MB-31 | DASHBOARD | backend | S | DASHBOARD_analyst | "PANIC SELL ALL" 404 |
| MB-32 | DASHBOARD | backend | S | DASHBOARD_analyst | No DRY_RUN/LIVE indicator |
| MB-33 | DASHBOARD | backend | M | DASHBOARD_backend | `_update_env_file` doesn't reach subprocesses |

*P1 items (P1-01 .. P1-50) follow the same structure in the P1 table above.*

---

*End of MASTER_BACKLOG.md — 33 P0s, 50 P1s, ~50 P2s across 8 modules. Live-blocker count: 33. Modules ready for `DRY_RUN=false`: 0. Next action: pick a Phase 2 prerequisite from §1-8 and start a small-commit PR ladder.*
