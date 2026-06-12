# NEW MODULE IDEAS — Strategic Decision Document

**Date:** 2026-06-12
**Author:** market-trading-analyst (senior desk review)
**Status:** PROPOSAL — no code in this wave; decision doc only
**Scope:** Genuinely NEW module architectures. Explicitly OUT of scope (owned elsewhere):
- Everything in `docs/agents/NEW_STRATEGY_BACKLOG.md` (cross-venue funding desk, smart-money flow, CEX-DEX latency arb, pairs/stat-arb) — those are *strategies inside existing modules*, not new modules.
- The `regime_allocator` (vol-regime capital weighting) currently being built.
- Anything the bot already has: DEX, Solana, Sniper, Futures (incl. funding-carry v2), Arbitrage, AI, Copy, Polymarket, Advisor, Dashboard, meta_controller, orchestrator_ai, portfolio_allocator, backtest_replay, allocation_guard, pnl_tracker, notification_engine/telegram, the `data/collectors/` set (whale_tracker, mempool_monitor, dexscreener, honeypot, social, token_sniffer, volume_analyzer, chain_data).

## Evaluation lens

A 24/7 multi-module bot at the AMBER→GREEN live-flip boundary does not die from a missing alpha source. It dies from: (a) unmeasured execution costs that quietly eat a thin edge, (b) shared-wallet balance chaos (a module starves for gas mid-exit), (c) a tail event no module owns (depeg, oracle break, venue outage), and (d) over-fit parameters tuned on paper fills. The proposals below are ranked with that in mind: **measurement and survival infrastructure first, new P&L surfaces second, moonshots honestly labeled.** Every proposal states why it might NOT work.

Each module follows the established pattern: `modules/<name>/` subprocess under `main.py`, `*_MODULE_ENABLED=true` opt-in (default OFF), own health port, own `CLAUDE.md`, `should_skip_live` + killswitch + pause-file compliance, config via ConfigManager DB keys, breaches logged through `monitoring/alerts.py`.

---

## TIER 1 — Near-term / low-risk (infrastructure that compounds)

### 1. `execution_quality` — Transaction Cost Analysis (TCA) module

**Thesis.** Nine trading modules report P&L, but none of them report *cost honesty*: quoted vs realized slippage, quote-to-fill latency, gas per successful trade, gas burnt on reverts, taker fees vs maker opportunities, Jito tip efficiency, partial-fill rates. With per-trade edges in the 10–80 bps range across DEX/Sniper/Arb, a 15 bps systematic execution leak is the difference between GREEN and slow bleed — and right now nobody would see it. TCA is the highest-information-per-dollar module available: it makes every other module's verdict honest and gives meta_controller and the new regime_allocator a cost-adjusted (not gross) performance signal.

**Where it slots in.** New `modules/execution_quality/` subprocess, read-only over the existing trade tables + a small set of new "intent" columns/side-tables (quoted price, quoted gas, quote timestamp) that each module writes at order time. Publishes per-module/per-venue/per-chain scorecards to the DB; dashboard panel; feeds meta_controller's health_scorer as a cost-adjustment input.

**Data + infra.** Already have: all fill data in Postgres, pool_engine RPC for receipt/gas lookups, pnl_tracker. Need: a normalized `execution_events` schema and a one-time "intent capture" wiring pass in each module's executor. No new external APIs, no capital.

**Feasibility: HIGH. Build size: M (2–3 waves).** The hard part is not analytics — it is that every module logs trades in a different schema (legacy `trades`, `solana_trades`, futures tables, sniper positions). The normalization layer is 70% of the work and must be fail-soft.

**Key risks / why it might not work.**
- *Garbage-in*: if intent capture (quoted price at decision time) is retrofitted sloppily, TCA reports become confidently wrong — worse than nothing. Mitigate: only score trades that have a captured quote; report coverage %.
- *Organizational risk*: scorecards that nobody acts on. Mitigate: wire directly into meta_controller scoring so cost decay automatically pressures PAUSE decisions.
- No edge decay / competition risk — this is internal measurement.

**Safety posture.** Read-only forever. Never touches an order path. Advisory output only. Can ship straight to always-on once coverage >80%.

---

### 2. `treasury` — Wallet, gas, and inventory management module

**Thesis.** Per `core/allocation_guard.py`, the EVM wallet is shared by DEX/Arb/AI/Futures and the Solana wallet by Solana/Sniper/Copy. allocation_guard caps *exposure*, but nothing manages the *actual balances*: native gas top-ups (a Sniper exit failing because Arb burnt the ETH is a real live-flip failure mode), sweeping realized profits to a cold/treasury address, maintaining CEX float for Futures, and stablecoin vs native inventory ratios per chain. Every professional desk separates trading P&L from treasury operations; this bot currently has zero treasury function, which means the first profitable month will sit as hot-wallet risk and the first gas drought will be discovered by a failed emergency exit.

**Where it slots in.** New `modules/treasury/` subprocess. Phase 1: balance observer + reconciler (reads all wallets/CEX accounts via existing pool_engine + ccxt creds, writes `treasury_snapshots`, alerts on gas-below-floor / hot-wallet-above-ceiling / balance-vs-DB-position drift). Phase 2 (separately gated): automated actions — gas top-up from a designated funding address, profit sweep above a watermark, CEX deposit/withdraw to target float. Phase 3 (own gate, much later): cross-chain rebalancing via canonical bridges (CCTP for USDC) — deliberately NOT in scope for v1; bridge risk is its own decision.

**Data + infra.** Already have: RPC pool, ccxt, secrets_manager, alerting. Need: a funding-wallet key policy, withdrawal-address whitelists on CEXs (operator task), and hard-coded destination allowlists in config (never free-form addresses).

**Feasibility: HIGH (Phase 1), MEDIUM (Phase 2). Build size: S for Phase 1, M for Phase 2.**

**Key risks / why it might not work.**
- *This module moves principal, not edge.* A bug here is a direct loss with no stop-loss. Phase 2 must be allowlist-only destinations, per-action and per-day USD caps, two confirmations (DB flag + dashboard ack) for anything above a threshold.
- *Key concentration*: an automated sweeper holding a key that can move everything is a bigger attack surface than any trading module. Mitigate: sweep destination is cold/whitelisted only; module never holds a key that can send to arbitrary addresses.
- *Operational complexity creep*: CEX withdrawal APIs, memo/tag handling, chain congestion. Keep Phase 2 scope brutal: gas top-up + profit sweep, nothing else.

**Safety posture.** Phase 1 observe-only (no keys needed beyond read). Phase 2 behind `treasury_actions_enabled=false` DB flag AND `should_skip_live` AND allowlists AND caps; every action alerted to Telegram before and after. Killswitch freezes all transfers.

---

### 3. `sentinel` — Cross-module anomaly detection and auto-freeze advisor

**Thesis.** The fleet's tail risks are not owned by any module: stablecoin depeg (USDC 2023 would have shredded every USD-denominated assumption simultaneously), oracle-vs-venue price deviation, exchange/API outage mid-position, funding-rate spikes that turn carry positions toxic in one interval, correlated drawdown across modules (everything is crypto-beta), and silent module death (heartbeat present but trade flow anomalous — e.g., 100% entry rejection for 6 hours, which has happened twice in this repo's history: DEX vol/liq gate, Futures volume gate). A sentinel that watches *distributions* rather than single trades catches the class of failure that per-trade RiskManager checks structurally cannot.

**Where it slots in.** New `modules/sentinel/` subprocess. Consumes: DB heartbeats, trade-flow rates per module, pool_engine endpoint health, a small price board (stables basket, BTC/ETH/SOL across two independent sources). Emits: graded alerts (INFO/WARN/CRITICAL) via notification_engine, and — exactly like meta_controller's autopilot pattern — optionally writes `logs/.pause_<module>` files when a CRITICAL rule fires, never the killswitch, never trades. Rules and thresholds are DB-configured.

**Data + infra.** Almost everything exists: heartbeats, alerts.py, notification_engine, pool_engine, price sources via existing collectors. Need: one cheap second price source for cross-validation (Coinbase/Kraken public REST is free) and a rules table.

**Feasibility: HIGH. Build size: S–M.**

**Key risks / why it might not work.**
- *False positives at the worst time*: auto-pausing Futures during a vol spike can strand a position that needed managing. Rule design must distinguish "freeze new entries" (pause file — safe, exits still run per module design) from "flatten" (never automated in v1).
- *Alert fatigue*: a noisy sentinel gets ignored within two weeks and is then worse than nothing. Ship with few, high-precision rules; expand only with measured precision.
- *Overlap risk* with meta_controller (slow, performance-based) — keep the boundary explicit: meta_controller scores *performance over days*; sentinel detects *anomalies over minutes*. Document this in both CLAUDE.md files.

**Safety posture.** Advisory-first (alerts only). Pause-file autopilot behind a DB flag defaulting false, mirroring meta_controller's gating exactly. No order paths, no keys.

---

### 4. `market_data_warehouse` — Unified historical market-data store

**Thesis.** The collectors in `data/collectors/` fetch data for live decisions and largely discard it. Funding-rate history, candle history, DEX liquidity snapshots, and orderbook-top snapshots are the raw material for every future improvement: backtest_replay fidelity, advisor/AI model retraining, the regime_allocator's vol estimation, TCA benchmarks (arrival price), and post-mortems after every losing streak. Buying this data later costs real money (Kaiko/Amberdata are $1k+/mo) and some of it (DEX pool states, pump.fun launches the bot itself observed) is *unbuyable*. The bot is already touching this data every minute; the warehouse just stops throwing it away.

**Where it slots in.** New `modules/market_data_warehouse/` subprocess (or a thin collector-scheduler) writing to TimescaleDB hypertables (already in the stack per infra notes). Consumers: backtest_replay, advisor, ml retraining scripts, regime_allocator, execution_quality (arrival-price benchmarks).

**Data + infra.** Already have: collectors, Postgres/TimescaleDB, Redis. Need: retention/compression policy (Timescale native), a schema for candles/funding/liquidity snapshots, and disk monitoring. No capital, no new keys.

**Feasibility: HIGH. Build size: M.**

**Key risks / why it might not work.**
- *Data swamp*: storage without named consumers is pure cost. Mitigate: only ingest series with a committed consumer (start with funding rates + 1m candles for traded symbols + Solana launch metadata).
- *Disk/ops burden* on the single deployment host; compression and retention must ship in v1, not "later".
- Zero market risk; the failure mode is wasted effort, not lost money.

**Safety posture.** No order paths, no keys. Always-on once shipped.

---

### 5. `catalyst_calendar` — Token unlocks, listings, and macro-event feed

**Thesis.** Several modules trade through *scheduled* events blind: token unlocks (supply cliffs reliably pressure price into and at the event), exchange listing/delisting announcements (Sniper and DEX candidates pump/dump on these), and macro prints (CPI/FOMC minutes regularly produce the funding/vol spikes that hurt Futures carry). A calendar module is a pure data feed: it cannot lose money itself, and it gives every consumer a cheap "reduce risk into known event" input. The advisor module also directly benefits (it already produces multi-market advice).

**Where it slots in.** New lightweight `modules/catalyst_calendar/` subprocess (or a collector under the warehouse) writing an `events` table: `(asset, event_type, timestamp, magnitude, source, confidence)`. Consumers read it as a risk input — e.g., Futures declines new carry entries within N hours of a large unlock; Sniper tags listing-adjacent tokens.

**Data + infra.** Need external sources: unlock data (CryptoRank/TokenUnlocks — free tiers exist but are rate-limited; paid ~$50–200/mo), exchange announcement pages (scraping — fragile), macro calendar (free, e.g. econdb/FMP). Everything else exists.

**Feasibility: HIGH-MEDIUM. Build size: S.**

**Key risks / why it might not work.**
- *Priced-in edge*: unlock effects are heavily studied; the *alpha* is mostly gone, but the *risk-avoidance* value (don't be long into a 5% supply cliff) survives even full pricing-in. Sell it internally as risk input, not signal.
- *Scraper fragility*: announcement pages change; treat scraped events as low-confidence, API events as high.
- *Stale calendar is worse than none* — consumers must check feed freshness before trusting it.

**Safety posture.** Pure data feed; no orders, no keys. Consumers treat absence-of-data as "no constraint" (fail-open for the feed, their own gates still apply).

---

## TIER 2 — Medium-term / new P&L or execution surfaces

### 6. `options_vol` — Crypto options module (hedging-first, Deribit)

**Thesis.** The fleet's aggregate book is structurally long crypto-beta (DEX/Solana/Sniper/Copy are long-only or long-biased; Futures can short but rarely offsets the rest). Options are the only instrument that buys *convex* protection against the correlated-drawdown scenario, and BTC/ETH options on Deribit are the one deep, liquid crypto derivatives market the bot doesn't touch. Hedging-first sequencing: (1) tail-hedge overlay — when fleet net delta exceeds a threshold, buy short-dated OTM puts sized to cap fleet drawdown; (2) later, defined-risk premium selling (covered calls against treasury inventory, put spreads — never naked) when realized-vs-implied spread is favorable. The hedge leg alone justifies the module: it converts the fleet's worst-case from "correlated wipeout" to "known premium cost".

**Where it slots in.** New `modules/options_vol/` subprocess. Reads fleet net exposure from pnl_tracker/allocation_guard aggregates; trades only on Deribit (new venue integration via ccxt, which supports Deribit options). Reports greeks (delta/vega/theta) to the dashboard as first-class numbers.

**Data + infra.** Need: Deribit account + API keys, options chain data (free via Deribit API), a greeks/IV library (py_vollib-class, small), margin-model understanding (Deribit portfolio margin). Capital: a dedicated sub-account; hedging budget is a premium *expense* line (e.g., ≤50–100 bps of fleet NAV per month, DB-capped).

**Feasibility: MEDIUM. Build size: L** (new venue, new instrument math, new risk dimension).

**Key risks / why it might not work.**
- *Premium bleed*: systematic put-buying is negative-EV in calm regimes; if the cap isn't enforced, the hedge eats more than the tail it insures. The monthly premium budget must be a hard DB cap.
- *Premium selling is the classic bot-killer*: short vol pays daily and dies yearly. Hence sequencing — selling only after the hedging leg has run shadow + live cleanly, only defined-risk structures, and `reduceOnly`-equivalent discipline on every exit.
- *Liquidity cliff outside BTC/ETH*: do not touch alt options; spreads there are uncrossable for a bot this size.
- *Operational*: options margin + expiry/settlement handling is genuinely harder than perps; a missed expiry is an unhedged weekend.

**Safety posture.** Shadow-first (log intended hedges + marked-to-market greeks for ≥4 weeks). Live behind `options_live_enabled=false` + `should_skip_live` + RiskManager-style notional caps + hard monthly premium budget. Selling strategies behind a *separate* second flag. Isolated sub-account so a margin error cannot touch spot/perp capital.

---

### 7. `yield_treasury` — Idle-capital carry (LST + blue-chip lending)

**Thesis.** Once the `treasury` module (idea #2) exists, the bot will have a measurable idle balance: SOL waiting for sniper opportunities, USDC float between trades. Parking idle SOL in jitoSOL/mSOL (~7% APY, instantly usable as collateral, thin unstake friction) and idle USDC in Aave v3 / Kamino main markets (~3–8%) converts dead inventory into carry with no directional risk added. This is not a get-rich module — it is the desk discipline of never holding unremunerated cash. On a $50k idle float it's ~$2–3k/yr; the point is it scales linearly with the fleet and the build is small *because it rides on treasury's allowlist machinery*.

**Where it slots in.** Extension wing of `modules/treasury/` (same subprocess, separate flag) rather than a standalone module — it shares the balance observer, allowlists, and caps. Only deposits to a short hard-coded venue allowlist (jitoSOL, mSOL, Aave v3 USDC, Kamino USDC main).

**Data + infra.** Already have: Solana + EVM execution, Jupiter for LST swaps. Need: Aave/Kamino deposit/withdraw call wiring, an LST/stable depeg monitor (sentinel rule — idea #3 synergy).

**Feasibility: MEDIUM-HIGH. Build size: S–M** (conditional on treasury Phase 2 existing).

**Key risks / why it might not work.**
- *Smart-contract risk is the whole story*: yields of 3–8% do not compensate a protocol exploit on concentrated treasury funds. Caps: max % of idle float deployed (e.g., 50%), max per venue (e.g., 25%), blue-chip venues only, no looping/leverage, no points farming, no new-protocol chasing — ever.
- *Liquidity exactly when needed*: stress events that drain treasury liquidity are the same events when trading modules need capital. Keep an undeployed floor (gas + N days of trading float) that yield deployment can never touch.
- *LST depeg/discount* during validator or market stress; sentinel must watch the LST/SOL ratio and trigger unwind alerts.
- Opportunity cost is near zero, but so is the upside — this should never be prioritized above a measurement or safety module.

**Safety posture.** Gated live behind `treasury_yield_enabled=false` AND treasury Phase 2 flags; venue allowlist hard-coded; per-venue and total caps in DB; every deposit/withdraw alerted; killswitch triggers no new deposits + withdrawal of lent stables.

---

### 8. `execution_gateway` — MEV-aware order-flow router (shared service)

**Thesis.** Four modules (DEX, Arbitrage, Sniper EVM leg, Copy EVM leg) broadcast EVM transactions through public RPC today, which means every sizeable swap is sandwich bait — a silent 10–50 bps tax that TCA (idea #1) will likely make visible. A shared execution gateway routes EVM transactions through private order flow (Flashbots Protect / MEV Blocker style RPC) with public-RPC fallback, centralizes nonce management and gas/priority-fee policy, and unifies the Jito tip policy already used on Solana. Architecturally it is the execution sibling of pool_engine: pool_engine answers "which RPC do I read from?", the gateway answers "how do I *send* safely?". Centralizing send policy also closes a live-readiness gap: today each module re-implements broadcast logic, so every DRY_RUN/killswitch check at the send boundary is duplicated code that can drift.

**Where it slots in.** Not a trading subprocess — a shared library + thin service under `trading/executors/` + `config/` (gateway policy in DB), adopted module-by-module behind per-module flags. Optionally a small monitor subprocess reporting inclusion latency and estimated sandwich savings (joint output with TCA).

**Data + infra.** Need: private-relay RPC endpoints (free: Flashbots Protect RPC, MEV Blocker), pool_engine extension for "send-class" endpoints. Already have: Jito on Solana, nonce handling per module (to be consolidated carefully).

**Feasibility: MEDIUM. Build size: M** — the code is modest; the *migration* of four live modules onto it without regressions is the real cost.

**Key risks / why it might not work.**
- *Latency vs protection trade-off*: private relays add inclusion delay; for Sniper, speed IS the edge, so Sniper likely opts out for entries and uses the gateway only for exits. This must be per-module, per-direction policy, not a blanket switch.
- *Chain coverage*: protect-style RPC is mature on Ethereum mainnet, patchy on L2s/alt-L1s where the bot actually does much of its volume (Base, etc. — sequencer-ordered chains have different MEV profiles and less sandwich risk anyway). Honest sizing: the benefit may be concentrated on a minority of volume.
- *Migration risk*: touching every module's send path is exactly where new bugs enter pre-live. Adopt one module at a time, DEX first (lowest latency sensitivity), with TCA before/after comparison as the acceptance test.

**Safety posture.** Library-level: inherits each caller's DRY_RUN/killswitch gating, and adds a single choke-point assertion of `should_skip_live` at send time (defense in depth, not replacement). Per-module adoption flags default to legacy path.

---

### 9. `clmm_lp` — Concentrated-liquidity market-making (Uniswap v3 / Orca)

**Thesis.** The bot already has deep plumbing on the *taker* side of AMMs; CLMM LPing is the maker side: deploy ranges on high-fee-tier pools the bot already understands (its own traded universe), earn fees, actively re-range. Done well, fees on volatile-pair CLMM positions can yield 20–80% APR. Done naively, impermanent loss eats it all. The honest framing: an LP position is a *short-gamma, short-vol position paid in fees* — it is a real trading desk with inventory risk, not passive yield.

**Where it slots in.** New `modules/clmm_lp/` subprocess. Heavy synergy with existing code: pool state reading (DEX module), Orca detection (Copy module already decodes Orca/Meteora), price validation (Solana PriceValidator), and — critically — the Futures module could delta-hedge LP inventory (an internal cross-module hedge, which meta layers can see via allocation_guard).

**Data + infra.** Have: RPC, pool math partially, venues. Need: position NFT management (Uni v3) / Orca whirlpool position accounts, IL accounting in pnl_tracker terms (fees earned vs HODL benchmark — must be first-class or the module will look profitable while losing), re-range cost model (gas + crossing spread every rebalance).

**Feasibility: MEDIUM-LOW. Build size: L.**

**Key risks / why it might not work.**
- *IL in trending markets*: a one-way move converts the position into the depreciating asset at the worst average price. Without delta-hedging this is a levered short-vol bet; with hedging it becomes an operationally complex MM desk (hedge slippage + funding cost can exceed fee income).
- *Toxic flow*: informed flow (arb bots — including this bot's own Arbitrage module, ironically) picks off stale ranges; retail fee flow is concentrated in a few pools where professional LPs already compete with better re-range latency.
- *Measurement trap*: fee APR looks great while IL accrues unrealized. If the HODL-benchmark accounting isn't built first, the module will be approved on fake numbers. This is the #1 reason to defer it until TCA + warehouse exist.

**Safety posture.** Shadow-first with full IL-vs-HODL simulation on live pool data (warehouse-dependent) for ≥4 weeks. Live behind dedicated flag + small fixed inventory cap + sentinel depeg/deviation rules. No leverage, no exotic pools, max 2 pools in v1.

---

## TIER 3 — Ambitious / high-risk (honest moonshot assessment)

### 10. `param_tuner` — Bandit-based self-tuning of module knobs (shadow-only)

**Thesis.** This repo's history is a graveyard of mis-set thresholds discovered weeks late: the DEX vol/liq gate blocking 100% of candidates, the Futures volume gate blocking 100% of signals, the AI confidence threshold mismatched to its own signal distribution. All of these were *DB-configurable knobs with a measurable objective* — exactly the setting where a contextual bandit (not deep RL) earns its keep. A param_tuner module would, for an explicit whitelist of non-risk knobs (entry thresholds, scoring weights, hold-time params), maintain counterfactual estimates ("had threshold been X, N more trades would have fired with estimated P&L Y" — computable because rejected candidates are already logged) and *propose* changes. The realistic v1 win is not autonomous optimization — it is institutionalizing the "Wave-13 audit" as a continuous process instead of a quarterly heroic effort.

**Where it slots in.** New `modules/param_tuner/` subprocess, architecturally a sibling of meta_controller: reads trade + rejected-candidate logs, writes proposals to a `param_proposals` table, surfaces them on the dashboard for one-click operator apply. Autopilot (auto-applying within pre-approved bounds) is a v3 question at the earliest.

**Data + infra.** Need: rejected-candidate logging with features (partially exists in some modules, missing in others — a prerequisite wiring pass), warehouse (idea #4) for market context, careful experiment bookkeeping. No capital, no keys.

**Feasibility: LOW-MEDIUM. Build size: L** (plus prerequisite logging work).

**Key risks / why it might not work.**
- *Non-stationarity*: crypto regimes shift faster than per-knob sample sizes accumulate; a threshold "optimized" on three weeks of chop is mis-set for the breakout. Counterfactual P&L on rejected trades is also biased (no market impact, fill assumptions).
- *Reward hacking / confounding*: the tuner moving knobs while regime_allocator moves capital while meta_controller pauses modules = three feedback controllers on one plant. Without coordination this oscillates. Tuner must run proposals-only until the other two layers are stable in production.
- *The seductive failure*: it will "work" in shadow (counterfactuals always look good) and overfit live. Acceptance must be out-of-sample: proposals scored on what happened *after* they were made, not on the data that generated them.
- Hard rule consistent with desk policy: risk knobs (stop widths, leverage caps, loss limits) are permanently outside the whitelist. A tuner that can widen stops is a martingale generator with extra steps.

**Safety posture.** Proposal-only for its entire v1/v2 life. No order paths, no keys, no direct config writes. Any future autopilot bounded to pre-approved ranges per key, gated behind a default-false DB flag, with meta_controller-style decision audit rows.

### 11. `intent_solver` — CoW Protocol solver / UniswapX filler

**Verdict up front: documented to explain why NOT to build it now.** Solving/filling intent auctions (CoW, UniswapX) is the natural endgame of the bot's routing + inventory + MEV skills, and winners earn real, market-neutral spread. But the competitive reality is brutal: top solvers run colocated infrastructure, private market-maker inventory, sub-100ms quoting across every venue simultaneously, and operate at single-digit-bps margins where one mispriced fill erases a day. CoW additionally requires a staked bond (six figures) and a vetting process; UniswapX filling without exclusive order flow means competing on pure latency against firms whose entire business this is. The bot has none of these advantages and the deficit is structural, not a build-size problem.

**Feasibility: LOW. Build size: XL.** **Decision: park it.** Revisit only if (a) the execution_gateway + TCA stack demonstrates top-decile internal execution quality for 6+ months, and (b) a niche emerges where the bot has genuinely private edge (e.g., long-tail Solana routes via its Jupiter/pump.fun infrastructure). Listed so future ideation waves don't re-litigate it from scratch.

---

## PRIORITIZED TABLE

Impact = expected contribution to fleet survivability + net P&L. Effort = build + migration + operational burden. Risk = probability the module loses money, misleads decisions, or rots unused.

| # | Module | Type | Impact | Effort | Risk | Score (I×/E×R) | Verdict |
|---|--------|------|--------|--------|------|------------------|---------|
| 1 | `execution_quality` (TCA) | Measurement | HIGH | M | LOW | **Best** | BUILD NEXT |
| 2 | `treasury` (Phase 1 observe → Phase 2 act) | Capital ops | HIGH | S→M | LOW→MED | **Best** | BUILD NEXT |
| 3 | `sentinel` (anomaly/auto-freeze) | Safety | HIGH | S–M | LOW | **Best** | BUILD NEXT |
| 4 | `market_data_warehouse` | Data infra | MED-HIGH (compounding) | M | LOW | Strong | Start ingestion early (cheap), grow with consumers |
| 5 | `catalyst_calendar` | Data feed | MED | S | LOW | Strong | Quick win after Tier-1 core |
| 8 | `execution_gateway` (MEV-aware send) | Execution infra | MED-HIGH | M | MED (migration) | Good | After TCA quantifies the leak |
| 6 | `options_vol` (hedge-first) | New P&L/hedge | MED-HIGH | L | MED | Good | First *new market* to add; hedge leg only at first |
| 7 | `yield_treasury` (LST/lending) | Carry | LOW-MED | S–M | MED (SC risk) | Moderate | Only after treasury Phase 2 proven |
| 9 | `clmm_lp` | New P&L | MED | L | HIGH (IL, measurement trap) | Weak now | Defer until warehouse + TCA exist |
| 10 | `param_tuner` (bandit, proposal-only) | Meta | MED | L | HIGH (overfit, controller conflict) | Weak now | Defer until meta_controller + regime_allocator stable live |
| 11 | `intent_solver` | New P&L | HIGH (if it worked) | XL | VERY HIGH | Reject | PARKED — structural competitive deficit |

## TOP-3 RECOMMENDATION & SEQUENCING

1. **`execution_quality` (TCA)** — build first. The whole fleet is at the live-flip boundary with thin per-trade edges; nothing else proposed (gateway, options sizing, meta scoring, regime weights) can be sized honestly until quoted-vs-realized costs are measured. It is read-only, so it can ship while live flips proceed, and its scorecards are the acceptance test for idea #8 later.
2. **`treasury` (Phase 1 observe-only)** — build in parallel (different surface, no contention with TCA). Shared-wallet gas starvation and unswept hot-wallet profits are the two most likely *non-market* loss events of the first live month; the observe/alert phase is small and de-risks live operations immediately. Phase 2 (gas top-up + sweep) follows once Phase 1 reconciliation runs clean for 2 weeks.
3. **`sentinel`** — build third, immediately after. It is small, reuses meta_controller's pause-file pattern, and covers the correlated/tail failure class (depeg, oracle break, silent module death) that no per-module gate can see. Shipping it before the fleet is fully live means the fleet's first stress event is watched.

Sequencing rationale: all three are measurement/survival infrastructure with near-zero market risk, deliberately ahead of any new P&L surface — the fleet already has nine ways to make (or lose) money and approximately zero ways to know its true costs, manage its shared capital, or catch a fleet-wide anomaly. The warehouse (#4) should start *passive ingestion* (funding + candles) during the same period because data has lead time; the first new *market* (options hedge leg, #6) enters only after Tier-1 is live and TCA has baselined execution. CLMM, param_tuner, and the intent solver are explicitly deferred with revisit conditions stated above.

