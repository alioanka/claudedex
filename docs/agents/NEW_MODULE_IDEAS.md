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
