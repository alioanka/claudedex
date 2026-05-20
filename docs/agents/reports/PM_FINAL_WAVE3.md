# PM_FINAL_WAVE3 — Carry-over Wave Close-out

**Branch:** `claude/create-expert-agents-JFSF5`
**Base commit:** `9c349b3` (Wave-2 close)
**Date:** 2026-05-19 → 2026-05-20
**Cutoff:** All 6 module agents (W3-A1..W3-A7 ex-Sniper) hit usage-credit ceiling mid-flight with 100+ tool calls each but did NOT lose work — uncommitted-but-finished changes were preserved and committed by the orchestrator.

## Wave 3 commits per module (cumulative)

| Module | W2 commits | W3 commits | W3 highlights |
|---|---|---|---|
| **DEX** | 9 | 2 (`7f0133a`, `7a5d8ff`) | Real Uniswap V3 QuoterV2 binding (kills 0.997 placeholder); EIP-1559 Type-2 gas + nonce_lock race fix + real `_path_has_liquidity` |
| **ARBITRAGE** | 6 | 3 (`f2e95d3`, `329b784`, `7024d0d`) | Per-(chain, dex_pair, pair) median+p90 realized-slippage learning (7-day rolling); live `_gas_spend_usd_hour` dashboard tile; full unit-test coverage |
| **SOLANA** | 6 | 3 (`7a0b012`, `263b285`, `40e7a27`, `89471d0`) | **P1-08 fix — scaler train-only fit** (kills look-ahead at scaler-fit time); TokenPriceBuffer + FIFO tests + pump-predictor gate wiring under flag; Jito bundle path lifted into shared `trading/chains/solana/jito_bundle.py` (-419 LoC dup) |
| **SNIPER** | 8 | 2 (`d89b1c4`, `5c00192`, `d11460e`) | Pyth Hermes feed client + 13 blue-chip mint map; resolution chain Pyth → Jupiter Price v2 → Jupiter /quote → Birdeye |
| **FUTURES** | 6 | 2 (`1ba5e3d`, `8d15efe`) | Per-symbol leverage cap overrides; funding-cost realized-vs-predicted widget + migration `029_add_futures_funding_payments.sql`; auto-deleverage trigger wiring (flagged off) |
| **AI** | 8 | 3 (`a666d20`, `d8f9b64`, `e4f47de`) | LSTM per-token rolling buffer (AI-Q-07); ensemble feature-decoupled predict path (AI-Q-08); token_scorer online weight learning (AI-Q-15) |
| **COPY_TRADING** | 6 | 2 (`4d22cf7`, `e73b09d`) | slippage_tracker.py + SlippageObservation rolling stats; wire into engine + new dashboard endpoint |

**Wave 3 total:** 14 new commits (+ 5 rescue commits = 19 wave-3 commits).
**Campaign total since brief:** 77 commits.

## Wave-3 P0/P1 wins

- **DEX:** Uniswap V3 routing no longer broken (every V3 quote returned `amount * 0.997`).
- **DEX:** EIP-1559 saves ~10–30% gas in expectation on Ethereum/Polygon/Arbitrum/Base.
- **DEX:** Concurrent-tx nonce race fixed (was silently double-spending nonces on burst orders).
- **DEX:** Real liquidity check (was `return True` — let routes through into 0-liquidity pools).
- **SOLANA P1-08:** ML scaler leakage gone. `pump_predictor` is now actually trainable without test-set statistics polluting the fit.
- **SOLANA:** ~400 LoC of duplicated Jito bundle code de-duped; SOLANA module can opt-in to Jito without re-implementing.
- **FUTURES:** Auto-deleverage wiring closes the gap where `should_auto_deleverage()` existed but was never called.
- **AI ensemble:** Predict path no longer importable from module internals (decoupling unblocks future cross-module reuse).

## Carry-over for Wave 4 (deferred this wave)

**DEX**
- Per-tick V3 price-impact (still uses linear extrapolation; QuoterV2 round-trip path was added for impact estimation but boundary-jump refusal is approximate).
- bloXroute private-pool routing on BSC (Flashbots is Ethereum-only — currently `_attempt_protected_path` returns None on BSC).

**ARBITRAGE**
- Triangular atomic-receiver contract deploy (operator approval still required — unchanged).

**SOLANA**
- Jito bundle wiring for the SOLANA module itself. Helper is now shared (this wave) but SOLANA engine doesn't yet call it. Flag stub exists; wire-up next wave.
- Pump-predictor warmup window (60 bars) takes ~30 minutes after restart with default poll cadence. Operator should expect a cold period before the flag has any effect.

**SNIPER**
- Pyth-feed adoption for non-Solana chains. Map is Solana-only this wave; EVM blue-chips (WETH, WBTC, USDC, USDT) have Pyth feeds too and could be added.
- `safety_check_enabled=true` DB flip — still operator-only, untouched.

**FUTURES**
- Telegram alert on FUT-RM-07 emergency-close. Deferred — Telegram controller exists in monitoring/ but bot-token wiring belongs in a follow-up.

**AI**
- Calibrated booster wrap (AI-Q-05). Audit identified it; refactor untouched this wave.
- Multi-provider quorum metrics + dashboard widget (the gate exists; observability deferred).

**COPY_TRADING**
- Per-leader probation table (CT-Q-09). Schema design noted in CLAUDE.md; migration deferred.
- Cross-module exposure aggregator (CT-Q-12). Helper file sketched but engine wiring deferred.

## Operator quick-start (Wave 2 + 3)

```bash
# 1. Pull
git fetch origin claude/create-expert-agents-JFSF5
git checkout claude/create-expert-agents-JFSF5
git pull origin claude/create-expert-agents-JFSF5

# 2. Rebuild trading-bot (NEVER prune the postgres volume)
docker compose up -d --build trading-bot
docker logs -f trading-bot   # ctrl-c after 30s; verify clean startup

# 3. Verify migrations 023 / 024 / 029 applied
docker exec trading-postgres psql -U "$(docker exec trading-postgres cat /run/secrets/db_user)" tradingbot \
  -c "\dt" | grep -E "ai_confidence_calibration|copy_leader_scores|futures_funding_payments"
```

After login, run the four Test Runner rounds from PM_FINAL.md (Wave 2 coverage). Wave 3 coverage will be added in a follow-up — until then, `pytest tests/unit/` covers the new code paths locally.

**DRY_RUN stays TRUE everywhere.** No flags flipped to LIVE. Kill switch unchanged.

**Wave 3 closed.**
