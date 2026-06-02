# ClaudeDex — Financial Advisor Module: Setup & Usage Guide

The Advisor is a **standalone, advice-only** module: it analyzes markets and tells you what it
thinks (Long/Short, buy/sell zones, short/mid/long horizons) — it **never places a trade**.
You execute manually on Midas. It has its own engines, dashboard, Telegram bot, and DB tables,
fully isolated from the trading modules. Default **OFF** until you enable it.

---

## 1. What it does
- **5 markets**, each with its own analyzer: US equities, Crypto, FX/metals (incl. gold/silver),
  BIST (Borsa İstanbul), Midas/TEFAS Turkish funds.
- **Per market × horizon** (short / mid / long) advice with entry zone, target, stop, rationale.
- **Multi-layer signal engine** (Wave-22): instead of "RSI says buy", every advice carries a
  transparent 7-layer score — Trend / Momentum / Volatility / Volume / Market-Regime / Risk /
  AI-forecast → a composite `action` (STRONG_BUY…STRONG_SELL) + `confidence` (layer-agreement).
- **Dual LLM rationale** (Wave-22): Anthropic (Opus 4.8 default) plus optional OpenAI second
  opinion; `consensus` mode flags when the two models DISAGREE on direction.
- **Kronos** K-line forecaster (optional, operator-installed) feeds the AI-forecast layer.
- **Simulate/backtest**: mark any advice as a DRY-RUN position ("$1000 TSLA, 6 months") and the
  module marks it to market over the horizon so you can measure advice quality before risking real money.
- **ML learning loop**: trains on closed sim outcomes once ≥50 exist (honest refuse-to-predict until then).
- **Own Telegram bot**: daily digest (not per-tick spam), separate token/chat.

---

## 2. Enable the module (minimum to get advice)
After deploying the branch (see DEPLOY_RUNBOOK_WAVES_18-21.md; remember **`--build`** — new pip deps):

```bash
# 1. .env on the VPS
ADVISOR_MODULE_ENABLED=true

# 2. Secure Credentials (Dashboard -> Secure Credentials), minimum:
ADVISOR_ANTHROPIC_API_KEY      # you can reuse your working Anthropic key

# 3. restart
docker compose down && docker compose up -d   # (--build only if not already built this deploy)

# 4. verify
docker compose ps                              # trading-bot Up (healthy)
curl -s localhost:8086/health                  # advisor health server
```
Then open **`/advisor/dashboard`**. US + Crypto + FX produce advice immediately on free data.

### Bump the LLM to the latest model (one-time, IMPORTANT)
Migration 058 was seeded with an older default and uses `ON CONFLICT DO NOTHING`, so the model
won't auto-update on an existing DB. Set it once:
```bash
PG() { docker exec trading-postgres psql -U "$(docker exec trading-postgres cat /run/secrets/db_user)" -d tradingbot -P pager=off "$@"; }
PG -c "UPDATE config_settings SET value='claude-opus-4-8' WHERE config_type='advisor_config' AND key='advisor_anthropic_model';"
```
(Or just set it in `/advisor/settings`. When a newer Claude ships, change this one value.)

---

## 3. Data-source status (what's real vs stubbed)
The dashboard shows a per-market badge: **AVAILABLE / DEGRADED / NOT_CONFIGURED**. Honest by design.

| Market | Source after `--build` | Status | To upgrade |
|---|---|---|---|
| US equities | yfinance (free EOD) | AVAILABLE | Alpaca/Polygon only if you later want real-time |
| Crypto | ccxt public (free) | AVAILABLE | — |
| FX / metals | yfinance (free) | AVAILABLE | `ADVISOR_FX_ALPHAVANTAGE_KEY` + `advisor_fx_data_source='alphavantage'`; TCMB official FX for USD/TRY |
| BIST | borsapy (TradingView-backed, ~15min delay) | AVAILABLE | `ADVISOR_BIST_API_KEY` + Matriks (paid, needs wiring) |
| Midas/TEFAS funds | tefas-crawler (2026 API) → tefasfon → manual | AVAILABLE / DEGRADED | manual NAV entry via `/advisor/portfolio` if upstream down |

If borsapy / tefas-crawler upstreams are unreachable, the analyzer fails **soft** to DEGRADED/manual —
the module never crashes, the badge just tells you the feed is limited.

---

## 4. Optional power-ups (operator steps)

### OpenAI second opinion (dual-advice)
```bash
# add ADVISOR_OPENAI_API_KEY in Secure Credentials, then:
PG -c "UPDATE config_settings SET value='consensus' WHERE config_type='advisor_config' AND key='advisor_dual_advice_mode';"  # off|both|consensus
PG -c "UPDATE config_settings SET value='gpt-4o' WHERE config_type='advisor_config' AND key='advisor_openai_model';"
```
- `both` = show Anthropic + OpenAI rationale side-by-side.
- `consensus` = agree → higher confidence; disagree → `providers_disagree` flag + lower confidence (a genuinely useful "the models don't agree, be careful" signal).

### Kronos K-line forecaster
```bash
pip install 'torch>=2.0' 'transformers>=4.38' huggingface_hub      # in the container/image
python scripts/download_kronos_weights.py --variant mini --dir /data/kronos
# set ADVISOR_KRONOS_WEIGHTS_PATH=/data/kronos/Kronos-mini in .env
PG -c "UPDATE config_settings SET value='true' WHERE config_type='advisor_config' AND key='advisor_kronos_enabled';"
# restart advisor
```
Kronos-mini is CPU-feasible (~50MB). Until installed, advice still flows (AI-forecast layer shows "unavailable").

### ML learning loop
1. Let sims accumulate (advice with simulate-on) until ≥50 are closed.
2. `python scripts/retrain_advisor_ml.py` (check metrics in output).
3. `PG -c "UPDATE config_settings SET value='true' WHERE config_type='advisor_config' AND key='advisor_ml_enabled';"` → restart.
   It refuses to predict on sparse data — by design, it won't fabricate confidence.

### vectorbt backtesting (heavy, optional)
Uncomment `vectorbt>=0.26.0` in `requirements.txt`, rebuild. Enables `vbt_backtest.run_strategy()`
(full date-range equity curve / Sharpe / drawdown). The lightweight `portfolio_engine.backtest()`
(single-advice replay) always works without it.

### Advisor Telegram bot (separate from trading bot)
```bash
# create a NEW bot via @BotFather, then in Secure Credentials:
ADVISOR_TELEGRAM_BOT_TOKEN
ADVISOR_TELEGRAM_CHAT_ID
# enable in /advisor/settings; sends a daily digest (de-duped, not per-tick).
```

---

## 5. The dashboard (`/advisor/*`)
- **/advisor/dashboard** — per-market advice cards, color-coded (green long / red short / amber hold),
  confidence bars, the 7-layer signal scores, regime label, data-source badge. Top band: portfolio
  value, open sims, best/worst.
- **/advisor/advice** — full advice history, filter by market/horizon/direction, expandable rationale
  (both LLMs if dual-advice on).
- **/advisor/simulations** — DRY-RUN sim tracker with live mark-to-market PnL.
- **/advisor/portfolio** — manually enter what you hold on Midas (context for advice; places no orders).
- **/advisor/settings** — every `advisor_config` key (model IDs, dual-advice mode, signal weights,
  sim defaults, ML flags, Kronos path, Telegram).

---

## 6. How to read an advice (and trust it)
1. Look at **action + confidence** first (composite of all 7 layers).
2. Check the **regime** — a STRONG_BUY in a PANIC regime is auto-downweighted; treat with caution.
3. Read **both rationales** if dual-advice is on; a `providers_disagree` flag means lower conviction.
4. **Don't act on it blind.** Turn on **simulate** for advice you're tempted by, and let the sim
   mark-to-market over the horizon. After a few weeks of closed sims you'll know whether the advice
   is actually any good — THEN consider executing on Midas with real capital. Same discipline that
   separated the winning trading modules (Solana/DEX) from the losers.

---

## 7. Honest limitations
- It's an **opinion engine**, not a profit oracle. LLM + technicals + Kronos = structured opinions.
- BIST / TEFAS data is delayed/unofficial on the free path — fine for mid/long-term advice, not for
  intraday precision. Paid vendors (Matriks/dxFeed) are the upgrade if you need production-grade.
- The ML loop and Kronos are dormant until you do the operator steps + accumulate data.
- Deferred (not built — see docs/agents/wave22/advisor_enhancement_plan.md): Qlib, Chronos-2,
  Moirai-2, FinGPT, borsa-mcp (scaffold/evaluate later). Skipped: FinRL, Lumibot, TradingView
  scraping, Alpaca/Polygon (advice-only / ToS / real-time-not-needed reasons documented).
