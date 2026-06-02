# ClaudeDex — Financial Advisor Module: Complete Setup & Usage Guide

The Advisor is a **standalone, advice-only** module: it analyzes markets and tells you what it
thinks (Long/Short, buy/sell zones, short/mid/long horizons) — it **never places a trade**.
You execute manually on Midas. It has its own engines, dashboard, Telegram bot, KAP event
engine, and DB tables, fully isolated from the trading modules. Default **OFF** until you enable it.

> **Two golden rules**
> 1. **`docker compose up --build`** (not plain restart) after pulling — `requirements.txt` carries
>    the advisor's pip deps (yfinance, anthropic, openai, borsapy, tefas-crawler, lightgbm…). The
>    rebuild installs them automatically. You do **not** `pip install` anything by hand for the base feature set.
> 2. **Two places to configure things:**
>    - **SECRETS** (API keys, tokens) → Dashboard → **Secure Credentials** panel (encrypted in DB), or `.env`.
>    - **SETTINGS** (models, modes, toggles, weights) → Dashboard → **Advisor → Settings** (`/advisor/settings`),
>      or SQL into `config_settings` (config_type=`advisor_config`).

---

## 0. TL;DR — minimum to get advice in 3 steps
```bash
# 1. deploy (on the VPS)
cd ~/claudedex && git pull origin claude/friendly-ramanujan-nMWNv
docker compose down && docker compose up --build -d
docker exec trading-bot python scripts/migrate_database.py    # applies advisor migrations 058,060,061,062,063

# 2. .env -> enable the module, then restart
#    ADVISOR_MODULE_ENABLED=true
docker compose down && docker compose up -d

# 3. Secure Credentials -> add ADVISOR_ANTHROPIC_API_KEY  (reuse your working Anthropic key)
#    then open  http://<host>:8080/advisor/dashboard
```
US equities + Crypto + FX produce advice immediately on free data. Everything else (OpenAI 2nd
opinion, BIST/funds real data, Kronos, ML, KAP, Telegram) is optional and covered below.

---

## 1. SECRETS — what to put in Secure Credentials (Dashboard → Secure Credentials)
Add only the ones for features you want. All resolve: Secure-Credentials/DB first, then `.env` fallback.

| Secret | Required for | Notes |
|---|---|---|
| `ADVISOR_ANTHROPIC_API_KEY` | **Core advice rationale** (all markets) | Reuse your existing Anthropic key. Without it → rule-based rationale only. |
| `ADVISOR_OPENAI_API_KEY` | **OpenAI 2nd opinion** (dual-advice) | ← *this is the one you couldn't find.* Set it here, then turn on dual-advice in Settings (§3). |
| `ADVISOR_TELEGRAM_BOT_TOKEN` | Advisor Telegram bot | A **NEW** bot from @BotFather, separate from the trading bot. |
| `ADVISOR_TELEGRAM_CHAT_ID` | Advisor Telegram bot | Your chat/channel id for advisor messages. |
| `ADVISOR_FX_ALPHAVANTAGE_KEY` | Real-time FX upgrade (optional) | Free tier 25 req/day; only if you set `advisor_fx_data_source='alphavantage'`. |
| `ADVISOR_BIST_API_KEY` | Paid BIST (Matriks) upgrade (optional) | Production-grade BIST; Matriks wiring is a later task. Free borsapy works without it. |

`.env` flags (not secrets, set in `.env`):
| Flag | Default | Meaning |
|---|---|---|
| `ADVISOR_MODULE_ENABLED` | `false` | Master on/off for the whole module. Set `true` to run it. |
| `ADVISOR_HEALTH_PORT` | `8086` | Advisor health server port. |
| `ADVISOR_KRONOS_WEIGHTS_PATH` | (unset) | Path to downloaded Kronos weights (see §5). |

---

## 2. SETTINGS — every `advisor_config` key (Dashboard → Advisor → Settings, or SQL)
Set helper for SQL:
```bash
PG() { docker exec trading-postgres psql -U "$(docker exec trading-postgres cat /run/secrets/db_user)" -d tradingbot -P pager=off "$@"; }
# generic: PG -c "UPDATE config_settings SET value='<v>' WHERE config_type='advisor_config' AND key='<k>';"
```

| Key | Default | What it controls |
|---|---|---|
| `advisor_anthropic_model` | `claude-opus-4-8` | Anthropic model (always-latest; change when newer ships). |
| `advisor_openai_model` | `gpt-4o` | OpenAI model for 2nd opinion. |
| `advisor_dual_advice_mode` | `off` | `off` / `both` / `consensus` — see §3. |
| `advisor_crypto_exchange` | `binance` | ccxt exchange for crypto data. |
| `advisor_fx_data_source` | `yfinance` | `yfinance` (free) or `alphavantage` (needs key). |
| `advisor_bist_data_source` | `` (auto) | `borsapy` / `yfinance` / `matriks`. Empty = auto (borsapy first). |
| `advisor_midas_data_source` | `` (auto) | `tefas_crawler` / `tefasfon` / `scrape` / `manual`. Empty = auto. |
| `signal_weight_trend` | `0.30` | Composite signal weighting (§6). |
| `signal_weight_momentum` | `0.25` | " |
| `signal_weight_volume` | `0.20` | " |
| `signal_weight_volatility` | `0.15` | " |
| `signal_weight_regime` | `0.10` | " |
| `advisor_kronos_enabled` | `false` | Turn on Kronos forecaster (after §5). |
| `advisor_kronos_variant` | `Kronos-mini` | mini (CPU) / small / base (GPU). |
| `advisor_kronos_device` | `cpu` | `cpu` or `cuda`. |
| `advisor_ml_enabled` | `false` | Turn on ML confidence model (after §7). |
| `advisor_ml_min_samples` | `50` | Min closed sims before ML predicts (honest gate). |
| `advisor_ml_retrain_days` | `7` | ML retrain cadence. |
| `advisor_ml_blend_weight` | `0.6` | How much ML adjusts confidence. |
| `advisor_telegram_enabled` | `false` | Turn on advisor Telegram bot (after secrets in §1). |
| `advisor_kap_enabled` | `false` | Turn on KAP event engine (§8). |
| `advisor_kap_poll_interval_s` | `60` | KAP disclosure poll frequency. |
| `kap_return_windows` | `1,3,5,10,30` | Forward-return days the accumulator computes. |
| `kap_classifier_enabled` | `false` | Turn on KAP event classification (§8). |
| `kap_classifier_llm_fallback` | `true` | Use LLM for events the rules can't classify. |

---

## 3. OpenAI 2nd opinion (dual-advice) — full steps
1. Secure Credentials → add `ADVISOR_OPENAI_API_KEY`.
2. Settings (or SQL):
```bash
PG -c "UPDATE config_settings SET value='consensus' WHERE config_type='advisor_config' AND key='advisor_dual_advice_mode';"
PG -c "UPDATE config_settings SET value='gpt-4o' WHERE config_type='advisor_config' AND key='advisor_openai_model';"
```
- `off` = Anthropic only.
- `both` = show Anthropic **and** OpenAI rationale side-by-side on each advice card.
- `consensus` = both models asked; **agree → higher confidence; disagree → `providers_disagree` flag + lower confidence.** (Recommended — the disagreement signal is genuinely useful.)
3. Fail-soft: if the OpenAI key is missing or a call fails, it silently falls back to Anthropic-only. Restart the advisor after changing the key.

---

## 4. Market data sources (what's real vs needs a key)
Dashboard shows a per-market badge: **AVAILABLE / DEGRADED / NOT_CONFIGURED** (honest by design).

| Market | After `--build` | Status | Upgrade |
|---|---|---|---|
| US equities | yfinance (free EOD) | AVAILABLE | Alpaca/Polygon only if you want real-time |
| Crypto | ccxt public | AVAILABLE | — |
| FX / metals | yfinance | AVAILABLE | `ADVISOR_FX_ALPHAVANTAGE_KEY` + set `advisor_fx_data_source='alphavantage'` |
| BIST | borsapy (~15min delay) | AVAILABLE | `ADVISOR_BIST_API_KEY` (Matriks, paid) |
| Midas/TEFAS funds | tefas-crawler → tefasfon → manual | AVAILABLE/DEGRADED | manual NAV entry via `/advisor/portfolio` if upstream down |

borsapy / tefas-crawler install automatically on `--build`. If an upstream is unreachable the
analyzer fails **soft** to DEGRADED/manual — the module never crashes; the badge tells you.

---

## 5. Kronos K-line forecaster (optional, operator step)
Kronos needs model weights (not auto-downloaded — they're large). One-time:
```bash
# inside the container (or rebuild image with these in requirements):
pip install 'torch>=2.0' 'transformers>=4.38' huggingface_hub
python scripts/download_kronos_weights.py --variant mini --dir /data/kronos
#   -> prints the path; put it in .env:  ADVISOR_KRONOS_WEIGHTS_PATH=/data/kronos/Kronos-mini
PG -c "UPDATE config_settings SET value='true' WHERE config_type='advisor_config' AND key='advisor_kronos_enabled';"
docker compose restart trading-bot
```
Kronos-mini is CPU-feasible (~50MB). Until enabled, advice still flows (AI-forecast layer = "unavailable").

---

## 6. The multi-layer signal engine (automatic — no setup)
Every advice already carries a transparent 7-layer score (no config needed):
Trend / Momentum / Volatility / Volume / Market-Regime / Risk / AI-forecast → composite
`action` (STRONG_BUY…STRONG_SELL) + `confidence`. Tune the layer weights via the `signal_weight_*`
keys (§2) if you want. **Confidence = layer agreement, not a profit prediction.**

---

## 7. ML learning loop (optional, needs data first)
```bash
# 1. let sims accumulate (advice with simulate-on) until >= 50 are CLOSED
# 2. train:
python scripts/retrain_advisor_ml.py        # prints metrics; --dry-run to preview counts
# 3. enable:
PG -c "UPDATE config_settings SET value='true' WHERE config_type='advisor_config' AND key='advisor_ml_enabled';"
docker compose restart trading-bot
```
It refuses to predict below `advisor_ml_min_samples` (50) — by design, no fabricated confidence on sparse data.

---

## 8. KAP Event Engine (BIST corporate-event intelligence) — Phase 1
Listens to Borsa İstanbul KAP disclosures, classifies them (bonus issue / rights issue / dividend /
tender win / SPK investigation / etc., 24 types — bedelsiz vs bedelli correctly distinguished),
and starts accumulating forward returns so that *over months* it builds real event-impact statistics.
```bash
# enable the engine + classifier
PG -c "UPDATE config_settings SET value='true' WHERE config_type='advisor_config' AND key='advisor_kap_enabled';"
PG -c "UPDATE config_settings SET value='true' WHERE config_type='advisor_config' AND key='kap_classifier_enabled';"
docker compose restart trading-bot

# OPTIONAL but recommended — backfill historical disclosures NOW so data accrues from day 1:
python scripts/crawl_kap_history.py --tickers THYAO,AKBNK,EREGL,HEDEF --since 2024-01-01
```
**Honest expectation (important):** Phase 1 gives you real-time **event alerts + classification**
("HEDEF: BONUS_ISSUE 300%, neutral-to-positive prior — watch volume/regime"). It does **NOT** yet
give "+4.8% / 74% confidence" impact scores — those need months of accumulated forward-return data.
The engine starts collecting that the moment you enable it.

---

## 9. Advisor Telegram bot (separate from the trading bot)
1. @BotFather → create a NEW bot → token.
2. Secure Credentials → `ADVISOR_TELEGRAM_BOT_TOKEN` + `ADVISOR_TELEGRAM_CHAT_ID`.
3. `PG -c "UPDATE config_settings SET value='true' WHERE config_type='advisor_config' AND key='advisor_telegram_enabled';"`
4. Sends a **daily digest** (per-market advice, de-duped, mobile-friendly) — not per-tick spam. Restart after enabling.

---

## 10. The dashboard (`/advisor/*`)
- **/advisor/dashboard** — per-market advice cards (color-coded), 7-layer scores, regime, data-source badge, KAP alerts.
- **/advisor/advice** — history, filter by market/horizon/direction, expandable rationale (both LLMs if dual-advice on).
- **/advisor/simulations** — DRY-RUN sim tracker with live mark-to-market PnL.
- **/advisor/portfolio** — manually enter Midas holdings (context only; places no orders).
- **/advisor/settings** — every `advisor_config` key with a Guide tab.

---

## 11. How to read advice & trust it
1. **action + confidence** first (composite of all layers).
2. Check **regime** — a STRONG_BUY in PANIC is auto-downweighted.
3. Read **both rationales** if dual-advice on; `providers_disagree` = lower conviction.
4. **Don't act blind.** Turn on **simulate** for advice you like; let it mark-to-market over the
   horizon. After weeks of closed sims you'll know if the advice is good — *then* commit real Midas capital.

---

## 12. What installs automatically vs what's manual
**Automatic on `docker compose up --build` + migrate:** the whole module, all 5 analyzers, signal
engine, dual-advice code, KAP engine, dashboard, Telegram code, all DB tables/config defaults, and
pip deps (yfinance/anthropic/openai/borsapy/tefas-crawler/lightgbm). You just flip flags + add keys.
**Manual operator steps (optional power-ups only):** Kronos weights download (§5), ML retrain after
≥50 sims (§7), KAP history backfill (§8), and any paid-data keys.

---

## 13. Honest limitations
- It's an **opinion engine**, not a profit oracle. Measure with simulate before trusting.
- BIST/TEFAS free data is delayed/unofficial — fine for mid/long advice, not intraday precision.
- KAP impact statistics need months to become meaningful; Phase 1 = alerts + classification only.
- Deferred (documented in `docs/agents/wave22/advisor_enhancement_plan.md`): Qlib, Chronos-2,
  Moirai-2, FinGPT, borsa-mcp. Skipped: FinRL, Lumibot, TradingView scraping, Alpaca/Polygon.
</content>
