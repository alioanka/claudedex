# ADVISOR Module

## What it does
Standalone financial advisor — **ADVICE-ONLY, no trade execution**. Generates Long/Short/Neutral signals for CRYPTO, US equities (NASDAQ/NYSE), BIST (Borsa Istanbul), FX including metals (XAU/XAG), and Turkish Midas funds. Operator executes manually on Midas exchange. Completely isolated from all trading modules.

## Entry point
`modules/advisor/main_advisor.py` — launched as a subprocess by `main.py` when `ADVISOR_MODULE_ENABLED=true` (default `false`). Health server on port 8086 (env override: `ADVISOR_HEALTH_PORT`).

## Key config (DB-backed, config_type='advisor_config')
| Key | Type | Default | What it does |
|---|---|---|---|
| `advisor_anthropic_model` | string | `claude-opus-4-5` | Anthropic model for LLM rationale. Loud WARN on 404; never silent fallback. |
| `enabled_markets` | string | `crypto,us_equities` | Comma-sep active markets: `crypto\|us_equities\|bist\|fx\|midas_funds` |
| `enabled_horizons` | string | `short,mid,long` | Time horizons to generate advice for |
| `run_interval_minutes` | int | 60 | Advice cycle cadence |
| `min_confidence` | float | 0.35 | Minimum confidence to publish advice |
| `max_sim_positions` | int | 20 | Open sim position cap |
| `sim_default_enabled` | bool | false | Auto-open sim position per advice |
| `sim_default_amount_usd` | float | 1000.0 | Default sim notional |
| `watchlist_crypto` | string | BTC/USDT,ETH/USDT,SOL/USDT | Crypto pairs (ccxt format) |
| `watchlist_us_equities` | string | AAPL,MSFT,NVDA,TSLA,AMZN | US tickers (yfinance format) |
| `watchlist_bist` | string | (empty) | BIST tickers — requires `advisor_bist_data_source` |
| `watchlist_fx` | string | EURUSD=X,XAUUSD=X | FX pairs + metals (yfinance format) |
| `watchlist_midas_funds` | string | (empty) | Tefas FONKODU codes |
| `advisor_crypto_exchange` | string | binance | ccxt exchange for crypto OHLCV |
| `advisor_bist_data_source` | string | (empty) | `yfinance` (degraded) or `matriks` (paid) |
| `advisor_fx_data_source` | string | yfinance | `yfinance` or `alphavantage` |
| `advisor_midas_data_source` | string | (empty) | `tefas_scrape` or `manual` |
| `advisor_telegram_enabled` | bool | true | Advisor Telegram bot master toggle |
| `advisor_telegram_bot_token` | string | (empty) | SEPARATE bot token (Secure Credentials) |
| `advisor_telegram_chat_id` | string | (empty) | Target chat ID |
| `advisor_kronos_enabled` | bool | false | Enable Kronos K-line forecast overlay |
| `advisor_kronos_variant` | string | Kronos-mini | mini (4.1M) / small (24.7M) / base (102.3M) |
| `advisor_kronos_device` | string | cpu | `cpu` or `cuda` |
| `advisor_ml_enabled` | bool | false | ML self-improvement loop (Wave-22) |

## Kill switch
- **Global**: `logs/.killswitch` — polled by BaseModule via `core.dry_run.start_killswitch_poller`.
- **Per-module**: `logs/.pause_advisor` — written by dashboard pause/resume.
- **Effect**: advice cycle stops; no partial cycle published.

## Logs
`logs/advisor/` — `advisor.log` (all), `advisor_errors.log` (WARNING+).

## Data source matrix

| Market | Source | Free? | Key required |
|---|---|---|---|
| CRYPTO | ccxt (public REST) | YES (no key) | None |
| US EQUITIES | yfinance (Yahoo Finance) | YES (no key) | None |
| BIST (degraded) | yfinance `.IS` suffix | YES | None (partial coverage) |
| BIST (full) | Matriks API | NO (paid) | `ADVISOR_BIST_API_KEY` in Secure Credentials |
| FX / Metals | yfinance (default) | YES | None |
| FX / Metals | Alpha Vantage | NO (free tier 25 req/day) | `ADVISOR_FX_ALPHAVANTAGE_KEY` in Secure Credentials |
| MIDAS FUNDS | tefas.gov.tr scrape | YES (fragile) | None (CSRF scraping, undocumented API) |
| MIDAS FUNDS | Manual operator entry | N/A | None (operator enters NAV in dashboard) |

## Secrets the operator must configure (Secure Credentials panel)
| Secret name | Used for |
|---|---|
| `ADVISOR_ANTHROPIC_API_KEY` | LLM rationale generation |
| `ADVISOR_TELEGRAM_BOT_TOKEN` | Separate advisor Telegram bot (NOT shared bot) |
| `ADVISOR_TELEGRAM_CHAT_ID` | Advisor alert target chat |
| `ADVISOR_BIST_API_KEY` | Matriks / paid BIST data (optional, if using paid source) |
| `ADVISOR_FX_ALPHAVANTAGE_KEY` | Alpha Vantage FX data (optional, free tier) |
| `ADVISOR_KRONOS_WEIGHTS_PATH` | Path to downloaded Kronos weights directory |

## Kronos integration (MB-19 fail-soft pattern)
Kronos is the NeoQuasar foundation model for K-line forecasting (MIT license). Three variants: Kronos-mini (4.1M, CPU-OK), Kronos-small (24.7M), Kronos-base (102.3M, GPU recommended). If `ADVISOR_KRONOS_WEIGHTS_PATH` is not set or the directory is missing, `predict()` returns `None` and the advice cycle continues without Kronos signal. Operator downloads weights separately:
```
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('NeoQuasar/Kronos-mini', local_dir='/data/kronos/Kronos-mini')"
```
Then set `ADVISOR_KRONOS_WEIGHTS_PATH=/data/kronos/Kronos-mini` in env and `advisor_kronos_enabled=true` in advisor_config.

## DB tables (migration 058)
- `advisor_advice` — every published advice event (market, symbol, horizon, direction, entry range, target, stop, confidence, rationale, model_id, kronos_signal)
- `advisor_sim_positions` — dry-run position tracking (seeded from advice; marked-to-market daily)
- `advisor_portfolio` — operator-reported holdings (manual entry via dashboard)
- `config_settings` rows with `config_type='advisor_config'` — all knobs above

## Architecture summary
```
main_advisor.py
  └── AdvisorApplication
        ├── AdvisorConfigManager         (loads advisor_config from DB)
        ├── AdviceEngine                 (orchestrates cycle)
        │     ├── CryptoAnalyzer         (ccxt — FREE)
        │     ├── USEquitiesAnalyzer     (yfinance — FREE, reference impl)
        │     ├── BISTAnalyzer           (stub — needs data source config)
        │     ├── FXAnalyzer             (yfinance default — FREE)
        │     └── MidasFundsAnalyzer     (stub — tefas scrape or manual)
        ├── KronosForecaster             (fail-soft — weights not auto-downloaded)
        ├── AdvisorRiskEngine            (advice gate: confidence, blocklist, sim cap)
        ├── AdvisorPortfolioEngine       (sim positions + operator holdings)
        └── AdvisorTelegramBot           (SEPARATE bot, STUB formatting Wave-21)
```

## Isolation guarantee
This module does NOT import or modify:
- `core/risk_manager.py` (trading risk manager)
- `config/pool_engine.py` (RPC pool)
- `monitoring/telegram_bot.py` (shared trading-module Telegram bot)
- Any trading module's engines or executors

It DOES reuse (read-only):
- `security/secrets_manager.py` — for resolving advisor's own API keys
- `core/dry_run.py` — for killswitch polling
- `config_settings` table — `config_type='advisor_config'` rows only

## Wave status
- Wave-20 (current): scaffold + interfaces + US equities reference analyzer + migration.
- Wave-21: wire ccxt crypto analyzer, BIST data, FX data, Midas tefas scrape, Kronos inference, Telegram rich formatting.
- Wave-22: ML daily learning loop.

## See also
- Architecture doc: `docs/agents/wave20/advisor_architecture.md`
- Migration: `migrations/058_advisor_module.sql`
