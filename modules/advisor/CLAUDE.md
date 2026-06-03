# ADVISOR Module

## What it does
Standalone financial advisor — **ADVICE-ONLY, no trade execution**. Generates Long/Short/Neutral signals for CRYPTO, US equities (NASDAQ/NYSE), BIST (Borsa Istanbul), FX including metals (XAU/XAG), and Turkish Midas funds. Operator executes manually on Midas exchange. Completely isolated from all trading modules.

## Advice quality: horizon-aware levels + real confidence (migration 067)
All analyzers now derive entry/target/stop and confidence from one shared, transparent
module: `modules/advisor/core/analyzers/levels.py`. This fixes two real defects observed
in live advice: (1) short/mid/long showed IDENTICAL entry/target/stop for a symbol, and
(2) confidence was frozen (looked stuck at ~43%) across symbols and horizons.

**Root cause.** Each analyzer previously used FIXED percentage bands (entry +/-0.5%,
target +/-5%, stop +/-3%) with NO horizon input, and confidence was `abs(vote)/3` (+0.1
vol boost) — a tiny discrete ladder that barely moved.

**Horizon-aware levels.** `levels.horizon_levels(signals, direction, horizon, config)`:
```
vol_unit            = clamp(atr_pct OR bb_half_width/2/close OR nav_return_stdev,
                            levels_vol_floor, levels_vol_ceiling)
entry_band_frac     = vol_unit * levels_entry_mult_<h>
target_distance_frac= vol_unit * levels_target_mult_<h> * levels_target_rr
stop_distance_frac  = vol_unit * levels_stop_mult_<h>
```
Multipliers grow short < mid < long, so target/stop distances widen with horizon, and a
more volatile asset gets wider bands than a quiet one. Volatility unit source per market:
crypto/us_equities/fx/bist use ATR-14 as a fraction of close (Bollinger-width proxy if
ATR is NaN); Midas funds (NAV-only, no OHLCV) use the daily NAV-return stdev.

**Real confidence.** `levels.signal_confidence(signals, horizon, config, providers_disagree)`:
```
agreement = abs(signed vote sum) / n_votes          # alignment strength [0,1]
magnitude = 0.7*RSI_extremity + 0.3*BB_extremity     # reading extremity [0,1]
volume    = clamp(vol_ratio - 1, 0, 1)               # volume confirmation [0,1]
base      = 0.55*agreement + 0.30*magnitude + 0.15*volume
conf      = base * horizon_factor (short 1.00 / mid 0.92 / long 0.85)
            - levels_conf_disagree_penalty (if dual-advice providers disagree)
conf      = clamp(conf, levels_conf_floor, levels_conf_ceiling)   # default [0.05, 0.95]
```
Every term is continuous, so confidence varies per symbol AND per horizon — it is no
longer a constant. **Dual-advice DISAGREE lowers confidence:** `AdviceEngine._overlay_dual_advice`
subtracts `levels_conf_disagree_penalty` (default 0.20) from `result.confidence` when the
anthropic vs openai directions disagree, before the `min_confidence` risk gate runs
(pre-disagree value stored in `extra['confidence_pre_disagree']`).

**Tunables** (migration `067_advisor_advice_quality.sql`, all `config_type='advisor_config'`,
operator-tunable, surface in dashboard Settings): `levels_entry_mult_{short,mid,long}`,
`levels_target_mult_{short,mid,long}`, `levels_stop_mult_{short,mid,long}`,
`levels_target_rr`, `levels_vol_floor`, `levels_vol_ceiling`, `levels_conf_floor`,
`levels_conf_ceiling`, `levels_conf_disagree_penalty`.

**HONESTY.** These are HEURISTIC levels and a heuristic confidence — NOT predictions or
guarantees. The defaults are a starting calibration (the LONG target multiplier of 20x
vol_unit can be aggressive for high-vol assets; lower it if targets look unrealistic).
ADVICE-ONLY: no orders are placed. Self-check: `python -m modules.advisor.core.analyzers.levels`.

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
Wave-24: the orchestrator's subprocess stdout/stderr capture is now ALSO pinned
to `logs/advisor/` (`main.py` previously derived it from the module display name
"Financial Advisor" → a separate `logs/financial_advisor/` folder). One folder
now. Telegram: `core/telegram_notifier.py` runs a `getMe` self-test at startup
and logs send failures (401/403/404 + body) to `advisor_errors.log`; the most
common silent-bot cause is the 403 "bot can't initiate conversation" — the
operator must press **Start** on the advisor bot DM once.

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
Seeded as cards by migration `065_advisor_secure_credentials.sql`. `main_advisor.py`
injects the ANTHROPIC/OPENAI/FX/BIST keys from `secrets_manager` (encrypted DB) into
the advisor config at startup; the advisor Telegram bot resolves its token/chat via
`secrets_manager` directly. All keep a `.env` (`os.getenv`) fallback.

| Secret name | Used for |
|---|---|
| `ADVISOR_ANTHROPIC_API_KEY` | LLM rationale generation (all markets) |
| `ADVISOR_OPENAI_API_KEY` | OpenAI dual-advice second opinion (enable in Advisor Settings) |
| `ADVISOR_TELEGRAM_BOT_TOKEN` | Separate advisor Telegram bot (NOT shared bot) |
| `ADVISOR_TELEGRAM_CHAT_ID` | Advisor alert target chat |
| `ADVISOR_BIST_API_KEY` | Matriks / paid BIST data (optional, if using paid source) |
| `ADVISOR_FX_ALPHAVANTAGE_KEY` | Alpha Vantage FX data (optional, free tier) |

`ADVISOR_KRONOS_WEIGHTS_PATH` is **not** a Secure Credential — it is a filesystem
path (not a secret) read via `os.getenv` only, so set it in `.env` (see
`docs/ADVISOR_SETUP_AND_USAGE.md` §5), not the credentials panel.

## Kronos integration (MB-19 fail-soft pattern)
Kronos is the NeoQuasar foundation model for K-line forecasting (MIT license). Three variants: Kronos-mini (4.1M, CPU-OK), Kronos-small (24.7M), Kronos-base (102.3M, GPU recommended). If `ADVISOR_KRONOS_WEIGHTS_PATH` is not set or the directory is missing, `predict()` returns `None` and the advice cycle continues without Kronos signal. Operator downloads weights separately:
```
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('NeoQuasar/Kronos-mini', local_dir='/data/kronos/Kronos-mini')"
```
Then set `ADVISOR_KRONOS_WEIGHTS_PATH=/data/kronos/Kronos-mini` in env and `advisor_kronos_enabled=true` in advisor_config.

## KAP engine (Borsa İstanbul disclosures) — Phase 1 (wired end-to-end)
KAP disclosure ingestion + classification + alerts + advice-overlay + dashboard, all gated by
`advisor_kap_enabled` (advisor_config, default `false`). ADVICE-ONLY; emits a documented
base_polarity PRIOR only — no market-impact score (impact stats accumulate over months in `kap_returns`).
- **Ingestion**: `core/kap/kap_listener.py` (`KapListener.run()`), polite rate-limited.
- **Classification worker**: `core/kap/classifier_worker.py` (`KapClassifierWorker.run()`) — third
  `_kap_tasks` entry in `main_advisor.py`. Every `advisor_kap_classify_interval_s` (default 60) pulls
  `kap_store.get_unclassified`, runs `classifier.classify` (rule + LLM fallback, fail-soft), persists
  via `kap_store.store_classification`.
- **Telegram alerts**: `AdvisorTelegramBot.send_kap_alert(...)` — fires for non-NEUTRAL disclosures
  with confidence ≥ `advisor_kap_alert_min_confidence` (0.5), capped at
  `advisor_kap_alert_max_per_cycle` (10) per cycle. Polarity-prior-only disclaimer on every alert.
- **BIST advice overlay**: `AdviceEngine._overlay_kap_context()` — BIST-only, after composite signal.
  Attaches recent classified disclosures (`advisor_kap_lookback_days`, default 7) to
  `AdviceResult.extra['kap_context']` as CONTEXT. Does NOT mutate action or numeric confidence.
- **Dashboard**: `/advisor/kap` page (`dashboard/templates/advisor_kap.html`), API
  `/api/advisor/kap/disclosures` in `monitoring/enhanced_dashboard.py`.
- **Config keys** seeded by **migration `064_kap_phase1_wiring.sql`**: `advisor_kap_classify_interval_s`,
  `advisor_kap_alert_min_confidence`, `advisor_kap_alert_max_per_cycle`, `advisor_kap_lookback_days`.
- **Schema fix**: `kap_store.store_classification` now writes the actual migration-063 columns
  (`base_polarity`, `params`, `classifier_stage`, `confidence`, `raw_subject`, `extra`) keyed on the
  BIGINT `disclosure_id` (= `kap_disclosures.id`); `get_unclassified` dedup join corrected to `c.disclosure_id = d.id`.

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
