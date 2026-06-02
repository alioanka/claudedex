# Financial Advisor Module — Architecture (Wave-20)

**Status**: Scaffold delivered. Specialist agents fill in engines against this contract.
**Date**: 2026-06-02
**Branch**: claude/friendly-ramanujan-nMWNv

---

## 1. Purpose and constraints

The advisor module is an **advice-only** research surface. It generates directional signals (Long/Short/Neutral) for five market segments and delivers them to the operator via Telegram and a dashboard page. The operator executes manually on Midas exchange.

Hard constraints (enforced at scaffold level):
- No import of any trading module engine (DEX, Futures, Solana, etc.).
- No wallet keys, exchange API keys with write permission, or order-placement code.
- No call to `pool_engine`, `risk_manager.validate_trade`, or the shared `TelegramBotController`.
- Default `ADVISOR_MODULE_ENABLED=false` — operator opts in explicitly.

---

## 2. Component diagram

```
 .env / Secure Credentials
  ADVISOR_ANTHROPIC_API_KEY
  ADVISOR_TELEGRAM_BOT_TOKEN  (separate bot, not the trading bot)
  ADVISOR_TELEGRAM_CHAT_ID
  ADVISOR_BIST_API_KEY         (optional, Matriks paid)
  ADVISOR_FX_ALPHAVANTAGE_KEY  (optional)
  ADVISOR_KRONOS_WEIGHTS_PATH  (path to downloaded HF weights dir)

 main.py
  └── ModuleProcess("advisor", script="modules/advisor/main_advisor.py",
                    enabled_env_var="ADVISOR_MODULE_ENABLED")   # default FALSE

 modules/advisor/main_advisor.py
  └── AdvisorApplication
        ├── AdvisorConfigManager          [config/advisor_config.py]
        │     Reads config_settings WHERE config_type='advisor_config'
        │
        ├── AdviceEngine                  [core/advice_engine.py]
        │     Runs every run_interval_minutes (default 60)
        │     For each (market, symbol, horizon):
        │       analyzer.analyze() → AdviceResult
        │       kronos.predict(klines_df) → float or None  (overlay)
        │       risk.should_publish() → bool
        │       persist to advisor_advice table
        │       if sim_enabled: portfolio.open_sim_position()
        │       telegram.send_advice()
        │     Mark-to-market open sim positions
        │     ML daily tick (stub — Wave-22)
        │
        ├── Market Analyzers              [core/analyzers/]
        │     BaseAnalyzer (ABC)
        │       ├── CryptoAnalyzer        ccxt public REST, no key
        │       ├── USEquitiesAnalyzer    yfinance, no key  ← REFERENCE IMPL
        │       ├── BISTAnalyzer          yfinance (.IS) degraded or Matriks paid
        │       ├── FXAnalyzer            yfinance default or Alpha Vantage
        │       └── MidasFundsAnalyzer    tefas.gov.tr scrape or manual
        │
        ├── KronosForecaster              [core/kronos_forecaster.py]
        │     Fail-soft: returns None when weights not present
        │     predict(df_klines: pd.DataFrame) → float or None
        │
        ├── AdvisorRiskEngine             [core/risk_engine.py]
        │     should_publish(result, open_sim_count) → (bool, reason)
        │     Gates: confidence floor, symbol blocklist, sim position cap
        │
        ├── AdvisorPortfolioEngine        [core/portfolio_engine.py]
        │     open_sim_position(AdviceResult) → sim_id
        │     mark_to_market(sim_id, price) → SimPosition
        │     close_sim_position(sim_id, price, reason) → SimPosition
        │     list_open_sims() → List[SimPosition]
        │
        ├── AdvisorTelegramBot            [core/telegram_notifier.py]
        │     Separate bot token (ADVISOR_TELEGRAM_BOT_TOKEN)
        │     send_advice(AdviceResult) → bool   [stub formatting Wave-21]
        │
        └── AdvisorHealthServer           [main_advisor.py]
              GET /health  → liveness + last cycle
              GET /status  → full diagnostics
              Port: 8086 (ADVISOR_HEALTH_PORT)

 DB tables (migration 058)
  ├── advisor_advice          — published advice events
  ├── advisor_sim_positions   — dry-run position tracking
  ├── advisor_portfolio       — operator-reported holdings
  └── config_settings         — config_type='advisor_config' rows
```

---

## 3. Per-market analyzer interface

Every market analyzer implements `BaseAnalyzer` (defined in `core/base_analyzer.py`):

```python
class BaseAnalyzer(ABC):
    market: Market                      # class-level constant

    async def analyze(self, symbol: str, horizon: Horizon) -> AdviceResult:
        """Never raises — return ERROR result on exception."""

    def data_source_status(self) -> DataSourceStatus:
        """AVAILABLE | NOT_CONFIGURED | DEGRADED | ERROR"""
```

**AdviceResult fields** (see `core/models.py`):
| Field | Type | Description |
|---|---|---|
| market | Market | CRYPTO / US_EQUITIES / BIST / FX / MIDAS_FUNDS |
| symbol | str | Ticker in market-native format |
| horizon | Horizon | short / mid / long |
| direction | Direction | long / short / neutral |
| entry_low / entry_high | float? | Suggested entry range |
| target_price | float? | Primary price target |
| stop_price | float? | Suggested stop-loss |
| confidence | float | [0.0, 1.0] |
| rationale | str | LLM-generated explanation (or rule-based fallback) |
| model_id | str | Anthropic model ID used |
| kronos_signal | float? | Kronos directional signal (None if unavailable) |
| data_source_status | DataSourceStatus | Feed availability |
| sim_enabled | bool | Whether to open a sim position |
| sim_amount_usd | float | Notional for sim |
| extra | dict | Arbitrary metadata; `klines_df` key holds pandas DF for Kronos |

---

## 4. Advice schema (DB: advisor_advice)

```sql
CREATE TABLE advisor_advice (
    id                  BIGSERIAL PRIMARY KEY,
    market              VARCHAR(32),    -- market enum value
    symbol              VARCHAR(64),
    horizon             VARCHAR(16),    -- short/mid/long
    direction           VARCHAR(16),    -- long/short/neutral
    entry_low           NUMERIC(20,8),
    entry_high          NUMERIC(20,8),
    target_price        NUMERIC(20,8),
    stop_price          NUMERIC(20,8),
    confidence          NUMERIC(5,4),
    rationale           TEXT,
    model_id            VARCHAR(128),
    kronos_signal       NUMERIC(12,6),  -- NULL if Kronos not loaded
    data_source_status  VARCHAR(32),
    sim_enabled         BOOLEAN,
    sim_amount_usd      NUMERIC(12,2),
    extra               JSONB,
    operator_notes      TEXT,           -- operator fills post-advice
    created_at          TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ
);
```

---

## 5. Simulate / backtest flow

```
AdviceResult (sim_enabled=true, sim_amount_usd=1000)
  │
  └── AdvisorPortfolioEngine.open_sim_position(result)
        INSERT INTO advisor_sim_positions
          (symbol, market, direction, horizon,
           entry_price, target_price, stop_price,
           notional_usd, status='open', opened_at=NOW())

  Next cycle (every run_interval_minutes):
  └── AdviceEngine._mark_to_market_open_sims()
        For each open sim:
          current_price = analyzer.last_price[symbol]  (cached from last analyze)
          AdvisorPortfolioEngine.mark_to_market(sim_id, current_price)
            UPDATE advisor_sim_positions SET current_price, pnl_pct, pnl_usd

  Dashboard / operator triggers close:
  └── AdvisorPortfolioEngine.close_sim_position(sim_id, exit_price, reason)
        UPDATE advisor_sim_positions SET status='closed', closed_at, pnl_pct, pnl_usd

  Backtest query (Wave-21 dashboard):
    SELECT symbol, horizon, direction,
           AVG(pnl_pct) FILTER (WHERE direction='long') AS avg_long_pnl,
           AVG(pnl_pct) FILTER (WHERE direction='short') AS avg_short_pnl,
           COUNT(*) FILTER (WHERE pnl_pct > 0) AS win_count,
           COUNT(*) AS total
    FROM advisor_sim_positions WHERE status='closed'
    GROUP BY symbol, horizon, direction;
```

---

## 6. ML self-improvement loop (scaffold, Wave-22)

Skeleton in `AdviceEngine._ml_learning_tick()`. Called once per advice cycle; is a no-op in Wave-20.

Wave-22 ML agent fills in:
1. **Outcome ingestion**: query `advisor_sim_positions WHERE status='closed'` for positions closed since last training run. Join to `advisor_advice` for the original signals.
2. **Feature store**: extract (market, symbol, horizon, sma_signal, rsi, bb_signal, vol_ratio, kronos_signal, confidence) at advice time → target = (pnl_pct > 0).
3. **Model refinement**: lightweight classifier (e.g. XGBoost or LightGBM) trained on accumulated outcomes. Persisted as `advisor_ml_model.pkl`.
4. **Confidence recalibration**: replace `_compute_confidence()` heuristic with model probability output.
5. **Toggle**: `advisor_ml_enabled=true` in advisor_config to activate; default false.

---

## 7. Kronos integration plan

### Feasibility verdict (Wave-20)

| Dimension | Detail |
|---|---|
| License | MIT — no restriction |
| Architecture | Decoder-only autoregressive Transformer; two-stage: tokenizer + transformer pre-training on 45+ exchange K-lines |
| Variants | mini (4.1M), small (24.7M), base (102.3M) — all open. large (499.2M) NOT open-sourced |
| Weights location | HuggingFace: `NeoQuasar/Kronos-mini`, `NeoQuasar/Kronos-small`, `NeoQuasar/Kronos-base` |
| Weights sizes | mini ~50 MB, small ~100 MB, base ~400 MB |
| Dependencies | Python 3.10+, PyTorch >= 2.0, transformers >= 4.38, pyqlib, pandas, numpy |
| GPU | mini + small: CPU-feasible (1-3s / batch); base: CPU works, GPU (8 GB VRAM) recommended; large: GPU only, not available |
| Inference API | `KronosPredictor(model_path).predict(df_ohlcv)` → float signal; `predict_batch(list_of_dfs)` for batches |
| Input format | DataFrame with columns: `open, high, low, close` (+ optional `volume`, `amount`). DateTime index, UTC, sorted ascending. Minimum 30 rows. |
| Output | Float; positive = bullish, negative = bearish. Magnitude varies by model. |

**Recommendation**: Start with Kronos-mini. CPU inference, 50 MB weights, MIT license. Upgrade to Kronos-small if backtesting shows meaningful accuracy improvement.

### Integration steps (Wave-21, quant agent)

1. Add to `requirements.txt`: `torch>=2.0 transformers>=4.38 pyqlib huggingface_hub`
2. Create `scripts/download_kronos_weights.py`:
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download("NeoQuasar/Kronos-mini", local_dir="/data/kronos/Kronos-mini")
   ```
3. Implement `_load_model(weights_dir, variant, device)` in `kronos_forecaster.py`:
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   # or from kronos import KronosTokenizer, KronosModel if custom classes
   tokenizer = KronosTokenizer.from_pretrained(str(weights_dir))
   model = KronosModel.from_pretrained(str(weights_dir)).to(device)
   model.eval()
   return model, tokenizer
   ```
4. Implement `_run_inference(model, tokenizer, df_klines)` — tokenize OHLCV, forward pass, extract directional scalar.
5. Verify klines_df is passed in `result.extra["klines_df"]` by each analyzer.
6. Set `ADVISOR_KRONOS_WEIGHTS_PATH` and `advisor_kronos_enabled=true` in advisor_config.

---

## 8. Data source matrix

| Market | Source | Free? | Status in Wave-20 | Required keys |
|---|---|---|---|---|
| CRYPTO | ccxt public REST | YES | Stub (not_configured) — Wave-21 | None |
| US EQUITIES | yfinance | YES | Reference impl (available) | None |
| BIST (degraded) | yfinance `.IS` suffix | YES | Stub — Wave-21 | None |
| BIST (production) | Matriks / Rasyonet API | NO (paid) | Stub — Wave-21 | `ADVISOR_BIST_API_KEY` in Secure Credentials |
| FX / Metals | yfinance | YES | Stub — Wave-21 | None |
| FX / Metals | Alpha Vantage | Free tier (25/day) | Stub — Wave-21 | `ADVISOR_FX_ALPHAVANTAGE_KEY` |
| FX / Metals | Stooq.com | YES (no SLA) | Stub — Wave-21 | None |
| MIDAS FUNDS | Tefas.gov.tr scrape | YES (fragile) | Stub — Wave-21 | None (CSRF session) |
| MIDAS FUNDS | Manual operator entry | N/A | Usable via portfolio table | None |

### Keys the operator must provision

| Secret name | Where to add | Purpose |
|---|---|---|
| `ADVISOR_ANTHROPIC_API_KEY` | Secure Credentials | LLM rationale for all markets |
| `ADVISOR_TELEGRAM_BOT_TOKEN` | Secure Credentials | Advisor Telegram bot (SEPARATE from trading bot) |
| `ADVISOR_TELEGRAM_CHAT_ID` | Secure Credentials | Target advisor chat |
| `ADVISOR_BIST_API_KEY` | Secure Credentials | Matriks paid BIST data (optional) |
| `ADVISOR_FX_ALPHAVANTAGE_KEY` | Secure Credentials | Alpha Vantage FX (optional, free tier) |
| `ADVISOR_KRONOS_WEIGHTS_PATH` | .env or Secure Credentials | Path to Kronos weights directory |

---

## 9. Separate Telegram design

The advisor uses a **completely separate** Telegram bot from the trading modules:

| Aspect | Trading modules | Advisor module |
|---|---|---|
| Bot token source | `TELEGRAM_BOT_TOKEN` (shared) | `ADVISOR_TELEGRAM_BOT_TOKEN` (separate) |
| Chat ID | `TELEGRAM_CHAT_ID` | `ADVISOR_TELEGRAM_CHAT_ID` |
| Controller class | `monitoring.telegram_bot.TelegramBotController` | `modules.advisor.core.telegram_notifier.AdvisorTelegramBot` |
| Import in advisor | NEVER | Own class only |
| Purpose | Trade alerts, emergency close | Advice notifications, sim performance |

**Why separate**: operator may want to route financial advice to a different channel/group (e.g. a personal investment channel) vs. the trading module alerts (bot operations channel).

---

## 10. Dashboard page plan (Wave-21)

Pages to add to the main dashboard (not a separate dashboard server):

| URL | What it shows |
|---|---|
| `/advisor` | Overview: last advice per market, analyzer health badges, sim P&L summary |
| `/advisor/advice` | Full advice history table with filters (market, horizon, direction, symbol) |
| `/advisor/sims` | Open sim positions with live P&L, close buttons; closed sims with outcome |
| `/advisor/portfolio` | Operator-reported holdings; add/edit/remove form |
| `/advisor/settings` | All advisor_config keys (editable, mirrors pattern from /futures/settings) |
| `/advisor/kronos` | Kronos health: loaded/variant/device, weights path, last signal |

API endpoints (advisor health server on 8086):
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness + last cycle time |
| `/status` | GET | Full diagnostics |

Main dashboard proxy endpoints (added to enhanced_dashboard.py — Wave-21):
| Endpoint | Method | Description |
|---|---|---|
| `/api/advisor/advice` | GET | Paginated advice history |
| `/api/advisor/sims` | GET | Open + recent sim positions |
| `/api/advisor/portfolio` | GET/POST | Operator holdings CRUD |
| `/api/advisor/settings` | GET/POST | advisor_config read/write |

---

## 11. Port assignments (existing + advisor)

| Port | Module |
|---|---|
| 8080 | Dashboard |
| 8081 | Futures Trading (FUTURES_HEALTH_PORT) |
| 8082 | Solana Trading (SOLANA_HEALTH_PORT) |
| 8085 | DEX Trading (DEX_HEALTH_PORT) |
| **8086** | **Financial Advisor (ADVISOR_HEALTH_PORT)** |

---

## 12. Prioritised follow-up specialist tasks

**Immediate (Wave-21) — specialists can start in parallel against this scaffold:**

| Priority | Task | Owner | Files |
|---|---|---|---|
| P0 | Wire ccxt OHLCV in CryptoAnalyzer | quant-algo | `core/analyzers/crypto.py` |
| P0 | Complete LLM rationale in USEquitiesAnalyzer (Anthropic call is stubbed) | quant-algo | `core/analyzers/us_equities.py::_llm_rationale` |
| P1 | Implement FXAnalyzer (yfinance path) | quant-algo | `core/analyzers/fx.py` |
| P1 | Implement BISTAnalyzer (yfinance degraded path) | backend | `core/analyzers/bist.py` |
| P1 | Tefas scrape for MidasFundsAnalyzer | backend | `core/analyzers/midas_funds.py` |
| P1 | Telegram rich HTML formatting | tg-specialist | `core/telegram_notifier.py::_format_advice_message` |
| P2 | Dashboard pages (/advisor/*) | backend | `modules/dashboard/enhanced_dashboard.py` + templates |
| P2 | Kronos weights download script | backend | `scripts/download_kronos_weights.py` |
| P3 | Kronos _load_model + _run_inference | quant-algo | `core/kronos_forecaster.py` |

**Wave-22:**

| Priority | Task | Owner |
|---|---|---|
| P1 | ML daily learning loop (outcome ingestion + model training) | ml-agent |
| P2 | Backtest replay against historical advisor_sim_positions | quant-algo |
| P3 | Kronos-small upgrade + GPU support | quant-algo |

---

## 13. Non-goals (explicitly out of scope)

- Auto-execution on Midas (the Midas API is not integrated; advice is read-only).
- Position sizing recommendations (operator decides allocation).
- Portfolio risk management (beyond the portfolio context display).
- Real-time streaming (advice runs on a schedule, not tick-by-tick).
