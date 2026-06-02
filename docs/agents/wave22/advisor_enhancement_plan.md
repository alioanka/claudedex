# Wave-22 Advisor Enhancement Plan
**Date**: 2026-06-02
**Branch**: claude/friendly-ramanujan-nMWNv
**Owner**: PM (architect)
**Status**: ACTIVE — specialists implement against this contract.

---

## 1. Triage Table

Every suggestion from the external ChatGPT review is classified BUILD-NOW, SCAFFOLD/EVALUATE, or SKIP.

### Data Sources

| Suggestion | Verdict | Reason |
|---|---|---|
| **borsapy** (BIST OHLCV library) | BUILD-NOW | Pure-Python BIST data library, MIT license, no key required. Fills the exact gap in BISTAnalyzer where yfinance (.IS) coverage is incomplete. Drop-in replacement for the degraded yfinance path. Small, well-scoped. |
| **tefas-crawler + tefasfon** (TEFAS funds) | BUILD-NOW | Both are lightweight Python packages wrapping the tefas.gov.tr API we already scrape manually. Replacing hand-rolled CSRF scraping with a maintained library reduces fragility. MidasFundsAnalyzer benefit is immediate and concrete. |
| **yfinance .IS** (BIST fallback) | BUILD-NOW | Already scaffolded in BISTAnalyzer. Wire the degraded path completely in Wave-22. No new dependency. |
| **dxFeed / Cbonds / Matriks paid** | SKIP | Paid APIs requiring vendor contracts. Matriks stub already exists; activation is an operator action (obtain key), not a build task. Cbonds is bond-specific, not in scope. dxFeed is institutional, over-scoped. |
| **borsa-mcp** (MCP server: BIST/US/TEFAS/crypto/FX/KAP/TCMB) | SCAFFOLD/EVALUATE | Wraps many sources we already cover. Could unify data adapters. BUT: MCP transport introduces new architectural layer; current analyzers are working; evaluate only after all five data sources stabilise. Target: Wave-24. |
| **Alpaca / Polygon** (US live) | SKIP | US equities covered by yfinance (free). Alpaca/Polygon add real-time streaming and order capabilities. Streaming is a non-goal; order capabilities violate advice-only constraint. |
| **tvdatafeed / TradingView scraping** | SKIP | Violates TradingView ToS. Fragile screen-scraping, no SLA, legally risky. |
| **TCMB** (Turkish Central Bank FX API) | BUILD-NOW | Free official REST API (evds.tcmb.gov.tr). Provides official USD/TRY and EUR/TRY rates. Useful context signal for BIST and FX analyzers. Single-endpoint integration, minimal effort. |
| **Alpha Vantage** (FX) | BUILD-NOW | Already stubbed in FXAnalyzer (advisor_fx_data_source='alphavantage' path exists). Wire the actual HTTP call. Key resolution already in place. 1-2 hours of quant work. |
| **TwelveData** (FX / equities) | SCAFFOLD/EVALUATE | Strong free tier (800 req/day), covers FX+equities+crypto. Good yfinance rate-limit backup. Evaluate in Wave-23 if yfinance reliability becomes an issue. |

### Models and Frameworks

| Suggestion | Verdict | Reason |
|---|---|---|
| **Kronos / Kronos-mini** (TSFM K-line) | BUILD-NOW | Already scaffolded in Wave-20/21. Wave-22 completes Kronos inference wiring (weights download script + load_model + run_inference integration test). MIT license, CPU-feasible. |
| **Chronos-2** (Amazon multivariate TSFM) | SCAFFOLD/EVALUATE | Strong academic TSFM. Architecture differs from Kronos (probabilistic quantile output). Significant effort — new inference pipeline. Plan Wave-24 spike: compare Chronos-2-small vs Kronos-mini on 30-day BTC/AAPL directional accuracy before committing. |
| **Moirai-2** (Salesforce TSFM) | SCAFFOLD/EVALUATE | Another strong TSFM. Combine evaluation with Chronos-2 spike in Wave-24. One spike covers both. Pick the winner for Wave-25 integration alongside Kronos. |
| **Microsoft Qlib** (alpha factors / backtest) | SCAFFOLD/EVALUATE | Powerful quant platform. Value is alpha factor library and backtest. Learning curve is steep; large dependency tree. Already partially installed as Kronos transitive dep (pyqlib). Evaluate for factor correlation analysis on closed sims in Wave-25 only if vectorbt proves insufficient. |
| **FinRL** (RL trading agents) | SKIP | RL framework for live trading. Advice-only module. Even for research, RL requires execution environment and reward functions tied to P&L. Wrong tool for this scope. |
| **FinGPT** (news/sentiment LLM) | SCAFFOLD/EVALUATE | Open-source financial NLP for news sentiment. Potentially useful but: (a) we already use Claude for rationale, which is strictly better at reasoning; (b) adds second LLM inference pipeline with own GPU/model management overhead. Evaluate only if operator specifically requests news sentiment layer. Wave-25+. |
| **vectorbt** (fast backtest) | BUILD-NOW | Lightweight vectorised backtest library. Pure Python, minimal dependencies. Directly enables validating sim position outcomes against historical data (architecture doc section 5 already calls for this). Runs on existing advisor_sim_positions closed rows. |
| **Lumibot** (multi-asset live trading framework) | SKIP | Full live-trading framework with broker integrations, order management, portfolio rebalancing. Directly and fundamentally violates advice-only constraint. No role in this module. |
| **pandas-ta / TA-Lib** (indicators) | SKIP | We already compute SMA, RSI, Bollinger Bands, and volume ratio natively. TA-Lib is already in requirements. pandas-ta would duplicate existing code and bloat the dependency tree. Add individual indicators as needed for the signal engine. |

### Architecture Idea

| Suggestion | Verdict | Reason |
|---|---|---|
| **Multi-layer signal engine** (6-layer composite scored output) | BUILD-NOW | Best idea in the review. Transforms the current 3-vote majority (SMA/RSI/BB) into a richer 6-layer composite score. No external dependencies. Fits inside existing extra dict on AdviceResult. Self-contained, high-leverage. |
| **Regime detection** (TRENDING/RANGING/PANIC/EUPHORIA/ACCUM/DISTRIB) | BUILD-NOW | Tightly coupled to signal engine. ADX, ATR, volume profile, RSI thresholds classify market state. No new dependencies. Implementable in ~150 lines alongside the signal engine. |

---

## 2. BUILD-NOW List (Prioritised)

Specialists implement in this order. No two items in the same priority batch touch the same file.

### P0 — Multi-Layer Signal Engine + Regime Detection (quant-algo)

Highest leverage. Replaces the 3-vote heuristic across all five analyzers with a richer, more defensible signal. Everything else (ML recalibration, backtest, LLM prompt improvement) improves once the signal is better.

| Item | Task | Target file(s) |
|---|---|---|
| P0-A | Implement `SignalEngine` + `RegimeDetector` (new file) | `modules/advisor/core/signal_engine.py` (NEW) |
| P0-B | Wire SignalEngine into all 5 analyzers; replace `_signals_to_direction` + `_compute_confidence` | `analyzers/crypto.py`, `us_equities.py`, `bist.py`, `fx.py`, `midas_funds.py` |
| P0-C | Write composite score into `result.extra["signal_layers"]` (schema in section 4) | `analyzers/*.py` |

### P1 — BIST Data: borsapy (quant-algo)

| Item | Task | Target file(s) |
|---|---|---|
| P1-A | Implement `_fetch_borsapy()` as primary AVAILABLE path in BISTAnalyzer | `analyzers/bist.py` |
| P1-B | Demote yfinance .IS to explicit fallback with DEGRADED status | `analyzers/bist.py` |
| P1-C | Add `borsapy` to requirements | `requirements.txt` |

### P1 — Tefas Libraries (backend-devops)

| Item | Task | Target file(s) |
|---|---|---|
| P1-D | Replace hand-rolled CSRF scrape in MidasFundsAnalyzer with `tefas-crawler` package | `analyzers/midas_funds.py` |
| P1-E | Add `tefas-crawler` to requirements; evaluate tefasfon as alternative | `requirements.txt` |

### P1 — Alpha Vantage FX Wiring (quant-algo)

| Item | Task | Target file(s) |
|---|---|---|
| P1-F | Complete `_fetch_alphavantage()` HTTP call in FXAnalyzer (stub → working) | `analyzers/fx.py` |

### P1 — TCMB Official FX Rate (backend-devops)

| Item | Task | Target file(s) |
|---|---|---|
| P1-G | Add `TCMBDataSource` helper (evds.tcmb.gov.tr/service/dataindex for USD/TRY, EUR/TRY) | `modules/advisor/core/data_sources/tcmb.py` (NEW) |
| P1-H | Wire TCMB rate as context signal in FXAnalyzer and BISTAnalyzer rationale prompts | `analyzers/fx.py`, `analyzers/bist.py` |

### P2 — vectorbt Backtest (quant-algo)

| Item | Task | Target file(s) |
|---|---|---|
| P2-A | Implement `scripts/backtest_advisor_sims.py` using vectorbt on closed advisor_sim_positions | `scripts/backtest_advisor_sims.py` (NEW) |
| P2-B | Add `vectorbt` to requirements | `requirements.txt` |

### P2 — Kronos Inference Completion (quant-algo)

| Item | Task | Target file(s) |
|---|---|---|
| P2-C | Integration test for KronosForecaster (already scaffolded Wave-21; needs end-to-end test with mock weights) | `core/kronos_forecaster.py`, `tests/integration/test_advisor_kronos.py` (NEW) |
| P2-D | Create `scripts/download_kronos_weights.py` if not already present | `scripts/download_kronos_weights.py` |

### P3 — ML Loop Wiring (quant-algo)

| Item | Task | Target file(s) |
|---|---|---|
| P3-A | Wire AdvisorMLModel constructor in main_advisor.py startup | `main_advisor.py` |
| P3-B | Extend `_FEATURE_NAMES` with `trend_score`, `momentum_score`, `regime_encoded`, `risk_score` from signal_engine | `core/advisor_ml.py` |

---

## 3. SCAFFOLD/EVALUATE Backlog

| Item | Notes | Target wave |
|---|---|---|
| borsa-mcp | Evaluate when all five analyzers stable. Could unify data layers but MCP is new architectural layer. | Wave-24 |
| Chronos-2 | Spike: run Chronos-2-small on BTC/AAPL 30d series vs Kronos-mini. Use `chronos` or `autogluon-timeseries`. | Wave-24 |
| Moirai-2 | Combine with Chronos-2 spike. Pick one winner for Wave-25. | Wave-24 |
| Microsoft Qlib | Evaluate for alpha factor extraction only. Qlib partially installed via pyqlib (Kronos dep). | Wave-25 |
| FinGPT | Evaluate only if operator requests news sentiment. Requires GPU; separate inference service if built. | Wave-25+ |
| TwelveData | Evaluate as yfinance rate-limit fallback. Compare free-tier limits to hourly polling cadence. | Wave-23 |

---

## 4. SKIP List

| Item | Reason |
|---|---|
| dxFeed | Paid institutional data. No operator contract. Over-scoped. |
| Cbonds | Bond-specific. No bond market in scope. |
| Matriks paid API | Already stubbed Wave-21. Activation requires operator key — operator action, not build task. |
| Alpaca | Order execution + streaming. Violates advice-only constraint. |
| Polygon | Real-time tick data. Our cadence is hourly; no value over yfinance for daily bars. |
| tvdatafeed / TradingView scraping | ToS violation. Fragile. Legally risky. |
| FinRL | RL trading framework. Requires execution environment. Fundamentally incompatible with advice-only. |
| Lumibot | Full live-trading framework. Directly violates advice-only constraint. |
| pandas-ta | Redundant. Existing native indicator implementations cover our needs. Import individual indicators as needed. |

---

## 5. Composite Signal Output Schema

This is the contract the quant specialist implements in `signal_engine.py` and writes into `AdviceResult.extra["signal_layers"]`.

### Output JSON shape (written to `result.extra["signal_layers"]`)

```json
{
  "trend_score":      82,
  "momentum_score":   77,
  "volatility_score": 61,
  "volume_score":     91,
  "regime":           "TRENDING",
  "ai_forecast":      0.42,
  "risk_score":       38,
  "confidence":       84,
  "action":           "strong_buy"
}
```

### Field definitions

| Field | Type | Range | Source | Description |
|---|---|---|---|---|
| `trend_score` | int | [0, 100] | SMA20/50 cross + EMA9 slope + ADX(14) | 0=max bearish, 100=max bullish. ADX>25 adds 20 pts to magnitude. |
| `momentum_score` | int | [0, 100] | RSI-14 + MACD histogram sign + Rate-of-Change(10) | RSI 30-70 neutral band maps to 40-60. Below 30 -> 0-30. Above 70 -> 70-100. |
| `volatility_score` | int | [0, 100] | Bollinger %B + ATR normalised vs 20d mean | High volatility = low score (risk signal). %B>0.8 or <0.2 -> score drops below 40. |
| `volume_score` | int | [0, 100] | Volume ratio vs 20d avg + OBV slope | vol_ratio>1.5 with directional agreement -> 80+. No volume data (spot FX) -> 50 neutral. |
| `regime` | str | enum | ADX + ATR + RSI + volume profile | One of: TRENDING, RANGING, PANIC, EUPHORIA, ACCUMULATION, DISTRIBUTION. |
| `ai_forecast` | float | [-1.0, 1.0] | Kronos signal normalised, or 0.0 if unavailable | Positive=bullish. Kronos raw signal / historical std. None maps to 0.0. |
| `risk_score` | int | [0, 100] | Inverse volatility | ATR_pct<1% -> 80+. ATR_pct>3% -> below 40. PANIC regime forces <=20. |
| `confidence` | int | [0, 100] | Weighted average of all layers, regime-adjusted | 0.25*trend + 0.20*momentum + 0.15*volatility + 0.15*volume + 0.10*risk + 0.15*abs(ai_forecast*100). PANIC multiplies by 0.5. |
| `action` | str | enum | confidence + direction + regime | One of: strong_buy, buy, neutral, sell, strong_sell. |

### Regime classification rules

| Regime | Condition |
|---|---|
| TRENDING | ADX(14) > 25 AND ATR(14)_pct < 3% |
| RANGING | ADX(14) < 20 AND ATR(14)_pct < 2% |
| PANIC | ATR(14)_pct > 5% AND RSI-14 < 25 AND vol_ratio > 3 |
| EUPHORIA | ATR(14)_pct > 4% AND RSI-14 > 75 AND vol_ratio > 2.5 |
| ACCUMULATION | ADX(14) < 20 AND vol_ratio > 1.5 AND RSI-14 < 45 |
| DISTRIBUTION | ADX(14) < 20 AND vol_ratio > 1.5 AND RSI-14 > 55 |
| Default | RANGING (when no rule matches) |

### Action derivation rules

Direction is set by majority vote: long if (trend_score + momentum_score + volume_score) > 150, short if < 150, else neutral.

| Condition | action |
|---|---|
| direction=long AND confidence>=75 AND regime!=PANIC | strong_buy |
| direction=long AND confidence>=50 | buy |
| direction=short AND confidence>=75 AND regime!=PANIC | strong_sell |
| direction=short AND confidence>=50 | sell |
| regime=PANIC (any direction) | neutral (PANIC overrides direction) |
| Otherwise | neutral |

### Implementation contract for `signal_engine.py`

```python
# modules/advisor/core/signal_engine.py
from dataclasses import dataclass

@dataclass
class CompositeSignal:
    trend_score: int        # [0, 100]
    momentum_score: int     # [0, 100]
    volatility_score: int   # [0, 100]
    volume_score: int       # [0, 100]
    regime: str             # TRENDING|RANGING|PANIC|EUPHORIA|ACCUMULATION|DISTRIBUTION
    ai_forecast: float      # [-1.0, 1.0]
    risk_score: int         # [0, 100]
    confidence: int         # [0, 100]
    action: str             # strong_buy|buy|neutral|sell|strong_sell

    def to_dict(self) -> dict:
        return {
            "trend_score": self.trend_score,
            "momentum_score": self.momentum_score,
            "volatility_score": self.volatility_score,
            "volume_score": self.volume_score,
            "regime": self.regime,
            "ai_forecast": self.ai_forecast,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "action": self.action,
        }


class SignalEngine:
    """
    Multi-layer signal scorer.

    Required signal keys: close, sma20, sma50, rsi, bb_upper, bb_lower.
    Optional keys: vol_ratio, adx, atr, macd_hist, roc10, obv_slope, ema9.
    kronos_signal: float or None — passed separately.

    Usage in each analyzer:
        signal = SignalEngine.compute(signals, kronos_signal=result.kronos_signal)
        result.extra["signal_layers"] = signal.to_dict()
        result.direction = _action_to_direction(signal.action)
        result.confidence = signal.confidence / 100.0
    """

    @staticmethod
    def compute(signals: dict, kronos_signal: float | None = None) -> "CompositeSignal":
        ...  # quant agent implements
```

### AdviceResult field mapping

After `SignalEngine.compute()`:
- `AdviceResult.direction` from `signal.action`: strong_buy/buy -> LONG; strong_sell/sell -> SHORT; neutral -> NEUTRAL
- `AdviceResult.confidence` = `signal.confidence / 100.0`
- `AdviceResult.extra["signal_layers"]` = `signal.to_dict()`
- `AdviceResult.kronos_signal` unchanged (raw Kronos float or None)

The `advisor_ml.py` feature set should be extended post-P0 with: trend_score, momentum_score, regime_encoded (int 0-5), risk_score.

---

## 6. OpenAI Dual-Advice Design

Each advice cycle can optionally call a second LLM provider (OpenAI) and merge the two rationale outputs.

### New config keys (to be seeded in migration 061 by backend-devops agent)

| Key | Type | Default | Description |
|---|---|---|---|
| `advisor_openai_enabled` | boolean | false | Enable OpenAI as secondary LLM for rationale |
| `advisor_openai_model` | string | gpt-4o | OpenAI model ID. Loud WARNING on 404; no silent fallback. |
| `advisor_openai_api_key` | string | (empty) | Resolved from config or ADVISOR_OPENAI_API_KEY env var. |
| `advisor_dual_advice_mode` | string | anthropic_primary | Merge strategy: anthropic_primary, openai_primary, best_of, both |

### API key resolution order (same pattern as Anthropic key)

1. `config["advisor_openai_api_key"]`
2. `os.getenv("ADVISOR_OPENAI_API_KEY")`

No key -> OpenAI path silently skipped; Anthropic rationale used alone.

### New functions in `rationale_helper.py`

```python
async def build_openai_rationale(
    symbol: str,
    market: Market,
    horizon: Horizon,
    signals: dict,
    direction: Direction,
    config: dict,
    caller_logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """
    Secondary OpenAI rationale. Returns None if key missing or call fails.
    Prompt identical to Anthropic prompt (same signal_summary block).
    Model: config["advisor_openai_model"] default "gpt-4o".
    Uses openai.OpenAI(api_key=...).chat.completions.create().
    Never raises — returns None on any exception.
    """
    ...

async def merge_rationale(
    anthropic_text: str,
    openai_text: Optional[str],
    mode: str,
) -> str:
    """
    Merge two rationale strings per advisor_dual_advice_mode.

    anthropic_primary : anthropic_text + "\\n\\n[OpenAI view: " + openai_text + "]"  if openai_text else anthropic_text
    openai_primary    : openai_text + "\\n\\n[Claude view: " + anthropic_text + "]"   if openai_text else anthropic_text
    best_of           : whichever text is longer (heuristic for more detailed)
    both              : "Claude: " + anthropic_text + "\\n\\nGPT: " + openai_text     if openai_text else anthropic_text
    """
    ...
```

### How `build_rationale()` changes

```python
async def build_rationale(..., config: dict, ...) -> str:
    # Step 1: Anthropic (existing path)
    anthropic_text = await _call_anthropic(...) or build_rule_based_rationale(...)

    # Step 2: OpenAI secondary (new)
    openai_text = None
    if str(config.get("advisor_openai_enabled", "false")).lower() == "true":
        openai_text = await build_openai_rationale(...)

    # Step 3: merge per mode
    mode = config.get("advisor_dual_advice_mode", "anthropic_primary")
    return await merge_rationale(anthropic_text, openai_text, mode)
```

### model_id for dual-advice

- Both providers used: `"claude-opus-4-8+gpt-4o"` (joined by `+`)
- Anthropic only: `"claude-opus-4-8"` (unchanged)
- OpenAI only (Anthropic fails, openai_primary mode): `"gpt-4o"`

### New secret for operator

| Secret name | Purpose |
|---|---|
| `ADVISOR_OPENAI_API_KEY` | OpenAI API key for secondary rationale |

### Dependency addition

```
requirements.txt: openai>=1.30   # lazy import; only loaded if advisor_openai_enabled=true
```

---

## 7. Model Default Bump (SMALL FIX A)

**Status**: COMPLETE (committed at eccd543)

**File changed**: `modules/advisor/core/rationale_helper.py` line 181
**Old default**: `"claude-opus-4-5"`
**New default**: `"claude-opus-4-8"` (DB-overridable via `advisor_anthropic_model` in advisor_config)

**Migration 058 updated**: seed value line 105 changed from `claude-opus-4-5` to `claude-opus-4-8`.

Note: migration 058 uses `ON CONFLICT DO NOTHING`. Existing installations will NOT be auto-updated.
Migration 061 (backend-devops) must include:

```sql
UPDATE config_settings
   SET value = 'claude-opus-4-8'
 WHERE config_type = 'advisor_config'
   AND key = 'advisor_anthropic_model'
   AND value = 'claude-opus-4-5';
```

---

## 8. Specialist Dispatch Order

| Round | Agent | Items | Scheduling constraint |
|---|---|---|---|
| 1 | quant-algo | P0-A: signal_engine.py (new file) | No existing file touched — can start immediately |
| 2 | quant-algo | P0-B + P0-C: wire SignalEngine into all 5 analyzers | After Round 1 (P0-A must exist) |
| 2 | backend-devops | P1-D + P1-E: tefas-crawler in MidasFundsAnalyzer | Parallel with Round 2 quant (different files) |
| 2 | backend-devops | P1-G: TCMB helper (new file) | Parallel with Round 2 (no overlap) |
| 3 | quant-algo | P1-A + P1-B + P1-C: borsapy in BISTAnalyzer | After Round 2 (bist.py touched in R2 by SignalEngine wire) |
| 3 | quant-algo | P1-F: Alpha Vantage FX wiring | After Round 2 (fx.py touched in R2 by SignalEngine wire) |
| 3 | backend-devops | P1-H: wire TCMB into fx.py + bist.py prompts | After Round 2 (both files touched in R2) |
| 4 | quant-algo | P2-A + P2-B: vectorbt backtest script | Independent of analyzer files |
| 4 | quant-algo | P2-C: Kronos integration test | Independent |
| 4 | backend-devops | P2-D: Kronos weights download script | Independent |
| 4 | backend-devops | Migration 061 (model bump UPDATE + OpenAI config seeds) | Can author after this plan is approved |
| 5 | quant-algo | P3-A + P3-B: ML loop wiring + extended features | After P0 complete (P3-B needs signal_layers) |
| 6 | quant-algo | rationale_helper.py dual-advice (OpenAI path) | After migration 061 seeds config keys |

---

## 9. PM-Owned Changes in This Wave

| File | Change |
|---|---|
| `modules/advisor/core/rationale_helper.py` | Default model bumped to claude-opus-4-8. Docstring updated. Committed at eccd543. |
| `migrations/058_advisor_module.sql` | Seed value updated to claude-opus-4-8. Committed at eccd543. |
| `docs/agents/wave22/advisor_enhancement_plan.md` | This plan document (created). |
