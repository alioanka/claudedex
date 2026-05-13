# AI_MODULE — Backend / DevOps Audit

**Author:** backend-devops-expert
**Date:** 2026-05-11
**Branch:** `claude/create-expert-agents-JFSF5`
**Scope:** infrastructure, secrets, configuration, observability, LLM-provider quality

---

## 1. Executive verdict — **AMBER** (acceptable for shadow-mode; not yet for live trading capital)

The AI module is in better shape than Futures on a few axes (a proper `AIProviderManager` with cost tracking, caching, rate limiting, and provider fallback; structured `ai_analysis_logs` table; per-provider rotating logs in `logs/ai_analysis/{openai_api,claude_api}.log`) but worse on others. The grade is dragged down by:

1. **Two parallel LLM call paths.** `AIProviderManager` (`core/ai_provider.py`) is the enterprise-grade path with caching/cost tracking — but `SentimentEngine` keeps a **legacy** `_analyze_with_llm()` / `_analyze_with_claude()` path (`core/sentiment_engine.py:505-682`) that does *not* go through the manager, has no cost tracking, no rate limiting, and no cache. It is reachable whenever `ai_provider_manager` initialization fails or returns `sentiment_score == 0.0` (legitimately neutral! — see bug below).
2. **Hard-coded prompts and models in code.** The sentiment prompt is built in three different places (`ai_provider.py:366-390`, `sentiment_engine.py:508-512, 618-622`) with subtly different wording. There is no prompt-template registry, no versioning, no unit test.
3. **Trade execution lives in the AI module** (`sentiment_engine.py:24-200` — `AITradeExecutor`) and ships signed orders directly to `fapi.binance.com` (`line 134`), bypassing the futures module's risk manager, audit log, and Prometheus counters entirely. **This is the single biggest risk** in the AI module.
4. **No PII / sensitive-input filter** before sending to external LLMs. Currently safe because input is public news headlines, but if/when we extend to user-supplied content or wallet logs, this becomes a P0.
5. **`ai_trading_engine.py` is a parallel universe.** It defines `Chain`, `TradeType`, `MarketCondition`, `AITradeSignal`, even an entire `AITradingEngine` orchestrator (1098 LOC) — but `main_ai.py` only instantiates `SentimentEngine`. The file is dead code or aspirational; either wire it in or delete.

Amber, not Red, because (a) `DRY_RUN=true` is the default everywhere, (b) the AI Provider Manager has a daily-budget kill-switch ($10/day default, `ai_provider.py:160`), (c) the module never moves real capital unless `direct_trading=True` AND `DRY_RUN=false` AND keys are configured.

---

## 2. Secrets / credentials audit

### Where each key currently comes from

| Secret | Read site | Source path | Encrypted? |
|---|---|---|---|
| `OPENAI_API_KEY` | `main_ai.py:91-95`, `core/ai_provider.py:179, 217-227`, `core/sentiment_engine.py:317, 329` | Secrets manager (DB / Docker secrets) → `.env` fallback | Yes in DB (Fernet); no in `.env` |
| `ANTHROPIC_API_KEY` | `main_ai.py:92-96`, `core/ai_provider.py:180, 217-227`, `core/sentiment_engine.py:322, 330` | Same | Same |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` (used by `AITradeExecutor`) | `core/sentiment_engine.py:38-42` | Secrets manager → `.env` | Same |
| `DATABASE_URL` | `main_ai.py:111-113` | Docker secrets → `.env` | File-mount (Docker) or plaintext |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | indirectly via `monitoring/telegram_bot.get_telegram_controller(db_pool)` (`main_ai.py:151`) | Secrets manager | Same |

Notably absent from the codebase: **Cohere, Perplexity, Mistral, Groq, Together** — the docstrings of the audit prompt mention them but no provider stubs exist. `AIProvider` enum (`core/ai_provider.py:25-29`) is `OPENAI | ANTHROPIC | COMBINED`. Plan accordingly.

### Findings

- **AI-BE-01 (P0): Hard `os.getenv()` fallback chains remain.** Eight call sites: `main_ai.py:95-96, 113`; `ai_provider.py:227`; `sentiment_engine.py:41-42, 237, 329-330`. Same anti-pattern as Futures (FUT-BE-01). If the DB is unreachable, the module silently uses a possibly-stale `.env` key without alerting. Add `WARNING` + `ai_module_secret_fallback_total{key}` counter.

- **AI-BE-02 (P0): Binance API key is loaded *both* by the Futures module and by `AITradeExecutor.__init__` (`sentiment_engine.py:38-42`).** This is a code-smell. The AI module should *call into* the Futures module's risk-checked executor, not maintain its own signed-request path that bypasses every risk control. Today the AI module's `_execute_binance_futures()` (`sentiment_engine.py:113-187`) directly POSTs `MARKET` orders with `quantity = amount_usd / price` — no `recvWindow`, no `newClientOrderId`, no precision check, no rate-limit handling, no audit log.

- **AI-BE-03 (P1): Pre-init secrets-manager bootstrap.** `sentiment_engine.py:313-314` calls `secrets.initialize(self.db_pool)` if the manager is in bootstrap mode — that's an awkward dance. The orchestrator (`main.py`) should pre-initialize the secrets manager once with the DB pool, and modules should treat it as ready.

- **AI-BE-04 (P2): API keys are logged in truncated form** (`main_ai.py:99-104` → `f"{openai_key[:20]}..."`). Twenty characters of an OpenAI key (`sk-proj-…`) is plenty to confirm the prefix but not enough to leak. Probably acceptable; some teams require zero prefix logging.

- **AI-BE-05 (P2): `BINANCE_API_KEY` discovered in `AITradeExecutor` does not honor testnet vs mainnet.** `sentiment_engine.py:134` is hard-coded `https://fapi.binance.com`. Even in `DRY_RUN=true` (which short-circuits at line 70), if an operator flips `DRY_RUN=false` while still on Binance testnet keys, this will fail signed-request validation in confusing ways.

---

## 3. Connector audit — LLM provider clients & exchange path

### `AIProviderManager` (`core/ai_provider.py`) — the good path

| Concern | Finding | File:line |
|---|---|---|
| **HTTP session reuse** | Single `aiohttp.ClientSession` created in `initialize()`, closed in `close()`. ✅ | `ai_provider.py:174-176` |
| **Timeout** | `total=60s`. Long but acceptable for LLMs. | `ai_provider.py:175` |
| **Auth** | `Bearer` header for OpenAI; `x-api-key` + `anthropic-version: 2023-06-01` for Claude. Correct. | `ai_provider.py:447-451, 487-491` |
| **Rate limiting** | Token-bucket `RateLimiter` per provider, 60 rpm default. Good but not driven from the provider's own header response. | `ai_provider.py:106-138, 190, 203` |
| **Caching** | SHA-256 keyed `ResponseCache(ttl=300s)`. Cache key = `model:prompt`. ✅ | `ai_provider.py:79-103` |
| **Cost tracking** | Per-model `input_cost`/`output_cost` baked into `AIModel` enum; daily budget enforced at `analyze_sentiment()` entry. ✅ | `ai_provider.py:160-162, 286-297` |
| **Retry / backoff** | None. First non-200 → mark provider failed and try next. Acceptable given fallback chain. | `ai_provider.py:461-466` |
| **Provider fallback** | `_get_ordered_providers()` returns providers by priority; loops until one succeeds. ✅ | `ai_provider.py:312-353, 355-364` |
| **DB logging** | `_log_api_call()` writes prompt/response previews (first 500 chars) into `ai_analysis_logs`. ✅ | `ai_provider.py:601-628` |
| **Error envelope** | "Budget exceeded" path returns a sentinel — caller must check `error` field; would be cleaner with raise. | `ai_provider.py:286-297` |

### Legacy path inside `SentimentEngine` — the bad path

| Concern | Finding | File:line |
|---|---|---|
| **New `aiohttp.ClientSession` per call** | A fresh session created on every API call → connection churn, extra TLS handshakes, no pooling. | `sentiment_engine.py:535, 646` |
| **No timeout** | The legacy session has no timeout. A hung LLM connection can block the cycle for as long as `aiohttp` waits. | `sentiment_engine.py:535` |
| **No retry / backoff** | First non-200 → return `0.0` (which then masquerades as a neutral sentiment). | `sentiment_engine.py:562-567, 673-678` |
| **No cost tracking** | The whole point of `AIProviderManager` is bypassed. The fallback (`sentiment_engine.py:437-451`) is reached every time the manager returns sentiment exactly `0.0` — including legitimate neutral scores. We pay twice for the same news cycle. | `sentiment_engine.py:438` |
| **No cache** | Identical headlines call OpenAI again. Recurring cost leak. | n/a |
| **Prompt drift** | Three nearly identical prompts (`ai_provider.py:366-390`, `sentiment_engine.py:508-512`, `:618-622`) — different temperature handling, different parsing logic. | n/a |
| **JSON parsing fragility** | `_parse_sentiment_response()` in `ai_provider.py:522` has a markdown-code-fence stripper and a fallback regex; legacy `_analyze_with_llm` just does `float(content)` — a single non-numeric char breaks it. | `sentiment_engine.py:551-561` |

### `AITradeExecutor._execute_binance_futures` (`sentiment_engine.py:113-187`) — UNSAFE for live capital

This duplicates the Futures connector inline:
- HMAC signing on a manually-built query string (`line 152-156`).
- No `recvWindow`.
- No `newClientOrderId`.
- No exchange-side rate-limit parsing.
- Hard-coded `https://fapi.binance.com` (mainnet only — `line 134`).
- `quantity = round(amount_usd / price, 3)` (`line 131`) — no per-symbol precision from `exchangeInfo`.
- No audit log; no Prometheus counter.

**AI-BE-06 (P0)**: Delete `_execute_binance_futures` and route AI trade intent through the Futures module's risk-validated `execute_order()`. The AI module should publish a `signal` (sentiment, confidence, symbol, side) — never sign exchange orders itself.

### Other connector issues

- **AI-BE-07 (P1)**: `_fetch_news()` (`sentiment_engine.py:488-503`) hits CryptoCompare without auth or pagination. If the CryptoCompare endpoint rate-limits the bot's IP, the engine sees zero headlines and silently sleeps — no metric, no alert. Add `ai_module_news_fetch_total{status}`.
- **AI-BE-08 (P2)**: `AIMarketAnalyzer._fetch_market_data()` (`ai_trading_engine.py:285+`) hits CoinGecko without a key (`ai_trading_engine.py:842-848`) — works at low volume, but free-tier rate-limit is ~5 rpm. If `analysis_interval` is short, we'll be 429-throttled.
- **AI-BE-09 (P2)**: `RateLimiter.acquire()` (`ai_provider.py:115-138`) uses `int(elapsed * rpm / 60)` for refill. At low rpm × short intervals, this rounds to zero and the bucket never refills until you exceed 1s. Use float math.

---

## 4. Configuration audit

### DB-backed (via `config_settings` with `config_type='ai_config'`)
Loaded in two places:
- `core/ai_provider.py:229-256` reads `daily_budget_usd`, `openai_model`, `claude_model`.
- `core/sentiment_engine.py:332-362` reads `direct_trading`, `confidence_threshold`, `trade_amount_usd`, `ai_provider`.

This is **not** managed by a Pydantic-backed `AIConfigManager` analogous to `FuturesConfigManager`. Each setting is hand-parsed (`float(val)`, `val.lower() in ('true', ...)`). No validation, no schema, no history. The Futures module already nailed this pattern; the AI module should adopt it.

### Hardcoded values that should be DB-backed
- `take_profit_pct=5.0`, `stop_loss_pct=-3.0`, `max_hold_hours=24` (`core/sentiment_engine.py:240-242`)
- `_cooldown_duration = timedelta(hours=1)` (`core/sentiment_engine.py:252`)
- `daily_budget_usd: float = 10.0` (`core/ai_provider.py:160`) — DB-readable but the *default* is hard-coded.
- Sentiment loop sleep `await asyncio.sleep(900)` (`core/sentiment_engine.py:476`).
- Cache TTL `ttl_seconds=300` (`core/ai_provider.py:82`).
- All prompt strings (no DB registry, no version).
- `symbol = "ETH"` hard-coded in `_execute_trade()` (`core/sentiment_engine.py:745`). **This means every sentiment-driven trade is on ETH regardless of which token the news was about.**

### `.env` direct reads
Eight call sites total — see `os.getenv` table in §2. All but `DRY_RUN` (`sentiment_engine.py:237`) should be DB-backed.

### Findings
- **AI-BE-10 (P0)**: Build `modules/ai_analysis/config/ai_config_manager.py` modeled on `FuturesConfigManager`. Pydantic schemas: `AIGeneralConfig`, `AIBudgetConfig` (daily/monthly $-cap, per-call max-token cap), `AIProviderConfig` (priority, model id, rpm), `AISentimentConfig` (loop sleep, cache TTL, headline source URLs), `AITradeConfig` (TP%, SL%, max-hold-hrs, cooldown, target symbol).
- **AI-BE-11 (P1)**: Move all prompt strings into a `prompts/` directory with semver tags and unit tests asserting that the JSON the model returns is parseable. Track prompt version in `ai_analysis_logs` so we can A/B-test prompts.

---

## 5. Observability audit

### Logging
- `main_ai.py:20-78` creates `logs/ai_analysis/{ai,ai_errors,openai_api,claude_api}.log` with rotation. Good.
- Per-call OpenAI and Claude request/response logs include token usage and latency (`sentiment_engine.py:546-549, 657-661`). Excellent for forensics.
- Logs are not JSON-structured.
- The `ai_analysis_logs` table also captures prompt/response previews (500 chars) (`ai_provider.py:609-625`). Good for DB-side analytics.

### Prometheus metrics
- **There are zero Prometheus metrics exposed by the AI module.** `grep prometheus_client` returns nothing. `main_ai.py` runs no health server, no `/metrics` endpoint.
- Hence none of the required metrics from the brief (`ai_module_llm_calls_total`, `ai_module_llm_tokens_total`, `ai_module_llm_cost_usd`, `ai_module_signal_latency_ms`) exist.
- `observability/prometheus.yml` has no AI target.
- `observability/alerts.yml` has no AI-specific rules.

This is a P0 — operationally we have **no real-time signal** that the LLM has fallen over, that the daily budget is nearing exhaustion, or that latency has blown out.

### Required metrics (must add)
- `ai_module_llm_calls_total{provider,model,outcome}` — outcome ∈ `ok|cache|error|budget_exceeded|rate_limited`
- `ai_module_llm_tokens_total{provider,model,type}` — type ∈ `input|output`
- `ai_module_llm_cost_usd{provider,model}` — counter
- `ai_module_llm_cost_usd_daily` — gauge (today's spend)
- `ai_module_signal_latency_ms` — histogram
- `ai_module_sentiment_score` — gauge (last computed score)
- `ai_module_news_fetch_total{status}` — counter
- `ai_module_active_positions` — gauge
- `ai_module_cache_hits_total` / `ai_module_cache_misses_total`

### Required alerts
- `AIBudgetWarning`: `ai_module_llm_cost_usd_daily / ai_daily_budget_usd > 0.8` for 5m.
- `AIBudgetExceeded`: `ai_module_llm_cost_usd_daily >= ai_daily_budget_usd`.
- `AILatencyHigh`: `histogram_quantile(0.95, ai_module_signal_latency_ms_bucket) > 10000`.
- `AIAllProvidersDown`: `rate(ai_module_llm_calls_total{outcome="error"}[5m]) / rate(ai_module_llm_calls_total[5m]) > 0.5`.
- `AINewsFeedDown`: `rate(ai_module_news_fetch_total{status="ok"}[15m]) == 0`.

### Findings
- **AI-BE-12 (P0)**: Add an aiohttp `/metrics` health server to `main_ai.py` (model after `main_futures.HealthServer`), wire `prometheus_client` counters and gauges throughout `AIProviderManager` and `SentimentEngine`, and add the scrape target to `observability/prometheus.yml`.
- **AI-BE-13 (P0)**: Add five AI alerts to `observability/alerts.yml`.
- **AI-BE-14 (P1)**: Daily $-cap alarm must page (Telegram + Prom) at 50% / 80% / 100%. Today the `budget exceeded` event only logs a `WARNING`.

---

## 6. Resource / lifecycle audit

- **aiohttp sessions**:
  - `AIProviderManager._session` is reused — good.
  - `SentimentEngine._analyze_with_llm` / `_analyze_with_claude` open a **new** session every cycle (`sentiment_engine.py:535, 646`). Should be reused.
  - `AITradeExecutor.session` is reused — OK.
  - `AIMarketAnalyzer._fetch_market_data` (`ai_trading_engine.py:298`) creates a new session per call. Wasteful.
- **Task cancellation**: `SentimentEngine.run()` (`sentiment_engine.py:364-486`) starts a `_monitor_positions()` task and cancels it on `KeyboardInterrupt`. There's no `stop()` method exposed via Telegram controller hook, despite `main_ai.py:157` calling `stop_method='stop'`. **Missing `SentimentEngine.stop()` will throw on Telegram /stop.** Verify or add.
- **DB pool**: `main_ai.py:121` calls `asyncpg.create_pool(db_url)` with default size (min=10, max=10 from `asyncpg`). No `command_timeout`. A hung query can starve the entire engine.
- **Unbounded structures**:
  - `AIProviderManager.call_history` capped at `_max_history=1000` (`ai_provider.py:166, 597-599`). ✅
  - `ResponseCache.cache` — no max size, only TTL. A long-running session with diverse headlines could grow unboundedly. Add LRU eviction with `maxsize=10000`.
  - `AIMarketAnalyzer.cache` (`ai_trading_engine.py:177`) — same problem, no eviction.
  - `SentimentEngine.active_positions` and `_symbol_cooldowns` are tiny — fine.
  - `AITradingEngine.active_positions` (in `ai_trading_engine.py:819`) — fine in volume but never persisted on restart.
- **DRY_RUN gate**: present in `AITradeExecutor.execute_trade` (`sentiment_engine.py:70`). Good. But `AITradingEngine._execute_live` (`ai_trading_engine.py:828`) is currently a stub `logger.info` — when someone fills this in, double-check the gate.

---

## 7. Profit-leak / loss-leak inventory (infra-driven only)

| # | Leak | Driver | File:line | Impact |
|---|---|---|---|---|
| 1 | **AI-module bypasses Futures risk manager** — `AITradeExecutor` signs and sends Binance orders directly with no daily-loss limit, no consecutive-loss circuit-breaker, no audit log. | `sentiment_engine.py:113-187` | Catastrophic if `direct_trading=True` + `DRY_RUN=false`. | Critical |
| 2 | **Duplicate LLM calls on neutral score** — legacy path triggers whenever `AIProviderManager` returns sentiment exactly `0.0`. A legitimately neutral market spends LLM budget twice every cycle (`900 s` = 96 cycles/day → up to ~96 redundant LLM calls/day). | `sentiment_engine.py:438` | $-leak proportional to neutrality. | Medium |
| 3 | **`symbol = "ETH"` is hard-coded for sentiment trades.** Every signal goes ETH-only regardless of which token the headlines actually discuss. | `sentiment_engine.py:745` | Foregone alpha. | High |
| 4 | **Hardcoded mainnet Binance URL in AI executor** — operator who flips `DRY_RUN=false` while still on testnet keys gets cryptic 401s; same operator flipping with mainnet keys but expecting testnet will execute real trades. | `sentiment_engine.py:134` | Foot-gun. | Medium |
| 5 | **No budget metric / alert** — when daily $-cap is hit, AI silently returns neutral. Sentiment-driven downstream consumers see "no signal" instead of "AI is broke". | `ai_provider.py:286-297` | Foregone alpha + silent failure. | Medium |
| 6 | **Cache TTL = 300 s** for fresh news is too short — we re-pay for nearly identical headlines every five minutes. Should be 30-60 min for daily news cycles, with explicit invalidation on major-event tags. | `ai_provider.py:82, 156` | Recurring $-leak. | Low |
| 7 | **News fetch failure invisibility** — if CryptoCompare 429s us, the cycle goes silent, but no metric fires. We could be deaf for hours before noticing. | `sentiment_engine.py:488-503` | Foregone alpha. | Medium |
| 8 | **Prompts not versioned** — a prompt tweak by one engineer silently changes confidence-threshold calibration; we can no longer A/B-compare. | `ai_provider.py:366-390` et al. | Insidious quality drift. | Medium |
| 9 | **`anthropic-version: 2023-06-01` is pinned** — when Anthropic deprecates or rate-limits old versions, we lose the Claude provider with no metric. | `ai_provider.py:489`, `sentiment_engine.py:637` | One-time future event. | Low |
| 10 | **`response_format={"type":"json_object"}`** is only on OpenAI calls; Claude calls expect JSON without forcing it. A free-form Claude reply trips `_parse_sentiment_response`. | `ai_provider.py:458, 493-497` | Sporadic loss of signal. | Low |
| 11 | **No PII / prompt-injection filter** — if we extend the input source to user-supplied text (Twitter handles the bot follows, etc.), prompt injection becomes possible. Not exploitable today; pre-emptive guard recommended. | `ai_provider.py:366-390` | Tail risk. | Low |

---

## 8. Dashboard integration

Existing pages under `modules/dashboard/templates/`:
- `dashboard_ai.html` ✅
- `performance_ai.html` ✅
- `sentiment_ai.html` ✅
- `settings_ai.html` ✅
- `logs_ai.html` ✅

Missing relative to the "every trading module" rule: **`trades_ai.html`** and **`positions_ai.html`** — the AI module *does* take positions (`SentimentEngine.active_positions`, `ai_trades` table), so these pages should exist for parity with the Futures module.

- **AI-BE-15 (P0)**: Add `trades_ai.html` and `positions_ai.html`. Backend should expose JSON endpoints `/ai/positions` and `/ai/trades` (today the only data is via the dashboard module's generic queries against `ai_trades`).
- **AI-BE-16 (P1)**: `settings_ai.html` Settings/Guide tabs must include: daily $ budget, model selection per provider, fallback order, confidence threshold, target symbol(s) (replace hard-coded `"ETH"`), prompt template version. PM/QA pass required.
- **AI-BE-17 (P2)**: Cost dashboard widget — daily/weekly/monthly spend by provider, cache hit rate, average latency, model cost-efficiency (USD per profitable signal).

---

## 9. Action backlog (ranked)

| ID | Priority | Action | Owner |
|---|---|---|---|
| AI-BE-01 | P0 | Remove `or os.getenv('OPENAI_API_KEY'/'ANTHROPIC_API_KEY'/'BINANCE_*')` fallbacks in `main_ai.py`, `core/ai_provider.py:227`, `core/sentiment_engine.py:41-42, 329-330`; emit `ai_module_secret_fallback_total` | backend |
| AI-BE-02 | P0 | Remove inline Binance execution from `AITradeExecutor`; route AI trade intents through the Futures module's risk-validated executor | backend + analyst |
| AI-BE-06 | P0 | Delete `_execute_binance_futures` (`sentiment_engine.py:113-187`); AI module signs **no** exchange orders | backend |
| AI-BE-10 | P0 | Build `AIConfigManager` (Pydantic, DB-backed) mirroring `FuturesConfigManager` | backend |
| AI-BE-12 | P0 | Add `/metrics` health server to `main_ai.py`; emit `ai_module_llm_*` counters & gauges; add scrape target to `observability/prometheus.yml` | backend |
| AI-BE-13 | P0 | Add five AI-specific alerts to `observability/alerts.yml` | backend |
| AI-BE-15 | P0 | Create `trades_ai.html` and `positions_ai.html` (parity with Futures) | backend |
| AI-BE-03 | P1 | Centralize secrets-manager initialization in `main.py` orchestrator; remove module-level bootstrap dance | backend |
| AI-BE-07 | P1 | Instrument `_fetch_news()` with `ai_module_news_fetch_total{status}` | backend |
| AI-BE-11 | P1 | Move prompts to versioned templates with unit tests; log `prompt_version` in `ai_analysis_logs` | backend + quant |
| AI-BE-14 | P1 | Budget warning at 50/80/100% via Telegram + Prom | backend |
| AI-BE-16 | P1 | Settings/Guide tabs on `settings_ai.html` with budget, model, target-symbol controls | backend + pm |
| AI-BE-18 | P1 | Kill the legacy fallback paths (`_analyze_with_llm`, `_analyze_with_claude`); fix the "sentiment==0.0 triggers fallback" bug at the same time (use an `Optional[float]` or a `success` flag, never overload `0.0`) | backend |
| AI-BE-19 | P1 | Add `command_timeout` and explicit `min_size/max_size` to `asyncpg.create_pool` in `main_ai.py:121` | backend |
| AI-BE-04 | P2 | Decide on policy for logging API-key prefixes; tighten or remove | backend |
| AI-BE-05 | P2 | Honor testnet flag in any (future) AI-side exchange path | backend |
| AI-BE-08 | P2 | Add CoinGecko Pro key (DB-backed) to `AIMarketAnalyzer._fetch_market_data` | backend |
| AI-BE-09 | P2 | Fix `RateLimiter` integer truncation (use float math) | backend |
| AI-BE-17 | P2 | Cost dashboard widget on `dashboard_ai.html` | backend |
| AI-BE-20 | P2 | Either wire `ai_trading_engine.py` into `main_ai.py` (full multi-chain orchestrator) or delete it (1098 LOC of dead code) | quant + backend |
| AI-BE-21 | P2 | Move `AIProviderManager.cache` to a Redis-backed cache so multiple AI workers can share | backend |
| AI-BE-22 | P2 | Add prompt-injection sanitizer once input sources include user-supplied content | backend |
| AI-BE-23 | P2 | Audit-log every LLM call with `request_id` to `security/audit_logger.py`; useful for compliance and incident forensics | backend |

---

## 10. Open questions

1. Is `ai_trading_engine.py` aspirational (a planned multi-chain orchestrator) or vestigial? It defines `Chain`, `AITradingEngine`, `AIStrategyGenerator`, `AIMarketAnalyzer`, `AIPerformanceTracker` (1098 LOC) but `main_ai.py` only loads `SentimentEngine`.
2. Is the AI module meant to ever sign exchange orders on its own, or is the design intent that AI emits signals to a downstream executor (Futures/DEX) only? My audit assumes the latter; recommendation AI-BE-02 / AI-BE-06 are written on that assumption.
3. Should we maintain a separate `ai_module_health_port` (e.g. 8082) for the metrics endpoint, or co-locate with the dashboard?
4. Is the hard-coded ETH symbol intentional (focused strategy on ETH alone) or a bug? If intentional, expose as a config; if bug, the AI module is generating ETH trades from news about every token.
5. Cohere/Perplexity/Mistral integration — is there a roadmap commitment, or should the brief's mention of those providers be dropped?
6. Daily $-budget of $10 (`ai_provider.py:160`) — what's the production-target spend? At gpt-4o-mini rates we get roughly 6700 sentiment calls/day for $10. Cycle every 15 minutes = 96 calls/day, well under. Budget seems generous; do we want to lower it as a guardrail?
