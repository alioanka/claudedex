# Wave-13 Agent 8 — AI Advisor Module Audit

## Module: modules/ai_analysis/

### Existing Edge Audit

The module's stated edge is LLM-driven sentiment from crypto news headlines, producing a directional float in [-1, 1] that gates Binance Futures entries. The pipeline is structurally sound: headline fetch -> sanitize -> LLM prompt -> score -> threshold gate -> BinanceFuturesExecutor. Previous waves (6–12) closed secrets resolution, executor delegation, prompt injection, and heartbeat observability.

Edge source: aggregate news sentiment provides a 15-minute leading signal on directional bias for BTC/ETH, exploited via a single-asset mean-reversion on high-confidence extremes.

### Bugs Found (Wave-12 Live Logs)

**BUG-1: Claude model 404 (DEAD PROVIDER)**

`ai_provider.py` line 41 and `sentiment_engine.py` line 620 hardcoded `claude-3-5-haiku-latest`. This alias returns `not_found_error` on accounts without haiku tier, causing a silent fallback to OpenAI. The dashboard showed "Provider: CLAUDE" even though every call went to OpenAI.

Fix shipped (commit `6f999d9`):
- `sentiment_engine._claude_model` / `._openai_model` are now instance attrs with defaults `claude-3-5-sonnet-20241022` / `gpt-4o-mini`, loaded from DB keys `claude_model` / `openai_model` in `ai_config`.
- `_call_llm_provider` uses `self._claude_model` (not hardcode).
- On HTTP 404 + `not_found_error`: emits a loud `WARNING` with the corrective DB query instead of silently falling back.
- `ai_provider.py` AIModel enum updated from `-latest` aliases to dated snapshots (`claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`, `claude-3-opus-20240229`).
- `_get_model` warns on unknown model IDs rather than silently returning GPT4O_MINI.

**BUG-2: News source failures**

- `cryptopanic.com/api/v1/posts/?public=true` returns HTTP 404 as of 2026 — dead endpoint, removed.
- CryptoCompare returned HTTP 200 with 0 parseable headlines in some cases; parser audited (was correct but missing explicit `isinstance(a, dict)` guard, now added).
- All source failures were previously at `logger.debug`, invisible in production logs.

Fix shipped: dead public CryptoPanic dropped; CoinTelegraph RSS added as 4th fallback; all failures now `logger.warning`; CryptoCompare parser hardened.

**BUG-3: Confidence threshold 0.50 blocks all trades**

Live LLM confidence clusters at 0.30–0.40. The 0.50 default caused 0 trades for months. The "0.5" comment in the old code referenced the dashboard *display* threshold (score >= 0.5 shown as BUY/SELL badge), not the execution gate.

Fix shipped: default lowered to 0.35. Configurable via DB key `confidence_threshold` in `ai_config`.

### Biases Found

- **Single-asset fixed symbol**: `_execute_trade` hardcodes `symbol = "ETH"`. Scores derived from broad market news (BTC dominance, regulatory headlines) are being applied to ETH only — a mismatch that can flip the sign of the trade. This is a known limitation (not fixed this wave); flagged for operator awareness.
- **15-minute cycle vs news velocity**: headlines used in cycle N may be stale by cycle N+1 given 15-minute sleep. No deduplication of already-analyzed headlines — the same 10 CoinDesk items could be re-analyzed until the next RSS refresh. Should cache by headline hash.
- **Response caching in AIProviderManager is prompt-level** (SHA256 of full prompt), which means the same 10 headlines (slightly reordered) get a fresh LLM call. Acceptable for now.
- **No outcome labeling**: sentiment signals are not labeled with next-period returns for calibration. The `ai_confidence_calibration` table exists (A6 E2) but is only written on position close. Without forward-fill of returns, we cannot retrain on it.

### Missing Features / Proposed Signals (Ranked)

1. **Headline deduplication** (P0 — free alpha leak fix): cache analyzed headline sets by sorted SHA256; skip LLM call if identical batch was analyzed within the last 2 cycles. Saves API cost and prevents the same stale news from locking the score at a stale level. Implementation: ~20 lines, no DB change.

2. **Per-symbol sentiment routing** (P1 — edge improvement): build a keyword-to-symbol map (`{"bitcoin", "BTC" -> "BTC"}, {"ethereum", "ETH" -> "ETH"}`). Filter headlines by symbol before LLM call; run separate score per symbol. Currently ETH trades on BTC news. Requires dashboard config for the target symbol list.

3. **Fear & Greed Index integration** (P1 — uncorrelated signal): `alternative.me/fng/` returns a 0–100 composite daily. Incorporate as a second feature alongside LLM score. High FGI (>75) + negative LLM score = strong short signal; cross-asset confirmation reduces false positives. Zero extra LLM cost.

4. **Confidence calibration feedback loop** (P2 — requires Agent 2 / ML): the `ai_confidence_calibration` table records entry score + realized PnL on close. A logistic regression on (score, abs(score), FGI, headline_count) -> (profitable: bool) would give a calibrated probability to replace the raw LLM output. Needs Agent 2 to add a retrain script. Flag: shared-model need.

5. **Multi-model ensemble** (P3 — diminishing returns): OpenAI and Claude already run in quorum mode (A6 E1). Adding a third provider (e.g. local Ollama mistral-7b as zero-cost tiebreaker) would require adding it to AIProviderManager — acceptable but only after calibration data exists.

### Profitability Checklist

- Edge: aggregate crypto news sentiment as a 15-minute leading indicator for BTC/ETH direction.
- Fee/slippage/MEV: Binance Futures has 0.02% maker / 0.04% taker. At $50 notional: ~$0.02 round-trip. Sentiment score needs to predict >0.04% move to cover fees. Given 15-min hold, this is achievable on high-confidence signals (|score| > 0.6).
- Sharpe/hit-rate: no live backtest available; confidence calibration table has 0 rows (no trades executed). After fixing bug-3 (threshold), operator should allow 1–2 weeks of dry-run data collection before enabling `direct_trading`.
- Capital allocation: fixed $50 USD per trade (config `trade_amount_usd`). Kelly sizing would require a calibrated win-rate — not yet available.
- Kill-switch: `logs/.killswitch` + `logs/.pause_ai`. Also: `max_hold_hours=24` auto-close, and `stop_loss_pct=-3.0`.

### DB-QUERY Block

```sql
-- 1. Discover AI config table schema
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'ai_runtime_stats'
ORDER BY ordinal_position;

-- 2. Show all current AI config settings (no keys)
SELECT key, value, value_type, updated_at
FROM config_settings
WHERE config_type = 'ai_config'
ORDER BY key;

-- 3. Check the configured Claude model (the root cause of BUG-1)
SELECT key, value
FROM config_settings
WHERE config_type = 'ai_config'
  AND key IN ('claude_model', 'openai_model', 'ai_provider', 'confidence_threshold');
-- Expected after wave-13: claude_model row absent (engine will use default
-- 'claude-3-5-sonnet-20241022') OR explicitly set. If 'claude-3-5-haiku-latest'
-- appears here, update it:
--   UPDATE config_settings SET value = 'claude-3-5-sonnet-20241022'
--   WHERE config_type = 'ai_config' AND key = 'claude_model';

-- 4. How many signals generated vs trades opened (last 7 days)?
SELECT
  COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '7 days') AS signals_7d,
  MIN(score) AS min_score,
  MAX(score) AS max_score,
  AVG(ABS(score)) AS avg_abs_score
FROM sentiment_logs
WHERE timestamp > NOW() - INTERVAL '7 days';

-- 5. AI trades table — any executed?
SELECT COUNT(*), MIN(entry_timestamp), MAX(entry_timestamp)
FROM ai_trades
WHERE entry_timestamp > NOW() - INTERVAL '30 days';

-- 6. Confidence calibration table (A6 E2) — any rows?
SELECT COUNT(*), AVG(confidence_score), AVG(realized_pnl_pct)
FROM ai_confidence_calibration;

-- 7. Last AI heartbeat
SELECT updated_at, stats->>'cycle' AS cycle,
       stats->>'has_anthropic' AS has_claude,
       stats->>'has_openai' AS has_openai,
       stats->>'direct_trading' AS direct_trading,
       stats->>'last_skip_reason' AS last_skip
FROM ai_runtime_stats
WHERE id = 1;
```

### Cross-Module / Dashboard Handoffs

**Dashboard agent (required)**:
- Add `claude_model` and `openai_model` to the AI Settings page (`/ai/settings` or `/settings/ai`). These are new `ai_config` DB keys with text inputs. Default display values: `claude-3-5-sonnet-20241022` and `gpt-4o-mini`. Note: changing `claude_model` clears the bad-model warn flag so the probe fires fresh next cycle.
- The `confidence_threshold` field already exists in the UI; verify it shows decimal (0.35) not percentage (35) form, because the DB stores a decimal.
- Add a "Claude model health" badge: green if last 10 Anthropic calls returned 200, red if any returned 404+not_found_error. Source: `ai_analysis_logs` table, `model` + `success` columns.

**Agent 2 (ML) flag**:
- The `ai_confidence_calibration` table needs a retrain script: logistic regression on (abs_score, fgi, headline_count) -> profitable_bool. Script would live at `scripts/retrain_ai_calibration.py`. Outcome rows are written by `_write_calibration_close` on position close (requires `direct_trading=true` first).

### Wave-13 Commits

- `6f999d9` — `[ai] wave-13: fix Claude model id, news sources, confidence threshold`
  - Files: `modules/ai_analysis/core/sentiment_engine.py`, `modules/ai_analysis/core/ai_provider.py`
  - Validation: syntax-checked via `ast.parse`; no backtest available (no live trades to backtest against); changes are defensive (wider model compatibility, more news sources, lower threshold to match observed signal distribution).
