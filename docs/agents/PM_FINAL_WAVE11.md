# PM Final — Wave 11 (2026-05-28)
Branch: `claude/create-expert-agents-JFSF5`
Verdict: **SHIP**

## Per-commit verdict

| SHA | Area | Verdict | Notes |
|---|---|---|---|
| `bc63b6f` | dashboard charts cap | VERIFIED | per-table SQL has `WHERE exit_ts >= $since` + `LIMIT $per_table_limit` on every branch incl. Solana; sniper+arb gated by `include_noisy`; 45s in-process cache keyed `(since_days, per_table_limit, include_noisy)`; 5MB hard-truncate trims `labels`/`datasets[].data` to last 500 and sets `meta._truncated`. Cache cardinality capped at 32. Compiles clean. |
| `f2c51ca` | telegram single-poller | VERIFIED | Only `start_polling` / `_poll_updates` gated on `module_name == poll_owner`; `sendMessage` and `_api_call` untouched. Default `TELEGRAM_POLL_OWNER=dashboard` means trading subprocesses do not poll → 409 spam ends immediately. All 7 `main_*.py` pass `module_name=` to `get_telegram_controller`. |
| `85388e0` | honeypot None-guard | VERIFIED | `__init__` falls back to `{}` API keys when `config_manager is None`; no gate logic loosened (GoPlus public + on-chain heuristics still run). Fixes the AI/Copy "RiskManager init failed" stderr line. |
| `03d7be0` | core engine tz | VERIFIED | `_as_utc` helper + 9 sites (155, 1068, 1696, 1730, 1732, 1820, 2006, 2117, 2363) all wrap `position['entry_time']` / `record.closed_at`. Remaining bare `datetime.now() - ...` at `core/engine.py:2664` is `stats['start_time']` (naive, process-local). |
| `8627953` | solana engine tz | VERIFIED + hygiene note | `_as_utc` helper + 9 `_as_utc(position.opened_at)` / `_as_utc(trade.closed_at)` sites. Remaining bare `datetime.now() - ...` matches are all process-local timers (`loss_block_start`, `failed_rpcs[url]`, `_price_cache_time`, `_last_fetch_time`, `_recent_tokens.created_at`) initialized in the same module with naive `datetime.now()` — naive↔naive. **Hygiene note:** this commit also accidentally swept the ARB 429 backoff ladder (`arbitrage_engine.py` + `modules/arbitrage/CLAUDE.md`) due to a shared-tree git-index race. ARB bit is correct: `429` / `too many requests` / `rate limit` triple-check, jittered exponential `[30, 60, 120, 300]` ladder, calls `pool_engine.report_rate_limit` (fail-soft), streak resets on success or non-rate-limit error. Mixed attribution accepted (reverting would force-push, forbidden). |
| `5bdd129` | ARB `self.chain` rename | VERIFIED | `grep -nE "self\.chain\b" modules/arbitrage/*.py` returns 0. Only `self.chain_name`, `self.chain_id`, `self.chain_config` remain. |
| `8b67fe2` | DEX derive wallet | VERIFIED | `_resolve_wallet_address` derives via `eth_account.Account.from_key(...).address`; stored address is reference-only; on mismatch `self.wallet_address_secret_mismatch=True` + CRITICAL log with masked addresses; derived address wins. Health endpoint (line 222) and `dex_runtime_stats` surface `wallet_address_secret_mismatch` + `wallet_address_stored`. Operator will see correct derived address. |
| `c07653f` | sniper wallets via executor | VERIFIED | `_persist_runtime_stats` now reads `getattr(self.executor, 'solana_wallet', None)` / `evm_wallet`; `_persist_runtime_stats()` called at end of `initialize()` so wallets surface to dashboard immediately. Masked-address confirmation log on startup. |
| `6fb1f50` | copy derive wallet | FIXED-BY-PM | Derivation + CRITICAL log shipped, but executor never exposed `wallet_address_secret_mismatch` / `stored_evm_wallet` attributes that `_persist_execution_wallets` could persist for the dashboard to read. Dashboard (`enhanced_dashboard.py:14673`) already reads these keys defensively. PM commit adds the two `self.*` attrs + extra UPSERT rows so COPY mismatch warning renders the same as DEX. |
| `ec8afbb` | AI heartbeat | VERIFIED | Migration 033 idempotent (`CREATE TABLE IF NOT EXISTS` + `INSERT ... ON CONFLICT DO NOTHING`, `id=1` PK with `CHECK (id=1)`). `_persist_ai_heartbeat` UPSERTs id=1 with `delegates_to='futures'` + diagnostics JSONB, fail-soft via `logger.debug` in `except`. Called at TOP of `run()` cycle **before** `_fetch_news()` (line 541), so news outages don't suppress liveness. |
| `0824fcd` | dashboard funding+modules | VERIFIED | AI branch reads `EXTRACT(EPOCH FROM (NOW() - updated_at))::int FROM ai_runtime_stats WHERE id=1` ≤1800s = healthy, with `to_regclass('public.ai_runtime_stats')` guard for migration-missing → `MIGRATION_MISSING` hint. `dex_solana` row reads `SOLANA_WALLET` (separate from `SOLANA_MODULE_WALLET`). DEX funding panel reads BOTH the derived address and `wallet_address_secret_mismatch` flag, renders WARNING badge with stale stored. Sniper total via `COUNT(*) FILTER (...)` aggregate (no rowset materialized). |

## 22MB cap estimate vs operator's actual data
Operator has: 671 DEX + 2437 Solana + 121 futures + 10 copy closed trades.
With defaults (`since_days=7`, `per_table_limit=5000`, `include_noisy=False`):
- Per-module SELECTs each `LIMIT 5000` AND `WHERE exit_ts >= now - 7d`; nothing approaches the cap.
- Unified rowset ≤ `671 + 2437 + 121 + 10 ≈ 3.2K` rows (sniper + arb excluded; 7-day window further trims).
- Each row ~150 bytes serialised → unified ~480 KB raw payload.
- 13 aggregate charts + 2 downsampled-to-500-pt equity/drawdown series.
- **Predicted response size: well under 1 MB.** No truncation; cache hit on every poll within 45s.
- 5 MB hard-ceiling is a backstop; even with `include_noisy=1 + per_table_limit=20000` (worst legal knobs) we are at ≤50K unified rows ≈ 7.5 MB pre-truncation — truncate fires, response capped at the per-chart 500-element ceiling, ends well under 5 MB. **Cap will hold.**

## DEX wallet-bug verdict
**YES — operator will see correct address `0x802fa1AD...` after rebuild.**
- `main_dex.py:_resolve_wallet_address` derives from `PRIVATE_KEY` via `eth_account.Account.from_key(...).address` and uses that address as `self.wallet_address` (line 756).
- Stored secret `WALLET_ADDRESS` is stored on `self.wallet_address_stored` and surfaced as a separate dashboard field — never as the wallet to fund.
- Health endpoint `/health` (line 222) returns the derived address as `wallet_address` plus `wallet_address_secret_mismatch=True` + `wallet_address_stored=<stale>`.
- Dashboard `/api/funding/accounts` (enhanced_dashboard.py:14500-14509) reads both, displays correct derived address as fundable, and renders the WARNING badge with the stored stale address.

## Safety invariants
`git diff bc63b6f~1..0824fcd` filtered for non-comment / non-docstring mutations of `should_skip_live`, `DRY_RUN`, `killswitch`, `SNIPER_SAFETY_CHECK_ENABLED`, `min_score`, `max_rug_prob`, `validate_trade`: **zero mutations**. All references are either comment text, documentation, or arguments forwarded unchanged.

## py_compile
All 16 changed files compile clean:
`monitoring/enhanced_dashboard.py`, `monitoring/telegram_bot.py`, `core/engine.py`, `modules/solana_trading/core/solana_engine.py`, `modules/arbitrage/arbitrage_engine.py`, `modules/dex_trading/main_dex.py`, `modules/sniper/core/sniper_engine.py`, `modules/sniper/main_sniper.py`, `modules/copy_trading/copy_engine.py`, `modules/copy_trading/main_copy.py`, `modules/ai_analysis/core/sentiment_engine.py`, `modules/ai_analysis/main_ai.py`, `modules/futures_trading/main_futures.py`, `modules/arbitrage/main_arbitrage.py`, `modules/solana_trading/main_solana.py`, `data/collectors/honeypot_checker.py`.

## Git hygiene
`git log --pretty=fuller` shows authored+committed by the same SHA on every Wave-11 commit; no force-push, no `--amend`, no `--no-verify`, no hook bypass.

## PM fix shipped this wave
`[pm] copy: persist wallet-mismatch flag to dashboard funding panel` — adds `self.stored_evm_wallet` + `self.wallet_address_secret_mismatch` on `CopyTradeExecutor`, then UPSERTs both into `config_settings('copytrading_diagnostics', ...)` so dashboard `/api/funding/accounts` renders the same WARNING badge for COPY that it already does for DEX. Closes a documented soft gap (dashboard had defensive read code with comment "if the copy_engine ever starts persisting it the same way DEX does").

## SHIP / DO-NOT-SHIP
**SHIP.** Operator can safely:
```
git pull && docker compose down && docker compose up -d --build && docker exec trading-bot python scripts/migrate_database.py
```
Migration 033 is idempotent. All safety primitives untouched. The four reported symptoms (22MB dashboard CPU loop, tz crashes, ARB AttributeError spam, telegram 409 conflicts, DEX wrong-address display) are fixed end-to-end.

## Wave-12 carry-over (DO NOT START)
- Real risk-feature collectors (the underlying data sources that would make the heuristic gates richer).
- `solana_positions.entry_usd` column (Wave-4 follow-up so the cross-module exposure aggregator stops nooping on open SOL positions).
- Dashboard wires for the four CT-Q-09/CT-Q-12 probation knobs (engine already consumes them via `ConfigManager`).
