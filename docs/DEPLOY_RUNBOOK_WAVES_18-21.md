# ClaudeDex — Deploy Runbook: Waves 18–21

**What this covers:** one consolidated deploy of all branch work since your last VPS deploy —
Wave 18 (cleanup + disable losers), Wave 19 (Telegram redesign), Waves 20–21 (Financial Advisor module).

**Branch:** `claude/friendly-ramanujan-nMWNv`
**Migrations applied this deploy:** 053, 054, 055, 056, 057, 058, 060
  (048 and 059 don't exist — harmless numbering gaps; the runner is filename-ordered, not sequential.)
**Key fact:** `requirements.txt` changed (yfinance/anthropic/lightgbm/xgboost) → **you MUST `--build`**, not just restart.
**Safety:** everything new is **default-OFF / opt-in**. Deploying changes nothing about your running
trading modules until you flip the specific flags in steps 4–6.

`PG()` helper (paste once per shell session):
```bash
PG() { docker exec trading-postgres psql -U "$(docker exec trading-postgres cat /run/secrets/db_user)" -d tradingbot -P pager=off "$@"; }
```

---

## STEP 1 — Deploy code + migrations
```bash
cd ~/claudedex
git pull origin claude/friendly-ramanujan-nMWNv
docker compose down && docker compose up --build -d      # --build is REQUIRED (new pip deps)
docker exec trading-bot python scripts/migrate_database.py
```
**Expect:** migrations 053–058 + 060 each `✅ Applied`, summary `Failed: 0`, and
`docker compose ps` shows `trading-bot` **Up (healthy)`. If the migrator reports any failure,
STOP and paste the output before continuing.

---

## STEP 2 — (Wave 18) Confirm the losers are neutralized
Migration 055 forced sniper + arbitrage allocation budgets to 0. Verify:
```bash
PG -c "SELECT key,value FROM config_settings WHERE config_type='allocation_guard_config' AND key IN ('budget_usd_sniper','budget_usd_arbitrage');"
# expect both = 0
```
**Optional — fully stop their subprocess overhead** (budget=0 stops their trades; these stop the processes):
edit `.env`:
```
SNIPER_MODULE_ENABLED=false
ARB_MODULE_ENABLED=false
```
then `docker compose down && docker compose up -d`.

---

## STEP 3 — Surgical data clean (only modules that fundamentally changed)
Wave 16/17/18 changed sniper (delayed-entry) and AI (multi-symbol + reversal) enough that old rows
are not comparable. KEEP solana_trades / trades(DEX) / futures_trades — those are baselines + ML data.
```bash
PG -c "TRUNCATE sniper_trades;"
PG -c "DELETE FROM ai_trades;"
```
Then clear logs (W18/19 hygiene means they'll stay readable now):
```bash
docker exec trading-bot sh -c 'find /app/logs -name "*.log" -exec truncate -s 0 {} \;'
```

---

## STEP 4 — (Wave 19) Activate the Telegram redesign  [OPTIONAL — legacy DM mode works until you do this]
Full guide: `docs/TELEGRAM_SETUP.md`. Short version:
1. Create the supergroup, enable **Topics** (forum mode), add your bot as admin (Manage Topics + Post Messages).
2. Create topics: DEX, Futures, Solana, AI, Sniper, Arbitrage, Copy, Full Dashboard, Summary, Error.
3. Get the group id (`-100…`) and each topic's `message_thread_id` via:
   `curl https://api.telegram.org/bot<TOKEN>/getUpdates` (read `message_thread_id` per topic).
4. Enter them in **Dashboard → Telegram Alerts** (`/telegram/settings`), or SQL:
```bash
PG -c "UPDATE config_settings SET value='-1001234567890' WHERE config_type='telegram_config' AND key='telegram_group_id';"
PG -c "UPDATE config_settings SET value='<thread_id>' WHERE config_type='telegram_config' AND key='topic_thread_id_dex';"
# ...repeat for futures/solana/ai/sniper/arbitrage/copy + dashboard/summary/error
```
5. Verify a test message lands in each topic. **Until `telegram_group_id` is set, the bot stays in
   legacy single-DM mode — nothing breaks.** Solana defaults to summary-only (anti-spam).

---

## STEP 5 — (Waves 20–21) Activate the Financial Advisor module  [OPTIONAL — default OFF]
Advice-only. Fully isolated. Won't touch your trading bot.
1. `.env`: `ADVISOR_MODULE_ENABLED=true`
2. **Secure Credentials** (Dashboard → Secure Credentials, or the credentials flow):
   - `ADVISOR_ANTHROPIC_API_KEY` — can reuse your working Anthropic key
   - `ADVISOR_TELEGRAM_BOT_TOKEN` + `ADVISOR_TELEGRAM_CHAT_ID` — a NEW dedicated bot (separate from trading bot)
3. `docker compose down && docker compose up -d` (no `--build` needed if step 1 already built)
4. Visit **`/advisor/dashboard`**. US equities + Crypto + FX work immediately (free data);
   BIST + Midas funds show DEGRADED/NOT_CONFIGURED badges until paid keys are added.
5. Confirm advisor health: `curl -s localhost:8086/health` (or check the dashboard).

**Optional advisor upgrades (later):**
- FX real-time: `ADVISOR_FX_ALPHAVANTAGE_KEY` + set `advisor_fx_data_source='alphavantage'`
- BIST full: `ADVISOR_BIST_API_KEY` (Matriks `_fetch_matriks` wiring is a Wave-22 task)
- Kronos forecasts: `pip`-side `torch>=2.0 transformers>=4.38 huggingface_hub`, then
  `python scripts/download_kronos_weights.py --variant mini --dir /data/kronos`,
  set `ADVISOR_KRONOS_WEIGHTS_PATH=/data/kronos/Kronos-mini`,
  `PG -c "UPDATE config_settings SET value='true' WHERE config_type='advisor_config' AND key='advisor_kronos_enabled';"`, restart.
- Advisor ML: accumulate ≥50 closed sims, then `python scripts/retrain_advisor_ml.py`,
  then set `advisor_ml_enabled=true`.

---

## STEP 6 — Post-deploy verification (what each fix should now show)
```bash
# Sniper: no crash spam, budget-zero suppression, watchlist demoted to DEBUG
docker exec trading-bot sh -c 'grep -iE "SNIPER|budget=0" logs/sniper/sniper.log | tail -10'
# AI: opens BTC/ETH/SOL, reverses on strong opposite signal, PnL card non-zero
docker exec trading-bot sh -c 'grep -iE "ai-reversal|Signal|position_cap" logs/ai_analysis/ai.log | tail -15'
# Arbitrage: thin_pool_artifact now DEBUG (quiet at INFO), idle by design
# Futures: funding-gate refusals throttled to 1/symbol/5min
# Copy: rate-limit WARNING throttled to 1/5min
# Solana + DEX: should be the only active earners — watch their trade flow
```

---

## STEP 7 — Run a clean window, then send Round-6 data
After ~24h:
```bash
bash docs/agents/wave13/wave13_db_queries.sh
```
Send that output + any error logs. This window measures: gated/disabled sniper+arb are quiet,
AI reversal works, Solana+DEX hold their edge, and (if enabled) the advisor produces advice + sims.

---

## Rollback (if anything goes wrong)
All migrations are idempotent and additive; config changes are reversible. To revert behavior:
- Re-enable arb/sniper budgets: `PG -c "UPDATE config_settings SET value='150' WHERE config_type='allocation_guard_config' AND key='budget_usd_sniper';"` (etc.)
- Turn advisor off: `.env ADVISOR_MODULE_ENABLED=false` + restart.
- Telegram: clear `telegram_group_id` → reverts to legacy DM.
Code rollback: `git checkout <prior-sha> && docker compose up --build -d` (migrations already applied are harmless to leave).
