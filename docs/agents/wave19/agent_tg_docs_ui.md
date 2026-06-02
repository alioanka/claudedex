# Wave-19 Agent Report: Telegram Docs + UI

**Agent:** tg-docs-ui
**Date:** 2026-06-02
**Branch:** claude/friendly-ramanujan-nMWNv

---

## Deliverables

### Task 1 — Dashboard UI

**New template:** `dashboard/templates/settings_telegram.html`
Route: `GET /telegram/settings` (renders `page='telegram_settings'`).

**New handlers in `monitoring/enhanced_dashboard.py`:**
- `_telegram_settings` — page render
- `api_get_telegram_settings` — `GET /api/telegram/settings`
- `api_save_telegram_settings` — `POST /api/telegram/settings`

**Nav entry added to `dashboard/templates/base.html`:**
"Telegram Alerts" item with `fab fa-telegram` icon, positioned between
Global Settings and RPC/API Config.

**Config keys rendered and saved (`config_type='telegram_config'`):**

| Key | Type | UI widget | Default |
|-----|------|-----------|---------|
| `notifications_enabled` | bool | toggle | false |
| `telegram_group_id` | text | text input | '' |
| `topic_thread_id_dex/futures/solana/ai/sniper/arbitrage/copy` | int | number input | null |
| `topic_thread_id_dashboard/summary/error` | int | number input | null |
| `dashboard_interval_hours` | int | number input | 3 |
| `summary_interval_hours` | int | number input | 6 |
| `error_dedup_window_s` | int | number input | 900 |
| `notify_dex/futures/ai/sniper/arbitrage/copy_mode` | string | dropdown | all |
| `notify_solana_mode` | string | dropdown | summary |

**Save behaviour:** null topic IDs are stripped before the POST so they do not
write empty rows that would shadow future migration 057 seeds.

**Test-send button:** not implemented as a live route. The template includes a
copy-pasteable curl snippet the operator can run manually, with a follow-up
note pointing to MASTER_BACKLOG.md. Rationale: wiring a live send-test button
requires calling the notification engine directly from the dashboard process,
which would couple two agents' code in the same file — violates the wave-19
split. The engine agent owns that surface.

**py_compile:** passed.

### Task 2 — Setup Guide

**New file:** `docs/TELEGRAM_SETUP.md`

Sections:
1. Create bot via @BotFather — token storage path (`TELEGRAM_BOT_TOKEN` in
   Secure Credentials vault, not `.env`).
2. Create supergroup, enable Topics (forum mode), add bot as admin.
3. Create the 10 topics with recommended names.
4. How to get each topic's `message_thread_id`: `getUpdates` curl with worked
   JSON example; desktop Copy Link method as alternative.
5. How to get the group ID from `getUpdates`.
6. How to enter config: dashboard UI (`/telegram/settings`) and SQL fallback
   with complete INSERT statements.
7. Per-module verbosity guidance — Solana `summary` default explained
   (~300-1000 msgs/2-3 days at `all`).
8. Verification: per-topic curl test; error de-dup smoke check.
9. Legacy DM fallback: blank group ID or blank topic IDs fall back to
   `TELEGRAM_CHAT_ID` single-DM.

---

## Deferred to Engine Agent

- Actual scheduler / dispatch logic that reads `telegram_config` from DB and
  routes messages to topic threads — that is the notification engine's concern
  (`monitoring/notification_engine.py`, owned by sibling agent).
- Migration 057 that seeds the `telegram_config` rows — sibling agent owns it.
- Live "Send Test Message" dashboard button that calls the engine — deferred;
  documented in MASTER_BACKLOG.md.
- `telegram_bot.py` was not modified (read-only for grounding).

---

## Files Changed

- `dashboard/templates/settings_telegram.html` — new
- `monitoring/enhanced_dashboard.py` — route registrations + 3 new handler
  methods (~85 lines net)
- `dashboard/templates/base.html` — nav entry added (7 lines)
- `docs/TELEGRAM_SETUP.md` — new
- `docs/agents/wave19/agent_tg_docs_ui.md` — this report
