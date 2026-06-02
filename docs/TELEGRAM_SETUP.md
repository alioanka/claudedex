# Telegram Notifications Setup Guide

This guide walks you through connecting ClaudeDex to a Telegram supergroup with
forum topics so that each trading module posts to its own dedicated thread.
Estimated time: 20-30 minutes.

---

## Prerequisites

- A Telegram account with the desktop or mobile client installed.
- Admin access to the ClaudeDex dashboard (`/telegram/settings`).
- The bot token from step 1 below.

---

## Step 1 — Create the Bot via @BotFather

1. Open Telegram and search for `@BotFather`.
2. Send `/newbot`.
3. Choose a display name (e.g. `ClaudeDex Alerts`) and a username ending in `bot`
   (e.g. `claudedex_alerts_bot`).
4. BotFather replies with a token that looks like:
   ```
   5839201744:AAHq4wMfExampleTokenHerexxxxxxxxxxxxxxx
   ```
   Copy it — you will not see it again without `/token`.

**Where to store the token:**
The token goes in the Secure Credentials vault, NOT in `.env`.
Navigate to **Dashboard → Secure Credentials** (`/credentials`) and add the key
`TELEGRAM_BOT_TOKEN` with that value. The dashboard reads it via
`SecureSecretsManager` (the same resolution path used by the AI module for
`ANTHROPIC_API_KEY` — see `modules/ai_analysis/CLAUDE.md` for the pattern).

Never paste the token into the Telegram Settings form; that form only stores
routing configuration (group ID, topic IDs, verbosity).

---

## Step 2 — Create the Supergroup and Enable Forum Topics

1. In Telegram, tap the pencil / compose icon and choose **New Group**.
2. Add at least one other account (you can remove them later), set a name such as
   `ClaudeDex Notifications`, and create the group.
3. Upgrade to supergroup if needed: **Group Info → Edit → Supergroup** (this
   happens automatically on most clients when you enable Topics).
4. Enable **Topics (Forum mode)**:
   - Desktop: Group Info → Edit (pencil) → scroll to **Topics** → toggle on.
   - Mobile: Group Info → Edit → Topics → Enable.
5. Add the bot as an administrator:
   - Group Info → Administrators → Add Administrator → search for your bot's
     username.
   - Grant at minimum: **Manage Topics** and **Post Messages**.

---

## Step 3 — Create the Topics

Inside the group, create one topic for each routing target. Suggested names and
intended content:

| Topic name       | Content |
|------------------|---------|
| DEX Module       | DEX trade entries, exits, scan results |
| Futures Module   | Futures position opens, closes, funding events |
| Solana Module    | Solana spot trade events (recommend `summary` verbosity) |
| AI Module        | AI sentiment signals and trade executions |
| Sniper Module    | Sniper entries, safety checks, exits |
| Arbitrage Module | Arbitrage opportunity executions |
| Copy Module      | Copy-trade entries and exits |
| Full Dashboard   | Periodic full-dashboard snapshots (every N hours) |
| Summary          | Cross-module P&L digest (every N hours) |
| Error            | ERROR-level events from every module (de-duplicated) |

To create a topic: tap the **+** (or pencil) inside the group, choose **New Topic**,
enter the name, pick an icon/color, and confirm.

---

## Step 4 — Get Each Topic's `message_thread_id`

Every forum topic has a numeric ID (also called `message_thread_id`). You need
this to route messages correctly. There are two methods:

### Method A — getUpdates (most reliable)

1. Add your bot to the group as admin (step 2 already done).
2. Post a message in the topic you want to identify (e.g. type "test" in the
   **DEX Module** topic).
3. Run this curl command in a terminal, replacing `<TOKEN>` with your bot token:

   ```bash
   curl "https://api.telegram.org/bot<TOKEN>/getUpdates"
   ```

4. The response is JSON. Look for the message you just sent. The key
   `message_thread_id` is the topic ID:

   ```json
   {
     "ok": true,
     "result": [
       {
         "update_id": 123456789,
         "message": {
           "message_id": 42,
           "message_thread_id": 7,
           "from": { "id": 9876543210, "username": "you" },
           "chat": {
             "id": -1001234567890,
             "title": "ClaudeDex Notifications",
             "type": "supergroup"
           },
           "text": "test"
         }
       }
     ]
   }
   ```

   In this example:
   - `message.chat.id` = `-1001234567890` — this is your **group ID** (step 5).
   - `message.message_thread_id` = `7` — this is the **topic ID** to enter for
     the DEX Module field.

   Repeat for each topic: post a test message, run getUpdates, note the ID.

### Method B — Copy Link (desktop only)

1. Right-click a message inside the topic.
2. Choose **Copy Message Link**.
3. The link looks like: `https://t.me/c/1234567890/7/42`
   - The second number (`7`) is the `message_thread_id`.
   - The first number (`1234567890`) is the group ID without the `-100` prefix
     — so the full group ID is `-1001234567890`.

---

## Step 5 — Get the Group ID

If you used method A above you already have the group ID from `message.chat.id`
(e.g. `-1001234567890`). The group ID always starts with `-100` for supergroups.

Alternatively:

1. Post any message in the **General** topic of your group.
2. Run `getUpdates` as shown in step 4.
3. Read `message.chat.id` from the result — this is the group ID.

---

## Step 6 — Enter the Configuration

### Via the Dashboard (recommended)

1. Navigate to **Telegram Alerts** in the sidebar (`/telegram/settings`).
2. Enter:
   - **Telegram Group ID** — the `-100...` value from step 5.
   - One **Topic Thread ID** per module topic from step 4.
   - **Dashboard interval**, **Summary interval**, **Error dedup window** if
     you want non-default values.
   - Per-module **verbosity** settings (see step 7).
3. Click **Save Changes**.

### Via SQL (fallback — useful for scripted deployments)

Replace placeholder values with your real IDs:

```sql
INSERT INTO config_settings (config_type, key, value, value_type)
VALUES
  ('telegram_config', 'notifications_enabled',    'true',               'bool'),
  ('telegram_config', 'telegram_group_id',         '-1001234567890',    'string'),
  ('telegram_config', 'topic_thread_id_dex',        '2',                'int'),
  ('telegram_config', 'topic_thread_id_futures',    '3',                'int'),
  ('telegram_config', 'topic_thread_id_solana',     '4',                'int'),
  ('telegram_config', 'topic_thread_id_ai',         '5',                'int'),
  ('telegram_config', 'topic_thread_id_sniper',     '6',                'int'),
  ('telegram_config', 'topic_thread_id_arbitrage',  '7',                'int'),
  ('telegram_config', 'topic_thread_id_copy',       '8',                'int'),
  ('telegram_config', 'topic_thread_id_dashboard',  '9',                'int'),
  ('telegram_config', 'topic_thread_id_summary',    '10',               'int'),
  ('telegram_config', 'topic_thread_id_error',      '11',               'int'),
  ('telegram_config', 'dashboard_interval_hours',   '3',                'int'),
  ('telegram_config', 'summary_interval_hours',     '6',                'int'),
  ('telegram_config', 'error_dedup_window_s',       '900',              'int'),
  ('telegram_config', 'notify_dex_mode',            'all',              'string'),
  ('telegram_config', 'notify_futures_mode',        'all',              'string'),
  ('telegram_config', 'notify_solana_mode',         'summary',          'string'),
  ('telegram_config', 'notify_ai_mode',             'all',              'string'),
  ('telegram_config', 'notify_sniper_mode',         'all',              'string'),
  ('telegram_config', 'notify_arbitrage_mode',      'all',              'string'),
  ('telegram_config', 'notify_copy_mode',           'all',              'string')
ON CONFLICT (config_type, key) DO UPDATE SET value = EXCLUDED.value, value_type = EXCLUDED.value_type;
```

---

## Step 7 — Per-Module Verbosity

The `notify_<module>_mode` key controls how much each module sends:

| Mode      | Behaviour |
|-----------|-----------|
| `all`     | Every trade entry, exit, and signal is posted immediately. |
| `summary` | Only periodic summaries are posted (hourly or at the configured interval). |
| `off`     | No messages for this module, even if a topic ID is configured. |

**Why Solana defaults to `summary`:**
On an active bot, the Solana module can generate roughly 300-1 000 Jupiter trade
events over a 2-3 day window. At `all` verbosity, that saturates the topic thread
and makes it unreadable. `summary` mode collapses those into a periodic P&L digest.

If you are running Solana in DRY_RUN or paper-trade mode with a very low frequency
strategy, `all` may be acceptable.

---

## Step 8 — Verification

### Test each topic manually

For each topic ID you configured, send a test message:

```bash
TOKEN="5839201744:AAHq4wMfExampleTokenHerexxxxxxxxxxxxxxx"
GROUP_ID="-1001234567890"
TOPIC_ID="7"     # Change for each topic

curl "https://api.telegram.org/bot${TOKEN}/sendMessage" \
  -d "chat_id=${GROUP_ID}" \
  -d "message_thread_id=${TOPIC_ID}" \
  -d "text=ClaudeDex test: DEX Module topic confirmed"
```

A successful response looks like:

```json
{
  "ok": true,
  "result": {
    "message_id": 55,
    "message_thread_id": 7,
    "chat": { "id": -1001234567890, ... },
    "text": "ClaudeDex test: DEX Module topic confirmed"
  }
}
```

If `ok` is `false`, common error responses:
- `"Bad Request: message thread not found"` — topic ID is wrong or the topic was
  deleted. Re-run `getUpdates` to confirm the ID.
- `"Forbidden: bot is not a member of the supergroup chat"` — the bot was not
  added as admin. Repeat step 2.
- `"Unauthorized"` — the bot token is wrong. Check the Secure Credentials vault.

### Verify error de-duplication

Intentionally trigger the same warning twice within the de-dup window
(`error_dedup_window_s`, default 900 s). Only the first occurrence should appear
in the Error topic; the second should be silently dropped. Check the dashboard
logs (`logs/dashboard/`) for `[tg-dedup] suppressed` log entries confirming it
worked.

---

## Step 9 — Legacy Single-DM Fallback

If `telegram_group_id` is blank **or** all topic thread IDs are left empty, the
notification engine falls back to the original behaviour: it sends all messages
to the single chat configured via `TELEGRAM_CHAT_ID` in the Secure Credentials
vault (the same key used by `monitoring/telegram_bot.py`). There is no group or
topic setup required in this mode.

This means you can adopt topic routing gradually: configure only the topics you
care about first, and leave the rest blank.

---

## Troubleshooting

| Symptom | Check |
|---------|-------|
| No messages in any topic | `notifications_enabled` is `true`; bot token is valid; group ID is correct |
| Messages go to wrong topic | Re-run `getUpdates` and confirm the `message_thread_id` matches what is in the dashboard |
| Bot sends to General instead of topic | The group is not in forum/topics mode — repeat step 2 |
| `"ETELEGRAM: 400 Bad Request"` in logs | Group ID or topic ID is incorrect |
| Duplicate errors still appearing | Increase `error_dedup_window_s`; check the key is saved in DB with `SELECT value FROM config_settings WHERE config_type='telegram_config' AND key='error_dedup_window_s'` |
| Token missing at runtime | Check `SELECT value FROM secure_credentials WHERE key='TELEGRAM_BOT_TOKEN'`; re-enter via `/credentials` if blank |

---

*Last updated: wave-19 (2026-06-02). Owned by the docs/dashboard engineer.*
