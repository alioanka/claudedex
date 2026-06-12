"""
Wave-19 Telegram Notification Engine
=====================================
Central notification layer that routes messages to Telegram forum topics per module,
with rate-limiting/batching, error de-duplication, and periodic dashboard/summary jobs.

Public API
----------
notify(module, category, text, level='info')
    Resolve the right message_thread_id from config and call sendMessage with it.
    Categories: trade | summary | dashboard | error
    Levels: info | warning | error | critical

format_header(module, title)
    Return a consistent "EMOJI [MODULE] title" header string for MarkdownV2.

TelegramNotificationEngine.start_periodic_jobs(db_pool)
    Launch the background tasks that post the rich dashboard (3h) and compact
    summary (6h) messages.

Design notes
------------
- BACKWARD COMPATIBLE: when telegram_group_id is empty OR topic thread IDs are
  empty, the engine falls back to the existing single-chat_id behavior so nothing
  breaks for operators still on the old single-DM setup.
- Config is read from DB config_settings config_type='telegram_config' (migration 057).
  Config is cached for 5 minutes to avoid hammering the DB on every message.
- Error de-dup: in-memory dict keyed by sha256(module+error_text[:200]); suppresses
  repeated sends within error_dedup_window_s (default 900s). When the window expires
  the rollup "still occurring xN" is sent once.
- Rate-limiting: per-module per-category (trade) throttle in seconds from
  throttle_<module>_s config keys.
- Thread safety: all state is asyncio-loop-local; no cross-thread sharing.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger("NotificationEngine")


def _setup_notification_logging() -> None:
    """Give the notification engine its OWN log file instead of leaking into
    whichever module subprocess happens to call it (e.g. futures_errors.log).

    Writes to logs/notifications/notifications.log with the module/category
    context that each message now carries. propagate=False keeps these lines
    out of the host module's error log. Idempotent — safe to call repeatedly.
    """
    if getattr(logger, "_notif_handler_attached", False):
        return
    try:
        from logging.handlers import RotatingFileHandler
        from pathlib import Path
        log_dir = Path("logs/notifications")
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_dir / "notifications.log", maxBytes=5 * 1024 * 1024, backupCount=3,
            delay=True,
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # keep out of the host module's log files
        logger._notif_handler_attached = True  # type: ignore[attr-defined]
    except Exception:
        # Never let logging setup break notifications.
        pass

# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------

MODULE_EMOJI: Dict[str, str] = {
    "dex":       "📈",
    "futures":   "📊",
    "solana":    "◎",
    "ai":        "🧠",
    "sniper":    "🎯",
    "arbitrage": "⚖️",
    "copy":      "👯",
    "dashboard": "🖥️",
    "system":    "🔧",
}

_MDV2_SPECIAL = r'_*[]()~`>#+-=|{}.!'


def escape_mdv2(text: str) -> str:
    """Escape all MarkdownV2 special characters."""
    result = []
    for ch in str(text):
        if ch in _MDV2_SPECIAL:
            result.append('\\')
        result.append(ch)
    return ''.join(result)


def format_header(module: str, title: str) -> str:
    """
    Return a standardised MarkdownV2 header for every module notification.

    Format: <emoji> *[MODULE]* title
    Example: 📈 *[DEX]* Trade Entry
    """
    emoji = MODULE_EMOJI.get(module.lower(), "🔔")
    mod_upper = escape_mdv2(module.upper())
    title_escaped = escape_mdv2(title)
    return f"{emoji} *\\[{mod_upper}\\]* {title_escaped}"


# ---------------------------------------------------------------------------
# Config cache
# ---------------------------------------------------------------------------

_CONFIG_CACHE_TTL = 300  # 5 minutes


@dataclass
class _TgConfig:
    """Snapshot of telegram_config from DB."""
    notifications_enabled: bool = True
    telegram_group_id: str = ""
    topic_thread_ids: Dict[str, str] = field(default_factory=dict)
    dashboard_interval_hours: int = 3
    summary_interval_hours: int = 6
    error_dedup_window_s: int = 900
    notify_mode: Dict[str, str] = field(default_factory=dict)
    throttle_s: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Error de-dup state
# ---------------------------------------------------------------------------

@dataclass
class _ErrorEntry:
    first_sent: float
    last_sent: float
    count: int = 1


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class TelegramNotificationEngine:
    """
    Topic-routed Telegram notification engine.

    Instantiate once per process; share across all modules via get_engine().
    """

    def __init__(self, db_pool=None):
        _setup_notification_logging()
        self.db_pool = db_pool

        # Credentials are resolved lazily from secrets_manager/env on first use
        self._bot_token: Optional[str] = None
        self._chat_id: Optional[str] = None  # fallback DM chat_id

        # Config cache
        self._cfg: Optional[_TgConfig] = None
        self._cfg_loaded_at: float = 0.0

        # Error de-dup: key -> _ErrorEntry
        self._error_entries: Dict[str, _ErrorEntry] = {}

        # Per-module last-sent timestamps for trade throttle
        # key: "<module>:<category>"
        self._last_sent: Dict[str, float] = {}

        # Background periodic jobs
        self._dashboard_task: Optional[asyncio.Task] = None
        self._summary_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Credential resolution
    # ------------------------------------------------------------------

    async def _resolve_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        """Resolve bot_token and chat_id via secrets_manager, then env."""
        if self._bot_token and self._chat_id:
            return self._bot_token, self._chat_id

        token: Optional[str] = None
        chat_id: Optional[str] = None

        try:
            from security.secrets_manager import SecureSecretsManager
            sm = SecureSecretsManager.get_instance()
            if self.db_pool:
                sm.initialize(self.db_pool)
            token = await sm.get_async('TELEGRAM_BOT_TOKEN', log_access=False)
            chat_id = await sm.get_async('TELEGRAM_CHAT_ID', log_access=False)
        except Exception:
            pass

        if not token:
            import os
            token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not chat_id:
            import os
            chat_id = os.getenv('TELEGRAM_CHAT_ID')

        if token:
            self._bot_token = token
        if chat_id:
            self._chat_id = chat_id

        return self._bot_token, self._chat_id

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    async def _load_config(self) -> _TgConfig:
        """Load telegram_config from DB, cached for _CONFIG_CACHE_TTL seconds."""
        now = time.monotonic()
        if self._cfg is not None and (now - self._cfg_loaded_at) < _CONFIG_CACHE_TTL:
            return self._cfg

        cfg = _TgConfig()

        if not self.db_pool:
            self._cfg = cfg
            self._cfg_loaded_at = now
            return cfg

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT key, value FROM config_settings "
                    "WHERE config_type = 'telegram_config'"
                )
            kv: Dict[str, str] = {r['key']: r['value'] for r in rows}

            def bval(k: str, default: bool) -> bool:
                v = kv.get(k, '').lower()
                return (v == 'true') if v in ('true', 'false') else default

            def ival(k: str, default: int) -> int:
                try:
                    return int(kv.get(k, default))
                except (ValueError, TypeError):
                    return default

            cfg.notifications_enabled = bval('notifications_enabled', True)
            cfg.telegram_group_id = kv.get('telegram_group_id', '').strip()
            cfg.dashboard_interval_hours = ival('dashboard_interval_hours', 3)
            cfg.summary_interval_hours = ival('summary_interval_hours', 6)
            cfg.error_dedup_window_s = ival('error_dedup_window_s', 900)

            topic_keys = [
                'dex', 'futures', 'solana', 'ai', 'sniper', 'arbitrage', 'copy',
                'dashboard', 'summary', 'error',
            ]
            for mod in topic_keys:
                val = kv.get(f'topic_thread_id_{mod}', '').strip()
                if val:
                    cfg.topic_thread_ids[mod] = val

            for mod in ('dex', 'futures', 'solana', 'ai', 'sniper', 'arbitrage', 'copy'):
                cfg.notify_mode[mod] = kv.get(f'notify_{mod}_mode', 'all').strip()
                cfg.throttle_s[mod] = ival(f'throttle_{mod}_s', 0)

        except Exception as e:
            logger.warning(f"NotificationEngine: failed to load config from DB: {e}")

        self._cfg = cfg
        self._cfg_loaded_at = now
        return cfg

    def invalidate_config_cache(self):
        """Force config reload on next use."""
        self._cfg_loaded_at = 0.0

    # ------------------------------------------------------------------
    # Core sendMessage
    # ------------------------------------------------------------------

    async def _send_raw(
        self,
        text: str,
        chat_id: str,
        thread_id: Optional[str] = None,
        parse_mode: str = 'MarkdownV2',
        *,
        ctx: str = '',
    ) -> bool:
        """Low-level sendMessage call with optional message_thread_id.

        If a thread_id is supplied but Telegram rejects it with
        'message thread not found' (the configured topic id is wrong or the
        group is not a forum), retry ONCE without the thread so the message
        still lands in the group's general topic instead of being lost +
        spamming the error log. `ctx` is a 'module/category' label for logs.
        """
        if not self._bot_token:
            return False

        # Truncate to Telegram's 4096-char limit
        if len(text) > 4000:
            text = text[:3990] + "\n\\.\\.\\.(truncated)"

        payload: Dict[str, Any] = {
            'chat_id': chat_id,
            'text': text,
            'parse_mode': parse_mode,
        }
        if thread_id:
            try:
                payload['message_thread_id'] = int(thread_id)
            except (ValueError, TypeError):
                pass

        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 200:
                        return True
                    body = await resp.text()
                    # Graceful fallback: bad topic thread id -> resend to the
                    # group's general topic (no thread) so the message is not
                    # lost. Logged once at INFO, not as a recurring WARNING.
                    if (
                        resp.status == 400
                        and 'message_thread_id' in payload
                        and 'message thread not found' in body.lower()
                    ):
                        logger.info(
                            "[%s] topic thread_id=%s not found in chat %s — "
                            "delivering to general topic instead. Fix "
                            "topic_thread_id_* in Telegram settings to route "
                            "per-module.",
                            ctx or 'notify', payload.get('message_thread_id'), chat_id,
                        )
                        payload.pop('message_thread_id', None)
                        async with session.post(
                            url, json=payload,
                            timeout=aiohttp.ClientTimeout(total=15)
                        ) as resp2:
                            if resp2.status == 200:
                                return True
                            body2 = await resp2.text()
                            logger.warning(
                                "[%s] Telegram sendMessage %s (no-thread retry): %s",
                                ctx or 'notify', resp2.status, body2[:200],
                            )
                            return False
                    logger.warning(
                        "[%s] Telegram sendMessage %s: %s",
                        ctx or 'notify', resp.status, body[:200],
                    )
                    return False
        except Exception as e:
            logger.warning("[%s] Telegram sendMessage error: %s", ctx or 'notify', e)
            return False

    # ------------------------------------------------------------------
    # Topic routing
    # ------------------------------------------------------------------

    async def _resolve_target(
        self,
        module: str,
        category: str,
        cfg: _TgConfig,
        chat_id_fallback: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Return (chat_id, thread_id) for (module, category).

        Priority:
          1. Group + per-topic thread_id
          2. Group with no thread (general group chat)
          3. Single DM chat_id (backward compat)
        """
        mod_lower = module.lower()

        # Determine which topic key to look up
        if category == 'error':
            topic_key = 'error'
        elif category in ('dashboard', 'summary'):
            topic_key = category
        else:
            topic_key = mod_lower

        if cfg.telegram_group_id:
            thread_id = cfg.topic_thread_ids.get(topic_key)
            return cfg.telegram_group_id, thread_id

        # Fallback: original single DM
        return chat_id_fallback, None

    # ------------------------------------------------------------------
    # Rate limiting / verbosity check
    # ------------------------------------------------------------------

    def _should_suppress(
        self,
        module: str,
        category: str,
        cfg: _TgConfig,
    ) -> bool:
        """
        Return True if this message should be suppressed based on verbosity
        mode or throttle settings.
        """
        mod_lower = module.lower()

        # Errors always pass (de-dup is separate)
        if category == 'error':
            return False

        # Check verbosity mode
        mode = cfg.notify_mode.get(mod_lower, 'all')
        if mode == 'off':
            return True
        if mode == 'summary' and category == 'trade':
            return True  # summary mode: individual trades suppressed

        # Check throttle for trade category
        if category == 'trade':
            throttle = cfg.throttle_s.get(mod_lower, 0)
            if throttle > 0:
                key = f"{mod_lower}:trade"
                last = self._last_sent.get(key, 0.0)
                if (time.monotonic() - last) < throttle:
                    return True
                self._last_sent[key] = time.monotonic()

        return False

    # ------------------------------------------------------------------
    # Error de-dup
    # ------------------------------------------------------------------

    def _should_suppress_error(
        self,
        module: str,
        text: str,
        dedup_window_s: int,
    ) -> Tuple[bool, int]:
        """
        Return (suppress, repeat_count).
        suppress=True  means skip sending (still within dedup window).
        When the window has expired, return suppress=False and count>1 so
        the caller can add a "still occurring xN" suffix.
        """
        sig = hashlib.sha256(
            f"{module.lower()}:{text[:200]}".encode()
        ).hexdigest()[:16]

        now = time.monotonic()
        entry = self._error_entries.get(sig)

        if entry is None:
            self._error_entries[sig] = _ErrorEntry(first_sent=now, last_sent=now, count=1)
            return False, 1

        entry.count += 1

        if (now - entry.last_sent) < dedup_window_s:
            # Still within window — suppress
            return True, entry.count

        # Window expired — allow a rollup send, reset last_sent
        entry.last_sent = now
        return False, entry.count

    # ------------------------------------------------------------------
    # Public notify()
    # ------------------------------------------------------------------

    async def notify(
        self,
        module: str,
        category: str,
        text: str,
        level: str = 'info',
    ) -> bool:
        """
        Send a notification for the given module and category.

        Parameters
        ----------
        module   : module name (dex, futures, solana, ai, sniper, arbitrage, copy,
                   dashboard, system, …)
        category : trade | summary | dashboard | error
        text     : MarkdownV2-formatted message body (WITHOUT the header —
                   format_header() is applied automatically for trade/error)
        level    : info | warning | error | critical

        Returns True if the message was sent.
        """
        bot_token, chat_id_fb = await self._resolve_credentials()
        if not bot_token:
            return False

        cfg = await self._load_config()
        if not cfg.notifications_enabled:
            return False

        # Error de-dup
        if category == 'error':
            suppress, count = self._should_suppress_error(
                module, text, cfg.error_dedup_window_s
            )
            if suppress:
                return False
            # If count > 1 it's a rollup; append note
            if count > 1:
                text = text + f"\n\n_\\(still occurring — seen {escape_mdv2(str(count))} times\\)_"
        else:
            # Verbosity / throttle gate
            if self._should_suppress(module, category, cfg):
                return False

        # Resolve target chat + thread
        target_chat, thread_id = await self._resolve_target(
            module, category, cfg, chat_id_fb
        )
        if not target_chat:
            return False

        sent = await self._send_raw(
            text, target_chat, thread_id, ctx=f"{module}/{category}"
        )

        # Audit trail: record every dispatched notification (what / where).
        # Previously notifications.log only ever showed "periodic jobs
        # started" because successful sends were never logged — only
        # failures were WARNINGs. Now each delivery is an INFO line with
        # module, category, level, chat + topic thread. Fail-soft: logging
        # must never affect the notification result.
        try:
            if sent:
                logger.info(
                    "SENT module=%s category=%s level=%s chat=%s thread=%s",
                    module, category, level, target_chat,
                    thread_id if thread_id else "general",
                )
            else:
                logger.info(
                    "NOT-SENT (suppressed/failed) module=%s category=%s level=%s",
                    module, category, level,
                )
        except Exception:
            pass

        return sent

    # ------------------------------------------------------------------
    # Convenience: send_notification (wraps header + body)
    # ------------------------------------------------------------------

    async def send_notification(
        self,
        module: str,
        category: str,
        title: str,
        body: str,
        level: str = 'info',
    ) -> bool:
        """
        Build a standardised header + body message and route via notify().

        title and body should be plain text; escaping is applied here.
        For fully custom MarkdownV2 (e.g. from existing alert helpers),
        use notify() directly with a pre-escaped string.
        """
        header = format_header(module, title)
        body_escaped = escape_mdv2(body)
        ts = escape_mdv2(
            datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        )
        separator = "\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-"
        text = f"{header}\n{separator}\n{body_escaped}\n\n_⏰ {ts}_"
        return await self.notify(module, category, text, level)

    # ------------------------------------------------------------------
    # Periodic jobs (dashboard process only)
    # ------------------------------------------------------------------

    async def start_periodic_jobs(self):
        """
        Launch background asyncio tasks for the periodic dashboard (3h) and
        summary (6h) messages.  Call from the dashboard process only.
        """
        if self._dashboard_task is None or self._dashboard_task.done():
            self._dashboard_task = asyncio.create_task(
                self._periodic_dashboard_loop(), name="tg_dashboard_loop"
            )
        if self._summary_task is None or self._summary_task.done():
            self._summary_task = asyncio.create_task(
                self._periodic_summary_loop(), name="tg_summary_loop"
            )
        logger.info("NotificationEngine: periodic jobs started")

    async def stop_periodic_jobs(self):
        """Cancel background periodic tasks."""
        for task in (self._dashboard_task, self._summary_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("NotificationEngine: periodic jobs stopped")

    async def _periodic_dashboard_loop(self):
        """Post full dashboard message every dashboard_interval_hours."""
        while True:
            try:
                cfg = await self._load_config()
                interval_s = max(1, cfg.dashboard_interval_hours) * 3600
                await asyncio.sleep(interval_s)
                await self._post_full_dashboard()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dashboard loop error: {e}")
                await asyncio.sleep(300)

    async def _periodic_summary_loop(self):
        """Post compact summary message every summary_interval_hours."""
        while True:
            try:
                cfg = await self._load_config()
                interval_s = max(1, cfg.summary_interval_hours) * 3600
                await asyncio.sleep(interval_s)
                await self._post_compact_summary()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Summary loop error: {e}")
                await asyncio.sleep(300)

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    async def _fetch_module_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Query per-module WR/PnL/last-5-trades for dashboard messages.
        Returns dict keyed by module name.
        """
        stats: Dict[str, Dict[str, Any]] = {}
        if not self.db_pool:
            return stats

        # Module -> (table, pnl_col, ts_col, closed_filter).
        #
        # IMPORTANT: column/table names below are the SCHEMA-VERIFIED names
        # taken from each module's INSERT statements and the dashboard's own
        # summary queries (monitoring/enhanced_dashboard.py:api_dashboard_summary).
        # The previous mapping guessed `entry_timestamp` + `pnl_usd`/`profit_usd`
        # for every table, but the real columns differ per module:
        #   - trades (DEX)  : profit_loss / entry_timestamp (status='closed')
        #   - solana_trades : pnl_sol  / entry_time      (no status col)
        #   - futures_trades: pnl      / entry_time
        #   - sniper_trades : profit_loss / entry_timestamp (status='closed')
        #   - arbitrage_trades  : profit_loss / entry_timestamp
        #   - copytrading_trades: profit_loss / entry_timestamp (status='closed')
        #   - ai_trades     : profit_loss / entry_timestamp (status='closed')
        # NOTE: DEX writes to the legacy `trades` table — `dex_trades` does
        # NOT exist. The previous schema-verification pass dropped DEX
        # entirely when `dex_trades` failed to verify, which is why DEX was
        # the one module missing from the Full-Dashboard and Summary digests.
        # The old mapping made EVERY query raise (caught at debug) -> empty
        # stats -> the periodic summary/dashboard builders returned early and
        # NOTHING was ever posted to the summary/dashboard topics.
        #
        # closed_filter is an optional extra WHERE clause (already prefixed
        # with AND) restricting to settled trades so PnL is meaningful.
        module_tables = {
            'dex':       ('trades',             'profit_loss', 'entry_timestamp', " AND status = 'closed'"),
            'futures':   ('futures_trades',     'pnl',         'entry_time',      ''),
            'solana':    ('solana_trades',      'pnl_sol',     'entry_time',      " AND exit_time IS NOT NULL"),
            'ai':        ('ai_trades',          'profit_loss', 'entry_timestamp', " AND status = 'closed'"),
            'sniper':    ('sniper_trades',      'profit_loss', 'entry_timestamp', " AND status = 'closed'"),
            'arbitrage': ('arbitrage_trades',   'profit_loss', 'entry_timestamp', ''),
            'copy':      ('copytrading_trades', 'profit_loss', 'entry_timestamp', " AND status = 'closed'"),
        }

        # EVERY module is pre-seeded with a zero placeholder BEFORE its query
        # runs, so a per-module query error (e.g. the legacy `trades` table not
        # yet created in this DB, a transient DB hiccup, or a schema drift) can
        # NEVER silently drop a module from the digest — the exact failure mode
        # that kept DEX out of the Summary + Full-Dashboard topics. A module
        # whose query fails renders an honest "data unavailable" line instead of
        # vanishing; a module with no closed trades renders a "0 trades" line.
        for mod in module_tables:
            stats[mod] = {
                'total_trades': 0, 'wins': 0, 'wr_pct': 0.0,
                'total_pnl': 0.0, 'pnl_24h': 0.0, 'streak': '',
                'available': True,
            }

        for mod, (tbl, pnl_col, ts_col, closed) in module_tables.items():
            try:
                async with self.db_pool.acquire() as conn:
                    row = await conn.fetchrow(f"""
                        SELECT
                            COUNT(*) AS total_trades,
                            COALESCE(SUM(CASE WHEN {pnl_col} > 0 THEN 1 ELSE 0 END), 0) AS wins,
                            COALESCE(SUM({pnl_col}), 0)::float AS total_pnl,
                            COALESCE(SUM(CASE WHEN {ts_col} > NOW() - INTERVAL '24 hours'
                                         THEN {pnl_col} ELSE 0 END), 0)::float AS pnl_24h
                        FROM {tbl}
                        WHERE {ts_col} IS NOT NULL{closed}
                    """)
                    recent = await conn.fetch(f"""
                        SELECT {pnl_col} AS pnl
                        FROM {tbl}
                        WHERE {ts_col} IS NOT NULL{closed}
                        ORDER BY {ts_col} DESC LIMIT 5
                    """)

                if row:
                    total = int(row['total_trades'] or 0)
                    wins = int(row['wins'] or 0)
                    wr = (wins / total * 100) if total > 0 else 0.0
                    last5 = [float(r['pnl'] or 0) for r in recent]
                    streak = ''.join('W' if p > 0 else 'L' for p in last5)
                    stats[mod] = {
                        'total_trades':  total,
                        'wins':          wins,
                        'wr_pct':        round(wr, 1),
                        'total_pnl':     round(float(row['total_pnl'] or 0), 2),
                        'pnl_24h':       round(float(row['pnl_24h'] or 0), 2),
                        'streak':        streak,
                        'available':     True,
                    }
            except Exception as e:
                # Keep the pre-seeded placeholder but mark it unavailable so the
                # module STILL appears in the digest (with an honest note) rather
                # than being dropped. WARNING-level: a recurring line here points
                # at a missing/renamed table for that module.
                stats[mod]['available'] = False
                logger.warning(f"Stats query failed for {mod} ({tbl}): {e}")

        return stats

    # ------------------------------------------------------------------
    # Dashboard/summary message builders
    # ------------------------------------------------------------------

    async def _post_full_dashboard(self):
        """Build and send the rich full-dashboard message."""
        stats = await self._fetch_module_stats()
        if not stats:
            return

        now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        lines: List[str] = [
            "🖥️ *\\[FULL DASHBOARD\\]* All Modules Summary",
            escape_mdv2("─" * 28),
            "",
        ]

        total_pnl_all = 0.0
        pnl_24h_all = 0.0
        for mod, s in sorted(stats.items(), key=lambda x: -x[1]['total_pnl']):
            emoji = MODULE_EMOJI.get(mod, "🔔")
            mod_upper = escape_mdv2(mod.upper())
            if not s.get('available', True):
                # Module's trade table couldn't be queried this cycle — show it
                # honestly rather than dropping it from the digest entirely.
                lines.append(f"{emoji} *{mod_upper}* — {escape_mdv2('data unavailable')}")
                lines.append("")
                continue
            wr = s['wr_pct']
            wr_str = escape_mdv2(f"{wr:.1f}%")
            tp = escape_mdv2(f"${s['total_pnl']:+.2f}")
            p24 = escape_mdv2(f"${s['pnl_24h']:+.2f}")
            trades = escape_mdv2(str(s['total_trades']))
            streak = escape_mdv2(s['streak'] or "N/A")
            lines.append(
                f"{emoji} *{mod_upper}* — {trades} trades \\| WR: {wr_str}"
            )
            lines.append(f"   PnL all\\-time: {tp} \\| 24h: {p24}")
            lines.append(f"   Last 5: `{streak}`")
            lines.append("")
            total_pnl_all += s['total_pnl']
            pnl_24h_all += s['pnl_24h']

        lines.append(escape_mdv2("─" * 28))
        lines.append(
            f"💰 *TOTAL PnL* all\\-time: {escape_mdv2(f'${total_pnl_all:+.2f}')} "
            f"\\| 24h: {escape_mdv2(f'${pnl_24h_all:+.2f}')}"
        )
        lines.append(f"_⏰ {escape_mdv2(now_str)}_")

        text = "\n".join(lines)
        await self.notify("dashboard", "dashboard", text, level="info")

    async def _post_compact_summary(self):
        """Build and send the compact summary message."""
        stats = await self._fetch_module_stats()
        if not stats:
            return

        now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        lines: List[str] = [
            "📋 *\\[SUMMARY\\]* Module Performance",
            "",
        ]

        for mod, s in sorted(stats.items(), key=lambda x: -x[1]['pnl_24h']):
            emoji = MODULE_EMOJI.get(mod, "🔔")
            mod_upper = escape_mdv2(mod.upper())
            if not s.get('available', True):
                lines.append(f"{emoji} *{mod_upper}* {escape_mdv2('data unavailable')}")
                continue
            wr_str = escape_mdv2(f"{s['wr_pct']:.1f}%")
            p24 = escape_mdv2(f"${s['pnl_24h']:+.2f}")
            trades = escape_mdv2(str(s['total_trades']))
            lines.append(
                f"{emoji} *{mod_upper}* {trades}T \\| WR {wr_str} \\| 24h {p24}"
            )

        lines.append("")
        lines.append(f"_⏰ {escape_mdv2(now_str)}_")

        text = "\n".join(lines)
        await self.notify("dashboard", "summary", text, level="info")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine_instance: Optional[TelegramNotificationEngine] = None


def get_engine(db_pool=None) -> TelegramNotificationEngine:
    """
    Return the process-wide TelegramNotificationEngine singleton.

    Pass db_pool on first call (typically from main_dashboard or the module's
    main_*.py) to enable DB-backed config and stats.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TelegramNotificationEngine(db_pool=db_pool)
    elif db_pool is not None and _engine_instance.db_pool is None:
        _engine_instance.db_pool = db_pool
    return _engine_instance
