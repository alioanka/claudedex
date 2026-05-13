# DASHBOARD_MODULE — Operator-UX / Kill-Switch / Risk-Surfacing Audit

Owner of this view: market-trading-analyst (secondary; PRIMARY per PLAN.md is backend-devops-expert). Angle: **operator-UX, kill-switch wiring, alerts surfacing, drawdown surfacing, DRY_RUN visibility**. Not infra.
Scope: `modules/dashboard/main_dashboard.py`, `monitoring/{module_routes,alerts,telegram_bot}.py`, `dashboard/templates/{base,dashboard,module_control,pro_controls,global_settings,positions_*,dashboard_*,settings_*,logs}.html`, `dashboard/static/js/{dashboard,modules,websocket}.js`, `core/risk_manager.py` (status surface).

## Executive verdict: AMBER → RED for live trading

There is a real dashboard with real controls — an always-visible Emergency Exit button (`base.html:1024`), per-module enable/disable/pause/start UI (`module_control.html:287-296`), and an Emergency Exit API endpoint (`module_routes.py:76, 852-883`). Telegram backstop exists with `/stop_all` and `/emergency` (`telegram_bot.py:124-137`). On surface the kill-switch story is OK.

Behind that surface, several controls are theatrical:

1. **Standalone dashboard is decoupled from live trading state.** `main_dashboard.py:138-151` instantiates `DashboardEndpoints(..., risk_manager=None, portfolio_manager=None, alerts_system=None, trading_engine=None)`. Comments admit "Will show as 'Module Offline'". Dashboard reads everything from the database directly. Result: every "live" widget shows whatever the modules **last wrote** to DB — there is no socket between dashboard and the running engine. If a module hangs but the DB still serves yesterday's row, dashboard shows green.
2. **No always-visible DRY_RUN / LIVE indicator.** `grep -nE "dry|DRY|paper" base.html` returns one hit (the simulator nav link). The main dashboard does not display whether the bot is in dry-run. The copy-trading sub-dashboard hides its DRY_RUN badge by default and only reveals it if at least one trade row has `dry_run=true` (`dashboard_copytrading.html:429-430, 643-644`). Operator can absolutely flip `DRY_RUN=false` in `.env`, restart, and the dashboard will not visibly distinguish from paper mode.
3. **Per-module pause is cosmetic.** `base_module.py:286-298` `pause()`/`resume()` only flip `self.status = ModuleStatus.PAUSED`. Engines do not poll `self.status` inside their main loops. Copy-trading (`copy_engine.py:572` `while self.is_running:`) never checks paused state. Hitting "Pause" on the UI changes the badge from green to orange while orders continue to fire. **P0 lie.**
4. **Pro-controls panic button URL mismatch.** `pro_controls.html:145` posts to `/api/bot/emergency_exit` (underscore). The actual endpoint registered in `module_routes.py:76` is `/api/bot/emergency-exit` (dash). 404. The big red "🚨 PANIC SELL ALL 🚨" button does **nothing** except show a misleading success alert. **P0.**
5. **Emergency exit skips copy_trading silently.** `bot_emergency_exit` (`module_routes.py:852-883`) iterates `module.get_positions()` and `module.close_position(position)`. Copy-trading's engine implements neither at the engine level. The base class will return empty positions for it, then call `module.stop()`. Net effect: emergency-exit closes EVM/Futures/Solana positions but leaves copy-trading mirror positions wide open. **P0 for cross-module flatten.**
6. **No DRY_RUN-aware widget on /positions pages.** `positions_copytrading.html` displays unrealized P&L (line 381) but no stop-loss, no take-profit, no close-position button, no DRY_RUN tag. Operator can't see *whether the P&L number is real*. The base.html dashboard's `Max Drawdown` (line 358) is a single all-time number with no day/week/month breakdown and no freeze-threshold overlay.
7. **No real-time P&L latency widget.** `grep -nE "last_update|stale|heartbeat|latency"` across base.html, dashboard.js, websocket.js → zero hits. There is no "data is 2s old vs 30s old" indicator. WebSocket reconnect happens silently (`websocket.js:9-19`); reconnect failure displays a single toast and stops trying after `MAX_RECONNECT_ATTEMPTS` (line 59-60). After that the dashboard is frozen and operator may not notice.
8. **No trade-rejection surface.** Risk-manager denials, RPC failures, slippage-exceeded errors all log to file via `monitoring/alerts.py`. Critical-priority alerts pop a toast on the dashboard (`websocket.js:277-281`), but there is no persistent rejection panel, no count, no module breakdown. If a circuit-breaker is tripped (`core/risk_manager.py:281-282`), the breaker status surface (`get_circuit_breaker_status` line 352) is exposed by the engine but **not consumed by any dashboard widget** (no `circuit_breaker` element in any template).
9. **Alerts have no acknowledgement workflow.** The `read` flag on alerts (`websocket.js:271`) is client-side only — refreshing the page resets it. There is no DB-backed "I have seen this alert, do not show again" workflow. Critical alerts are toasts that vanish.
10. **Global settings page is a raw key-value editor.** `global_settings.html:36-46` renders every config key as a string input. No validation, no help text per key, no grouping by risk severity (knobs that move millions sit next to ones that change a label). One mistyped digit on `breaker.max_drawdown_pct` and the bot loses its main safety net silently.

For a paper-mode monitoring console the dashboard is acceptable. For live capital it is **not yet operator-grade**; the lies around pause and the dead panic-sell button are the kind of issues that turn a 10% drawdown into a 50% drawdown because the operator thought trading was halted when it wasn't.

## DRY_RUN propagation audit (operator-visibility angle)

| Surface | Where | DRY_RUN visible? | Notes |
|---|---|---|---|
| Top nav (always visible) | `base.html:1000-1028` | NO | Start/Stop/Restart/Emergency-Exit buttons present. No mode badge. |
| Main dashboard hero card | `dashboard.html` (no match) | NO | The big numbers (Total P&L, drawdown, Sharpe) display the same regardless of mode. |
| Per-module sub-dashboard hero | `dashboard_copytrading.html:429-430` | PARTIAL (hidden) | `<span class="mode-badge mode-dry" id="modeBadge" style="display: none;">` — revealed by JS only if recent trades have `is_simulated=true`. Empty state shows nothing. |
| Per-module sub-dashboard hero | `dashboard_futures.html`, `dashboard_solana.html`, etc. | UNVERIFIED | Pattern likely similar but not audited here. |
| Module control page (`/module-control`) | `module_control.html` | NO | Shows ENABLED/RUNNING/PAUSED/DISABLED, never LIVE vs DRY. |
| Settings forms | `settings_copytrading.html:27-31` | YES (per-module toggle) | The dry-run checkbox exists in the form but **the engine does not honor DB-write reads at runtime for copy_trading** — see COPY_TRADING_analyst.md item CT-RM-04. |
| Telegram `/status` | `telegram_bot.py:_cmd_status` | UNVERIFIED | Probably reports module status, unsure if it includes DRY_RUN. **P1: verify.** |
| Backend module init logging | `copy_engine.py:555` | LOG ONLY | Engines log `Mode: DRY_RUN (Simulated) / LIVE TRADING` but only to logfile. |

**Recommendation P0:** add a sticky badge in `base.html` top-bar (next to Emergency Exit) that reads `MODE: LIVE` or `MODE: DRY-RUN` driven by `/api/system/mode` consolidating every module's effective `dry_run`. Red background for LIVE, green for DRY. Mismatch (some modules live, some paper) shows orange "MIXED" badge with a tooltip listing the modules per state.

## Kill-switch wiring

### Emergency Exit (panic-sell + halt)

| Component | File:Line | Status |
|---|---|---|
| Sidebar button (every page) | `base.html:1023-1028` | OK |
| `controlBot('emergency')` JS | `base.html:1126-1150` | OK — uses `confirm()` (browser native) and posts to `/api/bot/emergency-exit` |
| Endpoint registration | `module_routes.py:76` | OK |
| Handler `bot_emergency_exit` | `module_routes.py:852-883` | PARTIAL — iterates `module.get_positions()` and `module.close_position(position)`; works for engines that implement these. Copy-trading engine does not implement `close_position` (see COPY_TRADING_analyst.md, item CT-RM-05) so its mirror positions stay open. |
| Pro-controls panic | `pro_controls.html:142-147` | **BROKEN** — calls `/api/bot/emergency_exit` (underscore). 404. |
| Telegram backstop | `telegram_bot.py:_cmd_emergency` line 768 | OK — falls through to `_cmd_stop_all` |

The user reaching for emergency exit during a real incident may be on the operator desktop *or* on mobile. Two of three entry-points are healthy. The pro-controls one is a tripwire — fixing the URL is a one-character change.

### Per-module freeze (stop new entries, hold positions)

This is the more important control during drawdown. The persona-required behavior is "freeze new entries but keep existing positions for managed exit". The dashboard exposes a Pause button (`module_control.html:295-296`) hitting `POST /api/modules/{name}/pause`. The handler `pause_module` (`module_routes.py:469-500`) calls `self.module_manager.pause_module(name)` → `BaseModule.pause()` (`base_module.py:286-291`) which **only** sets `self.status = ModuleStatus.PAUSED`. Neither `copy_engine.py:run()` nor any other engine's main loop checks `self.status`. Pause is a UI lie.

What "freeze" should actually do per persona:
- Stop new entries (block any signal that would open a new position).
- Hold existing positions; keep monitoring SL/TP.
- Continue heartbeats to dashboard so the operator can see the engine still alive.

What "pause" currently does: cosmetic status change. **P0 ops fix:** rename to "Freeze (no new entries)" + actually wire each engine to consult `self.status == PAUSED` and short-circuit entry logic. Alternative: keep "pause" as a hard halt (process keeps running but loop sleeps); add a separate "freeze" semantic.

### Kill-switch reachability matrix

| Surface | Emergency-Exit | Per-module Stop | Per-module Freeze | Telegram |
|---|---|---|---|---|
| Sidebar (every page) | YES | NO (need to go to /module-control) | NO | N/A |
| Module Control page | YES | YES | LIE | N/A |
| Pro Controls page | BROKEN | YES (start/stop) | NO | N/A |
| Positions pages | NO | NO | NO | N/A |
| Mobile (sidebar) | YES via overlay | NO (must drill in) | NO | N/A |
| Telegram | YES (`/stop_all`, `/emergency`) | YES (`/stop <module>`) | NO | N/A |

Per-position close buttons are missing on every positions page (`positions_copytrading.html` only renders P&L stats, no actions). Operator who wants to close *one* bad position cannot do so from the dashboard — must wait for module-wide stop. **P1.**

## Drawdown surfacing

`dashboard.html:357-358` shows a single "Max Drawdown" metric in the Risk Metrics card. That's it. The persona requirement: "today, week, month, all-time, with the freeze-thresholds marked".

Current state:
- No today-day DD widget.
- No weekly DD widget.
- No monthly DD widget.
- No freeze-threshold overlay (config has `breaker.max_drawdown_pct=15` and `breaker.max_daily_loss_pct=10` per `core/risk_manager.py:288-289` but these are never rendered alongside the actual DD).
- No per-module DD breakdown.
- No circuit-breaker status surface (`get_circuit_breaker_status` exists at `core/risk_manager.py:352-370` but no template consumes it).

**P0 ops add:** four cards on the main dashboard — Today DD, Week DD, Month DD, All-time DD — each with a horizontal bar showing current DD vs threshold (orange when ≥50% of threshold, red when ≥80%, hard-red flashing when tripped). Also a circuit-breaker status pill that shows tripped state, reason, and time-until-auto-reset.

## Alert surfacing

`monitoring/alerts.py:130-1320` is a fully-featured alerts service: priorities (INFO/WARNING/CRITICAL), channels (Telegram, Discord, Email, Webhook, Slack, SMS, Pushover), rate-limiting, aggregation. It includes alert types `DRAWDOWN_ALERT`, `STOP_LOSS_HIT`, `TRAILING_STOP_UPDATED`, and a queue worker (`_process_alerts` line 541). Solid backend.

On the dashboard side:
- `websocket.js:27` subscribes to `alerts_update`; `:255-281` displays them in a notification panel and pops a toast for `priority === 'critical'`.
- The notification badge (`websocket.js:269-273`) shows unread count.
- No persistent alert table — no `/alerts` HTML page.
- No filtering by module / type / priority.
- No acknowledge workflow. Refresh and counters reset.
- No "snooze" or "mute" per alert type.
- No backend record of which alerts were seen vs unseen by operator.

**P0:** persist a `dashboard_alerts` table with `seen_at`, `acknowledged_at`, `acknowledged_by` fields. Add an `/alerts` page with filters and a per-row acknowledge button.

**P1:** add an "Active Issues" pill in the top-bar that shows count of unacknowledged CRITICAL alerts. Click → opens alert panel.

**P1:** add a *trade-rejection* feed surfacing risk-manager denials, slippage-exceeded errors, and RPC failures distinctly from "trading alerts". Right now these are silent log lines unless the dev wired a specific `alerts_system.send_alert` call. `core/risk_manager.py:281-282` returns `(False, reason)` but the caller (likely `core/engine.py`) decides what to do with it.

## Operator-UX micro-issues

1. **`confirm()` modal is browser-native** (`base.html:1135`). Some kiosk browsers / iframes block native `confirm`. There is a `custom-modal` div defined (`base.html:1058-1072`) but never used by the bot-control buttons. Inconsistent.
2. **Emergency Exit confirmation message** (`base.html:1132`) is one line of text. No explicit listing of "you are about to close N positions across M modules totalling $X". Operator under stress wants to see exposure before pulling the trigger.
3. **Alert toasts auto-dismiss** with no log of recently-shown toasts.
4. **No on-call mode banner** for night-time / weekend operations.
5. **Logs page** (`logs.html:1-42`) is server-fetch + search-box. Cannot confirm from template alone whether it streams live (likely uses `logs.js` polling). Filter by level exists; filter by module does not.
6. **Mobile responsiveness** — `base.html` includes `media (max-width: 768px)` in some sections but the bot-control sidebar `nav-control` items don't have explicit mobile styling. Sidebar overlay exists (`base.html:1033`) so mobile collapse works. Adequate but not battle-tested.
7. **Discovery_copytrading page** (`discovery_copytrading.html`) and `wallets_copytrading.html` exist but the engine never consumes their data — discovery is a UI without a backend job.
8. **Global settings editor renders raw strings** (`global_settings.html:36-46`) including booleans as checkboxes — no numeric validation. Operator setting `max_drawdown_pct` to `1.5` instead of `15` shrinks the safety net 10x with no warning.

## Profit-leak / loss-leak inventory (dashboard-driven)

1. **Pause-button lie** — operator thinks trading is frozen, places no follow-up SL adjustment, trades continue. Realized losses during pause window. *Loss-leak.*
2. **Broken pro-controls panic button** — operator thinks ALL positions are closing, takes phone call, returns to find positions still open. *Catastrophic loss-leak in crisis.*
3. **Emergency-exit skips copy_trading** — mirror positions remain after operator believes bot is flat. *Loss-leak proportional to mirror size.*
4. **No DRY_RUN indicator** — operator flips live, forgets it, runs a "test" that fires real orders. Or vice-versa: thinks live, is paper, misses moving market. *Both directions of loss-leak.*
5. **No real-time P&L staleness widget** — DB row is 30s old; operator reacts to old numbers. *Slippage on operator decisions.*
6. **No drawdown ladder + threshold overlay** — DD blows through hidden freeze-line; operator unaware. *Loss-leak as breaker trips silently.*
7. **No trade-rejection feed** — risk-denied trades are invisible. Operator does not know risk-manager is rejecting their alpha; or worse, not denying it when it should. *Either misses edge or runs through risk gate.*
8. **WebSocket silent failure after max-reconnect** — operator stares at dashboard showing yesterday's numbers. Trading continues in background. *Decision-blind.*
9. **No per-position close button** — must use module-wide stop to close one position. *Forces over-flat actions, leaving good positions to die with bad ones.*

## Live-readiness checklist (dashboard-angle)

- [ ] Always-visible Emergency Exit button — **YES** (`base.html:1024`).
- [ ] Confirmation modal — **YES** (`confirm()` — but native, not the custom modal).
- [ ] Per-module Freeze (no new entries, hold positions) — **LIE** (pause is cosmetic).
- [ ] Per-module Stop (close all + halt) — **PARTIAL** (works for some engines, skips copy_trading).
- [ ] Pro-controls panic button works — **NO** (URL bug).
- [ ] DRY_RUN/LIVE always visible — **NO**.
- [ ] Real-time P&L latency widget — **NO**.
- [ ] Drawdown today/week/month/all-time with thresholds — **NO** (only single max-DD).
- [ ] Circuit-breaker status surface — **NO** (data exists, no widget).
- [ ] Trade-rejection feed — **NO**.
- [ ] Alert acknowledge workflow — **NO**.
- [ ] WebSocket reconnect status visible — **PARTIAL** (toast on final failure, no persistent indicator).
- [ ] Mobile usable for on-call — **YES (adequate)**.
- [ ] Logs streaming + filter by level + module — **PARTIAL** (level yes, module no, streaming unverified).
- [ ] Per-position close button on positions pages — **NO**.

**4 of 15 green, 4 partial, 7 missing.** Not live-grade.

## Action backlog (ranked, ID prefix `DASH-RM-`)

P0 (blockers):
- **DASH-RM-01** — Fix pro-controls panic URL: change `/api/bot/emergency_exit` → `/api/bot/emergency-exit` (`pro_controls.html:145`). 1-line fix.
- **DASH-RM-02** — Wire per-engine `pause()` to actually freeze entries. Each engine's main loop must check `self.status == ModuleStatus.PAUSED` before opening any new position; existing positions continue to be monitored for SL/TP. Add a `frozen_at` field for audit.
- **DASH-RM-03** — Implement `get_positions()` and `close_position()` at `BaseModule` and override in `CopyTradingEngine` so `bot_emergency_exit` actually flattens copy-trading mirrors.
- **DASH-RM-04** — Always-visible MODE badge in `base.html` top-bar (LIVE/DRY-RUN/MIXED). New endpoint `/api/system/mode` aggregates per-module `dry_run`.
- **DASH-RM-05** — Drawdown panel on main dashboard: today, week, month, all-time, with threshold overlay from `core/risk_manager.py` config. Surface `get_circuit_breaker_status()`.
- **DASH-RM-06** — Trade-rejection feed on main dashboard. Subscribe to risk-manager denials + RPC failures + slippage exceeded.

P1 (live trading gate):
- **DASH-RM-07** — Alert acknowledge workflow. New DB table `dashboard_alerts` with `seen_at`, `acknowledged_by`. New `/alerts` page with filters and ack buttons.
- **DASH-RM-08** — Real-time P&L latency widget: timestamp of last db write, "stale > N seconds" red indicator.
- **DASH-RM-09** — Persistent WebSocket connection-status pill in top-bar (green/yellow/red + last-update time).
- **DASH-RM-10** — Per-position close button on `positions_*.html`. Endpoint `POST /api/positions/{id}/close`.
- **DASH-RM-11** — Replace the global-settings raw editor with a per-category form that validates numeric ranges, marks risk-critical keys in red, and requires "type RISK to confirm" for those.
- **DASH-RM-12** — Custom `confirmModal` used for Emergency Exit, with explicit listing of "N positions, $X exposure, M modules" before confirm.

P2 (polish):
- **DASH-RM-13** — Logs page: filter by module, level, time-range; tail-mode.
- **DASH-RM-14** — Mobile-first pass on `nav-control` buttons and positions tables. Add swipe-to-close on positions list.
- **DASH-RM-15** — On-call mode banner toggled per schedule.
- **DASH-RM-16** — Discovery → wallets workflow: wire `discovery_copytrading.html` to a backend job that scores wallets daily.

## Go-live sequence (operator-UX)

1. DASH-RM-01..03 ship — kill-switches actually work end-to-end. Smoke-test: pause module, verify zero new orders; emergency exit, verify every module flat (including copy-trading); pro-controls panic, verify endpoint response.
2. DASH-RM-04..06 ship — operator can see at a glance: mode, drawdown vs thresholds, why trades are being rejected.
3. DASH-RM-07..09 ship — operator can acknowledge alerts, knows when data is stale, knows when WS dies.
4. DASH-RM-10..12 ship — single-position close + custom confirm + risk-knob guards.
5. Tabletop incident drill: simulated 8% intra-day drawdown → operator clicks Pause → verify entries halt within 1 polling cycle → click "Close All in DEX" → verify positions flat → review alert acknowledgement audit trail.

## Open questions

1. Is the dashboard intended to be the **primary control surface** or a **read-only monitor with Telegram for actions**? Current state is half-and-half. Choose explicitly.
2. Should drawdown be tracked against deployed capital or NAV including unrealized P&L? Today's max-DD widget (`dashboard.html:1509`) reads `hist.max_drawdown` — origin of that number is unclear; needs traceability.
3. Should the dashboard authenticate per-action separately from per-page? Pulling the emergency switch should ideally require 2FA or a session-level "trader mode" toggle, not just a logged-in session.
4. WebSocket vs polling: `websocket.js` uses Socket.IO. Some environments behind corporate VPNs strip WS. Need a polling fallback that the UI surfaces as "Degraded".
5. Multi-operator coordination: if two operators both hit pause, who owns resume? No locking visible.
