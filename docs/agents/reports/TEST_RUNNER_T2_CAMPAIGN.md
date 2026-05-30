# Test Runner T2 Campaign — FUTURES / AI / COPY_TRADING

**Agent:** T2 (backend-devops-expert)
**Branch:** `claude/create-expert-agents-JFSF5`
**Base:** `ffeda0a`
**Date:** 2026-05-19
**Catalog file:** `monitoring/test_runner_routes.py`

## Scope
T2 owns the FUTURES, AI, and COPY_TRADING modules per PM_PLAN.md
"T1 / T2 brief" section. T1 owns DEX / ARB / SOLANA / SNIPER and appends
to the same file in parallel — rebase-before-commit cadence kept the two
streams clean.

## Commits
| Hash | Module batch | Entries |
|---|---|---|
| `9642626` | FUTURES wave-2 (FUT-RM-01/05/06/07) | 5 |
| `941911b` | AI wave-2 (E1 quorum / E2 calibration / E3 bandit) | 5 |
| `913c720` | COPY_TRADING wave-2 (wallet_discovery + leader_scorer) | 6 |

Total: 16 new TEST_CATALOG entries. Net LoC ≈683 across 3 commits
(each commit individually ≤ ~320 LoC, all comment-heavy by design so
the operator can read the entry without leaving the dashboard).

## Entries added (full list)

### FUTURES (5)
| Test ID | Kind | What it proves |
|---|---|---|
| `db_futures_leverage_caps` | db_query | FUT-RM-01 wiring: futures_max_leverage / max_positions / capital_allocation / default_leverage rows present in `config_settings` so main_futures.py merges real values into `risk_cfg` instead of the dataclass defaults (3x / 3 / 500). |
| `db_futures_funding_gate` | db_query | FUT-RM-05 directional gate keys: `skip_long_funding_bps`, `skip_short_funding_bps`. Defaults are dataclass fail-open if rows missing. |
| `db_futures_atr_sizing` | db_query | FUT-RM-06 ATR-based per-symbol sizing: `atr_sizing_enabled`, `atr_risk_pct`, `atr_stop_multiplier`. Opt-in. |
| `db_futures_isolated_enforce` | db_query | FUT-RM-07 post-fill ISOLATED-margin verifier toggle: `enforce_isolated_margin` (default True). |
| `api_settings_futures_post_wave2` | api probe | GET /api/settings/futures returns 200 + the wave-2 keys end-to-end through `FuturesConfigManager.get_*`. |

### AI (5)
| Test ID | Kind | What it proves |
|---|---|---|
| `db_ai_calibration_table` | db_query | Migration 023 `ai_confidence_calibration` table present + 8 expected columns (trade_id / provider / predicted_score / predicted_confidence / realized_pnl_pct / realized_won / quorum_required / closed_at). |
| `db_ai_calibration_sample` | db_query | Last 90 days of calibration rows: total / closed / quorum-active counts + AVG predicted-confidence vs AVG win-rate (drift = miscalibration). |
| `db_ai_quorum_bandit_config` | db_query | E1 quorum + E3 bandit tunables in `ai_config`: `quorum_required`, `quorum_max_disagreement`, `bandit_enabled`, `bandit_epsilon`, `ai_provider`. |
| `api_ai_calibration` | api probe | GET /api/ai/calibration returns reliability bins + Brier score (graceful empty when no data). |
| `db_ai_bandit_state` | db_query | Per-arm bandit state from `ai_feature_store.feature_vector.bandit_v1` — selections, mean reward, last-used. Convergence and ε-greedy exploration ratio visible. |

### COPY_TRADING (6 — operator-priority)
| Test ID | Kind | What it proves |
|---|---|---|
| `db_copy_leader_scores_table` | db_query | Migration 024 `copy_leader_scores` present + 11 expected columns (chain / wallet_address / source / realized_pnl_usd_30d / sharpe_30d / hit_rate / max_drawdown_pct / score / kelly_fraction / raw_metrics / last_scored_at). |
| `db_copy_leader_scores_top` | db_query | Top-10 by composite score fed to /copytrading/leaders + Kelly sizing in copy_engine. Empty = either migration missing or no refresh yet. |
| `db_copy_leader_scores_by_source` | db_query | wallet_discovery 5-source sweep distribution (dexscreener / birdeye / gmgn / helius / manual). Skew surfaces rate-limited or unkeyed sources. |
| `api_copytrading_leaders_list` | api probe | GET /api/copytrading/leaders returns cached ranked list (never hits the network). |
| `api_copytrading_leaders_refresh` | api probe | Probes the admin-only POST route registration via GET (expects 405 or 403 — both prove the route exists). |
| `db_copy_leader_scores_post_refresh` | db_query | Run AFTER `POST /api/copytrading/leaders/refresh` — confirms rows landed via `fresh_5m` count. Empty = refresh never ran or DiscoveryConfig errored. |

## Coverage map vs PM brief deliverables
- New HTTP endpoints in this wave (`git diff ffeda0a..HEAD -- monitoring/enhanced_dashboard.py | grep router.add_`):
  - `GET /api/ai/calibration` → `api_ai_calibration`
  - `GET /copytrading/leaders` (page) → covered indirectly via `api_copytrading_leaders_list` (the page only fetches data through the JSON endpoint)
  - `GET /api/copytrading/leaders` → `api_copytrading_leaders_list`
  - `POST /api/copytrading/leaders/refresh` → `api_copytrading_leaders_refresh` + paired `db_copy_leader_scores_post_refresh` for the write check
- New DB tables in this wave:
  - `ai_confidence_calibration` (migration 023) → `db_ai_calibration_table` + `db_ai_calibration_sample`
  - `copy_leader_scores` (migration 024) → `db_copy_leader_scores_table` + 3 readers + 1 post-refresh check
- COPY_TRADING operator-priority depth: wallet_discovery has 4 of the 6 entries directly tied to it (source distribution, refresh route, post-refresh freshness, ranked list).

## Notes / non-goals
- The catalog has no `POST` probe kind. The refresh probe uses GET against the POST route and treats HTTP 405/403 as healthy. Once a `kind: "http_post"` exists in the executor, swap `api_copytrading_leaders_refresh` to issue the real POST with `{"mock": true}` and assert the JSON shape.
- `dashboard/templates/test_runner.html` did not need to grow — the existing `auto-fit, minmax(280px, 1fr)` grid absorbs the new 16 buttons cleanly.
- All entries are read-only or rate-limited admin operations. None mutate `config_settings` or trigger live trades.

## Rebase cadence
Each batch was pulled-rebased twice (before edit + before commit) to keep
in sync with T1's parallel pushes. No conflicts surfaced because both
agents appended to the same trailing `]` of `TEST_CATALOG` and neither
modified entries the other owned.
