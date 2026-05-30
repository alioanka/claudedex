# Test Runner T1 Campaign Report (Wave 2)

**Agent:** T1 — backend-devops-expert
**Branch:** `claude/create-expert-agents-JFSF5`
**Modules covered:** DEX / ARBITRAGE / SOLANA / SNIPER
**Date:** 2026-05-19

## Scope
T1 owns Test Runner coverage for the four on-chain modules:
`modules/dex_trading`, `modules/arbitrage`, `modules/solana_trading`,
`modules/sniper`. T2 (parallel agent) owns FUTURES / AI / COPY_TRADING.

## Catalog entries added — 22 total

### DEX (6 entries — commit `cbe5910`)

| Catalog ID | Kind | Guards |
|---|---|---|
| `script_dex_decimals_unit_tests` | bash | `869eed3` — pytest regression for MB-01 decimals (USDC 6, WBTC 8, WETH 18) + route-quality scoring (`_score_quote` prefers lower gas / lower impact). |
| `script_dex_web3_v6_imports` | bash | `48d5f20` — DirectDEXExecutor + MEVProtection import cleanly under web3>=6; v6 snake_case helpers + PoA middleware shim resolve. |
| `script_dex_mev_unbound_check` | bash | `e872121` — `bundle_id = None` default present + `chain in (..., 'ethereum', ...)` Flashbots gate present. Catches re-introduction of the UnboundLocalError on the low-risk ADVANCED branch. |
| `db_dex_recent_trades_24h` | db | `f7d7941` + `162f711` — per-chain DEX trade flow + avg slippage/gas. Slippage column populated proves `self.max_slippage` init survived. avg_gas absurdly above ceiling = per-chain gwei cap misconfigured. |
| `db_dex_settings_keys` | db | `f7d7941` + `162f711` + `e872121` — confirms DEX max_slippage / gas-ceiling / mev_protection rows are seeded. |
| `db_dex_open_positions` | db | `23d860d` — open-position snapshot per chain × dex. `amount_in` must be in token-native units (post-decimals fix). |

### ARBITRAGE (5 entries — commit `5fb94d1`)

| Catalog ID | Kind | Guards |
|---|---|---|
| `db_arb_cost_profile_keys` | db | `744ee48` + `4adcd29` + `89175d4` — wave-2 ARB knobs: min_profit_spread (UI knob), adaptive_min_profit, gas_budget_usd_per_hour, chain_cost_profile. |
| `db_arb_flash_loan_receiver_secrets` | db | `89175d4` — per-chain Aave V3 receiver-contract addresses present in encrypted `config_sensitive` table (not just `os.getenv`). |
| `db_arb_recent_pnl_costs` | db | `8cf0143` — replaces $15 + 30%-of-spread magic constants with chain-aware live gas + slippage. avg_gas_usd should differ per chain. |
| `api_arb_settings_get` | api | `744ee48` — GET `/api/arbitrage/settings` surfaces `min_profit_spread` so the dashboard can write through to the engine. |
| `script_arb_nameerror_regression` | bash | `9e6a7d1` — grep-asserts `forward_output` / `final_output` (renamed identifiers) don't reappear in `_check_arb_opportunity`. Their return would silently drop every spatial-arb opp. |

### SOLANA (5 entries — commit `00e8f6c`)

| Catalog ID | Kind | Guards |
|---|---|---|
| `db_solana_adaptive_priority_fee` | db | `83df4ad` — adaptive priority-fee controller + Jupiter quote freshness TTL (default 10s). |
| `db_solana_drift_guards` | db | `661cee6` — MB-15 four fail-closed Drift guards: leverage cap, funding sanity cap, oracle-deviation cap, Pyth confidence cap. |
| `db_solana_ml_rug_gate` | db | `b1b358f` — P1-07 ML rug-classifier threshold + opt-in flag (`solana_ml_enabled`). |
| `db_solana_recent_trades_decimals` | db | `09a5c85` — solana_trades 24h roll-up. amount_sol staying within configured position_size proves the close-path decimals fix held. |
| `db_solana_position_size_caps` | db | snapshot of capital / position_size / max_positions / daily_loss_limit / stop_loss / take_profit so the operator can verify SolanaConfigManager DB rows exist. |

### SNIPER (5 entries via T2 commit `913c720` + 1 script via `16e75c4`)

| Catalog ID | Kind | Guards |
|---|---|---|
| `db_sniper_processed_hit_ratio` | db | `6612be2` — two-stage `processed`→`confirmed` getTransaction readback. Surfaces `processed_hit / (processed_hit + processed_miss_fallback)` from `sniper_runtime_stats`. >80% = fast-path saving ~5s per call. |
| `db_sniper_safety_check_errors` | db | `77b22e7` — safety_check_errors counter + jupiter/birdeye fallback counters preserved across the 1-min window flip. Non-zero growth = external safety provider outage. |
| `db_sniper_wss_carry_over` | db | `87c5523` — wss_dispatched + wss_inflight_peak carry-over fix (R3). inflight_peak approaching SNIPER_WSS_CONCURRENCY = RPC backup saturating the dispatch semaphore. |
| `db_sniper_quorum_outcomes_30m` | db | `4adcd29` (R4) — distribution of `sniper_trades.metadata.outcome`. Mix of rejected_safety / rejected_safety_error / success rows = both honeypot oracles healthy. |
| `api_sniper_timing_per_chain` | api | `5a0a3e9` — `/api/sniper/timing` source for the per-chain listener-health widget. |
| `script_sniper_listener_widget_present` | bash | `5a0a3e9` — grep-asserts the `listenerHealthPanel` + `listenerHealthBody` DOM ids + `/api/sniper/stats` URL in `performance_sniper.html`. |

## Scripts added under `scripts/`

| Script | Backs catalog entry |
|---|---|
| `scripts/dex_web3_v6_smoke.sh` | `script_dex_web3_v6_imports` |
| `scripts/dex_mev_unbound_check.sh` | `script_dex_mev_unbound_check` |
| `scripts/arb_nameerror_check.sh` | `script_arb_nameerror_regression` |
| `scripts/sniper_listener_widget_check.sh` | `script_sniper_listener_widget_present` |

All four are POSIX bash with `set -euo pipefail`, honor
`CLAUDEDEX_REPO_ROOT` (default `/app` for the docker mount), and exit
non-zero on any drift from the expected source/template content. The
v6 import smoke is a Python one-liner inline; the other three are
pure grep so they run in <100 ms.

## New HTTP endpoints introduced this wave (T1 scope)

`git diff ffeda0a..HEAD -- monitoring/enhanced_dashboard.py | grep
'router.add_'` shows the only new router calls this wave are
`/api/ai/calibration`, `/copytrading/leaders`, `/api/copytrading/leaders`,
and a `_copytrading_leaders` page — all owned by T2.

T1's four modules added zero new HTTP routes this wave; the wave-2
fixes were engine-internal (decimals, gas, latency, guards, quorum,
fallback feeds) and surfaced through existing dashboards. The
per-chain listener-health widget added by `5a0a3e9` is a client-side
consumer of the pre-existing `/api/sniper/stats` endpoint, which
already had an api probe (`api_sniper_stats`).

## Test-runner template

`dashboard/templates/test_runner.html` was not modified — the existing
grid (`grid-template-columns: repeat(auto-fit, minmax(280px, 1fr))`)
holds the additional categories without overflow. Section accordions
are auto-generated from `category` so DEX/ARB/SOLANA/SNIPER
sub-grouping appears automatically once entries are loaded.

## Commits (oldest → newest)

| Hash | Title |
|---|---|
| `cbe5910` | `[test-runner] DEX wave-2 catalog: 6 entries + 2 source-grep scripts` |
| `5fb94d1` | `[test-runner] ARBITRAGE wave-2 catalog: 5 entries + nameerror grep` |
| `00e8f6c` | `[test-runner] SOLANA wave-2 catalog: 5 entries` |
| `16e75c4` | `[test-runner] SNIPER widget-check script + tighten arb/dex grep regexes` |

Note: the 5 SNIPER catalog entries landed in T2's commit `913c720`
because of a workspace race between the two parallel agents (T2's
`git add` picked up T1's unstaged sniper block while staging COPY).
Content is correct; the attribution is split. `16e75c4` is the
authoritative T1 commit for the SNIPER bash script and follow-up
regex fixes on the prior grep scripts.

## Coverage matrix vs PM brief

PM brief listed 29 notable commits across the four modules. The
T1 catalog adds at least one probe per commit:

| Module | Notable commits | Catalog entries |
|---|---|---|
| DEX | `f7d7941`, `e872121`, `23d860d`, `48d5f20`, `162f711`, `a40f69a` | 6 (1:1) |
| ARB | `9e6a7d1`, `744ee48`, `8cf0143`, `89175d4` | 5 (4 commits + 1 PnL surface) |
| SOLANA | `09a5c85`, `661cee6`, `83df4ad`, `b1b358f` | 5 (4 commits + caps snapshot) |
| SNIPER | `6612be2`, `77b22e7`, `87c5523`, `4adcd29`, `adee9c2`, `5a0a3e9` | 6 (1:1) |

The `869eed3` test commit (DEX) is wrapped by
`script_dex_decimals_unit_tests`. The `014384d` ARB dashboard knob is
covered by `api_arb_settings_get`.

## Operational notes

- Every new probe is read-only (db_query is `SELECT`; api is `GET`;
  bash is grep/import smoke). Nothing in this batch performs writes
  or mutations to live state.
- All catalog entries pass `python -c "import ast; ast.parse(open(...).read())"`
  static parse. SQL is hand-validated against the column conventions
  documented in earlier T2 entries (e.g. `sniper_trades` uses
  `entry_timestamp`, `solana_trades` uses `entry_time`).
- Bash scripts run with `set -euo pipefail`, output `PASS` / `FAIL` on
  the last line for easy paste-back to the operator.
- The DRY_RUN-must-stay-TRUE rule is honored: no probe flips a module
  flag and no script alters config_settings rows.
