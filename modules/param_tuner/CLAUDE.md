# PARAM_TUNER Module

## What it does
Bandit-based, **SHADOW-ONLY** self-tuning of registered module knobs. For an
operator-owned whitelist of non-risk DB tunables (each with hard `[min, max]`
bounds), a deterministic UCB1 bandit accumulates rolling-performance evidence
per discretized value and writes value-change **PROPOSALS** to
`param_proposals` for operator approval. **PROPOSAL-ONLY BY DEFAULT** — it
never writes a live config row. When `auto_apply_enabled=true` (default
**false**) it may apply ONLY `kind='exploit'` proposals, ONLY within the
operator's bounds, ONLY to `config_settings` values — logged to
`config_history` and reversible. It **never trades, never touches code, env
flags, risk knobs, or `logs/.killswitch`.**

## HONESTY WARNING (read before trusting a proposal)
Rewards come from whatever value was *actually running* during a window, so:
- **Paper-tuned params overfit.** A value "optimized" on three weeks of chop
  is mis-set for the breakout. Every proposal — especially `explore` ones —
  needs **DRY_RUN validation** before it goes anywhere near LIVE settings.
- Windows are confounded by regime: a knob gets credit/blame for market moves
  it did not cause. The bandit's margin + min-pulls gates reduce, not remove,
  this.
- There is **no counterfactual inference**: values the operator never ran have
  no evidence. Proposals to such values are labeled `explore` = hypothesis
  only, and are never auto-applied.
- Keep registry ranges **tight and bounded**: the bandit can only propose
  inside `[min, max]`, so the bounds are the real safety rail. Shadow-only is
  the default posture for all of these reasons.

## Entry point
`modules/param_tuner/main_param_tuner.py` — launched by `main.py` when
`PARAM_TUNER_MODULE_ENABLED=true` (default **false**). Health server on port
8101 (`PARAM_TUNER_HEALTH_PORT`): `/health`, `/status` (last tick summary).
Bandit math: `core/bandit.py` (pure, deterministic, self-tested:
`python -m modules.param_tuner.core.bandit`). Engine/loop/persistence/apply:
`core/tuner_engine.py`.

## Bandit math (deterministic — every number operator-derivable)
Per tunable, the `[min, max]` range is discretized into `steps` arms. Each
reward observation (at most one per `reward_window_hours`) is the owning
module's rolling closed-trade score in [0,1] (same `score_track` math as
meta_controller, LIVE track preferred when it has `reward_min_trades` trades),
attributed to the arm nearest the currently configured value.
```
UCB(arm) = mean_reward(arm) + exploration_c * sqrt(2 ln N / n_arm)   (unpulled arm -> +inf)
exploit proposal: best arm has >= min_pulls_exploit obs AND mean gain >= improvement_margin
explore proposal: best arm under-sampled (UCB optimism, hypothesis only, never auto-applied)
```
UCB1 over Thompson sampling **by design**: no RNG, so identical state always
yields the identical proposal and the self-test needs no seeds. Changing a
registry entry's bounds/steps rebuilds the grid and carries evidence by
nearest-arm mapping (out-of-bounds evidence is dropped).

## Variant-challenge acceptance (migration 145; adapted from HKUDS/AI-Trader, MIT)
Answers the HONESTY WARNING above head-on. Instead of moving a bandit
proposal straight to the operator queue on its counterfactual-free but
regime-confounded reward history, the tuner opens a **variant challenge**:
BASELINE (current value) vs VARIANT (proposed value) are scored side by side
against the owning module's **real closed trades entered AFTER the challenge
opened** — out-of-sample by construction, no replay, no counterfactual reward.
Only a PASSED challenge reaches the existing `pending` operator queue; a
failed one auto-expires as `challenge_failed`. Pattern adapted (no verbatim
code) from HKUDS/AI-Trader's challenge/variant evaluation — same strategy,
parameter variants, one identical live mark-to-market harness (MIT-licensed;
attribution preserved in `core/challenge.py`).

- **Pure math**: `core/challenge.py` (routing + MTM reward + verdict; self-test
  `python -m modules.param_tuner.core.challenge`). **Persistence/loop**:
  `core/challenge_engine.py` (open → accumulate per tick → resolve; self-test
  `python -m modules.param_tuner.core.challenge_engine`).
- **Honesty constraint** — only ONE config value is ever actually live, so a
  variant is scoreable only via an observable proxy on realized trades. The one
  exact proxy is the **entry-gate proxy**: for a threshold knob whose only
  effect is WHICH candidates get taken, the variant's counterfactual book is a
  deterministic SUBSET of the realized book (variant takes a trade iff its
  recorded per-trade gate feature passes the variant threshold; a skipped trade
  scores neutral 0.5, and realized `pnl_pct` is the MTM truth, saturated at
  `challenge_pnl_cap_pct`). Two failure modes are enforced in code, not prose:
  - **Censored direction**: a *looser* min-gate would have taken trades that
    were never taken/recorded → unscoreable → BYPASS (old direct-to-pending),
    logged.
  - **Exempt knobs**: knobs needing unrecorded data have NO honest proxy →
    `CHALLENGE_EXEMPT` with a documented reason → bypass, keep old behavior.
- **Per-knob scoring status** (see `core/challenge.py`):
  | Knob | Route | Why |
  |---|---|---|
  | `ai_config/confidence_threshold` | **challenge** | `ai_trades.confidence_score` is the recorded gate feature; `profit_loss_pct` the MTM truth |
  | `trading/min_vol_liq_ratio` (dex) | exempt | vol/liq ratio at entry is not persisted per trade |
  | `sniper_config/max_hold_minutes` | exempt | time-exit re-mark needs a per-trade price path (not recorded) |
  | `solana_jupiter/jupiter_auto_exit` | exempt | same — time-exit needs the price at the candidate cutoff |
  | `futures_risk/atr_tp_rr_ratio` | exempt | nearer-TP fill needs MFE, which `futures_trades` does not record |
  | any unregistered knob | exempt | a knob is only challengeable once its honest proxy is written |
- **Verdict rule (conservative)**: at window end the variant must beat the
  baseline mean reward by `challenge_min_edge_pct` (relative) over at least
  `challenge_min_samples` out-of-sample observations, else `challenge_failed`
  — thin tape is a FAIL, never a pass. A failed `(knob, value)` pair cannot
  re-open for `challenge_retry_cooldown_hours` (the bandit is deterministic and
  would otherwise re-propose the identical value every tick).
- **Auto-apply is untouched**: challenge-routed proposals are NEVER
  auto-applied (they enter `pending` on pass, exactly like an operator-facing
  proposal); only exempt/censored knobs, or challenges globally disabled
  (`challenge_enabled=false`), retain the old auto-apply eligibility below.
  Making acceptance stricter is the only effect; nothing new is ever applied.

## Auto-apply safety chain (ALL links must hold; default OFF)
1. DB `auto_apply_enabled=true` (seeded **false**).
2. No `logs/.killswitch`, no `logs/.pause_param_tuner`.
3. Proposal is `kind='exploit'` (evidence-backed) — `explore` never applies.
4. Value re-clamped inside the operator's `[min, max]` from the registry.
5. Key still present in `tunable_registry` (operator can revoke any time).
6. Per-key `auto_apply_cooldown_hours` elapsed since the last auto-apply.
7. The ONLY write is `UPDATE config_settings SET value=... updated_by='param_tuner'`;
   the old value is preserved in the proposal row's `components.old_value` and
   in `config_history` (change_source `param_tuner`) — **revert = restore
   old_value** via the dashboard settings page or SQL.
Hard-coded in `tuner_engine.registry_entry_allowed`: config types
`risk_management`/`security`/`wallets` and keys containing `stop_loss`,
`leverage`, `max_loss`, `live_execution`, `dry_run`, `killswitch`,
`private_key`, `secret` are rejected even if the operator registers them. A
tuner that can widen stops is a martingale generator with extra steps.

## Key config (DB-backed, config_type='param_tuner'; migration 129)
| Key | Default | What it does |
|---|---|---|
| `auto_apply_enabled` | false | Master apply switch. false = proposals only |
| `tick_interval_seconds` | 3600 | Tuner cadence |
| `lookback_hours` | 24 | Rolling closed-trade reward window |
| `reward_window_hours` | 6 | Min hours between two rewards of the same tunable |
| `reward_min_trades` | 5 | Closed-trade floor below which no reward is recorded |
| `n_target` | 50 | Trades at which the track score saturates |
| `exploration_c` | 0.5 | UCB1 exploration constant |
| `min_pulls_exploit` | 3 | Evidence floor for an exploit proposal |
| `improvement_margin` | 0.05 | Mean-reward gain required to propose (anti-churn) |
| `max_open_proposals_per_key` | 1 | In-flight cap per tunable (counts `pending` + `challenge`) |
| `auto_apply_cooldown_hours` | 24 | Min hours between auto-applies of the same key |
| `tunable_registry` | 3 seeded knobs | JSON whitelist: module, config_type, key, min, max, steps, value_type |

Variant-challenge keys (migration 145, `config_type='param_tuner'`):

| Key | Default | What it does |
|---|---|---|
| `challenge_enabled` | true | Extra out-of-sample acceptance gate; false = old direct-to-pending flow |
| `challenge_window_hours` | 72 | Out-of-sample scoring window per challenge |
| `challenge_min_edge_pct` | 5 | Relative edge (% of baseline mean) the variant must beat to PASS |
| `challenge_min_samples` | 20 | Min out-of-sample observations by window end (else fail) |
| `challenge_retry_cooldown_hours` | 72 | Lockout before the same (knob, value) re-opens after a fail |
| `challenge_pnl_cap_pct` | 20 | Per-trade P&L% at which the MTM reward mapping saturates |

Seeded registry (all documented non-risk entry-side knobs): DEX
`trading/min_vol_liq_ratio` [0.01, 0.50], AI `ai_config/confidence_threshold`
[0.20, 0.60], SNIPER `sniper_config/max_hold_minutes` [15, 240].

Wave-F5 retargeting (mig 143): the three mig-129 targets closed ~0 trades
per 6h reward window, so the bandit was reward-starved (arms at `pulls: 0`,
`proposed=0` since Jun 16 — correct behavior, wrong knobs). Mig 143
conditionally appends two tunables owned by the modules that DO close
trades: FUTURES `futures_risk/atr_tp_rr_ratio` [0.8, 2.0] (TP R:R multiple —
a target knob; stop/leverage keys remain hard-excluded in code) and SOLANA
`solana_jupiter/jupiter_auto_exit` [0, 14400] (time-based exit seconds, the
solana analogue of sniper max_hold_minutes). Both target rows are seeded at
their exact code defaults (2.0 / 0) so the seed changes no runtime behavior;
the registry append only runs while the registry is still exactly the three
mig-129 entries (operator edits are never clobbered). Shadow-only posture
unchanged: `auto_apply_enabled` stays false.

## Kill switch
- Global: `logs/.killswitch` — tick skipped; auto-apply additionally re-checks
  it at the apply boundary.
- Per-module: `logs/.pause_param_tuner` — tick skipped.

## Logs
`logs/param_tuner/` — `param_tuner.log`, `param_tuner_errors.log`.

## DB tables (migrations 129, 145)
- `param_proposals` — one row per proposed change; status lifecycle
  `pending -> operator_applied | dismissed | superseded | auto_applied -> reverted`;
  challenge-routed proposals start as `challenge -> pending` (pass) or
  `challenge -> challenge_failed` (fail) (mig 145).
- `param_bandit_state` — durable per-tunable bandit state (JSON arms) +
  `last_reward_at` reward throttle.
- `param_variant_challenges` (mig 145) — one row per opened challenge:
  `proposal_id` FK, `baseline_value`/`variant_value`, running
  `n_samples`/`baseline_reward_sum`/`variant_reward_sum`, `last_scored_at`
  (incremental exit-time cursor), `started_at`/`window_hours`, `resolved_at`,
  `verdict` (`passed` | `challenge_failed` | `abandoned`).

## Dashboard surface (owned by the dashboard agent — NOT built here)
Read-only + fail-soft: pending `param_proposals` (kind, reason, bounds,
evidence) with approve/dismiss actions that write `status` and, on approve,
route through the existing settings-update path; bandit arm means per tunable.
Hide the panel when the tables are absent.

## Isolation / safety
Reuses read-only: `core/dry_run.py` (killswitch poll), meta_controller's
`score_track`/`collect_module_perf` (which single-source orchestrator_ai's
`_MODULE_QUERIES` schema map), the `*_trades` tables (also read read-only by
the variant-challenge harness for out-of-sample scoring). Writes only
`param_proposals`, `param_bandit_state`, `param_variant_challenges`, and
(auto-apply only, default OFF) one `config_settings` value + a
`config_history` audit row. Never imports a trading executor, never signs
anything, no paid LLM.
