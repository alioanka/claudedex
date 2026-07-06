-- Migration 139: Wave-F5 FUTURES geometry/gate + AI exit/gate tuning seeds
--
-- Source diagnosis: docs/agents/wave-f5/02_futures_ai.md (20-day DRY_RUN,
-- 691 futures trades PF 0.55 with ZERO take-profit exits; AI starved to ~1
-- stale reading/day by a bot-wide 10-call LLM budget).
--
-- DISCIPLINE (every statement below):
--   * conditional UPDATE ... WHERE value = <old default>  -> never clobbers an
--     operator override (a custom value simply won't match the WHERE clause).
--   * INSERT ... ON CONFLICT DO NOTHING                   -> creates the row on
--     a fresh DB without overwriting an existing one.
--   * NO live-execution flag is flipped. Every change is DRY_RUN-testable.
-- Single-quoted SQL literals only. Idempotent.

-- ─────────────────────────────────────────────────────────────────────────
-- FUTURES — geometry (F1), trailing arm (F3), entry hygiene (F4/F5/F8/F9),
-- rolling gate (F6/F7). All DB-backed via FuturesConfigManager.
-- ─────────────────────────────────────────────────────────────────────────

-- F1: atr_tp_rr_ratio 2.0 -> 1.0 (make TP1 reachable inside the 240-min hold).
UPDATE config_settings SET value = '1.0', updated_at = NOW()
 WHERE config_type = 'futures_risk' AND key = 'atr_tp_rr_ratio'
   AND value IN ('2.0', '2');
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES ('futures_risk', 'atr_tp_rr_ratio', '1.0', 'float',
        'Wave-F5 F1: TP1 = ratio x SL distance. Lowered 2.0->1.0 so TP1 (=SL '
        '=1.5% at the atr_sl_min_pct floor) is reachable inside max_hold_minutes '
        '(4h). At 2.0, TP1=3.0% never hit in 691 trades.',
        NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- F3: early trailing-stop arm at +0.75% price (new knob; TSL exits were 4/4
-- winners but only fired after TP2).
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES ('futures_risk', 'trailing_stop_arm_pct', '0.75', 'float',
        'Wave-F5 F3: arm the trailing stop as soon as the position is +this% '
        'in price (0.75 ~ 0.5xSL), independent of TP2. Trail distance stays the '
        'fixed trailing_stop_distance (no dynamic widening). 0 = disabled.',
        NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- F6: rolling_gate_bench_minutes 1440 -> 2880 (bleeders re-entered through the
-- expiring 24h bench).
UPDATE config_settings SET value = '2880', updated_at = NOW()
 WHERE config_type = 'futures_risk' AND key = 'rolling_gate_bench_minutes'
   AND value = '1440';
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES ('futures_risk', 'rolling_gate_bench_minutes', '2880', 'int',
        'Wave-F5 F6: symbol bench window doubled 1440->2880 (48h). Halves the '
        'AAVE/ADA/JUP/ETH/LINK re-entry bleed.',
        NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- F7: rolling_gate_max_win_rate 0.45 -> 0.48 (evict mid-tier bleeders one cycle
-- earlier; break-even WR at realized payoff is ~53.8%).
UPDATE config_settings SET value = '0.48', updated_at = NOW()
 WHERE config_type = 'futures_risk' AND key = 'rolling_gate_max_win_rate'
   AND value IN ('0.45', '0.450');
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES ('futures_risk', 'rolling_gate_max_win_rate', '0.48', 'float',
        'Wave-F5 F7: bench a symbol when trailing win rate < this AND net PnL < '
        'threshold. Raised 0.45->0.48.',
        NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- F4: min_signal_score DB 3 -> 4 (align with code default; cuts marginal
-- score-3 entries firing at RSI 31-33 / 0.1-0.4x volume).
UPDATE config_settings SET value = '4', updated_at = NOW()
 WHERE config_type = 'futures_strategy' AND key = 'min_signal_score'
   AND value = '3';
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES ('futures_strategy', 'min_signal_score', '4', 'int',
        'Wave-F5 F4: minimum |signal score| to enter. Restored 3->4 (code '
        'default was already 4; the DB row lagged).',
        NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- F8: hard volume gate at 0.25x (new knob; blocks dead tape 0.09-0.24x).
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES ('futures_strategy', 'min_volume_ratio', '0.25', 'float',
        'Wave-F5 F8: HARD reject entries with volume_ratio < this. Distinct '
        'from the diagnostic-only min_volume_multiplier (0.80x). Set at the '
        'bottom of the observed live range so only truly dead tape is blocked. '
        '0 = disabled.',
        NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- F5: blocked entry hours UTC = 3,4,5 (new knob; worst session -$66/33% win).
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES ('futures_strategy', 'blocked_entry_hours_utc', '3,4,5', 'string',
        'Wave-F5 F5: comma-separated UTC hours during which NEW entries are '
        'blocked (entries only; monitoring/exits unaffected). 03:00-06:59 UTC '
        'was the worst session. Empty = disabled.',
        NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- F9: SHORT sanity floor RSI >= 35 (new knob; shorts fired at RSI 30-33).
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES ('futures_strategy', 'short_min_rsi', '35', 'float',
        'Wave-F5 F9: require RSI >= this for any SHORT entry (blocks shorting '
        'into an already-oversold bounce, the pattern that filled the SL '
        'bucket). 0 = disabled.',
        NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- ─────────────────────────────────────────────────────────────────────────
-- AI (config_type = 'ai_config') — exit geometry (A6), concentration cap (A5),
-- confirmation filter (A4). LLM daily-budget default is code-side (150) — see
-- core/llm_budget.py; env BOT_LLM_DAILY_MAX_CALLS still wins.
-- ─────────────────────────────────────────────────────────────────────────

-- A6: stop_loss_pct -3 -> -2 (SL is internally -abs()). Match either sign.
UPDATE config_settings SET value = '-2', updated_at = NOW()
 WHERE config_type = 'ai_config' AND key = 'stop_loss_pct'
   AND value IN ('-3', '-3.0', '3', '3.0');
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES ('ai_config', 'stop_loss_pct', '-2', 'float',
        'Wave-F5 A6: AI stop-loss % (internally negative). Tightened -3->-2 to '
        'match majors realized daily vol so exits are informative, not 24h '
        'coin-flips.',
        NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- A6: take_profit_pct 6 -> 2.5.
UPDATE config_settings SET value = '2.5', updated_at = NOW()
 WHERE config_type = 'ai_config' AND key = 'take_profit_pct'
   AND value IN ('6', '6.0');
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES ('ai_config', 'take_profit_pct', '2.5', 'float',
        'Wave-F5 A6: AI take-profit %. Lowered 6->2.5 to majors daily-vol '
        'scale (TP never fired at 6% in the sample).',
        NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- A6: max_hold_hours 24 -> 48.
UPDATE config_settings SET value = '48', updated_at = NOW()
 WHERE config_type = 'ai_config' AND key = 'max_hold_hours'
   AND value IN ('24', '24.0');
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES ('ai_config', 'max_hold_hours', '48', 'int',
        'Wave-F5 A6: AI position auto-close timer. Widened 24->48h so the '
        'tighter TP/SL barriers get room to resolve before the timer.',
        NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- A5: per-signal position cap = 1 (one market-wide score must not open a
-- basket of correlated positions). New knob consumed in sentiment_engine.
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES ('ai_config', 'ai_max_positions_per_signal', '1', 'int',
        'Wave-F5 A5: max NEW positions a single sentiment score may open in one '
        'cycle. 1 = no more BTC+ETH+SOL correlated baskets on one score. 0 = '
        'disabled (falls back to the global ai_max_positions cap).',
        NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

-- A4: enable cross-source confirmation filter (filter-only; can only SKIP
-- entries, never add — no LLM spend). Flip the default false; operator custom
-- values other than 'false' are untouched.
UPDATE config_settings SET value = 'true', updated_at = NOW()
 WHERE config_type = 'ai_config' AND key = 'ai_confirmation_signal_enabled'
   AND value IN ('false', '0', 'no');
INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES ('ai_config', 'ai_confirmation_signal_enabled', 'true', 'bool',
        'Wave-F5 A4: cross-source tape-confirmation gate. Filter-only (skips '
        'entries whose per-symbol tape disagrees with the LLM sentiment); no '
        'new paid API calls.',
        NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;
