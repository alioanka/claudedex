# COPY_TRADING Module — Quant / Algorithm Audit

**Auditor:** quant-algo-expert (20+ yrs)
**Date:** 2026-05-11
**Scope:** `modules/copy_trading/copy_engine.py`, `modules/copy_trading/main_copy.py`, `core/portfolio_manager.py`, `core/decision_maker.py`, and the dashboard surface that consumes copy-trading metrics.

---

## 1. Executive verdict

The copy-trading module as it stands is **not a quant product**. It is a wallet-watcher that fires a fixed-size mirror order at a fixed 5-minute cadence, with **no leader-quality model, no Kelly sizing, no fractional-Kelly cap, no correlation handling, no per-leader attribution model, no slippage decay modeling, and no re-entry gate**. Specifically:

- Mirror size is a hand-coded `copy_ratio = 10 %` and a hand-coded `max_copy_amount = $100`, hard-wired in code (`copy_engine.py:530-531`). Neither uses `ConfigManager`, neither is risk-adjusted, and neither is leader-conditional.
- The portfolio-level Kelly allocator in `core/portfolio_manager.py:79,405-406` and the decision-maker's Kelly call at `core/decision_maker.py:621,698-727` exist — but **copy_engine never calls them**. Copy trades bypass `PortfolioManager.allocate_capital()` entirely and write straight to `copytrading_trades` (`copy_engine.py:1187,1205`).
- There is **no `leader_score` field, no leader table, no Sharpe-of-leader, no DD-of-leader** anywhere in the module. The "stats" function (`copy_engine.py:1229-1246`) writes only a single integer (trade-count) per wallet to `config_settings('wallet_stats', wallet)`. That's it. There is no edge measurement, so a wallet that is up 1000% and a wallet that is down 90% are funded identically.
- The P&L recorded for SELL legs (`copy_engine.py:1148-1181`) is the **leader's nominal SOL/ETH value at our recorded entry vs. exit price**, not our actual fill — for sells the engine deliberately fabricates `tx_hash = f"SELL_TRACKED_{...}"` and never executes (`copy_engine.py:1098-1106`). The dashboard cannot tell which trades are real and which are "tracked."

Verdict: **C-/D quant content**, **B engineering plumbing**. The next 200-line PR should be a leader-score table + a fractional-Kelly mirror allocator — see CT-Q-01 / CT-Q-02.

---

## 2. Math / Stat audit (per shown metric)

### 2.1 Leader score / leader selection

**Formula found in code:** none.
**Where it is read on the dashboard:** the dashboard does not show a leader score. `monitoring/enhanced_dashboard.py:8676-8790` displays `unique_wallets`, `wallets_tracked`, aggregate `total_pnl`, `win_rate`, but no per-wallet edge. The wallet-detail loop at `enhanced_dashboard.py:8363-8396` does per-wallet trade counts and per-wallet PnL — that is the only attribution that exists.

**Verdict:** missing. Without a leader score we are running a uniform-weighted bag of strangers. There is no edge filter, no sample-size adjustment, no multi-horizon (30/90/180d) blending. This is the biggest single profitability lever — see CT-Q-01.

### 2.2 Mirror size / copy ratio

**Formula:**
```
copy_amount = min(original_value * 10 // 100,  max_copy_amount * 1e18)
```
at `copy_engine.py:904-907`, with `self.copy_ratio = 10` (`copy_engine.py:531`) and `self.max_copy_amount = 100.0` USD (`copy_engine.py:530`). For Solana, even simpler:
```
copy_lamports = int(min(100.0 / sol_price, 0.1) * 1e9)
```
at `copy_engine.py:1087-1088` — always min(100/sol_price, 0.1) ≈ 0.1 SOL ≈ $20 today.

**Issues:**
- Not a function of (our_capital, leader_capital, leader_DD, our_DD, signal_confidence). It is a constant.
- Hard-codes USD cap in code, not in `ConfigManager`. Violates working-rule #2.
- `copy_ratio` is an integer percentage (`// 100`), losing precision for small leaders.
- No fractional-Kelly cap. The Kelly fn in `decision_maker.py:709-712` is reachable for organic trades but **never called** here.
- The min() on the Solana path means the dollar amount of mirror trades is **functionally constant at $20**, independent of leader-trade size — so a leader putting in $1M and a leader putting in $1k both produce $20 of our exposure. That destroys signal.

### 2.3 Per-leader correlation handling

**Formula found:** none.
**Reality:** every wallet in `self.targets` is independently checked on every cycle (`copy_engine.py:572-594`). If five wallets long ETH on the same block, we fire five mirror buys for $20 each. The shared engine has a `_wallet_last_copy_time` cooldown of 300 s per wallet (`copy_engine.py:541-542,949-963`) — but that is per-leader, not per-token. So **token-level correlation is unmanaged**.

Compare to `core/portfolio_manager.py:140-142` which does compute `correlation_matrix` and has a `correlation_threshold = 0.7` — but that machinery is not invoked from copy_engine. The Kelly/Risk-Parity allocators in `portfolio_manager.py:399-484` are similarly unreached.

### 2.4 Slippage decay (leader fills cheaper than us)

**Formula found:** none. The leader's fill price is never compared to ours. The DEX swap path uses a fixed `slippage_bps=100` on Solana (`copy_engine.py:225`) and `slippage=10.0%` on EVM (`copy_engine.py:267`). 10 % is *very* permissive and exposes us to sandwiches and to the gap between leader's fill (often in the same block, ahead of us) and our late fill. The dashboard does not display "realized slippage vs. leader" anywhere.

### 2.5 Tail-risk / per-leader max loss

**Formula found:** none. There is no per-leader stop, no per-leader DD cap, no kick-out rule. A blowing-up leader will keep getting mirrored until either:
1. The 5-min cooldown is in force (only limits frequency, not magnitude), or
2. The 100 USD `max_copy_amount` caps a single trade. But the per-leader cumulative exposure has **no cap**.

### 2.6 Re-entry rule after leader DD

**Formula found:** none. Once a wallet is in `self.targets`, it stays in until manually removed via `config_settings`. There is no quarantine, no probationary half-size period, no "recovered Sharpe > X for N days" gate.

### 2.7 Kelly fractional cap (anywhere in the codebase)

`core/decision_maker.py:698-727` implements:
```
p = confidence ; q = 1 - p ; b = 2.5
kelly_fraction = (p * b - q) / b
safe_fraction  = kelly_fraction * 0.25     # 25% Kelly — line 712
if risk_score: safe_fraction *= (1 - risk_score.total_risk)
position_size = min(balance * safe_fraction, balance * 0.05)
```

Comments:
- `b = 2.5` is **hard-coded**, not derived from realized payoff distribution. This is in-sample optimism. A real Kelly needs `b = E[gain | win] / E[loss | loss]` measured empirically.
- 25% Kelly is a reasonable cap (typical range 0.25–0.5x), **good**.
- The 5% absolute cap (`max_position_pct = 0.05`) is reasonable.
- But again: **copy_engine.py never calls this function.** It's dead code for the copy module.

### 2.8 P&L attribution / per-leader culling pipeline

`_update_wallet_stats` at `copy_engine.py:1229-1246` writes only `COUNT(*)` per wallet. No win-rate, no PnL, no Sharpe. The dashboard reconstructs per-wallet PnL in `enhanced_dashboard.py:8363-8396` directly from `copytrading_trades` — fine, but **slow** (N queries) and **not used by the allocator**, since there is no allocator.

---

## 3. Bias inventory

| Bias | Where it lives | Severity |
|---|---|---|
| **Survivorship** | Wallets are added manually by the operator. The operator naturally picks wallets that are already up — by definition winners. | High |
| **Look-ahead** | The leader's trade is by definition past-tense (mined block, ≤120 s old: `copy_engine.py:760,836`). Look-ahead in *signal* is avoided, but look-ahead in *attribution* is real: we record the leader's entry as ours and ignore that our actual block is 1-15 s later, often at a worse price. | High |
| **P-hacking** | None visible — there's no model to overfit because there's no model at all. | None (by absence) |
| **Currency-of-quote** | Severe. P&L is computed as USD via CoinGecko `simple/price` (`copy_engine.py:50-97`) at trade-event time. (a) The same trade revalued tomorrow will produce a different "P&L". (b) `USDC`, `USDT`, `USD-of-day` and `USD-of-CoinGecko-cache` are conflated. (c) For SELL on a token still rallying, we use ETH/SOL spot at the time of *the leader's tx*, not at *our* fill. | High |
| **Dry-run leakage** | `is_simulated` column is set from `self.dry_run` (`copy_engine.py:1198,1217`). The dashboard's stats query in `enhanced_dashboard.py:8701-8732` does **NOT filter `is_simulated = false`**, so dry-run trades are aggregated with live trades. Live PnL is therefore polluted. | **Critical** |
| **Sell-without-buy** | `copy_engine.py:1183-1201` records standalone sells (no matching buy in our DB) with `entry_usd = 0, profit_loss = 0`. The dashboard SQL now correctly excludes these from win/loss (`enhanced_dashboard.py:8718-8730`) but they are still counted in `total_trades` — inflating denominators in any ratio that uses `total_trades`. | Medium |
| **Sample-size** | No N<30 warning anywhere. A wallet with 3 winning copies will read "100 % win rate" on the dashboard with no CI band. | Medium |

---

## 4. Attribution audit

- **Per-leader P&L:** computable on the fly in `enhanced_dashboard.py:8363-8396`, but **not stored** and **not used to weight allocations** because allocations are constant.
- **Per-strategy P&L:** N/A — copy is one "strategy" by nature; not broken down by chain, by DEX, by token class, by leader cluster. The strategy-bar chart in `dashboard/static/js/performance.js:138-167` would show one bucket called "copy_trading" if invoked.
- **Per-symbol P&L:** stored at row level (`token_address`) but never aggregated in the dashboard for the copy module.
- **Realized vs. unrealized:** for open positions, `enhanced_dashboard.py:8853-8857` sets `unrealized_pnl = 0` and `current_price = entry_price`. So **unrealized P&L is structurally always zero** until a sell is mirrored. That means the dashboard's "P&L" for copy is realized-only and lags the truth by minutes-to-days.

Conclusion: attribution is **aggregate-only**, **realized-only**, and **decision-disconnected**. We can see history; we cannot use it to allocate.

---

## 5. Capital-allocator math review (the centerpiece)

The intended allocator (decision-maker Kelly + portfolio-manager strategies) is **not wired in**. What runs is:

```python
# copy_engine.py:530-531
self.max_copy_amount = 100.0   # USD
self.copy_ratio = 10           # %
```

For EVM (line 904):
```python
copy_amount = min(original_value * 10 // 100,
                  int(100.0 * 1e18))   # = $100 in wei
```

The `int(100.0 * 1e18)` line is **a unit bug**: 100 wei-USD is treated as 100×10¹⁸ wei of ETH. At ETH = $2 500 that means a max cap of 100 ETH ≈ $250 000, not $100. So `min(0.1 × original_value, 100 ETH)` — the cap is never binding for any retail-sized leader trade. **The 10 % copy_ratio is effectively the only governor.** This is CT-Q-03 below.

For Solana (line 1087-1088):
```python
copy_lamports = int(min(100.0 / sol_price, 0.1) * 1e9)
```
`100/sol_price` at SOL=$200 → 0.5 SOL. `min(0.5, 0.1) = 0.1`. So Solana is hard-capped at 0.1 SOL (≈$20) regardless of `max_copy_amount`. The intent ($100) does not match the math (0.1 SOL).

**Correct mirror-size formula should be (proposal):**

```
target_$ = our_capital
         × leader_weight(leader_id)            # 0..1, sum=1 across leaders
         × signal_confidence                   # 0..1, from per-leader Sharpe & recency
         × kelly_frac(p_win, b)                # fractional Kelly, ≤0.25
         × (1 − our_drawdown_pct)              # de-risk on our DD
         × clip_corr(token, current_book)      # 0..1, penalize correlated exposure
         ÷ max(1, slippage_decay_multiplier)   # discount slow leaders
target_$ = min(target_$, per_leader_cap, per_token_cap, single_trade_cap)
```

None of this is in code today.

---

## 6. Profitability levers (ranked, biggest impact first)

1. **Add a `leader_score` table** with rolling 30/90/180-day Sharpe, win-rate, avg-trade-size, slippage-vs-us, and exclude leaders whose 30-day score has fallen below threshold. (CT-Q-01)
2. **Replace constant `copy_ratio` with fractional-Kelly mirror sizing** keyed off leader_score and our_drawdown. (CT-Q-02)
3. **Fix the unit bug** in `copy_engine.py:906` (`100.0 * 1e18` is wei-of-ETH, not USD). (CT-Q-03)
4. **Token-level cooldown / per-token concentration cap** — at present 5 leaders longing the same token = 5x exposure with no cap. (CT-Q-04)
5. **Filter `is_simulated = false`** in the dashboard PnL aggregation. Dry-run trades currently pollute the live curve. (CT-Q-05)
6. **Stop fabricating SELL_TRACKED tx_hash** when we never executed — flag as `status='leader_only'` and exclude from PnL. (CT-Q-06)
7. **Add per-leader max-loss kill-switch** (e.g. realized loss > 2× leader's avg-win → eject). (CT-Q-07)
8. **Move all magic numbers** (copy_ratio, max_copy_amount, cooldown_s, slippage_bps) to `ConfigManager`. Mandated by working-rule #2. (CT-Q-08)
9. **Re-entry gate**: ejected leader needs 14 d of paper-Sharpe > 1.0 before being re-mirrored. (CT-Q-09)
10. **Slippage-decay model**: store leader's tx timestamp vs. our fill timestamp and our fill price vs. leader's fill price; reject leaders whose decay > 50 bps median. (CT-Q-10)

---

## 7. Action backlog

| ID | Title | Files | Est. lines | Risk |
|---|---|---|---|---|
| CT-Q-01 | Create `copy_leaders` table + `leader_score()` (Sharpe-30d, win-rate, sample-size adj) | new migration, `copy_engine.py` | ~150 | low |
| CT-Q-02 | Replace `copy_ratio` constant with fractional-Kelly mirror sizing using leader_score | `copy_engine.py:886-947`, new helper | ~120 | medium |
| CT-Q-03 | Fix `100.0 * 1e18` unit bug at `copy_engine.py:906` | 1 line | ~5 | low |
| CT-Q-04 | Add per-token open-exposure cap (sum of mirrors per token ≤ X% balance) | `copy_engine.py`, new dict | ~60 | low |
| CT-Q-05 | Add `WHERE is_simulated = false` filter to copy stats SQL | `enhanced_dashboard.py:8701` | ~3 | low |
| CT-Q-06 | Replace fake SELL_TRACKED tx_hash with `status='leader_only'` rows | `copy_engine.py:1098-1115` | ~40 | low |
| CT-Q-07 | Per-leader kill-switch on cumulative realized loss | `copy_engine.py`, new helper | ~60 | medium |
| CT-Q-08 | Move copy_ratio, max_copy_amount, cooldown_s, slippage_bps to ConfigManager | `copy_engine.py`, config | ~30 | low |
| CT-Q-09 | Leader probation table & re-entry gate | new table, `copy_engine.py` | ~80 | medium |
| CT-Q-10 | Slippage-decay tracker: log leader_fill_price vs our_fill_price | `copy_engine.py`, new column | ~80 | medium |
| CT-Q-11 | Wire copy_engine through `PortfolioManager.allocate_capital()` so the Kelly machinery applies | `copy_engine.py`, `portfolio_manager.py` | ~150 | high |
| CT-Q-12 | Add per-leader Sharpe / win-rate columns to the copytrading dashboard | dashboard + `enhanced_dashboard.py:8676` | ~100 | low |

---

## 8. Open questions

1. **What is the policy intent for SELL?** Today we never sell on the mirror side for Solana (line 1098). Is that a deliberate "ride the leader on the way up only" policy, or a missing implementation? It is materially different: the former is a directional bet, the latter is a bug.
2. **Why is `slippage = 10 %` (`copy_engine.py:267`) on EVM?** At 10 % you are guaranteed to be sandwiched on any liquid pair. Was this set to "not get reverted" rather than "to keep edge"?
3. **Wallet whitelist source of truth.** `target_wallets` comes from `config_settings` as a JSON list of strings, optionally with `@chain` suffix (`copy_engine.py:672-692`). Should there be a per-wallet `weight` and `enabled` flag in the same row, instead of a flat list?
4. **Initial-balance for Kelly.** If we wire CT-Q-11, what is `our_capital` for the copy module? A dedicated bankroll or a fraction of the global portfolio? Today there is no answer.
5. **Cross-module exposure.** If DEX module is long PEPE and copy module mirrors a wallet that buys PEPE, the portfolio is doubled up. There is no cross-module aggregator.
6. **Re-org / failed-tx handling.** If a leader's tx is reorg'd, we still copy. There is no `confirmed_blocks >= N` gate.

---

## 9. Summary

The module ships **wallet-mirroring**, not **copy-trading**. The Kelly machinery exists in the project (`core/decision_maker.py:698`, `core/portfolio_manager.py:79-82`) but is unreachable from `copy_engine.py`. The single highest-impact change is CT-Q-01 + CT-Q-02 together — measure leader edge and size proportionally. Until then, every dollar allocated to this module is allocated under a uniform-weight, zero-information prior. That is fine for paper / discovery, **not** for live capital.
