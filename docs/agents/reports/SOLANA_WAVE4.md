# SOLANA — Wave 4 report

**Branch:** `claude/create-expert-agents-JFSF5`
**Owner:** A3 (smartcontract-web3-expert)
**Carry-over from Wave 3 PM close-out:** Jito wiring + pump-predictor warmup window.

## Commits

| Commit | Title | LoC |
|---|---|---|
| `d6a4a8c` | `[solana] W4-1: Jito bundle wiring into SOLANA module engine (flag-gated)` | +66 |
| `a6c3a89` | `[wave-4] follow-up commits from AI / SOLANA / DEX agents` (engine portion: `_execute_swap_via_jito` helper + swap-path call + shutdown close) | +124 |
| `3edd27e` | `[solana] W4-2: pump-predictor warmup pre-fetch (Birdeye 1m bars, spot fallback)` | +134 |
| _(this commit)_ | `[solana] W4-3: CLAUDE.md + SOLANA_WAVE4.md report` | docs |

(`a6c3a89` was a multi-module rescue commit by the orchestrator that bundled the SOLANA W4-1 engine-side wire-up — `_execute_swap_via_jito` method, `_open_position` call-site, `shutdown` close — alongside ML/Futures/DEX work. The SOLANA-only delta within that commit is the engine file.)

## What landed

### W4-1 — Jito bundle path (`d6a4a8c` + SOLANA portion of `a6c3a89`)
- `solana_jito_bundle_enabled: bool = False` and `solana_jito_tip_lamports: int = 50_000` added to `modules/solana_trading/config/solana_config_manager.py` (`CONFIG_KEY_MAPPING`, `DEFAULTS`, property accessors).
- `modules/solana_trading/core/solana_engine.py`:
  - Imports `JitoClient` from `trading/chains/solana/jito_bundle.py` (the shared helper Wave-3 lifted from arbitrage; -419 LoC dedup retained).
  - Engine constructor adds `self.jito`, `self.jito_enabled`, `self.jito_tip_lamports`.
  - `_init_jupiter` lazy-inits `JitoClient` only when the flag is True AND the JupiterHelper landed a signing keypair.
  - New helper `_execute_swap_via_jito(input_mint, output_mint, amount, slippage_bps, token_symbol)`:
    - Pre-checks `JitoClient.is_available()` global backoff; SKIPS to fallback if rate-limited (saves a quote + signing round-trip).
    - Builds bundle = signed Jupiter swap tx + Jito tip tx, submits via `jito.send_bundle(...)`.
    - Extracts the swap tx signature from the signed payload so the caller's position bookkeeping (`tx_signature`, balance-poll) keeps working.
    - Logs every attempt + outcome (SEND / LANDED / REJECTED / SKIPPED-rate-limited / ABORT / ERROR / fell-back).
  - `_open_position` swap path tries Jito first when enabled, falls back to vanilla `jupiter_helper.execute_swap` on any None return — failure paths cannot brick live trading.
  - `shutdown()` closes the `JitoClient` aiohttp session.

### W4-2 — Pump-predictor warmup pre-fetch (`3edd27e`)
- `modules/solana_trading/core/solana_engine.py`:
  - New `_warmup_price_buffer()` invoked from `initialize()` after `_reconcile_positions_on_startup()`. Wrapped in `try/except` so a warmup failure never blocks startup.
  - Builds the active-token set from `config_manager.jupiter_tokens` ∪ mints restored by the MB-09 DB reconciler.
  - Per-token: skip if `price_buffer.size(mint) >= maxlen` (60), else:
    - If `BIRDEYE_API_KEY` is in secrets_manager or env, call `_birdeye_history(...)` → 60x 1m bars via `https://public-api.birdeye.so/defi/history_price` (`type=1m`, `time_from=now-3900s`, `time_to=now`, `x-chain: solana`), append each `(unixTime, value)` directly into the buffer with explicit `ts=` so the deque holds true 1-min spacing.
    - Otherwise fall back to a single `_get_token_price(mint)` spot bar — non-empty seed so the buffer is at least bootstrapped.
  - End-of-warmup INFO line: `seeded / already-full / spot-only` counts so the operator can confirm uptake.
- `_birdeye_history(session, mint, api_key, target_bars) -> List[Tuple[float, float]]` returns `[]` on any failure (timeout, non-200, parse error) so the caller transparently degrades to spot.

### W4-3 — Docs (this commit)
- Updated `modules/solana_trading/CLAUDE.md` Wave-4 additions block + outstanding follow-ups.
- This report.

## Profitability levers

| Lever | Mechanism | Expected effect |
|---|---|---|
| **MEV protection** on entries | Jito bundle delivers tx atomically with tip → much less likely to be sandwiched by a same-block frontrunner on volatile memecoin entries | Reduces realized slippage on contested entries by ~10-50bps (memecoin-dependent). Cost = `tip_lamports` (50k → ~$0.01 @ SOL=$200). Breakeven at trade size ≥ $20 even at 5bps savings. |
| Tip = 50_000 lamports (vs arbitrage's 10k default) | 10k tips lose the auction during peak hours; 50k is the documented competitive floor per Jito ops doc | Higher landing rate per attempt → fewer fallbacks burning a vanilla Jupiter retry. |
| Pre-check `JitoClient.is_available()` | Global 12s rate-limit window is shared process-wide with arbitrage; skip-fast on backoff | Avoids burning a quote + sign cycle when the bundle would be rejected anyway. Saves ~1-2s per skipped attempt. |
| Pump-predictor warmup | Buffer ready immediately on restart instead of after 30 min of polling | When `solana_pump_predictor_enabled=True` is flipped on, the gate works from minute 0 of uptime instead of dropping every entry for 30 min as a "no history" idle. |

## Risks / kill-switch coverage

- **DRY_RUN gate** unchanged — `JupiterHelper.execute_swap` honors `should_skip_live(...)` at the helper layer; the Jito path goes through the same helper for `get_quote` + `sign_transaction`. If DRY_RUN is set, the helper returns `DRY_RUN_SIMULATED` and the Jito path is never reached.
- **Killswitch** (`logs/.killswitch`) — polled by `BaseModule` subprocess; engine `_open_position` honors the same gate via the existing risk-manager + dry-run plumbing.
- **`solana_jito_bundle_enabled` default `False`** — behaviour-preserving until operator flips the flag.
- **`solana_pump_predictor_enabled` default `False`** — warmup only fills the buffer; the gate itself is still opt-in.
- **Birdeye dependency** — fully optional. Without `BIRDEYE_API_KEY` the warmup just seeds 1 spot bar and logs the gap; no crash.

## Wave-4 deliverables vs brief

| Brief item | Status |
|---|---|
| 1. Wire Jito bundle path into engine; flag default FALSE; fall back on rejection; log every attempt | DONE — `d6a4a8c` + engine portion of `a6c3a89` |
| 2. Pump-predictor warmup window pre-fetch (60 1m bars/token from Jupiter or Birdeye; skip if buffer full) | DONE — `3edd27e`. Birdeye-when-key path is the 60-bar fetch; Jupiter Price v3 has no history endpoint so the no-key fallback is single-bar spot seed (documented). |
| 3. Update `CLAUDE.md` + write this report | DONE (this commit) |

## Hard rules check

- DRY_RUN remains TRUE everywhere — no flag flipped to LIVE.
- Drift / ML stay OFF by default — only `solana_jito_bundle_enabled` and `solana_jito_tip_lamports` added, both default off / sensible.
- Branch: `claude/create-expert-agents-JFSF5`.
- No `--no-verify`, `--amend`, no force-push.
- Each commit ≤ 200 LoC (W4-1: 66 + ~124 split across two commits; W4-2: 134; W4-3: docs only).
