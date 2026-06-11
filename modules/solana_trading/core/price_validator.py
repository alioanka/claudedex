# modules/solana_trading/core/price_validator.py
"""Fetch-time price denomination / sanity validation (the +2000% bug).

WHY THIS EXISTS
---------------
The multi-source price chain (DexScreener -> CoinGecko -> Jupiter v3)
intermittently returns a WRONG-DENOMINATION or WRONG-TOKEN quote for a
position's mint (e.g. ~$153 for a $0.03 token — roughly SOL/USD leaking
in). The monitor then computes a +4000..+5000% unrealized PnL, fires
"Take Profit", and the save-time guard pins the exit to 50x entry which
the pnl clamp turns into the recurring "+2000.00%" rows the operator
sees. Pinning at SAVE time is too late: the exit already fired on a
poisoned price.

This validator sits at FETCH time, per mint:

- Tracks the last KNOWN-GOOD price per mint (seeded from entry price on
  open / reconcile, then updated on every accepted quote).
- A quote within `soft_jump_ratio` (default 2x, either direction) of the
  last good price is accepted immediately.
- A quote beyond `soft_jump_ratio` is held: it must be CONFIRMED by
  `jump_confirmations` (default 2) consecutive consistent readings
  before it is accepted. Until confirmed, `validate()` returns the last
  good price so monitoring stays alive but NO exit can fire on the
  unconfirmed jump.
- A quote beyond `hard_jump_ratio` (default 5x) needs
  `hard_jump_confirmations` (default 3) consecutive consistent readings,
  OR `jump_confirmations` readings from MORE THAN ONE source (a
  denomination bug is source-specific, so cross-source agreement is the
  strongest signal the move is real).
- Every rejected/held quote logs WHICH SOURCE returned it so the
  offending feed is identifiable from the logs.
- If the last good price is older than `lastgood_ttl_s` (default 600s)
  it is considered too stale to anchor against and the incoming quote is
  accepted (bounded worst-case: a broken feed can never wedge a mint
  forever).
- A pending (unconfirmed) jump expires after `pending_window_s`
  (default 120s) so an isolated glitch doesn't linger as state.

Pure in-process, single-threaded engine loop — no locks needed.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class _LastGood:
    price: float
    ts: float
    source: str = ""


@dataclass
class _Pending:
    price: float          # first anomalous reading (reference for consistency)
    first_ts: float
    count: int = 1
    sources: Set[str] = field(default_factory=set)


class PriceValidator:
    """Per-mint last-good tracking + jump confirmation.

    All thresholds are mutable via `configure()` so the engine can
    refresh them from DB config on every poll without reconstructing.
    """

    # Two pending readings are "consistent" when within this relative band
    # of each other (max/min <= 1 + band). A wrong-denomination feed echoes
    # the same wrong value, and a genuine repricing holds its new level, so
    # 30% is generous for a 5s poll interval.
    CONSISTENCY_BAND = 0.30

    def __init__(
        self,
        soft_jump_ratio: float = 2.0,
        hard_jump_ratio: float = 5.0,
        jump_confirmations: int = 2,
        hard_jump_confirmations: int = 3,
        lastgood_ttl_s: float = 600.0,
        pending_window_s: float = 120.0,
    ):
        self._last_good: Dict[str, _LastGood] = {}
        self._pending: Dict[str, _Pending] = {}
        self.configure(
            soft_jump_ratio=soft_jump_ratio,
            hard_jump_ratio=hard_jump_ratio,
            jump_confirmations=jump_confirmations,
            hard_jump_confirmations=hard_jump_confirmations,
            lastgood_ttl_s=lastgood_ttl_s,
            pending_window_s=pending_window_s,
        )

    def configure(
        self,
        soft_jump_ratio: Optional[float] = None,
        hard_jump_ratio: Optional[float] = None,
        jump_confirmations: Optional[int] = None,
        hard_jump_confirmations: Optional[int] = None,
        lastgood_ttl_s: Optional[float] = None,
        pending_window_s: Optional[float] = None,
    ) -> None:
        """Update thresholds; clamps keep operator typos from disabling
        the guard entirely (soft >= 1.2, hard >= soft, confirmations >= 1)."""
        if soft_jump_ratio is not None:
            self.soft_jump_ratio = max(1.2, float(soft_jump_ratio))
        if hard_jump_ratio is not None:
            self.hard_jump_ratio = max(self.soft_jump_ratio, float(hard_jump_ratio))
        if jump_confirmations is not None:
            self.jump_confirmations = max(1, int(jump_confirmations))
        if hard_jump_confirmations is not None:
            self.hard_jump_confirmations = max(
                self.jump_confirmations, int(hard_jump_confirmations)
            )
        if lastgood_ttl_s is not None:
            self.lastgood_ttl_s = max(30.0, float(lastgood_ttl_s))
        if pending_window_s is not None:
            self.pending_window_s = max(10.0, float(pending_window_s))

    # ------------------------------------------------------------------
    def seed(self, mint: str, price: float, source: str = "entry") -> None:
        """Anchor the last-good price for a mint (entry / reconcile)."""
        try:
            price = float(price)
        except (TypeError, ValueError):
            return
        if not mint or price <= 0:
            return
        self._last_good[mint] = _LastGood(price=price, ts=time.time(), source=source)
        self._pending.pop(mint, None)

    def last_good(self, mint: str) -> Optional[float]:
        lg = self._last_good.get(mint)
        return lg.price if lg else None

    def drop(self, mint: str) -> None:
        """Forget a mint (position closed)."""
        self._last_good.pop(mint, None)
        self._pending.pop(mint, None)

    # ------------------------------------------------------------------
    def validate(
        self, mint: str, price: float, source: str = "unknown",
        now: Optional[float] = None,
    ) -> Tuple[Optional[float], bool]:
        """Validate a fetched quote against the per-mint last-good price.

        Returns (price_to_use, accepted):
          - accepted=True  -> price_to_use == price (fresh quote accepted,
            last-good updated).
          - accepted=False -> the quote is an UNCONFIRMED jump;
            price_to_use is the last-good price (hold) or None when the
            quote itself is unusable. Callers must treat accepted=False
            as "do not let this reading move state on its own" — in
            particular the caller should evict any price cache so the
            next poll re-fetches instead of echoing the same bad value.
        """
        if now is None:
            now = time.time()
        try:
            price = float(price)
        except (TypeError, ValueError):
            return None, False
        if not mint or price <= 0:
            return None, False

        lg = self._last_good.get(mint)

        # No anchor, or anchor too stale to be meaningful -> accept.
        if lg is None or (now - lg.ts) > self.lastgood_ttl_s:
            if lg is not None:
                logger.info(
                    "🧭 price-validator: last-good for %s is %.0fs old (> ttl %.0fs) — "
                    "accepting fresh quote $%.10g from %s without anchor",
                    mint[:8], now - lg.ts, self.lastgood_ttl_s, price, source,
                )
            self._last_good[mint] = _LastGood(price=price, ts=now, source=source)
            self._pending.pop(mint, None)
            return price, True

        ratio = price / lg.price if price >= lg.price else lg.price / price

        if ratio <= self.soft_jump_ratio:
            self._last_good[mint] = _LastGood(price=price, ts=now, source=source)
            self._pending.pop(mint, None)
            return price, True

        # ---- Suspicious jump: needs confirmation -------------------------
        pend = self._pending.get(mint)
        if pend is not None and (now - pend.first_ts) > self.pending_window_s:
            pend = None  # expired glitch — start over

        consistent = (
            pend is not None
            and max(price, pend.price) / max(min(price, pend.price), 1e-18)
            <= 1.0 + self.CONSISTENCY_BAND
        )
        if consistent:
            pend.count += 1
            pend.sources.add(source)
        else:
            pend = _Pending(price=price, first_ts=now, sources={source})
        self._pending[mint] = pend

        if ratio <= self.hard_jump_ratio:
            needed = self.jump_confirmations
        else:
            # Hard jump: cross-source agreement counts as strong evidence
            # (a denomination bug is source-specific).
            needed = (
                self.jump_confirmations
                if len(pend.sources) > 1
                else self.hard_jump_confirmations
            )

        if pend.count >= needed:
            logger.warning(
                "🧭 price-validator: CONFIRMED %.1fx jump for %s — $%.10g -> $%.10g "
                "after %d consistent reading(s) from %s; accepting",
                ratio, mint[:8], lg.price, price, pend.count,
                "/".join(sorted(pend.sources)),
            )
            self._last_good[mint] = _LastGood(price=price, ts=now, source=source)
            self._pending.pop(mint, None)
            return price, True

        logger.warning(
            "🛑 price-validator: REJECTED unconfirmed %.1fx jump for %s — source=%s "
            "returned $%.10g vs last-good $%.10g (from %s, %.0fs ago); "
            "holding last-good (%d/%d confirmations)",
            ratio, mint[:8], source, price, lg.price, lg.source or "?",
            now - lg.ts, pend.count, needed,
        )
        return lg.price, False
