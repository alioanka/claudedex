# modules/solana_trading/core/price_buffer.py
"""Per-token rolling price buffer for ML inference.

Wave-3 deliverable. The PumpPredictor's LSTM head needs a sequence of
recent (timestamp, price) points to score; the engine doesn't carry one
today. This module is the minimal in-process buffer:

- `TokenPriceBuffer` keeps one `collections.deque(maxlen=N)` per token.
- O(1) append, O(N) snapshot.
- Optional per-token recency filter: snapshots older than `max_age_s`
  drop stale tokens so memory bounds stay tight when memecoin candidates
  rotate every minute.
- Pure in-process, no asyncio primitives needed (single-threaded engine
  loop touches it). If the engine ever multi-tasks the writer, wrap
  appends in an asyncio.Lock at the call site.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PricePoint:
    ts: float
    price: float


class TokenPriceBuffer:
    """FIFO price buffer keyed by token mint.

    Args:
        maxlen: per-token deque capacity. The pump predictor's
            `sequence_length` default is 20 bars; we keep 3x headroom
            (60) so the gate has slack across cycle drift / missed
            ticks.
        max_age_s: optional eviction window. A token whose newest bar
            is older than this gets dropped wholesale on the next
            `evict_stale()` call. 0/None disables eviction.
    """

    def __init__(self, maxlen: int = 60, max_age_s: Optional[float] = None):
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive, got {maxlen}")
        if max_age_s is not None and max_age_s <= 0:
            raise ValueError(f"max_age_s must be positive when set, got {max_age_s}")
        self._maxlen = maxlen
        self._max_age_s = max_age_s
        self._buf: Dict[str, Deque[PricePoint]] = {}

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def append(self, mint: str, price: float, ts: Optional[float] = None) -> None:
        """Record (ts, price) for `mint`. Non-positive prices are dropped."""
        if not mint or price is None:
            return
        try:
            price = float(price)
        except (TypeError, ValueError):
            return
        if price <= 0:
            return
        if ts is None:
            ts = time.time()
        dq = self._buf.get(mint)
        if dq is None:
            dq = deque(maxlen=self._maxlen)
            self._buf[mint] = dq
        dq.append(PricePoint(ts=ts, price=price))

    def snapshot(self, mint: str) -> List[PricePoint]:
        """Return a copy of the deque for `mint`, oldest first."""
        dq = self._buf.get(mint)
        return list(dq) if dq else []

    def prices(self, mint: str) -> List[float]:
        """Just the prices in temporal order."""
        return [p.price for p in self.snapshot(mint)]

    def has_enough(self, mint: str, n: int) -> bool:
        """True iff the buffer holds at least `n` bars for `mint`."""
        dq = self._buf.get(mint)
        return dq is not None and len(dq) >= n

    def __len__(self) -> int:
        """Number of tokens currently tracked (NOT total points)."""
        return len(self._buf)

    def size(self, mint: str) -> int:
        dq = self._buf.get(mint)
        return len(dq) if dq else 0

    def drop(self, mint: str) -> None:
        self._buf.pop(mint, None)

    def evict_stale(self, now: Optional[float] = None) -> int:
        """Drop tokens whose newest bar is older than `max_age_s`.

        Returns the number of tokens evicted. No-op when `max_age_s`
        is unset.
        """
        if not self._max_age_s:
            return 0
        if now is None:
            now = time.time()
        cutoff = now - self._max_age_s
        stale = [
            mint
            for mint, dq in self._buf.items()
            if not dq or dq[-1].ts < cutoff
        ]
        for mint in stale:
            self._buf.pop(mint, None)
        return len(stale)
