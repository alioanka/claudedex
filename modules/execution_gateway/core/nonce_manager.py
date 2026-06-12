"""Per-(chain, sender) async nonce allocator. Network access is injected."""

import asyncio
from typing import Awaitable, Callable, Dict, Tuple


class NonceManager:
    """Serializes nonce hand-out per sender so concurrent sends from one
    module never collide. fetch_pending_nonce is injected (the gateway passes
    an RPC eth_getTransactionCount('pending') closure); tests pass a stub."""

    def __init__(self) -> None:
        self._locks: Dict[Tuple[str, str], asyncio.Lock] = {}
        self._next: Dict[Tuple[str, str], int] = {}

    def _key(self, chain: str, sender: str) -> Tuple[str, str]:
        return ((chain or '').lower(), (sender or '').lower())

    def _lock(self, key: Tuple[str, str]) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def reserve(
        self,
        chain: str,
        sender: str,
        fetch_pending_nonce: Callable[[], Awaitable[int]],
    ) -> int:
        key = self._key(chain, sender)
        async with self._lock(key):
            cached = self._next.get(key)
            if cached is None:
                cached = int(await fetch_pending_nonce())
            self._next[key] = cached + 1
            return cached

    def release(self, chain: str, sender: str, nonce: int) -> None:
        """Roll back a reservation whose broadcast never reached any node.
        Only undoes the head reservation; out-of-order failures resync instead."""
        key = self._key(chain, sender)
        if self._next.get(key) == nonce + 1:
            self._next[key] = nonce
        else:
            self._next.pop(key, None)

    def resync(self, chain: str, sender: str) -> None:
        """Drop the cache; next reserve() refetches from the chain."""
        self._next.pop(self._key(chain, sender), None)


if __name__ == '__main__':
    async def _test():
        nm = NonceManager()
        calls = {'n': 0}

        async def fetch():
            calls['n'] += 1
            return 7

        a = await nm.reserve('ethereum', '0xAbC', fetch)
        b = await nm.reserve('ethereum', '0xabc', fetch)  # case-insensitive key
        assert (a, b) == (7, 8) and calls['n'] == 1
        # concurrent reservations are unique
        got = await asyncio.gather(*[nm.reserve('ethereum', '0xabc', fetch) for _ in range(5)])
        assert sorted(got) == [9, 10, 11, 12, 13] and calls['n'] == 1
        # head rollback reuses the nonce
        nm.release('ethereum', '0xabc', 13)
        assert await nm.reserve('ethereum', '0xabc', fetch) == 13
        # out-of-order release forces a refetch
        nm.release('ethereum', '0xabc', 9)
        assert await nm.reserve('ethereum', '0xabc', fetch) == 7 and calls['n'] == 2
        # resync drops cache; other chains are independent
        nm.resync('ethereum', '0xabc')
        assert await nm.reserve('base', '0xabc', fetch) == 7
        print('nonce_manager self-test OK')

    asyncio.run(_test())
