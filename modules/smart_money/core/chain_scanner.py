"""SMART_MONEY data acquisition — FREE sources only, read-only.

- Pair discovery + USD prices: DexScreener public API (no key).
- Swap events: EVM ``eth_getLogs`` over the watched pairs, RPC URLs ONLY via
  config.pool_engine.PoolEngine (get_endpoint + report_success/failure).
- Wallet attribution: ``eth_getTransactionByHash`` -> tx ``from``, budget-capped
  to the largest swaps so free-tier RPC quotas are respected.

Coverage limit (honest): v1 scans EVM chains only. Solana swap attribution
needs program-level parsing and is documented as a follow-up in CLAUDE.md.
Everything here is fail-soft: any error returns empty results, never raises.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("smart_money")

DEXSCREENER_BASE = "https://api.dexscreener.com"

# keccak topic0 of UniswapV2 Swap(address,uint256,uint256,uint256,uint256,address)
V2_SWAP_TOPIC = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
# keccak topic0 of UniswapV3 Swap(address,address,int256,int256,uint160,uint128,int24)
V3_SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

# Approximate seconds per block, for estimating event time from block offset.
BLOCK_SECONDS = {"ethereum": 12.0, "base": 2.0, "arbitrum": 0.3, "bsc": 3.0,
                 "polygon": 2.0, "avalanche": 2.0, "fantom": 1.0, "cronos": 6.0}

DEFAULT_SEARCH_QUERY = {"ethereum": "WETH", "base": "WETH", "arbitrum": "WETH",
                        "bsc": "WBNB", "polygon": "WMATIC"}

_GETLOGS_CHUNK_BLOCKS = 450          # stay under public-RPC getLogs range caps
_DECIMALS_SELECTOR = "0x313ce567"    # decimals()

# (chain, token) -> decimals; tokens are few (watched pairs only), cache forever
_decimals_cache: Dict[tuple, int] = {}


def _u256(word: str) -> int:
    return int(word, 16)


def _i256(word: str) -> int:
    v = int(word, 16)
    return v - (1 << 256) if v >= (1 << 255) else v


def _words(data: str) -> List[str]:
    d = data[2:] if data.startswith("0x") else data
    return [d[i:i + 64] for i in range(0, len(d), 64)]


# ───────────────────────────── DexScreener ──────────────────────────────────

async def discover_pairs(session, chain: str, cfg: dict) -> List[dict]:
    """Top watched pairs for a chain by 24h volume, liquidity-filtered."""
    query = str(cfg.get(f"search_query_{chain}",
                        DEFAULT_SEARCH_QUERY.get(chain, "WETH")))
    min_liq = float(cfg.get("min_pair_liquidity_usd", 100000))
    min_vol = float(cfg.get("min_pair_volume_24h_usd", 250000))
    cap = int(cfg.get("watch_pairs_per_chain", 12))
    try:
        async with session.get(f"{DEXSCREENER_BASE}/latest/dex/search",
                               params={"q": query}, timeout=15) as resp:
            if resp.status != 200:
                logger.warning("dexscreener search %s -> HTTP %d", chain, resp.status)
                return []
            payload = await resp.json()
    except Exception as exc:
        logger.warning("dexscreener search %s fail-soft: %s", chain, exc)
        return []
    out = []
    for p in (payload or {}).get("pairs") or []:
        try:
            if p.get("chainId") != chain:
                continue
            liq = float((p.get("liquidity") or {}).get("usd") or 0)
            vol = float((p.get("volume") or {}).get("h24") or 0)
            price = float(p.get("priceUsd") or 0)
            if liq < min_liq or vol < min_vol or price <= 0:
                continue
            out.append({
                "pair_address": p["pairAddress"].lower(),
                "base_token": p["baseToken"]["address"].lower(),
                "base_symbol": p["baseToken"].get("symbol", "?"),
                "quote_token": p["quoteToken"]["address"].lower(),
                "price_usd": price, "volume_h24": vol,
            })
        except Exception:
            continue
    out.sort(key=lambda x: x["volume_h24"], reverse=True)
    return out[:cap]


async def fetch_token_prices(session, chain: str,
                             tokens: List[str]) -> Dict[str, float]:
    """Current USD price per token (highest-liquidity pair on the chain).
    Used for LATE forward-return marks — never for anticipating the future."""
    prices: Dict[str, float] = {}
    for i in range(0, len(tokens), 30):                 # API max 30 per call
        batch = tokens[i:i + 30]
        try:
            async with session.get(
                    f"{DEXSCREENER_BASE}/latest/dex/tokens/{','.join(batch)}",
                    timeout=15) as resp:
                if resp.status != 200:
                    continue
                payload = await resp.json()
        except Exception as exc:
            logger.warning("dexscreener tokens fail-soft: %s", exc)
            continue
        best_liq: Dict[str, float] = {}
        for p in (payload or {}).get("pairs") or []:
            try:
                if p.get("chainId") != chain:
                    continue
                token = p["baseToken"]["address"].lower()
                liq = float((p.get("liquidity") or {}).get("usd") or 0)
                price = float(p.get("priceUsd") or 0)
                if price > 0 and liq >= best_liq.get(token, -1.0):
                    best_liq[token] = liq
                    prices[token] = price
            except Exception:
                continue
    return prices


# ───────────────────────────── EVM JSON-RPC ─────────────────────────────────

async def _rpc(session, pool_engine, chain: str, method: str,
               params: list) -> Optional[object]:
    """One JSON-RPC call through the shared PoolEngine (single RPC source)."""
    provider_type = f"{chain.upper()}_RPC"
    url = await pool_engine.get_endpoint(provider_type)
    if not url:
        return None
    t0 = time.monotonic()
    try:
        async with session.post(url, json={"jsonrpc": "2.0", "id": 1,
                                           "method": method, "params": params},
                                timeout=20) as resp:
            if resp.status == 429:
                await pool_engine.report_rate_limit(provider_type, url,
                                                    duration_seconds=60)
                return None
            body = await resp.json()
        if "error" in (body or {}):
            await pool_engine.report_failure(provider_type, url,
                                             error=str(body["error"])[:200])
            return None
        await pool_engine.report_success(
            provider_type, url, latency_ms=(time.monotonic() - t0) * 1000)
        return body.get("result")
    except Exception as exc:
        try:
            await pool_engine.report_failure(provider_type, url, error=str(exc)[:200])
        except Exception:
            pass
        return None


async def _token_decimals(session, pool_engine, chain: str, token: str) -> int:
    key = (chain, token)
    if key in _decimals_cache:
        return _decimals_cache[key]
    res = await _rpc(session, pool_engine, chain, "eth_call",
                     [{"to": token, "data": _DECIMALS_SELECTOR}, "latest"])
    dec = 18
    try:
        if res and res != "0x":
            dec = int(res, 16)
            if not (0 < dec <= 36):
                dec = 18
    except Exception:
        dec = 18
    _decimals_cache[key] = dec
    return dec


def _decode_base_amount(log: dict, base_is_token0: bool) -> Optional[Tuple[str, int]]:
    """-> (side, base_amount_raw) from a V2/V3 Swap log; None = not decodable."""
    topic0 = (log.get("topics") or [""])[0].lower()
    w = _words(log.get("data") or "0x")
    try:
        if topic0 == V2_SWAP_TOPIC and len(w) >= 4:
            a0_in, a1_in, a0_out, a1_out = (_u256(w[0]), _u256(w[1]),
                                            _u256(w[2]), _u256(w[3]))
            base_in = a0_in if base_is_token0 else a1_in
            base_out = a0_out if base_is_token0 else a1_out
            if base_out > 0 and base_in == 0:
                return "buy", base_out          # pool pays base out -> buyer
            if base_in > 0 and base_out == 0:
                return "sell", base_in
            return None
        if topic0 == V3_SWAP_TOPIC and len(w) >= 2:
            base_amt = _i256(w[0]) if base_is_token0 else _i256(w[1])
            if base_amt < 0:
                return "buy", -base_amt         # negative = pool sends base out
            if base_amt > 0:
                return "sell", base_amt
    except Exception:
        return None
    return None


async def scan_chain(session, pool_engine, chain: str, pairs: List[dict],
                     cursor: Optional[int], cfg: dict) -> Tuple[List[dict], Optional[int]]:
    """Scan Swap logs on the watched pairs since cursor; attribute wallets.

    Returns (events, new_cursor). Events carry est_ts approximated from the
    head-block timestamp and block spacing; events older than
    max_event_age_minutes are dropped so price_usd_at_event (the CURRENT
    DexScreener price) stays an honest at-event mark.
    """
    if not pairs:
        return [], cursor
    head_hex = await _rpc(session, pool_engine, chain, "eth_blockNumber", [])
    head_block_obj = await _rpc(session, pool_engine, chain,
                                "eth_getBlockByNumber", ["latest", False])
    if head_hex is None or head_block_obj is None:
        return [], cursor
    head = int(head_hex, 16)
    head_ts = int(head_block_obj.get("timestamp", "0x0"), 16) or time.time()
    blk_secs = BLOCK_SECONDS.get(chain, 12.0)
    max_blocks = int(cfg.get("max_blocks_per_tick", 600))
    start = (cursor + 1) if cursor is not None else \
        head - int(cfg.get("initial_lookback_blocks", 300))
    start = max(start, head - max_blocks, 0)
    if start > head:
        return [], cursor

    pair_meta = {p["pair_address"]: p for p in pairs}
    min_usd = float(cfg.get("min_event_usd", 2000))
    max_age_s = float(cfg.get("max_event_age_minutes", 30)) * 60.0
    lookup_budget = int(cfg.get("max_wallet_lookups_per_tick", 60))
    now = time.time()
    raw_swaps: List[dict] = []

    frm = start
    while frm <= head:
        to = min(frm + _GETLOGS_CHUNK_BLOCKS - 1, head)
        logs = await _rpc(session, pool_engine, chain, "eth_getLogs", [{
            "fromBlock": hex(frm), "toBlock": hex(to),
            "address": list(pair_meta.keys()),
            "topics": [[V2_SWAP_TOPIC, V3_SWAP_TOPIC]],
        }])
        if logs is None:
            return [], cursor       # keep cursor; retry the range next tick
        for log in logs:
            meta = pair_meta.get((log.get("address") or "").lower())
            if not meta:
                continue
            base_is_token0 = meta["base_token"] < meta["quote_token"]
            decoded = _decode_base_amount(log, base_is_token0)
            if not decoded:
                continue
            side, raw_amt = decoded
            blk = int(log.get("blockNumber", "0x0"), 16)
            est_ts = head_ts - (head - blk) * blk_secs
            if now - est_ts > max_age_s:
                continue
            dec = await _token_decimals(session, pool_engine, chain,
                                        meta["base_token"])
            amount_usd = (raw_amt / (10 ** dec)) * meta["price_usd"]
            if amount_usd < min_usd:
                continue
            raw_swaps.append({
                "chain": chain, "token": meta["base_token"],
                "token_symbol": meta["base_symbol"], "side": side,
                "amount_usd": amount_usd, "price_usd": meta["price_usd"],
                "est_ts": est_ts, "tx_hash": log.get("transactionHash"),
                "log_index": int(log.get("logIndex", "0x0"), 16),
            })
        frm = to + 1

    # Wallet attribution: largest swaps first, capped by the RPC budget.
    raw_swaps.sort(key=lambda s: s["amount_usd"], reverse=True)
    tx_from: Dict[str, Optional[str]] = {}
    events: List[dict] = []
    for s in raw_swaps:
        txh = s["tx_hash"]
        if txh not in tx_from:
            if len(tx_from) >= lookup_budget:
                continue
            tx = await _rpc(session, pool_engine, chain,
                            "eth_getTransactionByHash", [txh])
            tx_from[txh] = (tx or {}).get("from", "").lower() or None
        wallet = tx_from.get(txh)
        if wallet:
            events.append({**s, "wallet": wallet})
    return events, head
