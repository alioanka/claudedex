"""ExecutionGateway — MEV-aware EVM send policy (the execution sibling of
config/pool_engine: pool_engine answers 'which RPC do I read from', this
answers 'how do I SEND safely'). Routes signed txs through private order flow
(Flashbots Protect / MEV-Blocker-style RPC) with public-RPC fallback, and
centralizes nonce + gas policy + the should_skip_live live-gate at the send
boundary. Library only — it never originates a trade."""

import inspect
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from core.dry_run import should_skip_live
from modules.execution_gateway.core.gas_policy import (
    GasQuote, LEGACY_GAS_CHAINS, compute_gas,
)
from modules.execution_gateway.core.nonce_manager import NonceManager
from modules.execution_gateway.core.route_policy import (
    ROUTE_PRIVATE, ROUTE_PUBLIC, RoutePlan, select_route,
)

logger = logging.getLogger("execution_gateway")

CONFIG_TYPE = 'execution_gateway'
_CONFIG_TTL_SECONDS = 60.0


class GatewayError(Exception):
    pass


class GatewayRPCError(GatewayError):
    def __init__(self, message: str, *, rate_limited: bool = False):
        super().__init__(message)
        self.rate_limited = rate_limited


@dataclass
class TxIntent:
    chain: str                                   # 'ethereum', 'base', ...
    tx: Optional[dict] = None                    # unsigned tx fields
    signed_raw: Optional[str] = None             # 0x-hex raw signed tx
    sign_fn: Optional[Callable[[dict], Any]] = None  # tx dict -> raw signed
    sender: Optional[str] = None                 # required for unsigned intents
    direction: str = 'entry'                     # 'entry' | 'exit'
    notional_usd: Optional[float] = None
    urgency: str = 'normal'                      # 'normal' | 'fast' | 'rescue'
    prefer_private: Optional[bool] = None        # per-call policy override
    tag: Optional[str] = None                    # caller correlation id


@dataclass
class SendResult:
    ok: bool
    simulated: bool = False
    route: Optional[str] = None
    tx_hash: Optional[str] = None
    fallback_used: bool = False
    error: Optional[str] = None
    gas: Optional[GasQuote] = None
    nonce: Optional[int] = None
    reason: str = ''


class ExecutionGateway:
    """One instance per calling module. db_pool (asyncpg) is optional —
    without it config comes from env defaults and auditing is skipped."""

    def __init__(self, module: str, *, db_pool=None):
        self.module = module
        self.db_pool = db_pool
        self.nonces = NonceManager()
        self.counters: Dict[str, int] = {
            'sent_private': 0, 'sent_public': 0, 'fallbacks': 0,
            'simulated': 0, 'errors': 0,
        }
        self._cfg: dict = {}
        self._cfg_loaded_at = 0.0

    # ----- config -----------------------------------------------------------
    async def load_config(self, force: bool = False) -> dict:
        now = time.monotonic()
        if not force and self._cfg and now - self._cfg_loaded_at < _CONFIG_TTL_SECONDS:
            return self._cfg
        cfg: dict = {}
        if self.db_pool is not None:
            try:
                rows = await self.db_pool.fetch(
                    "SELECT key, value FROM config_settings WHERE config_type=$1",
                    CONFIG_TYPE,
                )
                cfg = {r['key']: r['value'] for r in rows}
            except Exception as exc:
                logger.warning("execution_gateway config load fail-soft: %s", exc)
        self._cfg, self._cfg_loaded_at = cfg, now
        return cfg

    def _private_url(self, chain: str, cfg: dict) -> Optional[str]:
        url = (cfg.get(f'private_rpc_url_{chain}') or '').strip()
        if url:
            return url
        if chain == 'ethereum':
            return (os.getenv('FLASHBOTS_RPC') or '').strip() or None
        return None

    # ----- RPC plumbing -----------------------------------------------------
    async def _rpc_call(self, url: str, method: str, params: list,
                        timeout: float = 15.0) -> Any:
        import aiohttp  # lazy: keeps the module import-safe without aiohttp
        payload = {'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 429:
                    raise GatewayRPCError(f'{method}: HTTP 429', rate_limited=True)
                if resp.status >= 400:
                    raise GatewayRPCError(f'{method}: HTTP {resp.status}')
                body = await resp.json(content_type=None)
        if isinstance(body, dict) and body.get('error'):
            raise GatewayRPCError(f"{method}: {body['error']}")
        return body.get('result') if isinstance(body, dict) else None

    async def _public_rpc(self, chain: str) -> Optional[str]:
        try:
            from config.pool_engine import get_pool
            pool = await get_pool()
            return await pool.get_endpoint(f'{chain.upper()}_RPC')
        except Exception as exc:
            logger.error("pool_engine endpoint lookup failed for %s: %s", chain, exc)
            return None

    async def _report_public(self, chain: str, url: str, ok: bool,
                             rate_limited: bool = False, error: str = None) -> None:
        try:
            from config.pool_engine import get_pool
            pool = await get_pool()
            ptype = f'{chain.upper()}_RPC'
            if ok:
                await pool.report_success(ptype, url)
            elif rate_limited:
                await pool.report_rate_limit(ptype, url, error_message=error)
            else:
                await pool.report_failure(ptype, url, error_message=error)
        except Exception:
            pass

    # ----- tx preparation ---------------------------------------------------
    async def _prepare_unsigned(self, intent: TxIntent, cfg: dict,
                                public_rpc: str) -> tuple:
        """Fill gas + nonce on a copy of intent.tx and sign it. Returns
        (raw_hex, gas_quote, reserved_nonce)."""
        chain = intent.chain.lower()
        tx = dict(intent.tx or {})
        gas: Optional[GasQuote] = None

        has_fees = ('gasPrice' in tx) or (
            'maxFeePerGas' in tx and 'maxPriorityFeePerGas' in tx)
        if not has_fees:
            base_fee = 0
            try:
                blk = await self._rpc_call(public_rpc, 'eth_getBlockByNumber',
                                           ['latest', False])
                base_fee = int(blk.get('baseFeePerGas', '0x0'), 16) if blk else 0
            except Exception as exc:
                logger.warning("baseFee fetch failed (%s); using priority floor only", exc)
            gas = compute_gas(base_fee, cfg, chain=chain, urgency=intent.urgency)
            if chain in LEGACY_GAS_CHAINS and gas.legacy_gas_price:
                tx['gasPrice'] = gas.legacy_gas_price
            else:
                tx['maxFeePerGas'] = gas.max_fee_per_gas
                tx['maxPriorityFeePerGas'] = gas.max_priority_fee_per_gas

        reserved = None
        if 'nonce' not in tx:
            async def _fetch() -> int:
                res = await self._rpc_call(
                    public_rpc, 'eth_getTransactionCount',
                    [intent.sender, 'pending'])
                return int(res, 16)
            reserved = await self.nonces.reserve(chain, intent.sender, _fetch)
            tx['nonce'] = reserved

        signed = intent.sign_fn(tx)
        if inspect.isawaitable(signed):
            signed = await signed
        raw = getattr(signed, 'raw_transaction', None) or \
            getattr(signed, 'rawTransaction', None) or signed
        if isinstance(raw, (bytes, bytearray)):
            raw = '0x' + raw.hex()
        if not isinstance(raw, str) or not raw.startswith('0x'):
            raise GatewayError('sign_fn did not return raw signed tx hex/bytes')
        return raw, gas, reserved

    # ----- the choke point --------------------------------------------------
    async def send(self, intent: TxIntent, *, dry_run: bool) -> SendResult:
        """Single safe-send entry point. Honors should_skip_live (module
        DRY_RUN / global killswitch / logs/.pause_<module>) BEFORE any
        broadcast, by construction."""
        chain = (intent.chain or '').strip().lower()
        cfg = await self.load_config()
        plan = select_route(
            chain, cfg, direction=intent.direction,
            notional_usd=intent.notional_usd,
            prefer_private=intent.prefer_private,
            private_url=self._private_url(chain, cfg),
        )

        if should_skip_live(dry_run, module=self.module):
            self.counters['simulated'] += 1
            res = SendResult(ok=True, simulated=True, route=plan.route,
                             reason=f'should_skip_live; would route: {plan.reason}')
            await self._audit(intent, res)
            return res

        raw, gas, reserved = intent.signed_raw, None, None
        public_rpc = await self._public_rpc(chain)
        try:
            if raw is None:
                if not (intent.tx and intent.sign_fn and intent.sender):
                    raise GatewayError(
                        'unsigned intent requires tx + sign_fn + sender')
                if public_rpc is None:
                    raise GatewayError(f'no public RPC available for {chain}')
                raw, gas, reserved = await self._prepare_unsigned(
                    intent, cfg, public_rpc)
            res = await self._broadcast(plan, chain, raw, cfg, public_rpc)
        except Exception as exc:
            if reserved is not None:
                self.nonces.release(chain, intent.sender, reserved)
            self.counters['errors'] += 1
            res = SendResult(ok=False, route=plan.route, error=str(exc)[:500],
                             reason=plan.reason)
        else:
            if not res.ok and reserved is not None:
                self.nonces.release(chain, intent.sender, reserved)
            if res.error and 'nonce' in res.error.lower() and intent.sender:
                self.nonces.resync(chain, intent.sender)
        res.gas, res.nonce = gas, reserved
        await self._audit(intent, res)
        return res

    async def _broadcast(self, plan: RoutePlan, chain: str, raw: str,
                         cfg: dict, public_rpc: Optional[str]) -> SendResult:
        try:
            timeout = float(cfg.get('private_send_timeout_seconds', 30) or 30)
        except (TypeError, ValueError):
            timeout = 30.0

        if plan.route == ROUTE_PRIVATE:
            url = self._private_url(chain, cfg)
            try:
                tx_hash = await self._rpc_call(
                    url, 'eth_sendRawTransaction', [raw], timeout=timeout)
                self.counters['sent_private'] += 1
                return SendResult(ok=True, route=ROUTE_PRIVATE, tx_hash=tx_hash,
                                  reason=plan.reason)
            except Exception as exc:
                logger.warning("private send failed (%s); fallback=%s",
                               exc, plan.fallback)
                if plan.fallback != ROUTE_PUBLIC:
                    self.counters['errors'] += 1
                    return SendResult(ok=False, route=ROUTE_PRIVATE,
                                      error=f'private send failed, no fallback: {exc}'[:500],
                                      reason=plan.reason)
                self.counters['fallbacks'] += 1
                res = await self._send_public(chain, raw, public_rpc)
                res.fallback_used = True
                res.reason = f'{plan.reason}; private failed -> public fallback'
                return res

        res = await self._send_public(chain, raw, public_rpc)
        res.reason = plan.reason
        return res

    async def _send_public(self, chain: str, raw: str,
                           public_rpc: Optional[str]) -> SendResult:
        url = public_rpc or await self._public_rpc(chain)
        if not url:
            self.counters['errors'] += 1
            return SendResult(ok=False, route=ROUTE_PUBLIC,
                              error=f'no public RPC available for {chain}')
        try:
            tx_hash = await self._rpc_call(url, 'eth_sendRawTransaction', [raw])
            await self._report_public(chain, url, True)
            self.counters['sent_public'] += 1
            return SendResult(ok=True, route=ROUTE_PUBLIC, tx_hash=tx_hash)
        except GatewayRPCError as exc:
            await self._report_public(chain, url, False,
                                      rate_limited=exc.rate_limited,
                                      error=str(exc))
            self.counters['errors'] += 1
            return SendResult(ok=False, route=ROUTE_PUBLIC, error=str(exc)[:500])
        except Exception as exc:
            await self._report_public(chain, url, False, error=str(exc))
            self.counters['errors'] += 1
            return SendResult(ok=False, route=ROUTE_PUBLIC, error=str(exc)[:500])

    # ----- audit (fail-soft, optional) --------------------------------------
    async def _audit(self, intent: TxIntent, res: SendResult) -> None:
        if self.db_pool is None:
            return
        cfg = self._cfg or {}
        if str(cfg.get('audit_enabled', 'true')).lower() not in ('true', '1', 'yes', 'on'):
            return
        try:
            await self.db_pool.execute(
                """INSERT INTO execution_gateway_sends
                   (module, chain, direction, route, status, tx_hash,
                    fallback_used, simulated, nonce, max_fee_per_gas,
                    max_priority_fee_per_gas, notional_usd, tag, error)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)""",
                self.module, intent.chain, intent.direction, res.route,
                'ok' if res.ok else 'error', res.tx_hash, res.fallback_used,
                res.simulated, res.nonce,
                res.gas.max_fee_per_gas if res.gas else None,
                res.gas.max_priority_fee_per_gas if res.gas else None,
                intent.notional_usd, intent.tag,
                res.error,
            )
        except Exception as exc:
            logger.debug("gateway audit skipped (fail-soft): %s", exc)
