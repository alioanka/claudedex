"""Shared Jito MEV-bundle client for Solana.

Lifted unchanged from `modules/arbitrage/solana_engine.py:297` so that
both the ARBITRAGE engine and the SOLANA spot engine can submit MEV-
protected bundles via the same class state. The class-level rate
counters (`_global_last_request_time`, `_global_backoff_until`,
`_global_consecutive_429s`) are now genuinely process-global -- any
module that constructs a `JitoClient` from this module shares the same
backoff window.

Public API (stable, do not break -- callers in arbitrage already exist):
    - `JitoClient()` constructor (no args)
    - `is_available()` classmethod -> `(bool, seconds_until)`
    - `initialize(keypair=None)` / `close()` lifecycle
    - `send_bundle(transactions, tip_lamports)` -> bundle_id or None
    - `create_tip_transaction(payer_keypair, tip_lamports, recent_blockhash)`
    - `get_random_tip_account()`
    - Class constants: `JITO_ENDPOINTS`, `JITO_TIP_ACCOUNTS`,
      `MIN_REQUEST_INTERVAL`.

Behavior is byte-for-byte the pre-lift implementation. No new logic.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional, Tuple

import aiohttp

from config.rpc_provider import RPCProvider
from security.secrets_manager import secrets

logger = logging.getLogger(__name__)


class JitoClient:
    """Jito MEV protection client (Solana's Flashbots).

    Lifted from arbitrage solana_engine -- see module docstring.
    """

    # Regional Jito block engines for failover (from official docs)
    # https://docs.jito.wtf/lowlatencytxnsend/
    JITO_ENDPOINTS = [
        "https://mainnet.block-engine.jito.wtf",
        "https://amsterdam.mainnet.block-engine.jito.wtf",
        "https://dublin.mainnet.block-engine.jito.wtf",
        "https://frankfurt.mainnet.block-engine.jito.wtf",
        "https://london.mainnet.block-engine.jito.wtf",
        "https://ny.mainnet.block-engine.jito.wtf",
        "https://slc.mainnet.block-engine.jito.wtf",
        "https://singapore.mainnet.block-engine.jito.wtf",
        "https://tokyo.mainnet.block-engine.jito.wtf",
    ]

    # Official Jito tip accounts (from getTipAccounts API response)
    JITO_TIP_ACCOUNTS = [
        "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
        "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
        "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
        "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
        "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
        "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
        "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
        "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
    ]

    # Global rate limit - Jito limits are SHARED across all endpoints
    _global_last_request_time = 0.0
    _global_backoff_until = 0.0
    _global_consecutive_429s = 0
    MIN_REQUEST_INTERVAL = 12.0  # seconds between bundle requests

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.keypair = None
        # secrets manager (DB-encrypted) first, env fallback, default endpoint last
        self.primary_endpoint = (
            secrets.get('JITO_BLOCK_ENGINE_URL', log_access=False)
            or os.getenv('JITO_BLOCK_ENGINE_URL', self.JITO_ENDPOINTS[0])
        )
        self.current_endpoint_idx = 0
        self._last_429_time = 0
        self._backoff_seconds = 0

    @classmethod
    def is_available(cls) -> Tuple[bool, float]:
        """Return (is_available, seconds_until_available)."""
        import time
        now = time.time()
        if cls._global_backoff_until > now:
            return False, cls._global_backoff_until - now
        time_since_last = now - cls._global_last_request_time
        if time_since_last < cls.MIN_REQUEST_INTERVAL:
            return False, cls.MIN_REQUEST_INTERVAL - time_since_last
        return True, 0.0

    async def initialize(self, keypair=None):
        timeout = aiohttp.ClientTimeout(total=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        self.keypair = keypair
        logger.info(f"   Jito endpoint: {self.primary_endpoint}")
        logger.info(f"   Jito rate limit: {self.MIN_REQUEST_INTERVAL}s between bundles")

    async def close(self):
        if self.session:
            await self.session.close()

    def _get_next_endpoint(self) -> str:
        self.current_endpoint_idx = (self.current_endpoint_idx + 1) % len(self.JITO_ENDPOINTS)
        return self.JITO_ENDPOINTS[self.current_endpoint_idx]

    def _handle_rate_limit(self, source: str):
        """Set global backoff -- Jito limits are GLOBAL, not per-endpoint."""
        import time
        JitoClient._global_consecutive_429s += 1
        backoff_seconds = min(10 * (2 ** (JitoClient._global_consecutive_429s - 1)), 60)
        JitoClient._global_backoff_until = time.time() + backoff_seconds
        logger.warning(
            f"⚠️ Jito rate limited ({source}): 429 "
            f"#{JitoClient._global_consecutive_429s}, global backoff {backoff_seconds}s"
        )

    async def send_bundle(
        self,
        transactions: List[str],
        tip_lamports: int = 10000,
    ) -> Optional[str]:
        """Send signed-base64 transactions as a Jito bundle.

        Returns bundle ID on landed confirmation, None otherwise. The
        tip_lamports arg is kept for signature compatibility; the actual
        tip transaction must already be in `transactions`.
        """
        import time
        now = time.time()

        if JitoClient._global_backoff_until > now:
            remaining = JitoClient._global_backoff_until - now
            logger.warning(
                f"⏳ Jito GLOBAL backoff: {remaining:.1f}s remaining "
                f"(after {JitoClient._global_consecutive_429s} 429s)"
            )
            return None

        time_since_last = now - JitoClient._global_last_request_time
        if time_since_last < self.MIN_REQUEST_INTERVAL:
            wait_time = self.MIN_REQUEST_INTERVAL - time_since_last
            logger.info(f"⏳ Jito rate limit: waiting {wait_time:.1f}s before next bundle")
            await asyncio.sleep(wait_time)
            now = time.time()

        JitoClient._global_last_request_time = now

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendBundle",
            "params": [transactions, {"encoding": "base64"}],
        }

        endpoints_to_try = [self.primary_endpoint, self._get_next_endpoint()]
        for i, endpoint in enumerate(endpoints_to_try):
            try:
                async with self.session.post(
                    f"{endpoint}/api/v1/bundles", json=payload
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if 'result' in result:
                            bundle_id = result.get('result')
                            logger.info(f"🛡️ Jito bundle submitted: {bundle_id}")
                            JitoClient._global_consecutive_429s = 0
                            JitoClient._global_backoff_until = 0
                            confirmed = await self._confirm_bundle(bundle_id, endpoint)
                            if confirmed:
                                logger.info(f"✅ Jito bundle CONFIRMED on-chain: {bundle_id}")
                                return bundle_id
                            logger.warning(
                                f"⚠️ Jito bundle NOT confirmed: {bundle_id} - bundle may have been dropped"
                            )
                            return None
                        elif 'error' in result:
                            error = result.get('error', {})
                            error_code = error.get('code', 'N/A')
                            error_msg = error.get('message', 'Unknown error')
                            if error_code == -32097 or 'rate limit' in error_msg.lower():
                                self._handle_rate_limit("API error")
                                if i < len(endpoints_to_try) - 1:
                                    await asyncio.sleep(5)
                                    continue
                                return None
                            logger.error(f"❌ Jito bundle error: {error_msg}")
                            logger.error(f"   Error code: {error_code}")
                            return None
                    elif resp.status == 429:
                        self._handle_rate_limit(f"HTTP 429 on {endpoint[:30]}")
                        if (
                            JitoClient._global_consecutive_429s <= 2
                            and i < len(endpoints_to_try) - 1
                        ):
                            await asyncio.sleep(5)
                            continue
                        return None
                    else:
                        try:
                            error_body = await resp.text()
                            logger.error(f"❌ Jito HTTP {resp.status} on {endpoint[:30]}...")
                            logger.error(f"   Response: {error_body[:500]}")
                        except Exception:
                            logger.error(f"❌ Jito HTTP {resp.status} on {endpoint[:30]}...")
                        continue
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Jito timeout on {endpoint[:30]}... trying next")
                continue
            except Exception as e:
                logger.warning(f"Jito error on {endpoint[:30]}...: {e}")
                continue

        logger.error("❌ All Jito endpoints failed or rate limited")
        return None

    async def _confirm_bundle(
        self,
        bundle_id: str,
        endpoint: str,
        timeout_seconds: int = 30,
    ) -> bool:
        """Poll getBundleStatuses until landed/finalized or timeout."""
        import time
        start_time = time.time()
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBundleStatuses",
            "params": [[bundle_id]],
        }
        check_interval = 1.0
        while (time.time() - start_time) < timeout_seconds:
            try:
                async with self.session.post(
                    f"{endpoint}/api/v1/bundles",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if 'result' in result and 'value' in result['result']:
                            statuses = result['result']['value']
                            if statuses and len(statuses) > 0:
                                bundle_status = statuses[0]
                                if bundle_status is None:
                                    logger.debug(
                                        f"Bundle {bundle_id[:16]}... not found yet, waiting..."
                                    )
                                else:
                                    cs = bundle_status.get('confirmation_status')
                                    err = bundle_status.get('err')
                                    if err:
                                        logger.error(f"❌ Jito bundle failed: {err}")
                                        return False
                                    if cs in ('finalized', 'confirmed'):
                                        slot = bundle_status.get('slot', 'unknown')
                                        logger.info(f"   Bundle landed in slot {slot}")
                                        return True
                                    if cs == 'processed':
                                        logger.debug(
                                            f"Bundle {bundle_id[:16]}... processing..."
                                        )
                                    else:
                                        logger.debug(f"Bundle status: {cs}")
            except asyncio.TimeoutError:
                logger.debug("Bundle status check timeout, retrying...")
            except Exception as e:
                logger.debug(f"Bundle status check error: {e}")
            await asyncio.sleep(check_interval)
            check_interval = min(check_interval * 1.5, 5.0)

        logger.warning(
            f"⚠️ Bundle confirmation timeout after {timeout_seconds}s - bundle may have been dropped"
        )
        return False

    def get_random_tip_account(self) -> str:
        """Random Jito tip account (secrets manager > env > random)."""
        import random
        env_tip = secrets.get('JITO_TIP_ACCOUNT', log_access=False) or os.getenv('JITO_TIP_ACCOUNT')
        if env_tip:
            return env_tip
        return random.choice(self.JITO_TIP_ACCOUNTS)

    async def create_tip_transaction(
        self,
        payer_keypair,
        tip_lamports: int = 10000,
        recent_blockhash: Optional[str] = None,
    ) -> Optional[str]:
        """Build + sign the mandatory Jito tip tx.

        The tip should be the LAST transaction in the bundle. Min 1000
        lamports per Jito docs. Returns a base64-encoded signed tx, or
        None on any error.
        """
        try:
            import base64

            from solders.hash import Hash
            from solders.message import Message
            from solders.pubkey import Pubkey
            from solders.system_program import TransferParams, transfer
            from solders.transaction import Transaction

            if not payer_keypair:
                logger.error("No keypair provided for tip transaction")
                return None

            tip_lamports = max(tip_lamports, 1000)
            tip_account_str = self.get_random_tip_account()
            tip_account = Pubkey.from_string(tip_account_str)
            payer_pubkey = payer_keypair.pubkey()
            logger.debug(
                f"Creating tip tx: {tip_lamports} lamports to {tip_account_str[:12]}..."
            )

            if not recent_blockhash:
                rpc_url = (
                    await RPCProvider.get_rpc('SOLANA_RPC')
                    or os.getenv('SOLANA_RPC_URL')
                    or 'https://api.mainnet-beta.solana.com'
                )
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        rpc_url,
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "getLatestBlockhash",
                            "params": [{"commitment": "finalized"}],
                        },
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            recent_blockhash = (
                                data.get('result', {}).get('value', {}).get('blockhash')
                            )

            if not recent_blockhash:
                logger.error("Failed to get recent blockhash for tip transaction")
                return None

            transfer_ix = transfer(
                TransferParams(
                    from_pubkey=payer_pubkey,
                    to_pubkey=tip_account,
                    lamports=tip_lamports,
                )
            )
            blockhash = Hash.from_string(recent_blockhash)
            message = Message.new_with_blockhash([transfer_ix], payer_pubkey, blockhash)
            tx = Transaction.new_unsigned(message)
            tx.sign([payer_keypair], blockhash)
            tx_bytes = bytes(tx)
            tx_b64 = base64.b64encode(tx_bytes).decode('utf-8')
            logger.info(
                f"✅ Tip transaction created: {tip_lamports} lamports to {tip_account_str[:12]}..."
            )
            return tx_b64
        except Exception as e:
            logger.error(f"Failed to create tip transaction: {e}")
            return None


__all__ = ['JitoClient']
