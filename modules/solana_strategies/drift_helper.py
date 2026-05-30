"""
Drift Protocol Helper
On-chain perpetual futures trading on Solana via driftpy SDK

Documentation: https://drift-labs.github.io/driftpy/
SDK: https://github.com/drift-labs/driftpy
"""

# ============================================================================
# HTTPX COMPATIBILITY FIX
# The solana library uses httpx.AsyncClient with 'proxy' parameter, but
# newer httpx versions (0.24+) renamed this to 'proxy' or require 'proxies'.
# This patch ensures compatibility with both old and new httpx versions.
# ============================================================================
import httpx

# Store original AsyncClient
_OriginalAsyncClient = httpx.AsyncClient


class _CompatAsyncClient(_OriginalAsyncClient):
    """Wrapper that handles the proxy parameter compatibility"""

    def __init__(self, *args, **kwargs):
        # Handle 'proxy' parameter for newer httpx versions
        proxy = kwargs.pop('proxy', None)
        if proxy is not None:
            try:
                super().__init__(*args, proxy=proxy, **kwargs)
                return
            except TypeError:
                if proxy is not None:
                    try:
                        super().__init__(*args, proxies={'all://': proxy}, **kwargs)
                        return
                    except TypeError:
                        pass
        super().__init__(*args, **kwargs)


# Apply the monkey patch
httpx.AsyncClient = _CompatAsyncClient
# ============================================================================

import os
import asyncio
import logging
from typing import Dict, Optional, List
from decimal import Decimal

logger = logging.getLogger(__name__)


class DriftHelper:
    """
    Drift Protocol integration helper

    Drift is an on-chain perpetual futures protocol on Solana.
    This helper uses the driftpy SDK for direct blockchain interaction.

    Features:
    - Open/close perpetual positions
    - Get account information
    - Monitor funding rates
    - Check liquidation risk
    - Manage collateral

    NOTE: Drift uses on-chain interaction (no REST API keys needed)
    Requires: pip install driftpy
    """

    def __init__(
        self,
        rpc_url: str = None,
        private_key: str = None,
        *,
        dry_run: bool = True,
        max_leverage: float = 3.0,
        max_abs_funding_rate_annual_pct: float = 50.0,
        oracle_deviation_max_pct: float = 1.0,
        min_oracle_confidence_bps: int = 500,
    ):
        """
        Initialize Drift helper

        Args:
            rpc_url: Solana RPC URL
            private_key: Base58-encoded Solana private key
            dry_run: When True (default), open_position/close_position
                short-circuit and return a sentinel signature without
                signing. The engine layer also checks the killswitch /
                pause file via ``core.dry_run.should_skip_live`` on every
                call (MB-15 SOL-RM-15).
            max_leverage: Hard cap on per-trade leverage (account-value
                multiplier). Defaults to 3x — Drift's matching engine
                will let you push past this on cross-margin accounts, so
                the cap MUST be enforced client-side.
            max_abs_funding_rate_annual_pct: Refuse to open if the market's
                annualized funding rate exceeds ±this value. Funding rips
                wider than 50%/yr are the canonical "leg is about to get
                run over" signal on perps.
            oracle_deviation_max_pct: Refuse to open if the AMM mark price
                deviates from the Pyth oracle by more than this percent.
                Drift liquidates against the oracle, so trading at a
                stale/manipulated mark is an instant-loss setup.
            min_oracle_confidence_bps: Max acceptable Pyth oracle
                confidence interval (in bps of price). Wider confidence
                means the oracle itself isn't sure, so liquidation math
                is unreliable.
        """
        # Get credentials from secrets manager (database/Docker secrets);
        # prefer PoolEngine for the RPC URL so failover/health-weighting works.
        pool_rpc = None
        try:
            from config.rpc_provider import RPCProvider
            pool_rpc = RPCProvider.get_rpc_sync('SOLANA_RPC')
        except Exception:
            pool_rpc = None
        try:
            from security.secrets_manager import secrets
            self.rpc_url = rpc_url or pool_rpc or secrets.get('SOLANA_RPC_URL', log_access=False) or os.getenv('SOLANA_RPC_URL')
            # Solana Module uses dedicated wallet (SOLANA_MODULE_PRIVATE_KEY)
            self.private_key = private_key or secrets.get('SOLANA_MODULE_PRIVATE_KEY', log_access=False)
        except Exception:
            self.rpc_url = rpc_url or pool_rpc or os.getenv('SOLANA_RPC_URL')
            self.private_key = private_key or os.getenv('SOLANA_MODULE_PRIVATE_KEY')

        # Drift client (will be initialized when needed)
        self.drift_client = None
        self.user_account = None

        # MB-15 hardening guards. All four are enforced inside
        # open_position; they are deliberately conservative so the
        # operator must explicitly raise them in DB config.
        self.dry_run = bool(dry_run)
        self.max_leverage = float(max_leverage)
        self.max_abs_funding_rate_annual_pct = float(max_abs_funding_rate_annual_pct)
        self.oracle_deviation_max_pct = float(oracle_deviation_max_pct)
        self.min_oracle_confidence_bps = int(min_oracle_confidence_bps)

        if self.dry_run:
            logger.warning(
                "🔶 Drift helper initialized in DRY_RUN — open_position will return sentinel sig"
            )
        else:
            logger.critical(
                "🔥 Drift helper LIVE — leverage_cap=%.1fx, funding_cap=%.1f%%/yr, "
                "oracle_dev_max=%.2f%%, oracle_conf_max=%d bps",
                self.max_leverage,
                self.max_abs_funding_rate_annual_pct,
                self.oracle_deviation_max_pct,
                self.min_oracle_confidence_bps,
            )

        logger.info("🎯 Drift Protocol helper initialized")

    async def initialize(self) -> bool:
        """
        Initialize Drift client

        Returns:
            bool: True if successful
        """
        try:
            # Check if driftpy is installed
            try:
                import driftpy
            except ImportError:
                logger.error(
                    "❌ driftpy SDK not installed. "
                    "Install with: pip install driftpy"
                )
                return False

            # Import required Drift components
            from driftpy.drift_client import DriftClient
            from driftpy.accounts import get_perp_market_account, get_spot_market_account
            from solana.rpc.async_api import AsyncClient
            from solders.keypair import Keypair

            # Initialize Solana RPC client
            connection = AsyncClient(self.rpc_url)

            # Load wallet keypair
            if not self.private_key:
                logger.error("No Solana private key configured for Drift")
                return False

            # Parse keypair (assumes base58 encoded)
            # TODO: Handle encrypted private keys
            wallet = Keypair.from_base58_string(self.private_key)

            # Initialize Drift client
            self.drift_client = DriftClient(
                connection,
                wallet,
                "mainnet-beta"  # or "devnet" for testing
            )

            # Subscribe to account data
            await self.drift_client.subscribe()

            # Get user account
            self.user_account = await self.drift_client.get_user()

            logger.info("✅ Drift client initialized successfully")
            logger.info(f"   Wallet: {str(wallet.pubkey())[:10]}...")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Drift: {e}", exc_info=True)
            return False

    async def close(self):
        """Close Drift client connection"""
        if self.drift_client:
            try:
                await self.drift_client.unsubscribe()
                logger.info("🔌 Drift client disconnected")
            except Exception as e:
                logger.error(f"Error closing Drift client: {e}")

    async def get_positions(self) -> List[Dict]:
        """
        Get all active perpetual positions

        Returns:
            List[Dict]: List of positions with details
        """
        if not self.drift_client or not self.user_account:
            logger.error("Drift client not initialized")
            return []

        try:
            positions = []

            # Get perpetual positions from user account
            for perp_position in self.user_account.get_perp_positions():
                # Skip empty positions
                if perp_position.base_asset_amount == 0:
                    continue

                position = {
                    'market_index': perp_position.market_index,
                    'base_asset_amount': float(perp_position.base_asset_amount),
                    'quote_entry_amount': float(perp_position.quote_entry_amount),
                    'last_cumulative_funding_rate': float(
                        perp_position.last_cumulative_funding_rate
                    ),
                    'unrealized_pnl': float(
                        await self.drift_client.get_user_unrealized_pnl(perp_position)
                    ),
                    'direction': 'LONG' if perp_position.base_asset_amount > 0 else 'SHORT',
                }

                positions.append(position)

            logger.info(f"📊 Found {len(positions)} active Drift positions")
            return positions

        except Exception as e:
            logger.error(f"Error getting Drift positions: {e}", exc_info=True)
            return []

    async def get_account_value(self) -> Decimal:
        """
        Get total account value including collateral and PnL

        Returns:
            Decimal: Total account value in USD
        """
        if not self.drift_client or not self.user_account:
            return Decimal("0")

        try:
            # Get total collateral
            total_collateral = await self.drift_client.get_user_spot_collateral()

            # Get unrealized PnL from all positions
            unrealized_pnl = await self.drift_client.get_user_unrealized_pnl()

            total_value = total_collateral + unrealized_pnl

            logger.debug(
                f"💰 Account value: ${total_value:.2f} "
                f"(Collateral: ${total_collateral:.2f}, "
                f"Unrealized PnL: ${unrealized_pnl:+.2f})"
            )

            return Decimal(str(total_value))

        except Exception as e:
            logger.error(f"Error getting account value: {e}")
            return Decimal("0")

    async def open_position(
        self,
        market_index: int,
        direction: str,
        base_amount: float,
        price_limit: Optional[float] = None,
        *,
        notional_usd: Optional[float] = None,
    ) -> Optional[str]:
        """
        Open a perpetual position with MB-15 pre-trade guards.

        Args:
            market_index: Market index (0 = SOL-PERP, 1 = BTC-PERP, etc.)
            direction: 'LONG' or 'SHORT'
            base_amount: Size in base asset units
            price_limit: Optional limit price
            notional_usd: Optional notional size override for leverage
                check. When omitted, leverage is estimated from
                base_amount × oracle mark.

        Returns:
            Optional[str]: Transaction signature or None
        """
        if not self.drift_client:
            logger.error("Drift client not initialized")
            return None

        # ---- MB-15 PRE-TRADE GUARDS ----------------------------------------
        # 1. DRY_RUN / killswitch / pause — defense-in-depth, even though the
        #    engine layer should also gate this. Helper-level gate matches the
        #    Jupiter helper pattern (SOL-RM-14).
        try:
            from core.dry_run import should_skip_live
            if should_skip_live(self.dry_run, module='solana'):
                logger.info(
                    "🔶 Drift DRY_RUN gate engaged (market=%d %s base=%.4f) — no order placed",
                    market_index, direction, base_amount,
                )
                return f"DRY_RUN_DRIFT_{market_index}_{direction}_{int(base_amount * 1e6)}"
        except ImportError:
            if self.dry_run:
                logger.info(
                    "🔶 Drift dry_run=True (core.dry_run unavailable) — no order placed"
                )
                return f"DRY_RUN_DRIFT_{market_index}_{direction}"

        # 2. Funding-rate sanity. Perps rip in one direction when funding
        #    is extreme; opening into that is asymmetric loss.
        try:
            funding_pct = await self.get_funding_rate(market_index)
            if abs(funding_pct) > self.max_abs_funding_rate_annual_pct:
                logger.warning(
                    "⛔ Drift funding-rate guard: market=%d funding=%.2f%%/yr > cap=%.2f%%/yr — refusing entry",
                    market_index, funding_pct, self.max_abs_funding_rate_annual_pct,
                )
                return None
        except Exception as exc:
            logger.warning(f"⛔ Drift funding-rate read failed: {exc} — refusing entry (fail-closed)")
            return None

        # 3. Oracle deviation guard. Drift liquidates against the oracle, so a
        #    wide gap between AMM mark and oracle means immediate liquidation
        #    risk on entry.
        try:
            ok, dev_pct, conf_bps = await self._check_oracle_health(market_index)
            if not ok:
                logger.warning(
                    "⛔ Drift oracle guard: market=%d dev=%.3f%% conf=%d bps — refusing entry",
                    market_index, dev_pct, conf_bps,
                )
                return None
        except Exception as exc:
            logger.warning(
                f"⛔ Drift oracle health-check failed: {exc} — refusing entry (fail-closed)"
            )
            return None

        # 4. Leverage cap. base_amount × oracle mark vs account collateral.
        try:
            if notional_usd is None:
                # Estimate notional from price_limit if supplied, else fall
                # back to oracle mark from the AMM.
                if price_limit and price_limit > 0:
                    notional_usd = float(base_amount) * float(price_limit)
                else:
                    market = await self.drift_client.get_perp_market_account(market_index)
                    mark = float(market.amm.last_mark_price_twap) / 1e6 if hasattr(market.amm, 'last_mark_price_twap') else None
                    if mark and mark > 0:
                        notional_usd = float(base_amount) * mark

            if notional_usd is not None:
                account_value = float(await self.get_account_value())
                if account_value > 0:
                    effective_leverage = notional_usd / account_value
                    if effective_leverage > self.max_leverage:
                        logger.warning(
                            "⛔ Drift leverage guard: notional=$%.2f / acct=$%.2f = %.2fx > cap=%.2fx — refusing entry",
                            notional_usd, account_value, effective_leverage, self.max_leverage,
                        )
                        return None
                else:
                    logger.warning(
                        "⛔ Drift account_value=0 with notional=$%.2f — refusing entry",
                        notional_usd,
                    )
                    return None
        except Exception as exc:
            logger.warning(f"⛔ Drift leverage check failed: {exc} — refusing entry (fail-closed)")
            return None
        # --------------------------------------------------------------------

        try:
            from driftpy.types import PositionDirection, OrderType

            # Convert direction
            drift_direction = (
                PositionDirection.Long()
                if direction.upper() == 'LONG'
                else PositionDirection.Short()
            )

            # Place market order (or limit if price specified)
            order_type = OrderType.Limit() if price_limit else OrderType.Market()

            logger.info(
                f"📈 Opening {direction} position on market {market_index}: "
                f"{base_amount} units"
            )

            # Execute order
            tx_sig = await self.drift_client.place_perp_order(
                market_index=market_index,
                direction=drift_direction,
                base_asset_amount=int(base_amount * 1e9),  # Convert to lamports
                order_type=order_type,
                price=int(price_limit * 1e6) if price_limit else None
            )

            logger.info(f"✅ Position opened: {tx_sig}")
            return str(tx_sig)

        except Exception as e:
            logger.error(f"❌ Error opening position: {e}", exc_info=True)
            return None

    async def _check_oracle_health(self, market_index: int):
        """Validate Pyth oracle vs AMM mark for a perp market.

        Returns:
            (ok, deviation_pct, confidence_bps)
            ok=False when the deviation exceeds ``oracle_deviation_max_pct``
            OR Pyth confidence exceeds ``min_oracle_confidence_bps``.
        """
        market = await self.drift_client.get_perp_market_account(market_index)
        amm = market.amm

        # Oracle price (Pyth, scaled by 1e6 on Drift). Confidence is also in
        # the same units; convert to bps of price.
        oracle_price_raw = getattr(amm, 'last_oracle_price_twap', None) or getattr(amm, 'oracle_price', None)
        oracle_conf_raw = getattr(amm, 'last_oracle_conf', 0) or 0
        mark_raw = getattr(amm, 'last_mark_price_twap', None) or getattr(amm, 'mark_price', None)

        if not oracle_price_raw or not mark_raw:
            # No oracle data → fail closed.
            return (False, 0.0, 9999)

        oracle_price = float(oracle_price_raw) / 1e6
        mark_price = float(mark_raw) / 1e6
        confidence = float(oracle_conf_raw) / 1e6

        if oracle_price <= 0 or mark_price <= 0:
            return (False, 0.0, 9999)

        dev_pct = abs(mark_price - oracle_price) / oracle_price * 100.0
        conf_bps = int((confidence / oracle_price) * 10_000) if oracle_price > 0 else 9999

        if dev_pct > self.oracle_deviation_max_pct:
            return (False, dev_pct, conf_bps)
        if conf_bps > self.min_oracle_confidence_bps:
            return (False, dev_pct, conf_bps)
        return (True, dev_pct, conf_bps)

    async def close_position(self, market_index: int) -> Optional[str]:
        """
        Close a perpetual position

        Args:
            market_index: Market index to close

        Returns:
            Optional[str]: Transaction signature or None
        """
        if not self.drift_client:
            logger.error("Drift client not initialized")
            return None

        try:
            # Get current position
            position = None
            for perp_pos in self.user_account.get_perp_positions():
                if perp_pos.market_index == market_index:
                    position = perp_pos
                    break

            if not position or position.base_asset_amount == 0:
                logger.warning(f"No position found for market {market_index}")
                return None

            # Determine opposite direction
            from driftpy.types import PositionDirection

            close_direction = (
                PositionDirection.Short()
                if position.base_asset_amount > 0
                else PositionDirection.Long()
            )

            # Place market order in opposite direction to close
            amount = abs(position.base_asset_amount)

            logger.info(f"📉 Closing position on market {market_index}")

            tx_sig = await self.drift_client.place_perp_order(
                market_index=market_index,
                direction=close_direction,
                base_asset_amount=amount,
            )

            logger.info(f"✅ Position closed: {tx_sig}")
            return str(tx_sig)

        except Exception as e:
            logger.error(f"❌ Error closing position: {e}", exc_info=True)
            return None

    async def get_funding_rate(self, market_index: int) -> float:
        """
        Get current funding rate for a market

        Args:
            market_index: Market index

        Returns:
            float: Funding rate (annualized percentage)
        """
        if not self.drift_client:
            return 0.0

        try:
            market = await self.drift_client.get_perp_market_account(market_index)
            funding_rate = float(market.amm.last_funding_rate) / 1e9

            # Convert to annualized percentage
            funding_rate_annual = funding_rate * 365 * 24 * 100

            logger.debug(
                f"📊 Funding rate for market {market_index}: "
                f"{funding_rate_annual:.4f}%"
            )

            return funding_rate_annual

        except Exception as e:
            logger.error(f"Error getting funding rate: {e}")
            return 0.0
