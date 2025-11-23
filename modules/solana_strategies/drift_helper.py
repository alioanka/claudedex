"""
Drift Protocol Helper
On-chain perpetual futures trading on Solana via driftpy SDK

Documentation: https://drift-labs.github.io/driftpy/
SDK: https://github.com/drift-labs/driftpy
"""
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

    def __init__(self, rpc_url: str = None, private_key: str = None):
        """
        Initialize Drift helper

        Args:
            rpc_url: Solana RPC URL
            private_key: Base58-encoded Solana private key
        """
        self.rpc_url = rpc_url or os.getenv('SOLANA_RPC_URL')
        self.private_key = private_key or os.getenv('SOLANA_PRIVATE_KEY')

        # Drift client (will be initialized when needed)
        self.drift_client = None
        self.user_account = None

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
        price_limit: Optional[float] = None
    ) -> Optional[str]:
        """
        Open a perpetual position

        Args:
            market_index: Market index (0 = SOL-PERP, 1 = BTC-PERP, etc.)
            direction: 'LONG' or 'SHORT'
            base_amount: Size in base asset units
            price_limit: Optional limit price

        Returns:
            Optional[str]: Transaction signature or None
        """
        if not self.drift_client:
            logger.error("Drift client not initialized")
            return None

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
