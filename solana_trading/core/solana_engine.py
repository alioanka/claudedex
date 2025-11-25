"""
Solana Trading Engine
Core engine for trading on Solana blockchain

Strategies:
- Jupiter V6: Token swaps via Jupiter aggregator
- Drift Protocol: Perpetuals trading
- Pump.fun: New token sniping
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import os
import base58

logger = logging.getLogger("SolanaTradingEngine")


class SolanaTradingEngine:
    """Core Solana trading engine"""

    def __init__(
        self,
        rpc_url: str,
        strategies: List[str],
        max_positions: int = 3,
        mode: str = "production"
    ):
        """
        Initialize Solana trading engine

        Args:
            rpc_url: Solana RPC URL
            strategies: List of enabled strategies (jupiter, drift, pumpfun)
            max_positions: Maximum concurrent positions
            mode: Operating mode
        """
        self.rpc_url = rpc_url
        self.strategies = [s.strip().lower() for s in strategies]
        self.max_positions = max_positions
        self.mode = mode
        self.is_running = False

        # Trading state
        self.active_positions: Dict = {}
        self.pending_orders: Dict = {}

        # Solana connection
        self.client = None
        self.wallet = None

        # Strategy modules
        self.jupiter_client = None
        self.drift_client = None
        self.pumpfun_monitor = None

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl_sol = 0.0

        logger.info(f"Solana engine initialized with strategies: {', '.join(self.strategies)}")

    async def initialize(self):
        """Initialize Solana connections and components"""
        try:
            logger.info(f"Initializing Solana connection to {self.rpc_url}...")

            # Initialize Solana client
            await self._init_solana_client()

            # Initialize enabled strategies
            if 'jupiter' in self.strategies:
                await self._init_jupiter()

            if 'drift' in self.strategies:
                await self._init_drift()

            if 'pumpfun' in self.strategies or 'pump.fun' in self.strategies:
                await self._init_pumpfun()

            logger.info("✅ Solana connection and strategies initialized")

        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            raise

    async def _init_solana_client(self):
        """Initialize Solana RPC client and wallet"""
        try:
            from solana.rpc.async_api import AsyncClient
            from solders.keypair import Keypair

            # Create RPC client
            self.client = AsyncClient(self.rpc_url)

            # Load wallet from encrypted private key
            # IMPORTANT: Solana MODULE uses SOLANA_MODULE_PRIVATE_KEY (separate from DEX module's SOLANA_PRIVATE_KEY)
            encrypted_key = os.getenv('SOLANA_MODULE_PRIVATE_KEY')
            encryption_key = os.getenv('ENCRYPTION_KEY')

            if not encrypted_key:
                raise ValueError("SOLANA_MODULE_PRIVATE_KEY required for Solana trading module")

            # Decrypt private key if encrypted (starts with gAAAAAB for Fernet)
            private_key = encrypted_key
            if encrypted_key.startswith('gAAAAAB') and encryption_key:
                try:
                    from cryptography.fernet import Fernet
                    f = Fernet(encryption_key.encode())
                    private_key = f.decrypt(encrypted_key.encode()).decode()
                    logger.info("✅ Successfully decrypted Solana module private key")
                except Exception as e:
                    logger.error(f"Failed to decrypt Solana module private key: {e}")
                    raise ValueError("Cannot decrypt SOLANA_MODULE_PRIVATE_KEY - check ENCRYPTION_KEY")

            # Decode private key (base58 or hex format)
            try:
                # Try base58 format first
                key_bytes = base58.b58decode(private_key)
                self.wallet = Keypair.from_bytes(key_bytes)
            except:
                # Try hex format
                key_bytes = bytes.fromhex(private_key)
                self.wallet = Keypair.from_bytes(key_bytes)

            wallet_address = str(self.wallet.pubkey())
            logger.info(f"✅ Solana wallet loaded: {wallet_address[:8]}...{wallet_address[-8:]}")

        except ImportError:
            logger.error("solana library not installed. Install: pip install solana solders")
            raise
        except Exception as e:
            logger.error(f"Solana client initialization failed: {e}")
            raise

    async def _init_jupiter(self):
        """Initialize Jupiter V6 aggregator client"""
        try:
            logger.info("Initializing Jupiter V6 aggregator...")

            # Jupiter V6 API configuration
            jupiter_api_key = os.getenv('JUPITER_API_KEY', '')  # Optional
            jupiter_tier = os.getenv('JUPITER_TIER', 'public')  # lite, public, ultra

            # Store config for later use
            self.jupiter_config = {
                'api_key': jupiter_api_key,
                'tier': jupiter_tier,
                'base_url': 'https://quote-api.jup.ag/v6'
            }

            logger.info(f"✅ Jupiter configured (tier: {jupiter_tier})")

        except Exception as e:
            logger.error(f"Jupiter initialization failed: {e}")
            raise

    async def _init_drift(self):
        """Initialize Drift Protocol client"""
        try:
            logger.info("Initializing Drift Protocol...")

            # Import driftpy
            try:
                from driftpy.drift_client import DriftClient
                from driftpy.accounts import get_perp_market_account, get_spot_market_account

                # Initialize Drift client
                # This will be implemented with actual Drift SDK
                logger.info("✅ Drift Protocol configured")

            except ImportError:
                logger.warning("driftpy not installed. Drift trading disabled.")
                self.strategies = [s for s in self.strategies if s != 'drift']

        except Exception as e:
            logger.error(f"Drift initialization failed: {e}")
            raise

    async def _init_pumpfun(self):
        """Initialize Pump.fun monitoring"""
        try:
            logger.info("Initializing Pump.fun monitor...")

            # Pump.fun configuration
            self.pumpfun_config = {
                'program_id': os.getenv('PUMPFUN_PROGRAM_ID', '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P'),
                'min_liquidity_sol': float(os.getenv('PUMPFUN_MIN_LIQUIDITY', '10')),
                'max_age_seconds': int(os.getenv('PUMPFUN_MAX_AGE_SECONDS', '300'))
            }

            logger.info("✅ Pump.fun monitor configured")

        except Exception as e:
            logger.error(f"Pump.fun initialization failed: {e}")
            raise

    async def run(self):
        """Main trading loop"""
        self.is_running = True
        logger.info("🚀 Solana trading engine started")

        try:
            while self.is_running:
                try:
                    # Main trading logic
                    await self._trading_cycle()

                    # Wait before next cycle
                    await asyncio.sleep(5)

                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    await asyncio.sleep(15)

        except Exception as e:
            logger.error(f"Critical error in trading engine: {e}")
            raise

        finally:
            logger.info("Solana trading engine stopped")

    async def _trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # 1. Monitor existing positions
            await self._monitor_positions()

            # 2. Check for new opportunities per strategy
            if 'jupiter' in self.strategies:
                await self._scan_jupiter_opportunities()

            if 'drift' in self.strategies:
                await self._scan_drift_opportunities()

            if 'pumpfun' in self.strategies or 'pump.fun' in self.strategies:
                await self._scan_pumpfun_opportunities()

            # 3. Execute pending orders
            await self._process_orders()

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")

    async def _monitor_positions(self):
        """Monitor active positions for exit signals"""
        if not self.active_positions:
            return

        for token_address, position in list(self.active_positions.items()):
            try:
                # Get current price (implementation depends on strategy)
                current_price = await self._get_token_price(token_address)

                # Calculate PnL
                entry_price = position['entry_price']
                amount = position['amount']

                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                position['pnl_pct'] = pnl_pct
                position['current_price'] = current_price

                # Check exit conditions
                if await self._should_exit_position(position):
                    await self._close_position(token_address, position)

            except Exception as e:
                logger.error(f"Error monitoring position {token_address}: {e}")

    async def _scan_jupiter_opportunities(self):
        """Scan for Jupiter arbitrage/swap opportunities"""
        if len(self.active_positions) >= self.max_positions:
            return

        try:
            # Jupiter strategy logic
            # Example: Monitor SOL/USDC, SOL/USDT spreads
            logger.debug("Scanning Jupiter opportunities...")

        except Exception as e:
            logger.error(f"Error scanning Jupiter: {e}")

    async def _scan_drift_opportunities(self):
        """Scan for Drift perpetuals opportunities"""
        if len(self.active_positions) >= self.max_positions:
            return

        try:
            # Drift strategy logic
            logger.debug("Scanning Drift opportunities...")

        except Exception as e:
            logger.error(f"Error scanning Drift: {e}")

    async def _scan_pumpfun_opportunities(self):
        """Scan for new Pump.fun token launches"""
        if len(self.active_positions) >= self.max_positions:
            return

        try:
            # Pump.fun monitoring logic
            # Monitor new token launches and snipe promising ones
            logger.debug("Scanning Pump.fun launches...")

        except Exception as e:
            logger.error(f"Error scanning Pump.fun: {e}")

    async def _process_orders(self):
        """Process pending orders"""
        if not self.pending_orders:
            return

        for order_id, order in list(self.pending_orders.items()):
            try:
                logger.info(f"Processing order: {order}")
                # Order execution logic
                pass
            except Exception as e:
                logger.error(f"Error processing order {order_id}: {e}")

    async def _should_exit_position(self, position: Dict) -> bool:
        """Determine if position should be closed"""
        pnl_pct = position['pnl_pct']

        # Stop loss
        stop_loss_pct = float(os.getenv('SOLANA_STOP_LOSS_PCT', '-10.0'))
        if pnl_pct <= stop_loss_pct:
            logger.info(f"Stop loss triggered: {pnl_pct:.2f}%")
            return True

        # Take profit
        take_profit_pct = float(os.getenv('SOLANA_TAKE_PROFIT_PCT', '50.0'))
        if pnl_pct >= take_profit_pct:
            logger.info(f"Take profit triggered: {pnl_pct:.2f}%")
            return True

        return False

    async def _close_position(self, token_address: str, position: Dict):
        """Close a position"""
        try:
            logger.info(f"Closing position: {token_address[:8]}... (PnL: {position['pnl_pct']:.2f}%)")

            # Calculate PnL in SOL
            pnl_sol = position['amount'] * (position['current_price'] - position['entry_price'])

            # Update stats
            self.total_pnl_sol += pnl_sol
            self.total_trades += 1

            if pnl_sol > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            # Remove from active positions
            del self.active_positions[token_address]

            logger.info(f"✅ Position closed: {token_address[:8]}..., PnL: {pnl_sol:.4f} SOL")

        except Exception as e:
            logger.error(f"Error closing position {token_address}: {e}")

    async def _get_token_price(self, token_address: str) -> float:
        """Get current price for token"""
        try:
            # Fetch from Jupiter price API or on-chain data
            # Placeholder implementation
            return 1.0
        except Exception as e:
            logger.error(f"Error fetching price for {token_address}: {e}")
            return 0.0

    async def close_all_positions(self):
        """Close all open positions"""
        logger.info("Closing all positions...")

        for token_address, position in list(self.active_positions.items()):
            await self._close_position(token_address, position)

        logger.info("✅ All positions closed")

    async def get_stats(self) -> Dict:
        """Get trading statistics"""
        win_rate = 0
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100

        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': f"{win_rate:.1f}%",
            'total_pnl': f"{self.total_pnl_sol:.4f} SOL",
            'active_positions': len(self.active_positions),
            'strategies': ', '.join(self.strategies)
        }

    async def shutdown(self):
        """Shutdown the engine"""
        logger.info("Shutting down Solana trading engine...")
        self.is_running = False

        # Close Solana client connection
        if self.client:
            try:
                await self.client.close()
            except:
                pass

        logger.info("✅ Engine shutdown complete")
