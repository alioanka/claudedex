# trading/executors/direct_dex.py
"""
Direct DEX Executor for On-chain Trading
Interacts directly with DEX smart contracts
"""

import asyncio
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from web3 import Web3
from web3.contract import Contract
from web3.middleware import geth_poa_middleware
from eth_account import Account
from eth_abi import encode_abi

from trading.orders.order_manager import Order, OrderType, OrderStatus
from trading.executors.base_executor import BaseExecutor
from utils.helpers import retry_async, measure_time, wei_to_ether, ether_to_wei
from utils.constants import DEX, DEX_ROUTERS, CHAIN_RPC_URLS

logger = logging.getLogger(__name__)

# DEX Router ABIs (simplified)
UNISWAP_V2_ABI = json.loads('[{"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}]')

UNISWAP_V3_ABI = json.loads('[{"inputs":[{"components":[{"internalType":"bytes","name":"path","type":"bytes"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"},{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMinimum","type":"uint256"}],"internalType":"struct ISwapRouter.ExactInputParams","name":"params","type":"tuple"}],"name":"exactInput","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"payable","type":"function"}]')

@dataclass
class DEXQuote:
    """Quote from direct DEX query"""
    dex: DEX
    path: List[str]
    amount_in: Decimal
    amount_out: Decimal
    price: Decimal
    price_impact: float
    gas_estimate: int
    liquidity: Decimal
    fee: Decimal

class DirectDEXExecutor(BaseExecutor):
    """
    Direct on-chain DEX trading executor
    Features:
    - Direct smart contract interaction
    - Multi-DEX support (Uniswap V2/V3, PancakeSwap, SushiSwap)
    - Optimal path finding
    - Sandwich attack protection
    - Gas optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # ✅ CRITICAL: DRY_RUN mode check
        self.dry_run = config.get('DRY_RUN', True)
        if self.dry_run:
            logger.warning("🔶 DIRECT DEX IN DRY RUN MODE - NO REAL TRANSACTIONS 🔶")
        else:
            logger.critical("🔥 DIRECT DEX IN LIVE MODE - REAL MONEY AT RISK 🔥")
        
        # Web3 connections for each chain
        self.w3_connections: Dict[str, Web3] = {}
        
        # DEX contracts
        self.dex_contracts: Dict[str, Dict[str, Contract]] = {}
        
        # Path finding parameters
        self.max_hops = config.get('max_hops', 3)
        self.common_bases = config.get('common_bases', {
            'ethereum': [
                '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',  # WETH
                '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',  # USDC
                '0xdAC17F958D2ee523a2206206994597C13D831ec7',  # USDT
                '0x6B175474E89094C44Da98b954EedeAC495271d0F',  # DAI
            ]
        })
        
        # Gas optimization
        self.gas_price_strategy = config.get('gas_strategy', 'fast')
        self.max_gas_price = config.get('max_gas_price', 500)  # gwei
        self.gas_buffer = config.get('gas_buffer', 1.2)
        
        # Sandwich protection
        self.use_commit_reveal = config.get('commit_reveal', False)
        self.randomize_gas = config.get('randomize_gas', True)
        self.delay_blocks = config.get('delay_blocks', 0)
        
        # Performance tracking
        self.route_cache: Dict[str, List[str]] = {}
        self.gas_estimates: Dict[str, int] = {}
        
        logger.info("Direct DEX Executor initialized")
        self.active_orders = {}  # Track active orders
        
    async def initialize(self) -> None:
        """Initialize Web3 connections and contracts"""
        try:
            # Initialize Web3 for each chain
            for chain, rpc_url in CHAIN_RPC_URLS.items():
                w3 = Web3(Web3.HTTPProvider(rpc_url))
                
                # Add middleware for PoA chains
                if chain in ['bsc', 'polygon']:
                    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                    
                if w3.isConnected():
                    self.w3_connections[chain] = w3
                    logger.info(f"Connected to {chain} at block {w3.eth.block_number}")
                    
                    # Initialize DEX contracts for this chain
                    await self._initialize_dex_contracts(chain, w3)
                else:
                    logger.warning(f"Failed to connect to {chain}")
                    
        except Exception as e:
            logger.error(f"Failed to initialize Direct DEX Executor: {e}")
            raise
            
    async def _initialize_dex_contracts(self, chain: str, w3: Web3) -> None:
        """Initialize DEX contracts for a chain"""
        self.dex_contracts[chain] = {}
        
        for dex in DEX:
            if chain in DEX_ROUTERS.get(dex, {}):
                router_address = DEX_ROUTERS[dex][chain]
                
                # Load appropriate ABI based on DEX type
                if 'v3' in dex.value.lower():
                    abi = UNISWAP_V3_ABI
                else:
                    abi = UNISWAP_V2_ABI
                    
                contract = w3.eth.contract(
                    address=Web3.toChecksumAddress(router_address),
                    abi=abi
                )
                
                self.dex_contracts[chain][dex.value] = contract
                logger.info(f"Initialized {dex.value} on {chain}")
                
    # PATCH for trading/executors/direct_dex.py - Complete the try block for get_best_quote

    @measure_time
    async def get_best_quote(
        self,
        token_in: str,
        token_out: str,
        amount: Decimal,
        chain: str = 'ethereum'
        ) -> Optional[DEXQuote]:
        """Get best quote across all DEXes"""
        try:
            quotes = []
            
            # Query each available DEX
            for dex_name, contract in self.dex_contracts.get(chain, {}).items():
                quote = await self._get_dex_quote(
                    dex_name,
                    contract,
                    token_in,
                    token_out,
                    amount,
                    chain
                )
                
                if quote:
                    quotes.append(quote)
                    
            if not quotes:
                logger.warning(f"No quotes found for {token_in} -> {token_out}")
                return None
            
            # Find the best quote based on output amount
            best_quote = max(quotes, key=lambda q: q.amount_out)
            
            logger.info(
                f"Best quote from {best_quote.dex.value}: "
                f"{amount} {token_in} -> "
                f"{best_quote.amount_out} {token_out}"
            )
            
            return best_quote
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting quotes for {token_in} -> {token_out}")
            return None
            
        except ConnectionError as e:
            logger.error(f"Network error getting quotes: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error getting quotes for {token_in} -> {token_out}: {e}")
            return None
            
        finally:
            # Log quote aggregation stats
            if 'quotes' in locals() and quotes:
                logger.debug(f"Aggregated {len(quotes)} quotes from DEXes")
            
    @measure_time
    async def execute_trade(self, order: Order, quote=None) -> Dict[str, Any]:
        """Execute trade directly on DEX"""
        try:
            # ✅ CRITICAL: DRY_RUN CHECK
            if self.dry_run:
                logger.info(f"🔶 DRY RUN: Simulating DEX trade for {order.token_in} -> {order.token_out}")
                return await self._simulate_dex_trade(order)
            
            # Get best quote if not provided
            if not quote:
                quote = await self.get_best_quote(
                    order.token_in,
                    order.token_out,
                    order.amount,
                    order.chain
                )
            
            if not quote:
                return {
                    'success': False,
                    'error': 'No valid quotes found'
                }
                
            # Validate slippage
            max_slippage = float(order.slippage) if order.slippage else 0.05
            if quote.price_impact > max_slippage:
                return {
                    'success': False,
                    'error': f'Price impact too high: {quote.price_impact:.2%}'
                }
            
            # ✅ NEW: Approve tokens if needed (for token -> token swaps)
            if order.token_in != self._get_native_token_address(order.chain):
                approved = await self._approve_token_if_needed(
                    order.token_in,
                    self.dex_contracts[order.chain][quote.dex.value].address,
                    order.amount,
                    order.chain
                )
                if not approved:
                    return {
                        'success': False,
                        'error': 'Token approval failed'
                    }
                
            # Build transaction
            tx = await self._build_swap_transaction(order, quote)
            
            # Apply MEV protection
            if self.use_commit_reveal:
                tx = await self._apply_commit_reveal(tx)
                
            if self.randomize_gas:
                tx = self._randomize_gas_price(tx)
            
            # ✅ FIXED: Get account from config, not from order
            w3 = self.w3_connections[order.chain]
            
            # Sign transaction with wallet from config
            from eth_account import Account
            account = Account.from_key(self.config.get('private_key'))
            signed_tx = account.sign_transaction(tx)
            
            # Send with retry logic
            tx_hash = await self._send_transaction_with_retry(
                signed_tx,
                order.chain
            )
            
            # ✅ FIXED: Increased timeout to 300s
            receipt = await self._wait_for_confirmation(tx_hash, order.chain, timeout=300)
            
            return {
                'success': receipt['status'] == 1,
                'tx_hash': tx_hash.hex(),
                'gasUsed': receipt['gasUsed'],
                'blockNumber': receipt['blockNumber'],
                'execution_price': float(quote.amount_out) / float(order.amount),
                'amount': float(order.amount),
                'token_amount': float(quote.amount_out),
                'dex': quote.dex.value,
                'slippage_actual': quote.price_impact,
                'route': str(quote.path)
            }
            
        except Exception as e:
            logger.error(f"Direct DEX execution failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'tx_hash': None
            }

    async def _simulate_dex_trade(self, order: Order) -> Dict[str, Any]:
        """Simulate DEX trade for paper trading"""
        try:
            import random
            import time
            
            # Get real quote
            quote = await self.get_best_quote(
                order.token_in,
                order.token_out,
                order.amount,
                order.chain
            )
            
            if not quote:
                return {
                    'success': False,
                    'error': 'No quotes available for simulation'
                }
            
            # Simulate delay
            await asyncio.sleep(random.uniform(1.0, 3.0))
            
            # Simulate gas
            simulated_gas = quote.gas_estimate
            w3 = self.w3_connections.get(order.chain)
            simulated_gas_price = w3.eth.gas_price if w3 else 50 * 10**9
            
            logger.info(f"✅ DRY RUN: Would swap {order.amount} on {quote.dex.value}")
            logger.info(f"   Output: {quote.amount_out} tokens")
            logger.info(f"   Price Impact: {quote.price_impact:.2%}")
            
            return {
                'success': True,
                'tx_hash': f"0xDRYRUN{int(time.time())}{random.randint(1000, 9999)}",
                'gasUsed': simulated_gas,
                'blockNumber': 0,
                'execution_price': float(quote.amount_out) / float(order.amount),
                'amount': float(order.amount),
                'token_amount': float(quote.amount_out),
                'dex': quote.dex.value,
                'slippage_actual': quote.price_impact,
                'route': str(quote.path),
                'metadata': {'dry_run': True}
            }
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return {
                'success': False,
                'error': f"Simulation failed: {str(e)}"
            }

    async def _approve_token_if_needed(
        self,
        token_address: str,
        spender: str,
        amount: Decimal,
        chain: str
    ) -> bool:
        """Approve token spending if needed"""
        try:
            w3 = self.w3_connections[chain]
            
            # Load ERC20 ABI (minimal)
            erc20_abi = [
                {
                    "constant": True,
                    "inputs": [
                        {"name": "owner", "type": "address"},
                        {"name": "spender", "type": "address"}
                    ],
                    "name": "allowance",
                    "outputs": [{"name": "", "type": "uint256"}],
                    "type": "function"
                },
                {
                    "constant": False,
                    "inputs": [
                        {"name": "spender", "type": "address"},
                        {"name": "amount", "type": "uint256"}
                    ],
                    "name": "approve",
                    "outputs": [{"name": "", "type": "bool"}],
                    "type": "function"
                }
            ]
            
            token = w3.eth.contract(
                address=Web3.toChecksumAddress(token_address),
                abi=erc20_abi
            )
            
            # Get wallet address
            from eth_account import Account
            account = Account.from_key(self.config.get('private_key'))
            wallet = account.address
            
            # Check current allowance
            amount_wei = ether_to_wei(amount)
            current_allowance = token.functions.allowance(wallet, spender).call()
            
            if current_allowance >= amount_wei:
                logger.info(f"✅ Token already approved")
                return True
            
            logger.info(f"🔄 Approving token {token_address} for {spender}")
            
            # Build approval tx
            approve_tx = token.functions.approve(
                spender,
                amount_wei
            ).build_transaction({
                'from': wallet,
                'gas': 100000,
                'gasPrice': w3.eth.gas_price,
                'nonce': w3.eth.get_transaction_count(wallet),
                'chainId': w3.eth.chain_id
            })
            
            # Sign and send
            signed = account.sign_transaction(approve_tx)
            tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
            
            # Wait for confirmation
            logger.info(f"⏳ Waiting for approval: {tx_hash.hex()}")
            receipt = await self._wait_for_confirmation(tx_hash, chain, timeout=300)
            
            if receipt['status'] == 1:
                logger.info(f"✅ Token approved successfully")
                return True
            else:
                logger.error(f"❌ Token approval failed")
                return False
                
        except Exception as e:
            logger.error(f"Token approval error: {e}", exc_info=True)
            return False

    async def get_quote(self, token_in: str, token_out: str, amount: float, chain: str = 'ethereum'):
        """
        Get quote for a trade (BaseExecutor interface)
        
        Args:
            token_in: Input token address
            token_out: Output token address
            amount: Amount to trade
            chain: Chain name
            
        Returns:
            Quote dictionary
        """
        quote = await self.get_best_quote(token_in, token_out, Decimal(str(amount)), chain)
        
        if not quote:
            return {
                'input_token': token_in,
                'output_token': token_out,
                'input_amount': amount,
                'output_amount': 0,
                'route': 'no_route_found'
            }
        
        return {
            'input_token': token_in,
            'output_token': token_out,
            'input_amount': amount,
            'output_amount': float(quote.amount_out),
            'route': quote.dex.value,
            'price': float(quote.price),
            'price_impact': quote.price_impact,
            'gas_estimate': quote.gas_estimate
        }

    def _get_native_token_address(self, chain: str) -> str:
        """Get native token address for chain"""
        native_tokens = {
            'ethereum': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',  # WETH
            'bsc': '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c',  # WBNB
            'polygon': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',  # WMATIC
            'arbitrum': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',  # WETH
            'base': '0x4200000000000000000000000000000000000006'  # WETH
        }
        return native_tokens.get(chain.lower(), native_tokens['ethereum'])


    async def _build_swap_transaction(
        self,
        order: Order,
        quote: DEXQuote
    ) -> Dict[str, Any]:
        """Build swap transaction"""
        try:
            w3 = self.w3_connections[order.chain]
            contract = self.dex_contracts[order.chain][quote.dex.value]
            
            # Calculate minimum output with slippage
            min_amount_out = int(
                quote.amount_out * (1 - float(order.slippage or self.max_slippage)) * 10**18
            )
            
            # Deadline (20 minutes from now)
            deadline = int((datetime.now() + timedelta(minutes=20)).timestamp())
            
            # Build transaction based on DEX type
            if 'v3' in quote.dex.value.lower():
                # Uniswap V3 exact input
                tx_data = contract.encodeABI(
                    fn_name='exactInput',
                    args=[{
                        'path': self._encode_v3_path(quote.path),
                        'recipient': order.recipient or order.wallet_address,
                        'deadline': deadline,
                        'amountIn': ether_to_wei(order.amount),
                        'amountOutMinimum': min_amount_out
                    }]
                )
            else:
                # Uniswap V2 style swap
                tx_data = contract.encodeABI(
                    fn_name='swapExactTokensForTokens',
                    args=[
                        ether_to_wei(order.amount),
                        min_amount_out,
                        quote.path,
                        order.recipient or order.wallet_address,
                        deadline
                    ]
                )
                
            # Get optimal gas price
            gas_price = await self._get_optimal_gas_price(order.chain)
            
            # Build transaction
            tx = {
                'from': order.wallet_address,
                'to': contract.address,
                'data': tx_data,
                'gas': int(quote.gas_estimate * self.gas_buffer),
                'gasPrice': gas_price,
                'nonce': w3.eth.get_transaction_count(order.wallet_address),
                'chainId': w3.eth.chain_id
            }
            
            return tx
            
        except Exception as e:
            logger.error(f"Error building transaction: {e}")
            raise
            
    async def _estimate_price_impact(
        self,
        dex: str,
        path: List[str],
        amount: int,
        chain: str
    ) -> float:
        """Estimate price impact of trade"""
        try:
            # Get output for small amount
            small_amount = amount // 1000
            small_output = await self._simulate_swap(dex, path, small_amount, chain)
            
            # Get output for actual amount
            actual_output = await self._simulate_swap(dex, path, amount, chain)
            
            # Calculate impact
            expected_output = (small_output * amount) // small_amount
            
            if expected_output > 0:
                impact = 1 - (actual_output / expected_output)
                return max(0, impact)
            
            return 0
            
        except Exception as e:
            logger.debug(f"Error estimating price impact: {e}")
            return 0
            
    async def _estimate_gas(
        self,
        dex: str,
        path: List[str],
        amount: int,
        chain: str
    ) -> int:
        """Estimate gas for swap"""
        # Base gas estimates
        base_gas = {
            'UNISWAP_V2': 150000,
            'UNISWAP_V3': 180000,
            'PANCAKESWAP': 150000,
            'SUSHISWAP': 160000
        }
        
        # Add gas per hop
        hop_gas = 30000
        
        estimated = base_gas.get(dex.upper(), 200000)
        estimated += (len(path) - 2) * hop_gas
        
        return estimated
        
    async def _get_liquidity(
        self,
        token_a: str,
        token_b: str,
        chain: str
    ) -> Decimal:
        """Get liquidity for pair"""
        # Would query pair contract for reserves
        # Simplified for now
        return Decimal('1000000')
        
    def _calculate_fee(self, dex: str, amount: Decimal) -> Decimal:
        """Calculate DEX fee"""
        fee_rates = {
            'UNISWAP_V2': Decimal('0.003'),
            'UNISWAP_V3': Decimal('0.003'),
            'PANCAKESWAP': Decimal('0.0025'),
            'SUSHISWAP': Decimal('0.003')
        }
        
        rate = fee_rates.get(dex.upper(), Decimal('0.003'))
        return amount * rate
        
    def _encode_v3_path(self, tokens: List[str]) -> bytes:
        """Encode path for Uniswap V3"""
        # Simplified encoding - would need fee tiers
        path = b''
        for i, token in enumerate(tokens):
            path += bytes.fromhex(token[2:])  # Remove 0x
            if i < len(tokens) - 1:
                # Add fee tier (3000 = 0.3%)
                path += (3000).to_bytes(3, 'big')
        return path
        
    def _randomize_gas_price(self, tx: Dict) -> Dict:
        """Randomize gas price for MEV protection"""
        import random
        
        # Add random variation to gas price (±5%)
        variation = random.uniform(0.95, 1.05)
        tx['gasPrice'] = int(tx['gasPrice'] * variation)
        
        return tx
        
    async def _apply_commit_reveal(self, tx: Dict) -> Dict:
        """Apply commit-reveal scheme for MEV protection"""
        # Would implement commit-reveal pattern
        # Simplified for now
        return tx
        
    async def _send_transaction_with_retry(
        self,
        signed_tx,
        chain: str,
        max_retries: int = 3
    ) -> str:
        """Send transaction with retry logic"""
        w3 = self.w3_connections[chain]
        
        for attempt in range(max_retries):
            try:
                tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                logger.info(f"Transaction sent: {tx_hash.hex()}")
                return tx_hash
                
            except Exception as e:
                if 'replacement transaction underpriced' in str(e):
                    # Increase gas price and retry
                    logger.warning(f"Transaction underpriced, retrying with higher gas")
                    await asyncio.sleep(1)
                elif attempt == max_retries - 1:
                    raise
                else:
                    await asyncio.sleep(2 ** attempt)
                    
        raise Exception("Failed to send transaction after retries")
        
    async def _wait_for_confirmation(
        self,
        tx_hash: str,
        chain: str,
        timeout: int = 300
    ) -> Dict:
        """Wait for transaction confirmation"""
        w3 = self.w3_connections[chain]
        
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            try:
                receipt = w3.eth.get_transaction_receipt(tx_hash)
                if receipt:
                    return receipt
            except:
                pass
                
            await asyncio.sleep(2)
            
        raise Exception(f"Transaction not confirmed after {timeout} seconds")
        
    async def _get_optimal_gas_price(self, chain: str) -> int:
        """Get optimal gas price for chain"""
        w3 = self.w3_connections[chain]
        
        if self.gas_price_strategy == 'fast':
            gas_price = w3.eth.gas_price * 1.2
        elif self.gas_price_strategy == 'standard':
            gas_price = w3.eth.gas_price
        else:  # slow
            gas_price = w3.eth.gas_price * 0.8
            
        # Apply max gas price limit
        max_gas = self.max_gas_price * 10**9  # Convert to wei
        
        return min(int(gas_price), max_gas)
        
    async def _quote_v3(
        self,
        contract: Contract,
        path: List[str],
        amount: int
    ) -> int:
        """Get quote from Uniswap V3"""
        # Would use quoter contract
        # Simplified simulation
        return int(amount * 0.997)  # Assume 0.3% fee
        
    async def cleanup(self) -> None:
        """Cleanup resources"""
        self.route_cache.clear()
        self.gas_estimates.clear()
        logger.info("Direct DEX Executor cleaned up")

            
    async def _get_dex_quote(
        self,
        dex_name: str,
        contract: Contract,
        token_in: str,
        token_out: str,
        amount: Decimal,
        chain: str
    ) -> Optional[DEXQuote]:
        """Get quote from specific DEX"""
        try:
            w3 = self.w3_connections[chain]
            
            # Find optimal path
            path = await self._find_path(
                dex_name,
                token_in,
                token_out,
                chain
            )
            
            if not path:
                return None
                
            # Get amounts out
            amount_wei = ether_to_wei(amount)
            
            if 'v3' in dex_name.lower():
                # Uniswap V3 quoter logic
                amount_out = await self._quote_v3(
                    contract,
                    path,
                    amount_wei
                )
            else:
                # Uniswap V2 style
                amounts = contract.functions.getAmountsOut(
                    amount_wei,
                    path
                ).call()
                amount_out = amounts[-1]
                
            if amount_out == 0:
                return None
                
            # Calculate price and impact
            amount_out_ether = wei_to_ether(amount_out)
            price = amount_out_ether / amount
            
            # Estimate price impact
            price_impact = await self._estimate_price_impact(
                dex_name,
                path,
                amount_wei,
                chain
            )
            
            # Estimate gas
            gas_estimate = await self._estimate_gas(
                dex_name,
                path,
                amount_wei,
                chain
            )
            
            return DEXQuote(
                dex=DEX[dex_name.upper()],
                path=path,
                amount_in=amount,
                amount_out=amount_out_ether,
                price=price,
                price_impact=price_impact,
                gas_estimate=gas_estimate,
                liquidity=await self._get_liquidity(path[0], path[-1], chain),
                fee=self._calculate_fee(dex_name, amount)
            )
            
        except Exception as e:
            logger.debug(f"Error getting {dex_name} quote: {e}")
            return None
            
    async def _find_path(
        self,
        dex: str,
        token_in: str,
        token_out: str,
        chain: str
    ) -> Optional[List[str]]:
        """Find optimal trading path"""
        try:
            # Check cache
            cache_key = f"{dex}:{chain}:{token_in}:{token_out}"
            if cache_key in self.route_cache:
                return self.route_cache[cache_key]
                
            token_in = Web3.toChecksumAddress(token_in)
            token_out = Web3.toChecksumAddress(token_out)
            
            # Direct path
            direct_path = [token_in, token_out]
            
            # Multi-hop paths through common bases
            multi_hop_paths = []
            for base in self.common_bases.get(chain, []):
                base = Web3.toChecksumAddress(base)
                if base != token_in and base != token_out:
                    # Try token_in -> base -> token_out
                    multi_hop_paths.append([token_in, base, token_out])
                    
                    # For 3 hops, try additional base
                    if self.max_hops >= 3:
                        for base2 in self.common_bases.get(chain, []):
                            base2 = Web3.toChecksumAddress(base2)
                            if base2 != base and base2 != token_in and base2 != token_out:
                                multi_hop_paths.append([token_in, base, base2, token_out])
                                
            # Test each path and find best
            best_path = None
            best_output = 0
            
            for path in [direct_path] + multi_hop_paths:
                try:
                    # Quick liquidity check
                    if await self._path_has_liquidity(path, chain):
                        # Simulate swap to check output
                        output = await self._simulate_swap(dex, path, ether_to_wei(Decimal('1')), chain)
                        if output > best_output:
                            best_output = output
                            best_path = path
                except:
                    continue
                    
            # Cache the result
            if best_path:
                self.route_cache[cache_key] = best_path
                
            return best_path
            
        except Exception as e:
            logger.error(f"Error finding path: {e}")
            return None
            
    async def _path_has_liquidity(self, path: List[str], chain: str) -> bool:
        """Check if path has sufficient liquidity"""
        # Implementation would check pair contracts
        # Simplified for now
        return True
        
    async def _simulate_swap(
        self,
        dex: str,
        path: List[str],
        amount: int,
        chain: str
    ) -> int:
        """Simulate swap to get output amount"""
        try:
            contract = self.dex_contracts[chain][dex]
            
            if 'v3' in dex.lower():
                # V3 simulation
                return await self._quote_v3(contract, path, amount)
            else:
                # V2 simulation
                amounts = contract.functions.getAmountsOut(amount, path).call()
                return amounts[-1]
        except:
            return 0

    # Missing methods for trading/executors classes
    # These methods should be added to the respective executor classes

    # ============================================
    # For DirectDEXExecutor and ToxiSolAPIExecutor
    # ============================================

    async def validate_order(self, order: Order) -> bool:
        """
        Validate order before execution
        
        Args:
            order: Order to validate
            
        Returns:
            True if order is valid
        """
        try:
            # Check chain is supported
            if order.chain not in self.w3_connections:
                logger.error(f"Chain {order.chain} not supported")
                return False
            
            # Check token addresses
            if not Web3.isAddress(order.token_in):
                logger.error(f"Invalid token_in address: {order.token_in}")
                return False
                
            if not Web3.isAddress(order.token_out):
                logger.error(f"Invalid token_out address: {order.token_out}")
                return False
            
            # Check amount
            if order.amount <= 0:
                logger.error(f"Invalid amount: {order.amount}")
                return False
            
            # Check slippage
            if order.slippage and (order.slippage < 0 or order.slippage > 1):
                logger.error(f"Invalid slippage: {order.slippage}")
                return False
            
            # Check wallet balance (only if not dry run)
            if not self.dry_run:
                w3 = self.w3_connections[order.chain]
                from eth_account import Account
                account = Account.from_key(self.config.get('private_key'))
                
                balance = w3.eth.get_balance(account.address)
                required = ether_to_wei(order.amount)
                
                if balance < required:
                    logger.error(f"Insufficient balance: {wei_to_ether(balance)} < {order.amount}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Order validation error: {e}", exc_info=True)
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get status of an order
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status
        """
        try:
            # Check if we have the order in tracking
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                
                # Check transaction status on-chain
                if order.tx_hash:
                    receipt = await self._get_transaction_receipt(order.tx_hash)
                    
                    if receipt:
                        if receipt['status'] == 1:
                            return OrderStatus.COMPLETED
                        else:
                            return OrderStatus.FAILED
                    else:
                        # Still pending
                        return OrderStatus.EXECUTING
                else:
                    return order.status
            else:
                logger.warning(f"Order {order_id} not found")
                return OrderStatus.UNKNOWN
                
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return OrderStatus.ERROR

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order (if possible)
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        try:
            if order_id not in self.active_orders:
                logger.error(f"Order {order_id} not found")
                return False
                
            order = self.active_orders[order_id]
            
            # Can only cancel if not yet executed
            if order.status in [OrderStatus.PENDING, OrderStatus.QUEUED]:
                order.status = OrderStatus.CANCELLED
                
                # Remove from active orders
                del self.active_orders[order_id]
                
                logger.info(f"Order {order_id} cancelled")
                return True
                
            elif order.status == OrderStatus.EXECUTING:
                # Try to cancel on-chain (replace with higher gas price tx doing nothing)
                # This is complex and chain-specific
                logger.warning(f"Cannot cancel executing order {order_id}")
                return False
                
            else:
                logger.warning(f"Order {order_id} in state {order.status} cannot be cancelled")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    async def modify_order(
        self, 
        order_id: str,
        modifications: Dict[str, Any]
    ) -> bool:
        """
        Modify an existing order (if not yet executed)
        
        Args:
            order_id: Order ID to modify
            modifications: Dictionary of fields to modify
            
        Returns:
            True if modified successfully
        """
        try:
            if order_id not in self.active_orders:
                logger.error(f"Order {order_id} not found")
                return False
                
            order = self.active_orders[order_id]
            
            # Can only modify pending orders
            if order.status != OrderStatus.PENDING:
                logger.error(f"Cannot modify order in status {order.status}")
                return False
                
            # Apply modifications
            allowed_fields = ['amount', 'slippage', 'gas_price', 'max_priority_fee']
            
            for field, value in modifications.items():
                if field in allowed_fields:
                    setattr(order, field, value)
                    logger.info(f"Modified {field} to {value} for order {order_id}")
                else:
                    logger.warning(f"Field {field} cannot be modified")
                    
            # Re-validate order
            if not await self.validate_order(order):
                logger.error("Modified order is invalid")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            return False


    # ============================================
    # Helper methods that might be needed
    # ============================================

    async def _get_transaction_receipt(self, tx_hash: str) -> Optional[Dict]:
        """
        Get transaction receipt from blockchain
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Receipt dictionary or None
        """
        try:
            # For DirectDEX
            if hasattr(self, 'w3_connections'):
                for chain, w3 in self.w3_connections.items():
                    try:
                        receipt = w3.eth.get_transaction_receipt(tx_hash)
                        if receipt:
                            return dict(receipt)
                    except:
                        continue
                        
            # For MEVProtectionLayer and ToxiSol
            elif hasattr(self, 'w3'):
                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                return dict(receipt) if receipt else None
                
            return None
            
        except Exception as e:
            logger.debug(f"Could not get receipt for {tx_hash}: {e}")
            return None

    async def _check_bundle_status(self, bundle_id: str) -> str:
        """
        Check Flashbots bundle status
        
        Args:
            bundle_id: Bundle hash
            
        Returns:
            Status string: 'pending', 'included', 'failed'
        """
        try:
            if not self.session:
                return 'unknown'
                
            async with self.session.get(
                f"{self.flashbots_relay}/relay/v1/bundle",
                params={'bundleHash': bundle_id}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('isIncluded'):
                        return 'included'
                    elif data.get('isFailed'):
                        return 'failed'
                    else:
                        return 'pending'
                        
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Error checking bundle status: {e}")
            return 'unknown'

    # Also add this helper method for building base transactions:
    async def _build_transaction(self, order: Order) -> Dict[str, Any]:
        """
        Build base transaction from order
        
        Args:
            order: Order to build transaction from
            
        Returns:
            Transaction dictionary
        """
        return {
            'from': order.wallet_address,
            'to': order.token_out,  # This would be router address in reality
            'value': 0,
            'data': '0x',  # Would be actual swap data
            'gas': 300000,  # Estimate
            'gasPrice': self.w3.eth.gas_price if hasattr(self, 'w3') else 50 * 10**9,
            'nonce': self.w3.eth.get_transaction_count(order.wallet_address) if hasattr(self, 'w3') else 0,
            'chainId': 1  # Would get from order.chain
        }

    # Format helper function (add to direct_dex.py):
    def format_token_amount(amount: Decimal, decimals: int) -> str:
        """Format token amount for display"""
        return f"{amount:.{decimals}f}".rstrip('0').rstrip('.')