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

from ..orders.order_manager import Order, OrderType, OrderStatus
from .base_executor import BaseExecutor
from ...utils.helpers import retry_async, measure_time, wei_to_ether, ether_to_wei
from ...utils.constants import DEX, DEX_ROUTERS, CHAIN_RPC_URLS

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
            best_quote = max(quotes, key=lambda q: q.output_amount)
            
            logger.info(
                f"Best quote from {best_quote.dex}: "
                f"{format_token_amount(amount, 18)} {token_in} -> "
                f"{format_token_amount(best_quote.output_amount, 18)} {token_out}"
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
    async def execute_trade(self, order: Order) -> Dict[str, Any]:
        """Execute trade directly on DEX"""
        try:
            # Get best quote
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
            if quote.price_impact > float(order.slippage or self.max_slippage):
                return {
                    'success': False,
                    'error': f'Price impact too high: {quote.price_impact:.2%}'
                }
                
            # Build transaction
            tx = await self._build_swap_transaction(order, quote)
            
            # Apply MEV protection
            if self.use_commit_reveal:
                tx = await self._apply_commit_reveal(tx)
                
            if self.randomize_gas:
                tx = self._randomize_gas_price(tx)
                
            # Sign and send
            w3 = self.w3_connections[order.chain]
            signed_tx = w3.eth.account.sign_transaction(tx, order.private_key)
            
            # Send with retry logic
            tx_hash = await self._send_transaction_with_retry(
                signed_tx,
                order.chain
            )
            
            # Wait for confirmation
            receipt = await self._wait_for_confirmation(tx_hash, order.chain)
            
            return {
                'success': receipt['status'] == 1,
                'transactionHash': tx_hash.hex(),
                'gasUsed': receipt['gasUsed'],
                'blockNumber': receipt['blockNumber'],
                'amountOut': quote.amount_out,
                'dex': quote.dex.value
            }
            
        except Exception as e:
            logger.error(f"Direct DEX execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
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
        timeout: int = 120
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
                
            # Return best quote (highest output)
            best_quote = max(quotes, key=lambda q: q.amount_out)
            logger.info(f"Best quote from {best_quote.dex.value}: {best_quote.amount_out}")
            
            return best_quote
            
        except Exception as e:
            logger.error(f"Error getting quotes: {e}")
            return None
            
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