"""
Advanced Trade Execution System
Multi-route execution with MEV protection
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from web3 import Web3
from eth_account import Account
import aiohttp
from decimal import Decimal

@dataclass
class TradeOrder:
    """Trade order structure"""
    token_address: str
    side: str  # 'buy' or 'sell'
    amount: float  # In base currency (ETH, BNB, etc.)
    token_amount: Optional[float] = None  # For sells
    slippage: float = 0.03  # 3% default
    deadline: int = 300  # 5 minutes
    gas_price_multiplier: float = 1.2
    use_mev_protection: bool = True
    use_private_mempool: bool = False
    split_order: bool = False
    max_splits: int = 3
    urgency: str = 'normal'  # 'low', 'normal', 'high', 'critical'
    metadata: Dict = field(default_factory=dict)

@dataclass
class ExecutionResult:
    """Trade execution result"""
    success: bool
    tx_hash: Optional[str]
    execution_price: float
    amount: float
    token_amount: float
    gas_used: float
    gas_price: float
    slippage_actual: float
    execution_time: float
    route: str
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

class ExecutionRoute(Enum):
    """Available execution routes"""
    UNISWAP_V2 = "uniswap_v2"
    UNISWAP_V3 = "uniswap_v3"
    PANCAKESWAP = "pancakeswap"
    SUSHISWAP = "sushiswap"
    ONE_INCH = "1inch"
    PARASWAP = "paraswap"
    TOXISOL = "toxisol"
    DIRECT = "direct"

class TradeExecutor:
    """Base trade executor with multiple routes"""
    
    def __init__(self, config: Dict):
        """
        Initialize trade executor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Web3 setup
        self.w3 = Web3(Web3.HTTPProvider(config.get('web3_provider_url')))
        # NEW:
        private_key = config.get('private_key') or config.get('security', {}).get('private_key')
        if not private_key:
            raise ValueError("PRIVATE_KEY not found in configuration")
        self.account = Account.from_key(private_key)
        self.wallet_address = self.account.address
        
        # Chain configuration
        self.chain_id = config.get('chain_id', 1)
        self.native_token = config.get('native_token', 'ETH')
        
        # Router addresses
        self.routers = {
            'uniswap_v2': config.get('uniswap_v2_router'),
            'uniswap_v3': config.get('uniswap_v3_router'),
            'pancakeswap': config.get('pancakeswap_router'),
            'sushiswap': config.get('sushiswap_router')
        }
        
        # DEX aggregator APIs
        self.aggregator_apis = {
            '1inch': config.get('1inch_api_key'),
            'paraswap': config.get('paraswap_api_key')
        }
        
        # ToxiSol configuration
        self.toxisol_config = config.get('toxisol', {})
        
        # MEV protection
        self.flashbots_relay = config.get('flashbots_relay')
        self.private_pool_endpoint = config.get('private_pool_endpoint')
        
        # Gas configuration
        self.max_gas_price = config.get('max_gas_price', 500)  # Gwei
        self.gas_limit = config.get('gas_limit', 500000)
        
        # Execution settings
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1)
        
        # Monitoring
        self.active_orders = {}
        self.execution_history = []
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_gas_spent': 0,
            'total_slippage': 0,
            'routes_used': {}
        }
        
    async def execute(self, order: TradeOrder) -> ExecutionResult:
        """
        Execute trade order
        
        Args:
            order: Trade order to execute
            
        Returns:
            Execution result
        """
        start_time = time.time()
        
        try:
            # Pre-execution checks
            if not await self._pre_execution_checks(order):
                return ExecutionResult(
                    success=False,
                    tx_hash=None,
                    execution_price=0,
                    amount=0,
                    token_amount=0,
                    gas_used=0,
                    gas_price=0,
                    slippage_actual=0,
                    execution_time=0,
                    route='',
                    error='Pre-execution checks failed'
                )
                
            # Select best route
            best_route = await self._select_best_route(order)
            
            if not best_route:
                return ExecutionResult(
                    success=False,
                    tx_hash=None,
                    execution_price=0,
                    amount=0,
                    token_amount=0,
                    gas_used=0,
                    gas_price=0,
                    slippage_actual=0,
                    execution_time=0,
                    route='',
                    error='No viable route found'
                )
                
            # Apply MEV protection if needed
            if order.use_mev_protection:
                order = await self._apply_mev_protection(order)
                
            # Execute trade
            result = await self._execute_with_retry(order, best_route)
            
            # Post-execution processing
            if result.success:
                await self._post_execution_processing(order, result)
                self.stats['successful_trades'] += 1
            else:
                self.stats['failed_trades'] += 1
                
            self.stats['total_trades'] += 1
            result.execution_time = time.time() - start_time
            
            # Record execution
            self.execution_history.append({
                'order': order,
                'result': result,
                'timestamp': datetime.now()
            })
            
            return result
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                tx_hash=None,
                execution_price=0,
                amount=0,
                token_amount=0,
                gas_used=0,
                gas_price=0,
                slippage_actual=0,
                execution_time=time.time() - start_time,
                route='',
                error=str(e)
            )
            
    async def _pre_execution_checks(self, order: TradeOrder) -> bool:
        """Perform pre-execution checks"""
        try:
            # Check wallet balance
            if order.side == 'buy':
                balance = self.w3.eth.get_balance(self.wallet_address)
                required = Web3.to_wei(order.amount, 'ether')
                
                if balance < required:
                    print(f"Insufficient balance: {balance} < {required}")
                    return False
                    
            else:  # sell
                # Check token balance
                token_balance = await self._get_token_balance(order.token_address)
                if token_balance < order.token_amount:
                    print(f"Insufficient token balance: {token_balance} < {order.token_amount}")
                    return False
                    
            # Check gas price
            gas_price = self.w3.eth.gas_price
            max_gas_wei = Web3.to_wei(self.max_gas_price, 'gwei')
            
            if gas_price > max_gas_wei:
                print(f"Gas price too high: {gas_price} > {max_gas_wei}")
                return False
                
            # Check token contract
            if not await self._verify_token_contract(order.token_address):
                print(f"Token contract verification failed: {order.token_address}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Pre-execution check error: {e}")
            return False
            
    async def _select_best_route(self, order: TradeOrder) -> Optional[ExecutionRoute]:
        """
        Select best execution route
        
        Args:
            order: Trade order
            
        Returns:
            Best execution route or None
        """
        try:
            # Get quotes from all routes
            quotes = await self._get_all_quotes(order)
            
            if not quotes:
                return None
                
            # Sort by expected output (considering gas)
            best_quote = max(quotes, key=lambda x: x['net_output'])
            
            return ExecutionRoute(best_quote['route'])
            
        except Exception as e:
            print(f"Route selection error: {e}")
            return ExecutionRoute.UNISWAP_V2  # Default fallback
            
    async def _get_all_quotes(self, order: TradeOrder) -> List[Dict]:
        """Get quotes from all available routes"""
        quotes = []
        
        # Get quotes in parallel
        tasks = []
        
        if self.routers.get('uniswap_v2'):
            tasks.append(self._get_uniswap_v2_quote(order))
            
        if self.routers.get('uniswap_v3'):
            tasks.append(self._get_uniswap_v3_quote(order))
            
        if self.aggregator_apis.get('1inch'):
            tasks.append(self._get_1inch_quote(order))
            
        if self.aggregator_apis.get('paraswap'):
            tasks.append(self._get_paraswap_quote(order))
            
        if self.toxisol_config:
            tasks.append(self._get_toxisol_quote(order))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict) and result.get('output'):
                quotes.append(result)
                
        return quotes
        
    async def _get_uniswap_v2_quote(self, order: TradeOrder) -> Dict:
        """Get Uniswap V2 quote"""
        try:
            # Load router contract
            router_abi = self._load_abi('uniswap_v2_router')
            router = self.w3.eth.contract(
                address=self.routers['uniswap_v2'],
                abi=router_abi
            )
            
            # Get amounts out
            path = [self._get_weth_address(), order.token_address] if order.side == 'buy' else [order.token_address, self._get_weth_address()]
            amount_in = Web3.to_wei(order.amount, 'ether') if order.side == 'buy' else order.token_amount
            
            amounts = router.functions.getAmountsOut(amount_in, path).call()
            output = amounts[-1]
            
            # Estimate gas
            gas_estimate = 150000  # Approximate
            gas_price = self.w3.eth.gas_price
            gas_cost = gas_estimate * gas_price
            
            return {
                'route': 'uniswap_v2',
                'output': output,
                'gas_estimate': gas_estimate,
                'gas_cost': gas_cost,
                'net_output': output - gas_cost if order.side == 'sell' else output
            }
            
        except Exception as e:
            print(f"Uniswap V2 quote error: {e}")
            return {}
            
    async def _get_1inch_quote(self, order: TradeOrder) -> Dict:
        """Get 1inch aggregator quote"""
        try:
            base_url = "https://api.1inch.exchange/v5.0"
            
            params = {
                'fromTokenAddress': self._get_weth_address() if order.side == 'buy' else order.token_address,
                'toTokenAddress': order.token_address if order.side == 'buy' else self._get_weth_address(),
                'amount': Web3.to_wei(order.amount, 'ether') if order.side == 'buy' else order.token_amount,
                'fromAddress': self.wallet_address,
                'slippage': int(order.slippage * 100)
            }
            
            headers = {
                'Authorization': f'Bearer {self.aggregator_apis["1inch"]}'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/{self.chain_id}/quote", params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return {
                            'route': '1inch',
                            'output': int(data['toTokenAmount']),
                            'gas_estimate': data.get('estimatedGas', 200000),
                            'gas_cost': 0,  # Included in quote
                            'net_output': int(data['toTokenAmount'])
                        }
                        
        except Exception as e:
            print(f"1inch quote error: {e}")
            
        return {}
        
    async def _execute_with_retry(self, order: TradeOrder, route: ExecutionRoute) -> ExecutionResult:
        """Execute trade with retry logic"""
        for attempt in range(self.max_retries):
            try:
                if route == ExecutionRoute.UNISWAP_V2:
                    result = await self._execute_uniswap_v2(order)
                elif route == ExecutionRoute.UNISWAP_V3:
                    result = await self._execute_uniswap_v3(order)
                elif route == ExecutionRoute.ONE_INCH:
                    result = await self._execute_1inch(order)
                elif route == ExecutionRoute.PARASWAP:
                    result = await self._execute_paraswap(order)
                elif route == ExecutionRoute.TOXISOL:
                    result = await self._execute_toxisol(order)
                else:
                    result = await self._execute_direct(order)
                    
                if result.success:
                    # Update statistics
                    self.stats['routes_used'][route.value] = self.stats['routes_used'].get(route.value, 0) + 1
                    self.stats['total_gas_spent'] += result.gas_used * result.gas_price
                    self.stats['total_slippage'] += result.slippage_actual
                    
                    return result
                    
                # If failed but not final attempt, wait before retry
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    
            except Exception as e:
                print(f"Execution attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries - 1:
                    return ExecutionResult(
                        success=False,
                        tx_hash=None,
                        execution_price=0,
                        amount=0,
                        token_amount=0,
                        gas_used=0,
                        gas_price=0,
                        slippage_actual=0,
                        execution_time=0,
                        route=route.value,
                        error=str(e)
                    )
                    
                await asyncio.sleep(self.retry_delay * (attempt + 1))
                
        return ExecutionResult(
            success=False,
            tx_hash=None,
            execution_price=0,
            amount=0,
            token_amount=0,
            gas_used=0,
            gas_price=0,
            slippage_actual=0,
            execution_time=0,
            route=route.value,
            error='Max retries exceeded'
        )
        
    async def _execute_uniswap_v2(self, order: TradeOrder) -> ExecutionResult:
        """Execute trade on Uniswap V2"""
        try:
            router_abi = self._load_abi('uniswap_v2_router')
            router = self.w3.eth.contract(
                address=self.routers['uniswap_v2'],
                abi=router_abi
            )
            
            # Calculate deadline
            deadline = int(time.time()) + order.deadline
            
            # Build transaction
            if order.side == 'buy':
                path = [self._get_weth_address(), order.token_address]
                amount_in = Web3.to_wei(order.amount, 'ether')
                
                # Calculate minimum output with slippage
                amounts_out = router.functions.getAmountsOut(amount_in, path).call()
                min_amount_out = int(amounts_out[-1] * (1 - order.slippage))
                
                # Build transaction
                tx = router.functions.swapExactETHForTokens(
                    min_amount_out,
                    path,
                    self.wallet_address,
                    deadline
                ).build_transaction({
                    'from': self.wallet_address,
                    'value': amount_in,
                    'gas': order.metadata.get('gas_limit', self.gas_limit),
                    'gasPrice': int(self.w3.eth.gas_price * order.gas_price_multiplier),
                    'nonce': self.w3.eth.get_transaction_count(self.wallet_address)
                })
                
            else:  # sell
                path = [order.token_address, self._get_weth_address()]
                amount_in = int(order.token_amount)
                
                # Approve router if needed
                await self._approve_token(order.token_address, self.routers['uniswap_v2'], amount_in)
                
                # Calculate minimum output with slippage
                amounts_out = router.functions.getAmountsOut(amount_in, path).call()
                min_amount_out = int(amounts_out[-1] * (1 - order.slippage))
                
                # Build transaction
                tx = router.functions.swapExactTokensForETH(
                    amount_in,
                    min_amount_out,
                    path,
                    self.wallet_address,
                    deadline
                ).build_transaction({
                    'from': self.wallet_address,
                    'gas': order.metadata.get('gas_limit', self.gas_limit),
                    'gasPrice': int(self.w3.eth.gas_price * order.gas_price_multiplier),
                    'nonce': self.w3.eth.get_transaction_count(self.wallet_address)
                })
                
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt['status'] == 1:
                # Parse execution details
                execution_price = amounts_out[-1] / amount_in if order.side == 'buy' else amount_in / amounts_out[-1]
                actual_slippage = abs(1 - (amounts_out[-1] / (amount_in * execution_price)))
                
                return ExecutionResult(
                    success=True,
                    tx_hash=tx_hash.hex(),
                    execution_price=execution_price,
                    amount=order.amount,
                    token_amount=amounts_out[-1] if order.side == 'buy' else amount_in,
                    gas_used=receipt['gasUsed'],
                    gas_price=tx['gasPrice'],
                    slippage_actual=actual_slippage,
                    execution_time=0,
                    route='uniswap_v2'
                )
                
            else:
                return ExecutionResult(
                    success=False,
                    tx_hash=tx_hash.hex(),
                    execution_price=0,
                    amount=0,
                    token_amount=0,
                    gas_used=receipt['gasUsed'],
                    gas_price=tx['gasPrice'],
                    slippage_actual=0,
                    execution_time=0,
                    route='uniswap_v2',
                    error='Transaction reverted'
                )
                
        except Exception as e:
            return ExecutionResult(
                success=False,
                tx_hash=None,
                execution_price=0,
                amount=0,
                token_amount=0,
                gas_used=0,
                gas_price=0,
                slippage_actual=0,
                execution_time=0,
                route='uniswap_v2',
                error=str(e)
            )
            
    async def _execute_1inch(self, order: TradeOrder) -> ExecutionResult:
        """Execute trade via 1inch aggregator"""
        try:
            base_url = "https://api.1inch.exchange/v5.0"
            
            # Prepare swap parameters
            params = {
                'fromTokenAddress': self._get_weth_address() if order.side == 'buy' else order.token_address,
                'toTokenAddress': order.token_address if order.side == 'buy' else self._get_weth_address(),
                'amount': Web3.to_wei(order.amount, 'ether') if order.side == 'buy' else order.token_amount,
                'fromAddress': self.wallet_address,
                'slippage': int(order.slippage * 100),
                'disableEstimate': False,
                'allowPartialFill': False
            }
            
            headers = {
                'Authorization': f'Bearer {self.aggregator_apis["1inch"]}'
            }
            
            async with aiohttp.ClientSession() as session:
                # Get swap data
                async with session.get(f"{base_url}/{self.chain_id}/swap", params=params, headers=headers) as response:
                    if response.status != 200:
                        error_data = await response.text()
                        raise Exception(f"1inch API error: {error_data}")
                        
                    swap_data = await response.json()
                    
                # Extract transaction data
                tx_data = swap_data['tx']
                tx_data['nonce'] = self.w3.eth.get_transaction_count(self.wallet_address)
                tx_data['gasPrice'] = int(self.w3.eth.gas_price * order.gas_price_multiplier)
                
                # Sign and send transaction
                signed_tx = self.account.sign_transaction(tx_data)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Wait for confirmation
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
                
                if receipt['status'] == 1:
                    return ExecutionResult(
                        success=True,
                        tx_hash=tx_hash.hex(),
                        execution_price=float(swap_data['toTokenAmount']) / float(swap_data['fromTokenAmount']),
                        amount=order.amount,
                        token_amount=float(swap_data['toTokenAmount']),
                        gas_used=receipt['gasUsed'],
                        gas_price=tx_data['gasPrice'],
                        slippage_actual=0,  # 1inch handles slippage
                        execution_time=0,
                        route='1inch',
                        metadata={'protocols': swap_data.get('protocols', [])}
                    )
                    
                else:
                    return ExecutionResult(
                        success=False,
                        tx_hash=tx_hash.hex(),
                        execution_price=0,
                        amount=0,
                        token_amount=0,
                        gas_used=receipt['gasUsed'],
                        gas_price=tx_data['gasPrice'],
                        slippage_actual=0,
                        execution_time=0,
                        route='1inch',
                        error='Transaction reverted'
                    )
                    
        except Exception as e:
            return ExecutionResult(
                success=False,
                tx_hash=None,
                execution_price=0,
                amount=0,
                token_amount=0,
                gas_used=0,
                gas_price=0,
                slippage_actual=0,
                execution_time=0,
                route='1inch',
                error=str(e)
            )
            
    async def _apply_mev_protection(self, order: TradeOrder) -> TradeOrder:
        """Apply MEV protection strategies"""
        protected_order = order
        
        # Strategy 1: Use private mempool
        if self.flashbots_relay and order.urgency != 'critical':
            protected_order.use_private_mempool = True
            
        # Strategy 2: Split large orders
        if order.amount > 10 and order.urgency != 'critical':  # > 10 ETH
            protected_order.split_order = True
            protected_order.max_splits = min(5, int(order.amount / 2))
            
        # Strategy 3: Add random delay
        if order.urgency == 'low':
            await asyncio.sleep(np.random.uniform(0.5, 2.0))
            
        # Strategy 4: Adjust gas price dynamically
        if order.urgency == 'critical':
            protected_order.gas_price_multiplier = 1.5
        elif order.urgency == 'high':
            protected_order.gas_price_multiplier = 1.3
            
        return protected_order
        
    async def _send_private_transaction(self, signed_tx) -> str:
        """Send transaction through private mempool"""
        if self.flashbots_relay:
            # Send to Flashbots
            bundle = {
                "jsonrpc": "2.0",
                "method": "eth_sendBundle",
                "params": [{
                    "txs": [signed_tx.rawTransaction.hex()],
                    "blockNumber": hex(self.w3.eth.block_number + 1)
                }],
                "id": 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.flashbots_relay, json=bundle) as response:
                    result = await response.json()
                    if 'result' in result:
                        return result['result']
                        
        # Fallback to regular mempool
        return self.w3.eth.send_raw_transaction(signed_tx.rawTransaction).hex()
        
    async def _approve_token(self, token_address: str, spender: str, amount: int):
        """Approve token spending"""
        token_abi = self._load_abi('erc20')
        token = self.w3.eth.contract(address=token_address, abi=token_abi)
        
        # Check current allowance
        current_allowance = token.functions.allowance(self.wallet_address, spender).call()
        
        if current_allowance < amount:
            # Build approval transaction
            tx = token.functions.approve(spender, amount).build_transaction({
                'from': self.wallet_address,
                'gas': 100000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.wallet_address)
            })
            
            # Sign and send
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for confirmation
            self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
            
    async def _get_token_balance(self, token_address: str) -> float:
        """Get token balance"""
        token_abi = self._load_abi('erc20')
        token = self.w3.eth.contract(address=token_address, abi=token_abi)
        
        balance = token.functions.balanceOf(self.wallet_address).call()
        decimals = token.functions.decimals().call()
        
        return balance / (10 ** decimals)
        
    async def _verify_token_contract(self, token_address: str) -> bool:
        """Verify token contract is valid"""
        try:
            # Check if address is valid
            if not Web3.is_address(token_address):
                return False
                
            # Check if contract exists
            code = self.w3.eth.get_code(token_address)
            if code == b'':
                return False
                
            # Try to get basic token info
            token_abi = self._load_abi('erc20')
            token = self.w3.eth.contract(address=token_address, abi=token_abi)
            
            # These calls should work for any ERC20 token
            token.functions.totalSupply().call()
            token.functions.decimals().call()
            
            return True
            
        except Exception:
            return False
            
    async def _post_execution_processing(self, order: TradeOrder, result: ExecutionResult):
        """Post-execution processing"""
        # Log execution
        print(f"Trade executed: {result.tx_hash}")
        
        # Update active orders
        if order.metadata.get('order_id'):
            del self.active_orders[order.metadata['order_id']]
            
        # Emit events or notifications
        if result.success and result.slippage_actual > order.slippage:
            print(f"Warning: Actual slippage ({result.slippage_actual:.2%}) exceeded target ({order.slippage:.2%})")
            
    def _get_weth_address(self) -> str:
        """Get WETH address for current chain"""
        weth_addresses = {
            1: '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',  # Ethereum
            56: '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c',  # BSC (WBNB)
            137: '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',  # Polygon (WMATIC)
            42161: '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',  # Arbitrum
            8453: '0x4200000000000000000000000000000000000006'  # Base
        }
        
        return weth_addresses.get(self.chain_id, weth_addresses[1])
        
    def _load_abi(self, contract_type: str) -> List:
        """Load contract ABI"""
        # In production, load from files
        # For now, return minimal ABIs
        
        if contract_type == 'uniswap_v2_router':
            return [
                {
                    "name": "swapExactETHForTokens",
                    "type": "function",
                    "inputs": [
                        {"name": "amountOutMin", "type": "uint256"},
                        {"name": "path", "type": "address[]"},
                        {"name": "to", "type": "address"},
                        {"name": "deadline", "type": "uint256"}
                    ],
                    "outputs": [{"name": "amounts", "type": "uint256[]"}]
                },
                {
                    "name": "swapExactTokensForETH",
                    "type": "function",
                    "inputs": [
                        {"name": "amountIn", "type": "uint256"},
                        {"name": "amountOutMin", "type": "uint256"},
                        {"name": "path", "type": "address[]"},
                        {"name": "to", "type": "address"},
                        {"name": "deadline", "type": "uint256"}
                    ],
                    "outputs": [{"name": "amounts", "type": "uint256[]"}]
                },
                {
                    "name": "getAmountsOut",
                    "type": "function",
                    "inputs": [
                        {"name": "amountIn", "type": "uint256"},
                        {"name": "path", "type": "address[]"}
                    ],
                    "outputs": [{"name": "amounts", "type": "uint256[]"}],
                    "stateMutability": "view"
                }
            ]
            
        elif contract_type == 'erc20':
            return [
                {
                    "name": "approve",
                    "type": "function",
                    "inputs": [
                        {"name": "spender", "type": "address"},
                        {"name": "amount", "type": "uint256"}
                    ],
                    "outputs": [{"name": "", "type": "bool"}]
                },
                {
                    "name": "allowance",
                    "type": "function",
                    "inputs": [
                        {"name": "owner", "type": "address"},
                        {"name": "spender", "type": "address"}
                    ],
                    "outputs": [{"name": "", "type": "uint256"}],
                    "stateMutability": "view"
                },
                {
                    "name": "balanceOf",
                    "type": "function",
                    "inputs": [{"name": "account", "type": "address"}],
                    "outputs": [{"name": "", "type": "uint256"}],
                    "stateMutability": "view"
                },
                {
                    "name": "totalSupply",
                    "type": "function",
                    "inputs": [],
                    "outputs": [{"name": "", "type": "uint256"}],
                    "stateMutability": "view"
                },
                {
                    "name": "decimals",
                    "type": "function",
                    "inputs": [],
                    "outputs": [{"name": "", "type": "uint8"}],
                    "stateMutability": "view"
                }
            ]
            
        return []
        
    async def emergency_sell_all(self):
        """Emergency sell all positions"""
        print("EMERGENCY SELL ALL TRIGGERED")
        
        # This would sell all token positions
        # Implementation would depend on position tracking
        pass
        
    def get_stats(self) -> Dict:
        """Get executor statistics"""
        return self.stats.copy()

    """
    PATCH FILE: Add missing methods to trading/executors/base_executor.py
    Add these methods to the TradeExecutor class for API compatibility
    """

    # Add these methods to the TradeExecutor class in base_executor.py

    async def execute_trade(self, order: TradeOrder) -> Dict:
        """
        Execute trade (wrapper for execute method)
        
        Args:
            order: Trade order to execute
            
        Returns:
            Execution result dictionary
        """
        # Use existing execute method
        result = await self.execute(order)
        
        # Convert ExecutionResult to Dict
        return {
            'success': result.success,
            'tx_hash': result.tx_hash,
            'execution_price': result.execution_price,
            'amount': result.amount,
            'token_amount': result.token_amount,
            'gas_used': result.gas_used,
            'gas_price': result.gas_price,
            'slippage_actual': result.slippage_actual,
            'execution_time': result.execution_time,
            'route': result.route,
            'error': result.error,
            'metadata': result.metadata
        }

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation successful
        """
        try:
            # Check if order exists and is pending
            if order_id not in self.active_orders:
                print(f"Order {order_id} not found")
                return False
                
            order = self.active_orders[order_id]
            
            # For market orders, cancellation might not be possible if already executing
            if order.get('status') == 'executing':
                print(f"Order {order_id} is already executing and cannot be cancelled")
                return False
                
            # Remove from active orders
            del self.active_orders[order_id]
            
            # Log cancellation
            self.execution_history.append({
                'order_id': order_id,
                'action': 'cancelled',
                'timestamp': datetime.now()
            })
            
            print(f"Order {order_id} cancelled successfully")
            return True
            
        except Exception as e:
            print(f"Error cancelling order {order_id}: {e}")
            return False

    async def modify_order(self, order_id: str, updates: Dict) -> bool:
        """
        Modify a pending order
        
        Args:
            order_id: ID of order to modify
            updates: Dictionary of fields to update
            
        Returns:
            True if modification successful
        """
        try:
            # Check if order exists
            if order_id not in self.active_orders:
                print(f"Order {order_id} not found")
                return False
                
            order = self.active_orders[order_id]
            
            # Check if order can be modified
            if order.get('status') == 'executing':
                print(f"Order {order_id} is executing and cannot be modified")
                return False
                
            # Validate updates
            allowed_updates = ['amount', 'slippage', 'gas_price_multiplier', 'deadline']
            for key in updates.keys():
                if key not in allowed_updates:
                    print(f"Field {key} cannot be modified")
                    return False
                    
            # Apply updates
            for key, value in updates.items():
                if key in order:
                    order[key] = value
                    
            # Update timestamp
            order['last_modified'] = datetime.now()
            
            # Log modification
            self.execution_history.append({
                'order_id': order_id,
                'action': 'modified',
                'updates': updates,
                'timestamp': datetime.now()
            })
            
            print(f"Order {order_id} modified successfully")
            return True
            
        except Exception as e:
            print(f"Error modifying order {order_id}: {e}")
            return False

    def validate_order(self, order: TradeOrder) -> bool:
        """
        Validate a trade order
        
        Args:
            order: Order to validate
            
        Returns:
            True if order is valid
        """
        try:
            # Check required fields
            if not order.token_address:
                print("Token address is required")
                return False
                
            # Validate token address
            if not Web3.is_address(order.token_address):
                print("Invalid token address")
                return False
                
            # Check side
            if order.side not in ['buy', 'sell']:
                print("Order side must be 'buy' or 'sell'")
                return False
                
            # Validate amounts
            if order.amount <= 0:
                print("Amount must be positive")
                return False
                
            if order.side == 'sell' and not order.token_amount:
                print("Token amount required for sell orders")
                return False
                
            # Check slippage
            if order.slippage < 0 or order.slippage > 1:
                print("Slippage must be between 0 and 1")
                return False
                
            # Check deadline
            if order.deadline <= 0:
                print("Invalid deadline")
                return False
                
            # Check gas multiplier
            if order.gas_price_multiplier <= 0:
                print("Invalid gas price multiplier")
                return False
                
            # Additional chain-specific validation
            if self.chain_id not in [1, 56, 137, 42161, 8453]:  # Supported chains
                print(f"Unsupported chain: {self.chain_id}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Order validation error: {e}")
            return False

    async def get_order_status(self, order_id: str) -> Dict:
        """
        Get status of an order
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status dictionary
        """
        try:
            # Check active orders
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                return {
                    'order_id': order_id,
                    'status': order.get('status', 'pending'),
                    'side': order.get('side'),
                    'amount': order.get('amount'),
                    'token_address': order.get('token_address'),
                    'created_at': order.get('created_at'),
                    'last_modified': order.get('last_modified'),
                    'metadata': order.get('metadata', {})
                }
                
            # Check execution history
            for record in self.execution_history:
                if isinstance(record, dict) and record.get('order', {}).get('metadata', {}).get('order_id') == order_id:
                    result = record.get('result')
                    if result:
                        return {
                            'order_id': order_id,
                            'status': 'completed' if result.success else 'failed',
                            'tx_hash': result.tx_hash,
                            'execution_price': result.execution_price,
                            'gas_used': result.gas_used,
                            'completed_at': record.get('timestamp'),
                            'error': result.error
                        }
                        
            # Order not found
            return {
                'order_id': order_id,
                'status': 'not_found',
                'error': 'Order not found'
            }
            
        except Exception as e:
            return {
                'order_id': order_id,
                'status': 'error',
                'error': str(e)
            }
        
async def test_web3_connection():
    """Test Web3 connectivity"""
    try:
        w3 = Web3(Web3.HTTPProvider(os.environ.get('WEB3_PROVIDER_URL')))
        if w3.is_connected():
            print(f"Web3 connected. Chain ID: {w3.eth.chain_id}")
            return True
        else:
            print("Web3 connection failed")
            return False
    except Exception as e:
        print(f"Web3 connection error: {e}")
        return False