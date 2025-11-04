"""
Chain Data Collector - On-chain data analysis via Web3
"""

import asyncio
from web3 import Web3, AsyncWeb3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import aiohttp
from collections import defaultdict

@dataclass
class TokenInfo:
    """On-chain token information"""
    address: str
    name: str
    symbol: str
    decimals: int
    total_supply: int
    owner: Optional[str] = None
    is_mintable: bool = False
    is_pausable: bool = False
    has_blacklist: bool = False
    has_fee_mechanism: bool = False
    buy_tax: float = 0
    sell_tax: float = 0
    max_wallet: Optional[int] = None
    max_tx: Optional[int] = None
    creation_time: Optional[datetime] = None
    creator_address: Optional[str] = None
    is_verified: bool = False
    
@dataclass
class ContractInfo:
    """Smart contract analysis"""
    address: str
    is_renounced: bool
    has_proxy: bool
    is_upgradeable: bool
    has_mint_function: bool
    has_pause_function: bool
    has_blacklist_function: bool
    hidden_functions: List[str] = field(default_factory=list)
    suspicious_functions: List[str] = field(default_factory=list)
    
@dataclass
class LiquidityInfo:
    """Liquidity pool information"""
    pool_address: str
    token0: str
    token1: str
    reserves0: int
    reserves1: int
    total_supply: int
    locked: bool
    lock_duration: Optional[int] = None
    lock_end: Optional[datetime] = None
    provider_count: int = 0
    
@dataclass
class TransactionInfo:
    """Transaction analysis"""
    hash: str
    from_address: str
    to_address: str
    value: int
    gas_used: int
    block_number: int
    timestamp: datetime
    method: Optional[str] = None
    status: bool = True

class ChainDataCollector:
    """Collect and analyze on-chain data"""
    
    # Standard ERC20 ABI
    ERC20_ABI = json.loads('[{"constant":true,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"type":"function"},{"constant":true,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"type":"function"},{"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},{"constant":true,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"},{"constant":true,"inputs":[],"name":"owner","outputs":[{"name":"","type":"address"}],"type":"function"}]')
    
    # UniswapV2 Pair ABI (simplified)
    PAIR_ABI = json.loads('[{"constant":true,"inputs":[],"name":"getReserves","outputs":[{"name":"_reserve0","type":"uint112"},{"name":"_reserve1","type":"uint112"},{"name":"_blockTimestampLast","type":"uint32"}],"type":"function"},{"constant":true,"inputs":[],"name":"token0","outputs":[{"name":"","type":"address"}],"type":"function"},{"constant":true,"inputs":[],"name":"token1","outputs":[{"name":"","type":"address"}],"type":"function"}]')
    
    def __init__(self, config: Dict):
        """
        Initialize chain data collector
        
        Args:
            config: Configuration with RPC endpoints
        """
        self.config = config
        
        # Initialize Web3 connections for different chains
        self.w3_connections = {}
        self._setup_connections()
        
        # Contract cache
        self.contract_cache = {}
        
        # Known router addresses
        self.routers = {
            'ethereum': {
                'uniswap_v2': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
                'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F'
            },
            'bsc': {
                'pancakeswap_v2': '0x10ED43C718714eb63d5aA57B78B54704E256024E',
                'pancakeswap_v3': '0x13f4EA83D0bd40E75C8222255bc855a974568Dd4'
            },
            'polygon': {
                'quickswap': '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff'
            }
        }
        
        # Known factory addresses
        self.factories = {
            'ethereum': {
                'uniswap_v2': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',
                'sushiswap': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac'
            },
            'bsc': {
                'pancakeswap_v2': '0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73'
            }
        }
        
        # Suspicious function signatures
        self.suspicious_functions = [
            'setFee', 'setTaxFee', 'setLiquidityFee',
            'includeInFee', 'excludeFromFee',
            'pause', 'unpause', 'setPaused',
            'blacklist', 'addToBlacklist', 'removeFromBlacklist',
            'mint', 'burn',
            'setMaxTx', 'setMaxWallet',
            'transferOwnership', 'renounceOwnership'
        ]
        
    def _setup_connections(self):
        """Setup Web3 connections for different chains"""
        try:
            # Ethereum
            if 'ethereum_rpc' in self.config:
                w3 = Web3(Web3.HTTPProvider(self.config['ethereum_rpc']))
                if w3.is_connected():
                    self.w3_connections['ethereum'] = w3
                    
            # BSC
            if 'bsc_rpc' in self.config:
                w3 = Web3(Web3.HTTPProvider(self.config['bsc_rpc']))
                w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                if w3.is_connected():
                    self.w3_connections['bsc'] = w3
                    
            # Polygon
            if 'polygon_rpc' in self.config:
                w3 = Web3(Web3.HTTPProvider(self.config['polygon_rpc']))
                w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                if w3.is_connected():
                    self.w3_connections['polygon'] = w3
                    
            # Arbitrum
            if 'arbitrum_rpc' in self.config:
                w3 = Web3(Web3.HTTPProvider(self.config['arbitrum_rpc']))
                if w3.is_connected():
                    self.w3_connections['arbitrum'] = w3
                    
        except Exception as e:
            print(f"Web3 connection setup error: {e}")
            
    def get_web3(self, chain: str) -> Optional[Web3]:
        """Get Web3 instance for chain"""
        return self.w3_connections.get(chain)
        
    async def get_token_info(self, token_address: str, chain: str = 'ethereum') -> Optional[TokenInfo]:
        """
        Get comprehensive token information
        
        Args:
            token_address: Token contract address
            chain: Blockchain network
            
        Returns:
            TokenInfo object or None
        """
        try:
            w3 = self.get_web3(chain)
            if not w3:
                return None
                
            # Get contract
            token_address = Web3.to_checksum_address(token_address)
            contract = w3.eth.contract(address=token_address, abi=self.ERC20_ABI)
            
            # Get basic info
            name = await self._safe_call(contract.functions.name())
            symbol = await self._safe_call(contract.functions.symbol())
            decimals = await self._safe_call(contract.functions.decimals())
            total_supply = await self._safe_call(contract.functions.totalSupply())
            
            # Try to get owner
            owner = await self._safe_call(contract.functions.owner()) if hasattr(contract.functions, 'owner') else None
            
            # Get contract code for analysis
            code = w3.eth.get_code(token_address)
            
            # Analyze contract features
            has_mint = self._check_function_in_code(code, 'mint')
            has_pause = self._check_function_in_code(code, 'pause')
            has_blacklist = self._check_function_in_code(code, 'blacklist')
            has_fee = self._check_function_in_code(code, 'fee')
            
            # Get creation info
            creation_info = await self._get_contract_creation(token_address, chain)
            
            return TokenInfo(
                address=token_address,
                name=name or 'Unknown',
                symbol=symbol or 'UNKNOWN',
                decimals=decimals or 18,
                total_supply=total_supply or 0,
                owner=owner,
                is_mintable=has_mint,
                is_pausable=has_pause,
                has_blacklist=has_blacklist,
                has_fee_mechanism=has_fee,
                creation_time=creation_info.get('timestamp') if creation_info else None,
                creator_address=creation_info.get('creator') if creation_info else None
            )
            
        except Exception as e:
            print(f"Token info error for {token_address}: {e}")
            return None
            
    async def analyze_contract(self, contract_address: str, chain: str = 'ethereum') -> Optional[ContractInfo]:
        """
        Analyze smart contract for red flags
        
        Args:
            contract_address: Contract address
            chain: Blockchain network
            
        Returns:
            ContractInfo object
        """
        try:
            w3 = self.get_web3(chain)
            if not w3:
                return None
                
            contract_address = Web3.to_checksum_address(contract_address)
            
            # Get contract code
            code = w3.eth.get_code(contract_address)
            code_hex = code.hex()
            
            # Check for suspicious patterns
            suspicious = []
            hidden = []
            
            for func in self.suspicious_functions:
                if func.lower() in code_hex.lower():
                    suspicious.append(func)
                    
            # Check for proxy pattern
            has_proxy = 'delegatecall' in code_hex.lower()
            is_upgradeable = 'upgrade' in code_hex.lower()
            
            # Check ownership
            is_renounced = await self._check_renounced_ownership(contract_address, chain)
            
            # Check for hidden functions (non-standard names)
            if 'function' in code_hex.lower():
                # This would need more sophisticated bytecode analysis
                # Simplified check for now
                pass
                
            return ContractInfo(
                address=contract_address,
                is_renounced=is_renounced,
                has_proxy=has_proxy,
                is_upgradeable=is_upgradeable,
                has_mint_function='mint' in code_hex.lower(),
                has_pause_function='pause' in code_hex.lower(),
                has_blacklist_function='blacklist' in code_hex.lower(),
                hidden_functions=hidden,
                suspicious_functions=suspicious
            )
            
        except Exception as e:
            print(f"Contract analysis error: {e}")
            return None
            
    async def get_liquidity_info(self, pair_address: str, chain: str = 'ethereum') -> Optional[LiquidityInfo]:
        """
        Get liquidity pool information
        
        Args:
            pair_address: LP pair address
            chain: Blockchain network
            
        Returns:
            LiquidityInfo object
        """
        try:
            w3 = self.get_web3(chain)
            if not w3:
                return None
                
            pair_address = Web3.to_checksum_address(pair_address)
            pair_contract = w3.eth.contract(address=pair_address, abi=self.PAIR_ABI)
            
            # Get reserves
            reserves = await self._safe_call(pair_contract.functions.getReserves())
            token0 = await self._safe_call(pair_contract.functions.token0())
            token1 = await self._safe_call(pair_contract.functions.token1())
            
            if not reserves:
                return None
                
            # Check if liquidity is locked
            lock_info = await self._check_liquidity_lock(pair_address, chain)
            
            return LiquidityInfo(
                pool_address=pair_address,
                token0=token0,
                token1=token1,
                reserves0=reserves[0],
                reserves1=reserves[1],
                total_supply=0,  # Would need to get from pair contract
                locked=lock_info.get('locked', False) if lock_info else False,
                lock_duration=lock_info.get('duration') if lock_info else None,
                lock_end=lock_info.get('unlock_time') if lock_info else None
            )
            
        except Exception as e:
            print(f"Liquidity info error: {e}")
            return None
            
    async def get_holder_distribution(self, token_address: str, chain: str = 'ethereum') -> Dict:
        """
        Analyze token holder distribution
        
        Args:
            token_address: Token contract address
            chain: Blockchain network
            
        Returns:
            Holder distribution analysis
        """
        try:
            # This would typically use a service like Etherscan API or Covalent
            # For now, returning placeholder
            
            return {
                'top_10_percentage': 0,
                'top_50_percentage': 0,
                'unique_holders': 0,
                'whale_concentration': 0,
                'distribution_score': 0  # 0-1, higher is better
            }
            
        except Exception as e:
            print(f"Holder distribution error: {e}")
            return {}
            
    async def get_recent_transactions(self, address: str, chain: str = 'ethereum',
                                    limit: int = 100) -> List[TransactionInfo]:
        """
        Get recent transactions for an address
        
        Args:
            address: Address to check
            chain: Blockchain network
            limit: Maximum number of transactions
            
        Returns:
            List of recent transactions
        """
        try:
            w3 = self.get_web3(chain)
            if not w3:
                return []
                
            # Get latest block
            latest_block = w3.eth.block_number
            
            transactions = []
            blocks_to_check = min(1000, limit * 10)  # Rough estimate
            
            for block_num in range(latest_block, latest_block - blocks_to_check, -1):
                if len(transactions) >= limit:
                    break
                    
                try:
                    block = w3.eth.get_block(block_num, full_transactions=True)
                    
                    for tx in block.transactions:
                        if tx['from'] == address or tx['to'] == address:
                            transactions.append(TransactionInfo(
                                hash=tx['hash'].hex(),
                                from_address=tx['from'],
                                to_address=tx['to'] or '',
                                value=tx['value'],
                                gas_used=tx['gas'],
                                block_number=block_num,
                                timestamp=datetime.fromtimestamp(block['timestamp'])
                            ))
                            
                            if len(transactions) >= limit:
                                break
                                
                except Exception:
                    continue
                    
            return transactions
            
        except Exception as e:
            print(f"Transaction fetching error: {e}")
            return []
            
    async def check_honeypot_onchain(self, token_address: str, chain: str = 'ethereum') -> Dict:
        """
        Check for honeypot characteristics on-chain
        
        Args:
            token_address: Token to check
            chain: Blockchain network
            
        Returns:
            Honeypot analysis results
        """
        try:
            w3 = self.get_web3(chain)
            if not w3:
                return {'is_honeypot': False, 'reason': 'No connection'}
                
            # Get contract code
            code = w3.eth.get_code(token_address)
            code_hex = code.hex()
            
            honeypot_indicators = {
                'has_pause': 'pause' in code_hex.lower(),
                'has_blacklist': 'blacklist' in code_hex.lower(),
                'has_whitelist': 'whitelist' in code_hex.lower(),
                'has_max_tx': 'maxtx' in code_hex.lower() or 'maxamount' in code_hex.lower(),
                'has_cooldown': 'cooldown' in code_hex.lower(),
                'has_fee_change': 'setfee' in code_hex.lower() or 'settax' in code_hex.lower(),
                'modifiable_transfer': 'transfer' in code_hex.lower() and 'onlyowner' in code_hex.lower()
            }
            
            # Count red flags
            red_flags = sum(honeypot_indicators.values())
            
            # Determine if honeypot
            is_honeypot = red_flags >= 3
            
            reasons = [key for key, value in honeypot_indicators.items() if value]
            
            return {
                'is_honeypot': is_honeypot,
                'confidence': min(red_flags * 0.2, 1.0),
                'red_flags': red_flags,
                'reasons': reasons,
                'details': honeypot_indicators
            }
            
        except Exception as e:
            print(f"Honeypot check error: {e}")
            return {'is_honeypot': False, 'error': str(e)}
            
    async def get_gas_price(self, chain: str = 'ethereum') -> Dict:
        """Get current gas prices"""
        try:
            w3 = self.get_web3(chain)
            if not w3:
                return {}
                
            gas_price = w3.eth.gas_price
            
            return {
                'standard': gas_price,
                'fast': int(gas_price * 1.2),
                'slow': int(gas_price * 0.8),
                'base_fee': w3.eth.get_block('latest').get('baseFeePerGas', 0)
            }
            
        except Exception as e:
            print(f"Gas price error: {e}")
            return {}
            
    async def estimate_transaction_cost(self, from_addr: str, to_addr: str,
                                       value: int, chain: str = 'ethereum') -> Dict:
        """Estimate transaction cost"""
        try:
            w3 = self.get_web3(chain)
            if not w3:
                return {}
                
            # Estimate gas
            gas_estimate = w3.eth.estimate_gas({
                'from': from_addr,
                'to': to_addr,
                'value': value
            })
            
            gas_price = w3.eth.gas_price
            
            return {
                'gas_limit': gas_estimate,
                'gas_price': gas_price,
                'estimated_cost': gas_estimate * gas_price,
                'estimated_cost_eth': w3.from_wei(gas_estimate * gas_price, 'ether')
            }
            
        except Exception as e:
            print(f"Cost estimation error: {e}")
            return {}
            
    async def _safe_call(self, func):
        """Safely call a contract function"""
        try:
            return func.call()
        except Exception:
            return None
            
    def _check_function_in_code(self, code: bytes, function_name: str) -> bool:
        """Check if function signature exists in bytecode"""
        try:
            # Simple check - would need proper bytecode analysis for accuracy
            return function_name.lower() in code.hex().lower()
        except:
            return False
            
    async def _get_contract_creation(self, address: str, chain: str) -> Optional[Dict]:
        """Get contract creation details"""
        try:
            # This would typically use Etherscan API or similar
            # Placeholder for now
            return None
        except:
            return None
            
    async def _check_renounced_ownership(self, address: str, chain: str) -> bool:
        """Check if contract ownership is renounced"""
        try:
            w3 = self.get_web3(chain)
            if not w3:
                return False
                
            # Try to get owner
            contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=self.ERC20_ABI)
            
            try:
                owner = contract.functions.owner().call()
                # Check if owner is zero address (renounced)
                return owner == '0x0000000000000000000000000000000000000000'
            except:
                # No owner function or error
                return False
                
        except Exception:
            return False
            
    async def _check_liquidity_lock(self, pair_address: str, chain: str) -> Optional[Dict]:
        """Check if liquidity is locked"""
        try:
            # This would check popular lock contracts like Unicrypt, Team.Finance
            # Placeholder for now
            return {'locked': False}
        except:
            return None

    # ============================================================================
    # PATCH FOR: chain_data.py
    # Add these methods to the ChainDataCollector class
    # ============================================================================

    async def get_block_number(self, chain: str = 'ethereum') -> int:
        """
        Get current block number
        
        Args:
            chain: Blockchain network
            
        Returns:
            Current block number
        """
        try:
            w3 = self.get_web3(chain)
            if not w3:
                return 0
                
            return w3.eth.block_number
            
        except Exception as e:
            print(f"Get block number error: {e}")
            return 0

    async def get_transaction(self, tx_hash: str, chain: str = 'ethereum') -> Dict:
        """
        Get transaction details
        
        Args:
            tx_hash: Transaction hash
            chain: Blockchain network
            
        Returns:
            Transaction details
        """
        try:
            w3 = self.get_web3(chain)
            if not w3:
                return {}
                
            tx = w3.eth.get_transaction(tx_hash)
            receipt = w3.eth.get_transaction_receipt(tx_hash)
            
            return {
                'hash': tx.hash.hex(),
                'from': tx['from'],
                'to': tx['to'],
                'value': tx['value'],
                'gas': tx['gas'],
                'gasPrice': tx.get('gasPrice', 0),
                'nonce': tx['nonce'],
                'blockNumber': tx.get('blockNumber'),
                'blockHash': tx.get('blockHash').hex() if tx.get('blockHash') else None,
                'status': receipt.status if receipt else None,
                'gasUsed': receipt.gasUsed if receipt else None,
                'input': tx['input']
            }
            
        except Exception as e:
            print(f"Get transaction error: {e}")
            return {}

    async def get_token_balance(self, address: str, token: str, chain: str = 'ethereum') -> Decimal:
        """
        Get token balance for an address
        
        Args:
            address: Wallet address
            token: Token contract address
            chain: Blockchain network
            
        Returns:
            Token balance
        """
        from decimal import Decimal
        
        try:
            w3 = self.get_web3(chain)
            if not w3:
                return Decimal('0')
                
            # Checksum addresses
            address = Web3.to_checksum_address(address)
            token = Web3.to_checksum_address(token)
            
            # Get contract
            contract = w3.eth.contract(address=token, abi=self.ERC20_ABI)
            
            # Get balance
            balance = await self._safe_call(contract.functions.balanceOf(address))
            
            if balance:
                # Get decimals
                decimals = await self._safe_call(contract.functions.decimals()) or 18
                
                # Convert to decimal
                return Decimal(balance) / Decimal(10 ** decimals)
                
            return Decimal('0')
            
        except Exception as e:
            print(f"Get token balance error: {e}")
            return Decimal('0')

    async def monitor_mempool(self, chain: str = 'ethereum'):
        """
        Monitor mempool for pending transactions (AsyncGenerator)
        
        Args:
            chain: Blockchain network
            
        Yields:
            Pending transaction data
        """
        try:
            w3 = self.get_web3(chain)
            if not w3:
                return
                
            # Subscribe to pending transactions
            # Note: This requires WebSocket connection
            pending_filter = w3.eth.filter('pending')
            
            while True:
                try:
                    # Get new pending transactions
                    pending_txs = pending_filter.get_new_entries()
                    
                    for tx_hash in pending_txs:
                        try:
                            # Get transaction details
                            tx = w3.eth.get_transaction(tx_hash)
                            
                            yield {
                                'hash': tx.hash.hex(),
                                'from': tx['from'],
                                'to': tx['to'],
                                'value': tx['value'],
                                'gas': tx['gas'],
                                'gasPrice': tx.get('gasPrice', 0),
                                'input': tx['input'],
                                'timestamp': datetime.now()
                            }
                            
                        except Exception:
                            continue
                            
                    await asyncio.sleep(1)  # Poll every second
                    
                except Exception as e:
                    print(f"Mempool monitoring error: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            print(f"Mempool setup error: {e}")
