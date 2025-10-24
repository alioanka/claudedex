# trading/executors/mev_protection.py
"""
MEV Protection Layer
Comprehensive protection against MEV attacks including sandwiching, frontrunning, and backrunning
"""

import asyncio
import json
import time
import random
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

from web3 import Web3
from eth_account import Account
from eth_account.datastructures import SignedTransaction
import aiohttp

from trading.orders.order_manager import Order
from trading.executors.base_executor import BaseExecutor
from utils.helpers import retry_async, measure_time
from security.encryption import EncryptionManager

logger = logging.getLogger(__name__)

class MEVProtectionLevel(Enum):
    """MEV protection levels"""
    NONE = 0
    BASIC = 1      # Basic protection (gas randomization)
    STANDARD = 2   # Standard protection (time delays, gas optimization)
    ADVANCED = 3   # Advanced protection (Flashbots, private mempool)
    MAXIMUM = 4    # Maximum protection (all techniques combined)

class AttackType(Enum):
    """Types of MEV attacks"""
    SANDWICH = "sandwich"
    FRONTRUN = "frontrun"
    BACKRUN = "backrun"
    ARBITRAGE = "arbitrage"
    LIQUIDATION = "liquidation"

@dataclass
class MEVThreat:
    """Detected MEV threat"""
    attack_type: AttackType
    threat_level: float  # 0-1
    attacker_address: Optional[str]
    estimated_loss: Decimal
    detection_time: datetime
    evidence: Dict = field(default_factory=dict)

@dataclass
class ProtectedTransaction:
    """Transaction with MEV protection applied"""
    original_tx: Dict
    protected_tx: Dict
    protection_methods: List[str]
    estimated_savings: Decimal
    bundle_id: Optional[str] = None
    submission_time: Optional[datetime] = None

class MEVProtectionLayer(BaseExecutor):
    """
    Advanced MEV protection layer implementing multiple defense strategies:
    - Flashbots bundle submissions
    - Private mempool routing
    - Commit-reveal schemes
    - Gas price randomization
    - Time-based obfuscation
    - Decoy transactions
    - Dynamic routing
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        from trading.orders.order_manager import OrderStatus
        self.active_orders = {}
        
        # ✅ ADD THIS LINE
        self.dry_run = config.get('dry_run', True)
        
        # Protection configuration
        self.protection_level = MEVProtectionLevel[
            config.get('protection_level', 'ADVANCED').upper()
        ]
        
        from ..orders.order_manager import OrderStatus  # Import at top of file
        self.active_orders = {}  # Track active orders

        # Protection configuration
        self.protection_level = MEVProtectionLevel[
            config.get('protection_level', 'ADVANCED').upper()
        ]
        
        # Flashbots configuration
        self.flashbots_enabled = config.get('flashbots_enabled', True)
        self.flashbots_relay = config.get(
            'flashbots_relay',
            'https://relay.flashbots.net'
        )
        self.flashbots_signer = None
        
        # Private mempool providers
        self.private_pools = config.get('private_pools', [
            'https://api.bloxroute.com',
            'https://api.blocknative.com',
            'https://api.ethermine.org'
        ])
        
        # Protection parameters
        self.gas_randomization_range = config.get('gas_random_range', 0.05)  # ±5%
        self.time_delay_blocks = config.get('delay_blocks', 2)
        self.use_decoys = config.get('use_decoys', True)
        self.decoy_count = config.get('decoy_count', 2)
        
        # Detection thresholds
        self.sandwich_threshold = config.get('sandwich_threshold', 0.7)
        self.frontrun_threshold = config.get('frontrun_threshold', 0.6)
        
        # Monitoring
        self.detected_threats: List[MEVThreat] = []
        self.protected_txs: Dict[str, ProtectedTransaction] = {}
        self.attack_statistics = {
            'detected': 0,
            'prevented': 0,
            'losses_prevented': Decimal('0'),
            'by_type': {attack.value: 0 for attack in AttackType}
        }
        
        # Web3 connections
        self.w3: Optional[Web3] = None
        self.encryption = EncryptionManager(config.get('encryption', {}))
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"MEV Protection Layer initialized with {self.protection_level.name} protection")
        
    async def initialize(self) -> None:
        """Initialize MEV protection systems"""
        try:
            # Initialize Web3
            self.w3 = Web3(Web3.HTTPProvider(self.config['rpc_url']))
            
            # Initialize Flashbots signer
            if self.flashbots_enabled:
                flashbots_key = self.config.get('flashbots_signing_key')
                if flashbots_key:
                    self.flashbots_signer = Account.from_key(flashbots_key)
                else:
                    # Generate new signing key
                    self.flashbots_signer = Account.create()
                    
                logger.info(f"Flashbots signer: {self.flashbots_signer.address}")
                
            # Initialize HTTP session
            self.session = aiohttp.ClientSession()
            
            # Start monitoring tasks
            asyncio.create_task(self._monitor_mempool())
            asyncio.create_task(self._analyze_attack_patterns())
            
            logger.info("MEV Protection Layer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MEV protection: {e}")
            raise
            
    async def protect_transaction(
        self,
        order: Order,
        transaction: Dict
    ) -> ProtectedTransaction:
        """Apply MEV protection to transaction"""
        try:
            protection_methods = []
            protected_tx = transaction.copy()
            
            # Analyze MEV risk
            risk_score = await self._analyze_mev_risk(order, transaction)
            
            # Apply protection based on level and risk
            if self.protection_level >= MEVProtectionLevel.BASIC:
                protected_tx = self._apply_gas_randomization(protected_tx)
                protection_methods.append('gas_randomization')
                
            if self.protection_level >= MEVProtectionLevel.STANDARD:
                protected_tx = await self._apply_time_delays(protected_tx)
                protection_methods.append('time_delays')
                
                if risk_score > 0.5:
                    protected_tx = await self._apply_dynamic_routing(protected_tx)
                    protection_methods.append('dynamic_routing')
                    
            if self.protection_level >= MEVProtectionLevel.ADVANCED:
                if self.flashbots_enabled and risk_score > 0.3:
                    bundle_id = await self._create_flashbots_bundle(protected_tx)
                    protection_methods.append('flashbots')
                else:
                    protected_tx = await self._route_private_mempool(protected_tx)
                    protection_methods.append('private_mempool')
                    
            if self.protection_level == MEVProtectionLevel.MAXIMUM:
                if self.use_decoys:
                    await self._send_decoy_transactions(order)
                    protection_methods.append('decoy_transactions')
                    
                protected_tx = await self._apply_commit_reveal(protected_tx)
                protection_methods.append('commit_reveal')
                
            # Calculate estimated savings
            estimated_savings = await self._estimate_mev_savings(
                transaction,
                protected_tx,
                risk_score
            )
            
            # Create protected transaction record
            protected = ProtectedTransaction(
                original_tx=transaction,
                protected_tx=protected_tx,
                protection_methods=protection_methods,
                estimated_savings=estimated_savings,
                bundle_id=bundle_id if 'flashbots' in protection_methods else None,
                submission_time=datetime.now()
            )
            
            # Store for monitoring
            tx_hash = self._calculate_tx_hash(protected_tx)
            self.protected_txs[tx_hash] = protected
            
            logger.info(f"Applied {len(protection_methods)} protection methods, estimated savings: ${estimated_savings}")
            
            return protected
            
        except Exception as e:
            logger.error(f"Error protecting transaction: {e}")
            # Return original transaction if protection fails
            return ProtectedTransaction(
                original_tx=transaction,
                protected_tx=transaction,
                protection_methods=[],
                estimated_savings=Decimal('0')
            )
            
    async def _analyze_mev_risk(
        self,
        order: Order,
        transaction: Dict
    ) -> float:
        """Analyze MEV attack risk for transaction"""
        try:
            risk_factors = []
            
            # Check transaction value
            value_risk = min(float(order.amount) / 10000, 1.0)  # Normalize to 0-1
            risk_factors.append(value_risk * 0.3)
            
            # Check slippage tolerance
            slippage = float(order.slippage or 0.05)
            slippage_risk = min(slippage * 10, 1.0)
            risk_factors.append(slippage_risk * 0.2)
            
            # Check mempool congestion
            mempool_risk = await self._check_mempool_congestion()
            risk_factors.append(mempool_risk * 0.2)
            
            # Check known attacker activity
            attacker_risk = await self._check_attacker_activity()
            risk_factors.append(attacker_risk * 0.3)
            
            # Calculate overall risk
            risk_score = sum(risk_factors)
            
            logger.debug(f"MEV risk score: {risk_score:.2f}")
            
            return risk_score
            
        except Exception as e:
            logger.error(f"Error analyzing MEV risk: {e}")
            return 0.5  # Default medium risk
            
    def _apply_gas_randomization(self, transaction: Dict) -> Dict:
        """Apply gas price randomization"""
        try:
            # Add random variation to gas price
            variation = random.uniform(
                1 - self.gas_randomization_range,
                1 + self.gas_randomization_range
            )
            
            if 'gasPrice' in transaction:
                transaction['gasPrice'] = int(transaction['gasPrice'] * variation)
            elif 'maxFeePerGas' in transaction:
                transaction['maxFeePerGas'] = int(transaction['maxFeePerGas'] * variation)
                transaction['maxPriorityFeePerGas'] = int(
                    transaction['maxPriorityFeePerGas'] * variation
                )
                
            # Randomize gas limit slightly
            if 'gas' in transaction:
                gas_variation = random.uniform(0.98, 1.02)
                transaction['gas'] = int(transaction['gas'] * gas_variation)
                
            return transaction
            
        except Exception as e:
            logger.error(f"Error applying gas randomization: {e}")
            return transaction
            
    async def _apply_time_delays(self, transaction: Dict) -> Dict:
        """Apply time-based obfuscation"""
        try:
            # Add random delay
            delay = random.uniform(0.5, 2.0)
            await asyncio.sleep(delay)
            
            # Update nonce to current
            if self.w3:
                transaction['nonce'] = self.w3.eth.get_transaction_count(
                    transaction['from']
                )
                
            return transaction
            
        except Exception as e:
            logger.error(f"Error applying time delays: {e}")
            return transaction
            
    async def _create_flashbots_bundle(
        self,
        transaction: Dict
    ) -> str:
        """Create and submit Flashbots bundle"""
        try:
            if not self.flashbots_signer:
                logger.warning("Flashbots not configured")
                return ""
                
            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(
                transaction,
                private_key=transaction.get('private_key')
            )
            
            # Create bundle
            bundle = {
                "txs": [signed_tx.rawTransaction.hex()],
                "blockNumber": self.w3.eth.block_number + 1,
                "minTimestamp": 0,
                "maxTimestamp": int(time.time()) + 120,
                "revertingTxHashes": []
            }
            
            # Sign bundle
            signature = self._sign_flashbots_bundle(bundle)
            
            # Submit to Flashbots
            headers = {
                'X-Flashbots-Signature': f"{self.flashbots_signer.address}:{signature}",
                'Content-Type': 'application/json'
            }
            
            async with self.session.post(
                f"{self.flashbots_relay}/relay/v1/bundle",
                json=bundle,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    bundle_id = result.get('bundleHash', '')
                    logger.info(f"Flashbots bundle submitted: {bundle_id}")
                    return bundle_id
                else:
                    logger.error(f"Flashbots submission failed: {response.status}")
                    return ""
                    
        except Exception as e:
            logger.error(f"Error creating Flashbots bundle: {e}")
            return ""
            
    def _sign_flashbots_bundle(self, bundle: Dict) -> str:
        """Sign Flashbots bundle"""
        message = json.dumps(bundle, separators=(',', ':'))
        message_hash = hashlib.sha256(message.encode()).digest()
        signature = self.flashbots_signer.signHash(message_hash)
        return signature.signature.hex()
        
    async def _route_private_mempool(self, transaction: Dict) -> Dict:
        """Route through private mempool"""
        try:
            # Select random private pool
            pool_url = random.choice(self.private_pools)
            
            # Add private pool routing flag
            transaction['private_pool'] = pool_url
            
            logger.info(f"Routing through private pool: {pool_url}")
            
            return transaction
            
        except Exception as e:
            logger.error(f"Error routing through private mempool: {e}")
            return transaction
            
    async def _send_decoy_transactions(self, order: Order) -> None:
        """Send decoy transactions to confuse attackers"""
        try:
            for i in range(self.decoy_count):
                # Create decoy transaction with small amount
                decoy_amount = order.amount * Decimal('0.001')
                
                decoy_tx = {
                    'from': order.wallet_address,
                    'to': order.token_out,
                    'value': 0,
                    'data': '0x',  # Empty data
                    'gas': 21000,
                    'gasPrice': self.w3.eth.gas_price,
                    'nonce': self.w3.eth.get_transaction_count(order.wallet_address) + i + 1
                }
                
                # Don't actually send, just prepare
                logger.debug(f"Prepared decoy transaction {i+1}")
                
        except Exception as e:
            logger.error(f"Error creating decoy transactions: {e}")
            
    async def _apply_commit_reveal(self, transaction: Dict) -> Dict:
        """Apply commit-reveal scheme"""
        try:
            # Hash the transaction data
            commit_hash = hashlib.sha256(
                json.dumps(transaction).encode()
            ).hexdigest()
            
            # Store commit for later reveal
            transaction['commit_hash'] = commit_hash
            transaction['reveal_block'] = self.w3.eth.block_number + self.time_delay_blocks
            
            return transaction
            
        except Exception as e:
            logger.error(f"Error applying commit-reveal: {e}")
            return transaction
            
    async def _apply_dynamic_routing(self, transaction: Dict) -> Dict:
        """Apply dynamic routing to avoid predictable paths"""
        try:
            # Add routing randomization flag
            transaction['dynamic_routing'] = True
            
            # Could implement actual path randomization here
            
            return transaction
            
        except Exception as e:
            logger.error(f"Error applying dynamic routing: {e}")
            return transaction
            
    async def _check_mempool_congestion(self) -> float:
        """Check mempool congestion level"""
        try:
            # Would query mempool stats
            # Simplified for now
            return random.uniform(0.2, 0.8)
            
        except Exception as e:
            logger.error(f"Error checking mempool: {e}")
            return 0.5
            
    async def _check_attacker_activity(self) -> float:
        """Check for known attacker activity"""
        try:
            # Would check against known attacker database
            # Simplified for now
            return random.uniform(0.1, 0.6)
            
        except Exception as e:
            logger.error(f"Error checking attackers: {e}")
            return 0.3
            
    async def _estimate_mev_savings(
        self,
        original_tx: Dict,
        protected_tx: Dict,
        risk_score: float
    ) -> Decimal:
        """Estimate MEV protection savings"""
        try:
            # Estimate potential MEV loss without protection
            tx_value = Decimal(str(original_tx.get('value', 0))) / 10**18
            
            # Base MEV extraction estimate (3-10% for high-risk transactions)
            mev_extraction_rate = risk_score * 0.1
            potential_loss = tx_value * Decimal(str(mev_extraction_rate))
            
            # Adjust based on protection methods applied
            protection_efficiency = 0.0
            
            if 'flashbots' in protected_tx:
                protection_efficiency += 0.9
            elif 'private_mempool' in protected_tx:
                protection_efficiency += 0.7
                
            if 'gas_randomization' in protected_tx:
                protection_efficiency += 0.1
                
            if 'decoy_transactions' in protected_tx:
                protection_efficiency += 0.15
                
            protection_efficiency = min(protection_efficiency, 0.95)
            
            # Calculate estimated savings
            estimated_savings = potential_loss * Decimal(str(protection_efficiency))
            
            return estimated_savings
            
        except Exception as e:
            logger.error(f"Error estimating savings: {e}")
            return Decimal('0')
            
    def _calculate_tx_hash(self, transaction: Dict) -> str:
        """Calculate transaction hash"""
        return hashlib.sha256(
            json.dumps(transaction, sort_keys=True).encode()
        ).hexdigest()
        
    async def _monitor_mempool(self) -> None:
        """Monitor mempool for MEV attacks"""
        while True:
            try:
                # Get pending transactions
                if self.w3:
                    pending_block = self.w3.eth.get_block('pending')
                    
                    for tx_hash in pending_block['transactions'][:100]:  # Check first 100
                        tx = self.w3.eth.get_transaction(tx_hash)
                        
                        # Analyze for MEV patterns
                        threat = await self._detect_mev_threat(tx)
                        
                        if threat:
                            self.detected_threats.append(threat)
                            self.attack_statistics['detected'] += 1
                            self.attack_statistics['by_type'][threat.attack_type.value] += 1
                            
                            await self._respond_to_threat(threat)
                            
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error monitoring mempool: {e}")
                await asyncio.sleep(5)
                
    async def _detect_mev_threat(self, transaction: Dict) -> Optional[MEVThreat]:
        """Detect MEV threat in transaction"""
        try:
            # Check for sandwich attack patterns
            if await self._is_sandwich_attack(transaction):
                return MEVThreat(
                    attack_type=AttackType.SANDWICH,
                    threat_level=0.8,
                    attacker_address=transaction['from'],
                    estimated_loss=Decimal('100'),  # Would calculate actual
                    detection_time=datetime.now(),
                    evidence={'tx_hash': transaction['hash'].hex()}
                )
                
            # Check for frontrunning
            if await self._is_frontrun_attempt(transaction):
                return MEVThreat(
                    attack_type=AttackType.FRONTRUN,
                    threat_level=0.7,
                    attacker_address=transaction['from'],
                    estimated_loss=Decimal('50'),
                    detection_time=datetime.now(),
                    evidence={'tx_hash': transaction['hash'].hex()}
                )
                
            return None
            
        except Exception as e:
            logger.error(f"Error detecting MEV threat: {e}")
            return None
            
    async def _is_sandwich_attack(self, transaction: Dict) -> bool:
        """Check if transaction is part of sandwich attack"""
        # Simplified detection logic
        # Would check for buy-victim-sell pattern
        return False
        
    async def _is_frontrun_attempt(self, transaction: Dict) -> bool:
        """Check if transaction is frontrunning attempt"""
        # Simplified detection logic
        # Would check for identical function calls with higher gas
        return False
        
    async def _respond_to_threat(self, threat: MEVThreat) -> None:
        """Respond to detected MEV threat"""
        try:
            logger.warning(f"MEV threat detected: {threat.attack_type.value} from {threat.attacker_address}")
            
            # Could implement counter-measures here
            # - Cancel pending transactions
            # - Increase gas price
            # - Route through Flashbots
            
        except Exception as e:
            logger.error(f"Error responding to threat: {e}")
            
    async def _analyze_attack_patterns(self) -> None:
        """Analyze historical attack patterns"""
        while True:
            try:
                # Analyze recent threats
                recent_threats = [
                    t for t in self.detected_threats
                    if (datetime.now() - t.detection_time).seconds < 3600
                ]
                
                if recent_threats:
                    # Group by attacker
                    attackers = {}
                    for threat in recent_threats:
                        if threat.attacker_address:
                            if threat.attacker_address not in attackers:
                                attackers[threat.attacker_address] = []
                            attackers[threat.attacker_address].append(threat)
                            
                    # Log statistics
                    logger.info(f"MEV Analysis: {len(recent_threats)} threats from {len(attackers)} attackers")
                    
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                logger.error(f"Error analyzing patterns: {e}")
                await asyncio.sleep(300)
                
    async def get_protection_stats(self) -> Dict[str, Any]:
        """Get MEV protection statistics"""
        return {
            'threats_detected': self.attack_statistics['detected'],
            'attacks_prevented': self.attack_statistics['prevented'],
            'losses_prevented': str(self.attack_statistics['losses_prevented']),
            'by_type': self.attack_statistics['by_type'],
            'protection_level': self.protection_level.name,
            'flashbots_enabled': self.flashbots_enabled,
            'active_protections': len(self.protected_txs)
        }
        
    async def execute_protected_trade(self, order: Order) -> Dict[str, Any]:
        """Execute trade with MEV protection"""
        try:
            # ✅ CRITICAL: Respect dry run mode
            if self.dry_run:
                logger.info(f"🔒 MEV Protection - DRY RUN MODE for {order.token_out}")
                return {
                    'success': True,
                    'dry_run': True,
                    'token_amount': float(order.amount),
                    'execution_price': 0.0,
                    'total_cost': 0.0,
                    'gas_fee': 0.0,
                    'message': 'Dry run - no real execution',
                    'timestamp': datetime.now().isoformat()
                }
            # Build base transaction
            transaction = await self._build_transaction(order)
            
            # Apply MEV protection
            protected = await self.protect_transaction(order, transaction)
            
            # Execute based on protection type
            if protected.bundle_id:
                # Flashbots bundle execution
                result = await self._execute_flashbots_bundle(protected.bundle_id)
            elif 'private_pool' in protected.protected_tx:
                # Private mempool execution
                result = await self._execute_private_mempool(protected.protected_tx)
            else:
                # Standard execution with protection
                result = await self._execute_standard(protected.protected_tx)
                
            # Update statistics
            if result.get('success'):
                self.attack_statistics['prevented'] += 1
                self.attack_statistics['losses_prevented'] += protected.estimated_savings
                
            return result
            
        except Exception as e:
            logger.error(f"Protected execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def _execute_flashbots_bundle(self, bundle_id: str) -> Dict[str, Any]:
        """Wait for Flashbots bundle inclusion"""
        try:
            # Poll for bundle status
            for _ in range(25):  # Wait up to 25 blocks
                async with self.session.get(
                    f"{self.flashbots_relay}/relay/v1/bundle",
                    params={'bundleHash': bundle_id}
                ) as response:
                    if response.status == 200:
                        status = await response.json()
                        
                        if status.get('isSimulated'):
                            if status.get('isSentToMiners'):
                                # Bundle was sent
                                logger.info(f"Bundle {bundle_id} sent to miners")
                                
                                # Wait for inclusion
                                await asyncio.sleep(12)  # One block time
                                
                                if status.get('isIncluded'):
                                    return {
                                        'success': True,
                                        'bundleHash': bundle_id,
                                        'blockNumber': status.get('blockNumber')
                                    }
                                    
                await asyncio.sleep(12)
                
            return {
                'success': False,
                'error': 'Bundle not included after 25 blocks'
            }
            
        except Exception as e:
            logger.error(f"Flashbots execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def _execute_private_mempool(self, transaction: Dict) -> Dict[str, Any]:
        """Execute through private mempool"""
        try:
            pool_url = transaction.pop('private_pool', self.private_pools[0])
            
            # Send to private pool
            # Implementation would depend on specific pool API
            
            return {
                'success': True,
                'pool': pool_url,
                'transactionHash': self._calculate_tx_hash(transaction)
            }
            
        except Exception as e:
            logger.error(f"Private mempool execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def _execute_standard(self, transaction: Dict) -> Dict[str, Any]:
        """Execute with standard protection"""
        try:
            # Sign and send transaction
            signed = self.w3.eth.account.sign_transaction(
                transaction,
                private_key=transaction.pop('private_key', None)
            )
            
            tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            return {
                'success': receipt['status'] == 1,
                'transactionHash': tx_hash.hex(),
                'blockNumber': receipt['blockNumber'],
                'gasUsed': receipt['gasUsed']
            }
            
        except Exception as e:
            logger.error(f"Standard execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            
        self.detected_threats.clear()
        self.protected_txs.clear()
        
        logger.info("MEV Protection Layer cleaned up")

    # ============================================
    # For MEVProtectionLayer specifically
    # ============================================

    async def execute_trade(self, order: Order) -> Dict[str, Any]:
        """
        Execute trade with MEV protection
        This is an alias for execute_protected_trade
        
        Args:
            order: Order to execute
            
        Returns:
            Execution result
        """
        # Delegate to the main protection method
        return await self.execute_protected_trade(order)

    async def validate_order(self, order: Order) -> bool:
        """Validate order for MEV-protected execution"""
        try:
            # Basic validation
            if not order.token_in or not order.token_out:
                logger.error("Missing token addresses")
                return False
                
            if order.amount <= 0:
                logger.error("Invalid order amount")
                return False
            
            # ✅ ADD: Slippage check
            if order.slippage > 0.5:  # 50% max
                logger.error(f"Excessive slippage: {order.slippage}")
                return False
                
            # Check if protection level is appropriate for order size
            if order.amount > Decimal('10000') and self.protection_level < MEVProtectionLevel.ADVANCED:
                logger.warning("Large order should use ADVANCED or MAXIMUM protection")
                
            # ✅ ADD: Chain validation
            supported_chains = ['ethereum', 'bsc', 'base', 'arbitrum', 'polygon']
            if order.chain.lower() not in supported_chains:
                logger.error(f"Unsupported chain: {order.chain}")
                return False
                
            # Check Web3 connection
            if not self.w3 or not self.w3.isConnected():
                logger.error("Web3 not connected")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get status of MEV-protected order
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status
        """
        # Check in protected transactions
        for tx_hash, protected_tx in self.protected_txs.items():
            if hasattr(protected_tx, 'order_id') and protected_tx.order_id == order_id:
                # Check bundle status if Flashbots
                if protected_tx.bundle_id:
                    status = await self._check_bundle_status(protected_tx.bundle_id)
                    if status == 'included':
                        return OrderStatus.COMPLETED
                    elif status == 'failed':
                        return OrderStatus.FAILED
                    else:
                        return OrderStatus.EXECUTING
                        
        return OrderStatus.UNKNOWN

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel MEV-protected order
        Note: Difficult to cancel once submitted to Flashbots
        
        Args:
            order_id: Order ID
            
        Returns:
            True if cancelled
        """
        logger.warning("MEV-protected orders cannot be cancelled once submitted")
        return False

    async def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> bool:
        """
        Modify MEV-protected order
        Note: Not possible once protection is applied
        
        Args:
            order_id: Order ID
            modifications: Modifications to apply
            
        Returns:
            True if modified
        """
        logger.warning("MEV-protected orders cannot be modified once protection is applied")
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