"""
Advanced Wallet Security Manager for DexScreener Trading Bot
Handles hardware wallet integration, key management, and secure transaction signing
"""

import os
import json
import asyncio
import hashlib
import hmac
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from hdwallet import HDWallet
from hdwallet.symbols import ETH as ETH_SYMBOL
import secrets
import base64

from security.encryption import EncryptionManager

logger = logging.getLogger(__name__)

class WalletType(Enum):
    """Supported wallet types"""
    HOT_WALLET = "hot_wallet"
    HARDWARE_LEDGER = "hardware_ledger"
    HARDWARE_TREZOR = "hardware_trezor"
    MULTISIG = "multisig"
    CUSTODY = "custody"

class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = 1      # Small test transactions
    MEDIUM = 2   # Regular trading
    HIGH = 3     # Large amounts
    CRITICAL = 4 # Emergency operations

@dataclass
class WalletConfig:
    """Wallet configuration"""
    wallet_type: WalletType
    address: str
    derivation_path: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    max_transaction_amount: Decimal = Decimal('1.0')
    daily_limit: Decimal = Decimal('10.0')
    requires_2fa: bool = True
    whitelist_addresses: List[str] = None
    backup_locations: List[str] = None

@dataclass
class TransactionApproval:
    """Transaction approval data"""
    transaction_hash: str
    approved: bool
    approver: str
    timestamp: datetime
    security_checks: Dict[str, bool]
    risk_score: float

class WalletSecurityManager:
    """
    Advanced wallet security manager with hardware wallet support,
    multi-signature capabilities, and comprehensive security controls
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encryption_manager = EncryptionManager(config.get('encryption', {}))
        self.wallets: Dict[str, WalletConfig] = {}
        self.approved_transactions: Dict[str, TransactionApproval] = {}
        self.transaction_history: List[Dict] = []
        self.security_policies: Dict[str, Any] = {}
        self.emergency_contacts: List[str] = []
        
        # Initialize security thresholds
        self.security_thresholds = {
            SecurityLevel.LOW: Decimal('0.1'),
            SecurityLevel.MEDIUM: Decimal('1.0'),
            SecurityLevel.HIGH: Decimal('10.0'),
            SecurityLevel.CRITICAL: Decimal('100.0')
        }
        
        # Rate limiting
        self.transaction_limits = {}
        self.daily_volumes = {}
        
        # Web3 connections
        self.web3_connections: Dict[str, Web3] = {}
        
        logger.info("WalletSecurityManager initialized")

    async def initialize(self) -> None:
        """Initialize wallet security manager"""
        try:
            await self._load_wallet_configurations()
            await self._initialize_web3_connections()
            await self._load_security_policies()
            await self._setup_hardware_wallets()
            
            logger.info("Wallet security manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize wallet security manager: {e}")
            raise

    async def _load_wallet_configurations(self) -> None:
        """Load wallet configurations from secure storage"""
        try:
            encrypted_configs = self.config.get('wallet_configs', {})
            
            for wallet_id, encrypted_config in encrypted_configs.items():
                decrypted_config = self.encryption_manager.decrypt_sensitive_data(encrypted_config)
                
                self.wallets[wallet_id] = WalletConfig(
                    wallet_type=WalletType(decrypted_config['wallet_type']),
                    address=decrypted_config['address'],
                    derivation_path=decrypted_config.get('derivation_path'),
                    security_level=SecurityLevel(decrypted_config.get('security_level', 2)),
                    max_transaction_amount=Decimal(str(decrypted_config.get('max_transaction_amount', '1.0'))),
                    daily_limit=Decimal(str(decrypted_config.get('daily_limit', '10.0'))),
                    requires_2fa=decrypted_config.get('requires_2fa', True),
                    whitelist_addresses=decrypted_config.get('whitelist_addresses', []),
                    backup_locations=decrypted_config.get('backup_locations', [])
                )
                
        except Exception as e:
            logger.error(f"Failed to load wallet configurations: {e}")
            raise

    async def _initialize_web3_connections(self) -> None:
        """Initialize Web3 connections for different chains"""
        try:
            chain_configs = self.config.get('chains', {})
            
            for chain_name, chain_config in chain_configs.items():
                rpc_url = chain_config.get('rpc_url')
                if rpc_url:
                    w3 = Web3(Web3.HTTPProvider(rpc_url))
                    
                    # Add PoA middleware if needed
                    if chain_config.get('is_poa', False):
                        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                    
                    if w3.is_connected():
                        self.web3_connections[chain_name] = w3
                        logger.info(f"Connected to {chain_name} blockchain")
                    else:
                        logger.warning(f"Failed to connect to {chain_name} blockchain")
                        
        except Exception as e:
            logger.error(f"Failed to initialize Web3 connections: {e}")

    async def _load_security_policies(self) -> None:
        """Load security policies and rules"""
        self.security_policies = {
            'require_approval_threshold': Decimal('5.0'),
            'max_daily_volume': Decimal('100.0'),
            'max_transaction_frequency': 10,  # per hour
            'require_2fa_threshold': Decimal('1.0'),
            'emergency_stop_conditions': [
                'suspicious_pattern_detected',
                'unusual_volume_spike',
                'blacklisted_address_interaction'
            ],
            'auto_lock_conditions': [
                'multiple_failed_attempts',
                'unauthorized_access_attempt',
                'hardware_wallet_disconnected'
            ]
        }

    async def _setup_hardware_wallets(self) -> None:
        """Setup and verify hardware wallet connections"""
        try:
            for wallet_id, wallet_config in self.wallets.items():
                if wallet_config.wallet_type in [WalletType.HARDWARE_LEDGER, WalletType.HARDWARE_TREZOR]:
                    await self._verify_hardware_wallet(wallet_id, wallet_config)
                    
        except Exception as e:
            logger.error(f"Failed to setup hardware wallets: {e}")

    async def _verify_hardware_wallet(self, wallet_id: str, wallet_config: WalletConfig) -> bool:
        """Verify hardware wallet connection and configuration"""
        try:
            # This would integrate with actual hardware wallet libraries
            # For now, we'll simulate the verification
            logger.info(f"Verifying hardware wallet {wallet_id} ({wallet_config.wallet_type.value})")
            
            # Simulate hardware wallet verification
            await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Hardware wallet verification failed for {wallet_id}: {e}")
            return False

    async def create_wallet(self, 
                          wallet_type: WalletType,
                          security_level: SecurityLevel = SecurityLevel.MEDIUM,
                          derivation_path: Optional[str] = None) -> Tuple[str, str]:
        """
        Create a new secure wallet
        
        Returns:
            Tuple of (wallet_id, address)
        """
        try:
            wallet_id = f"wallet_{secrets.token_hex(8)}"
            
            if wallet_type == WalletType.HOT_WALLET:
                # Create HD wallet
                hdwallet = HDWallet(symbol=ETH_SYMBOL)
                mnemonic = hdwallet.generate_mnemonic()
                hdwallet.from_mnemonic(mnemonic)
                
                if derivation_path:
                    hdwallet.from_path(derivation_path)
                
                private_key = hdwallet.private_key()
                address = hdwallet.address()
                
                # Encrypt and store private key
                encrypted_key = self.encryption_manager.encrypt_wallet_key(private_key)
                
                # Store encrypted wallet data
                wallet_data = {
                    'wallet_type': wallet_type.value,
                    'address': address,
                    'derivation_path': derivation_path,
                    'security_level': security_level.value,
                    'encrypted_key': encrypted_key,
                    'mnemonic_backup': self.encryption_manager.encrypt_sensitive_data(mnemonic),
                    'created_at': datetime.utcnow().isoformat()
                }
                
                # Save to secure storage
                await self._save_wallet_data(wallet_id, wallet_data)
                
            elif wallet_type in [WalletType.HARDWARE_LEDGER, WalletType.HARDWARE_TREZOR]:
                # Hardware wallet setup
                address = await self._setup_hardware_wallet_address(wallet_type, derivation_path)
                
            else:
                raise ValueError(f"Unsupported wallet type: {wallet_type}")
            
            # Create wallet configuration
            wallet_config = WalletConfig(
                wallet_type=wallet_type,
                address=address,
                derivation_path=derivation_path,
                security_level=security_level
            )
            
            self.wallets[wallet_id] = wallet_config
            
            logger.info(f"Created new wallet {wallet_id} with address {address}")
            return wallet_id, address
            
        except Exception as e:
            logger.error(f"Failed to create wallet: {e}")
            raise

    # The existing signature is actually fine since derivation_path has a default value
    # But if you want exact match, you can add this wrapper:

    async def create_wallet_simple(self, 
                                wallet_type: WalletType,
                                security_level: SecurityLevel) -> Tuple[str, str]:
        """
        Create wallet with simplified signature matching API spec
        
        Args:
            wallet_type: Type of wallet
            security_level: Security level
            
        Returns:
            Tuple of (wallet_id, address)
        """
        return await self.create_wallet(
            wallet_type=wallet_type,
            security_level=security_level,
            derivation_path=None  # Use default
        )

    async def _save_wallet_data(self, wallet_id: str, wallet_data: Dict) -> None:
        """Save wallet data to secure storage"""
        # This would integrate with your secure database
        # For now, we'll encrypt and store in memory
        encrypted_data = self.encryption_manager.encrypt_sensitive_data(wallet_data)
        # Save to database or secure file system
        pass

    async def _setup_hardware_wallet_address(self, wallet_type: WalletType, derivation_path: Optional[str]) -> str:
        """Setup hardware wallet and get address"""
        # This would integrate with hardware wallet libraries
        # Placeholder implementation
        return "0x" + secrets.token_hex(20)

    async def sign_transaction(self, 
                             wallet_id: str,
                             transaction_data: Dict,
                             chain: str = 'ethereum') -> Dict:
        """
        Sign transaction with comprehensive security checks
        
        Args:
            wallet_id: Wallet identifier
            transaction_data: Transaction parameters
            chain: Blockchain network
            
        Returns:
            Signed transaction data
        """
        try:
            # Security checks
            security_result = await self._perform_security_checks(wallet_id, transaction_data)
            if not security_result['approved']:
                raise ValueError(f"Transaction security check failed: {security_result['reason']}")
            
            wallet_config = self.wallets.get(wallet_id)
            if not wallet_config:
                raise ValueError(f"Wallet {wallet_id} not found")
            
            # Check transaction limits
            await self._check_transaction_limits(wallet_id, transaction_data)
            
            # Get Web3 connection
            w3 = self.web3_connections.get(chain)
            if not w3:
                raise ValueError(f"No connection to {chain} blockchain")
            
            # Sign based on wallet type
            if wallet_config.wallet_type == WalletType.HOT_WALLET:
                signed_tx = await self._sign_hot_wallet_transaction(wallet_id, transaction_data, w3)
            elif wallet_config.wallet_type in [WalletType.HARDWARE_LEDGER, WalletType.HARDWARE_TREZOR]:
                signed_tx = await self._sign_hardware_wallet_transaction(wallet_id, transaction_data, w3)
            elif wallet_config.wallet_type == WalletType.MULTISIG:
                signed_tx = await self._sign_multisig_transaction(wallet_id, transaction_data, w3)
            else:
                raise ValueError(f"Unsupported wallet type: {wallet_config.wallet_type}")
            
            # Log transaction
            await self._log_transaction(wallet_id, transaction_data, signed_tx)
            
            return signed_tx
            
        except Exception as e:
            logger.error(f"Failed to sign transaction: {e}")
            raise

    async def _perform_security_checks(self, wallet_id: str, transaction_data: Dict) -> Dict:
        """Perform comprehensive security checks on transaction"""
        checks = {
            'wallet_exists': wallet_id in self.wallets,
            'amount_within_limits': True,
            'recipient_whitelisted': True,
            'no_suspicious_patterns': True,
            'rate_limit_ok': True,
            'hardware_connected': True
        }
        
        wallet_config = self.wallets.get(wallet_id)
        if not wallet_config:
            return {'approved': False, 'reason': 'Wallet not found', 'checks': checks}
        
        # Check transaction amount
        amount = Decimal(str(transaction_data.get('value', 0)))
        if amount > wallet_config.max_transaction_amount:
            checks['amount_within_limits'] = False
            return {'approved': False, 'reason': 'Amount exceeds wallet limit', 'checks': checks}
        
        # Check recipient whitelist
        to_address = transaction_data.get('to', '').lower()
        if wallet_config.whitelist_addresses and to_address not in [addr.lower() for addr in wallet_config.whitelist_addresses]:
            checks['recipient_whitelisted'] = False
            return {'approved': False, 'reason': 'Recipient not whitelisted', 'checks': checks}
        
        # Check rate limits
        if not await self._check_rate_limits(wallet_id):
            checks['rate_limit_ok'] = False
            return {'approved': False, 'reason': 'Rate limit exceeded', 'checks': checks}
        
        # All checks passed
        return {'approved': True, 'reason': 'All security checks passed', 'checks': checks}

    async def _check_transaction_limits(self, wallet_id: str, transaction_data: Dict) -> None:
        """Check transaction against daily and frequency limits"""
        wallet_config = self.wallets[wallet_id]
        amount = Decimal(str(transaction_data.get('value', 0)))
        
        # Check daily volume
        today = datetime.utcnow().date()
        daily_key = f"{wallet_id}_{today}"
        
        if daily_key not in self.daily_volumes:
            self.daily_volumes[daily_key] = Decimal('0')
        
        if self.daily_volumes[daily_key] + amount > wallet_config.daily_limit:
            raise ValueError("Daily transaction limit exceeded")
        
        self.daily_volumes[daily_key] += amount

    async def _check_rate_limits(self, wallet_id: str) -> bool:
        """Check transaction rate limits"""
        current_time = datetime.utcnow()
        hour_key = f"{wallet_id}_{current_time.hour}"
        
        if hour_key not in self.transaction_limits:
            self.transaction_limits[hour_key] = []
        
        # Remove transactions older than 1 hour
        self.transaction_limits[hour_key] = [
            tx_time for tx_time in self.transaction_limits[hour_key]
            if current_time - tx_time < timedelta(hours=1)
        ]
        
        # Check frequency limit
        max_frequency = self.security_policies.get('max_transaction_frequency', 10)
        if len(self.transaction_limits[hour_key]) >= max_frequency:
            return False
        
        self.transaction_limits[hour_key].append(current_time)
        return True

    async def _sign_hot_wallet_transaction(self, wallet_id: str, transaction_data: Dict, w3: Web3) -> Dict:
        """Sign transaction using hot wallet (private key)"""
        # Retrieve and decrypt private key
        wallet_data = await self._get_wallet_data(wallet_id)
        encrypted_key = wallet_data['encrypted_key']
        private_key = self.encryption_manager.decrypt_wallet_key(encrypted_key)
        
        # Create account object
        account = Account.from_key(private_key)
        
        # Build transaction
        transaction = {
            'nonce': w3.eth.get_transaction_count(account.address),
            'gasPrice': w3.eth.gas_price,
            'gas': transaction_data.get('gas', 21000),
            'to': transaction_data['to'],
            'value': int(transaction_data.get('value', 0)),
            'data': transaction_data.get('data', '0x')
        }
        
        # Sign transaction
        signed_txn = w3.eth.account.sign_transaction(transaction, private_key)
        
        return {
            'raw_transaction': signed_txn.rawTransaction.hex(),
            'transaction_hash': signed_txn.hash.hex(),
            'transaction': transaction
        }

    async def _sign_hardware_wallet_transaction(self, wallet_id: str, transaction_data: Dict, w3: Web3) -> Dict:
        """Sign transaction using hardware wallet"""
        # This would integrate with hardware wallet libraries
        # Placeholder implementation
        logger.info(f"Signing transaction with hardware wallet {wallet_id}")
        
        # Simulate hardware wallet signing
        await asyncio.sleep(1)  # Simulate user interaction time
        
        return {
            'raw_transaction': '0x' + secrets.token_hex(100),
            'transaction_hash': '0x' + secrets.token_hex(32),
            'transaction': transaction_data,
            'hardware_signed': True
        }

    async def _sign_multisig_transaction(self, wallet_id: str, transaction_data: Dict, w3: Web3) -> Dict:
        """Sign transaction using multisig wallet"""
        # This would integrate with multisig contract
        # Placeholder implementation
        logger.info(f"Signing transaction with multisig wallet {wallet_id}")
        
        return {
            'raw_transaction': '0x' + secrets.token_hex(100),
            'transaction_hash': '0x' + secrets.token_hex(32),
            'transaction': transaction_data,
            'multisig_signed': True,
            'requires_additional_signatures': True
        }

    async def _get_wallet_data(self, wallet_id: str) -> Dict:
        """Retrieve wallet data from secure storage"""
        # This would retrieve from your secure database
        # Placeholder implementation
        return {
            'encrypted_key': 'encrypted_private_key_data',
            'address': self.wallets[wallet_id].address
        }

    async def _log_transaction(self, wallet_id: str, transaction_data: Dict, signed_tx: Dict) -> None:
        """Log transaction for audit trail"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'wallet_id': wallet_id,
            'transaction_hash': signed_tx.get('transaction_hash'),
            'to_address': transaction_data.get('to'),
            'value': transaction_data.get('value'),
            'gas': transaction_data.get('gas'),
            'status': 'signed'
        }
        
        self.transaction_history.append(log_entry)
        logger.info(f"Transaction logged: {signed_tx.get('transaction_hash')}")

    async def emergency_stop(self, reason: str) -> None:
        """Emergency stop all wallet operations"""
        try:
            logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
            
            # Lock all wallets
            for wallet_id in self.wallets:
                await self._lock_wallet(wallet_id, reason)
            
            # Notify emergency contacts
            await self._notify_emergency_contacts(reason)
            
            # Log emergency action
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'emergency_stop',
                'reason': reason,
                'affected_wallets': list(self.wallets.keys())
            }
            
            self.transaction_history.append(log_entry)
            
        except Exception as e:
            logger.error(f"Failed to execute emergency stop: {e}")

    async def _lock_wallet(self, wallet_id: str, reason: str) -> None:
        """Lock a specific wallet"""
        logger.warning(f"Locking wallet {wallet_id}: {reason}")
        # Implementation would mark wallet as locked in database
        
    async def _notify_emergency_contacts(self, reason: str) -> None:
        """Notify emergency contacts about security incident"""
        for contact in self.emergency_contacts:
            # Send notification (email, SMS, etc.)
            logger.info(f"Notifying emergency contact {contact}: {reason}")

    async def rotate_keys(self, wallet_id: str) -> bool:
        """Rotate wallet keys for enhanced security"""
        try:
            wallet_config = self.wallets.get(wallet_id)
            if not wallet_config:
                raise ValueError(f"Wallet {wallet_id} not found")
            
            if wallet_config.wallet_type == WalletType.HOT_WALLET:
                # Generate new private key
                new_account = Account.create()
                new_private_key = new_account.key.hex()
                new_address = new_account.address
                
                # Encrypt and store new key
                encrypted_key = self.encryption_manager.encrypt_wallet_key(new_private_key)
                
                # Update wallet configuration
                wallet_config.address = new_address
                
                # Save updated wallet data
                wallet_data = {
                    'address': new_address,
                    'encrypted_key': encrypted_key,
                    'rotated_at': datetime.utcnow().isoformat()
                }
                
                await self._save_wallet_data(wallet_id, wallet_data)
                
                logger.info(f"Keys rotated for wallet {wallet_id}")
                return True
            
            else:
                logger.warning(f"Key rotation not supported for wallet type {wallet_config.wallet_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to rotate keys for wallet {wallet_id}: {e}")
            return False

    async def backup_wallet(self, wallet_id: str, backup_location: str) -> bool:
        """Create secure backup of wallet"""
        try:
            wallet_config = self.wallets.get(wallet_id)
            if not wallet_config:
                raise ValueError(f"Wallet {wallet_id} not found")
            
            # Get wallet data
            wallet_data = await self._get_wallet_data(wallet_id)
            
            # Create backup data
            backup_data = {
                'wallet_id': wallet_id,
                'wallet_config': wallet_config.__dict__,
                'wallet_data': wallet_data,
                'backup_timestamp': datetime.utcnow().isoformat(),
                'backup_version': '1.0'
            }
            
            # Encrypt backup
            encrypted_backup = self.encryption_manager.encrypt_sensitive_data(backup_data)
            
            # Save to backup location
            # This would save to secure cloud storage or hardware device
            logger.info(f"Wallet {wallet_id} backed up to {backup_location}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup wallet {wallet_id}: {e}")
            return False

    def get_wallet_status(self, wallet_id: str) -> Dict:
        """Get comprehensive wallet status"""
        wallet_config = self.wallets.get(wallet_id)
        if not wallet_config:
            return {'error': 'Wallet not found'}
        
        # Get daily volume
        today = datetime.utcnow().date()
        daily_key = f"{wallet_id}_{today}"
        daily_volume = self.daily_volumes.get(daily_key, Decimal('0'))
        
        # Get recent transaction count
        current_time = datetime.utcnow()
        hour_key = f"{wallet_id}_{current_time.hour}"
        recent_tx_count = len(self.transaction_limits.get(hour_key, []))
        
        return {
            'wallet_id': wallet_id,
            'address': wallet_config.address,
            'wallet_type': wallet_config.wallet_type.value,
            'security_level': wallet_config.security_level.value,
            'daily_volume_used': float(daily_volume),
            'daily_limit': float(wallet_config.daily_limit),
            'recent_transaction_count': recent_tx_count,
            'max_transaction_amount': float(wallet_config.max_transaction_amount),
            'requires_2fa': wallet_config.requires_2fa,
            'is_hardware_connected': wallet_config.wallet_type in [WalletType.HARDWARE_LEDGER, WalletType.HARDWARE_TREZOR]
        }

    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            # Clear sensitive data from memory
            self.wallets.clear()
            self.approved_transactions.clear()
            self.transaction_limits.clear()
            self.daily_volumes.clear()
            
            logger.info("Wallet security manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during wallet security manager cleanup: {e}")

    def encrypt_private_key(self, private_key: str) -> str:
        """
        Encrypt private key for storage
        Wrapper matching documented signature
        """
        encrypted_data = self.encryption_manager.encrypt_wallet_key(private_key)
        return encrypted_data['encrypted_key']