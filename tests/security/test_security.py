# tests/security/test_security.py
"""
Security tests for trading bot
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
import hashlib
import hmac
import secrets

from security.encryption import EncryptionManager
from security.api_security import APISecurityManager
from security.wallet_security import WalletSecurityManager, SecurityLevel
from security.audit_logger import AuditLogger, AuditEventType, AuditSeverity

@pytest.mark.security
class TestSecurity:
    """Security test suite"""
    @pytest.fixture
    def mock_config(self):
        """Mock security configuration"""
        return {
            "security": {
                "encryption_key": "test_key_32_bytes_long_000000!",
                "jwt_secret": "test_jwt_secret",
                "rate_limits": {"default": {"requests": 100, "window": 60}}
            },
            "wallet": {
                "security_level": "high",
                "require_2fa": True
            }
        }
    
    @pytest.mark.asyncio
    async def test_encryption_manager(self, mock_config):
        """Test encryption and decryption"""
        manager = EncryptionManager(mock_config)
        
        # Test key generation
        key = manager.generate_key()
        assert len(key) == 44  # Base64 encoded 32-byte key
        
        # Test encryption/decryption
        sensitive_data = "private_key_12345"
        encrypted = manager.encrypt(sensitive_data)
        decrypted = manager.decrypt(encrypted)
        
        assert decrypted == sensitive_data
        assert encrypted != sensitive_data
        
        # Test that different encryptions produce different ciphertexts
        encrypted2 = manager.encrypt(sensitive_data)
        assert encrypted != encrypted2
        
        # But both decrypt to same value
        assert manager.decrypt(encrypted2) == sensitive_data
    
    @pytest.mark.asyncio
    async def test_api_security_rate_limiting(self, mock_config):
        """Test API rate limiting"""
        api_security = APISecurityManager(mock_config)
        await api_security.initialize()
        
        client_ip = "192.168.1.1"
        
        # Should allow initial requests
        for i in range(10):
            allowed = await api_security.check_rate_limit(client_ip)
            assert allowed == True
        
        # Simulate reaching rate limit (assuming 10 requests per minute)
        api_security.rate_limits["default"] = {"requests": 10, "window": 60}
        
        # Next request should be blocked
        allowed = await api_security.check_rate_limit(client_ip)
        assert allowed == False
        
        # Test rate limit reset
        await asyncio.sleep(1)  # Wait for window to pass
        # In real scenario, would wait full window
    
    @pytest.mark.asyncio
    async def test_api_authentication(self, mock_config):
        """Test JWT authentication"""
        api_security = APISecurityManager(mock_config)
        await api_security.initialize()
        
        # Generate token
        user_id = "user_123"
        token = await api_security.generate_token(user_id, ["trade", "read"])
        assert token is not None
        
        # Validate token
        payload = await api_security.validate_token(token)
        assert payload is not None
        assert payload["user_id"] == user_id
        assert "trade" in payload["permissions"]
        
        # Test invalid token
        invalid_token = "invalid.token.here"
        payload = await api_security.validate_token(invalid_token)
        assert payload is None
    
    @pytest.mark.asyncio
    async def test_wallet_security(self, wallet_security):
        """Test wallet security features"""
        # Create high security wallet
        public_key, wallet_id = await wallet_security.create_wallet(
            wallet_type="hot",
            security_level=SecurityLevel.HIGH
        )
        
        assert public_key is not None
        assert wallet_id is not None
        
        # Test transaction signing
        transaction = {
            "to": "0xrecipient",
            "value": "1000000000",
            "data": "0x",
            "nonce": 1
        }
        
        signature = await wallet_security.sign_transaction(
            wallet_id, 
            transaction, 
            "ethereum"
        )
        
        assert signature is not None
        assert "signature" in signature
        
        # Test key rotation
        success = await wallet_security.rotate_keys(wallet_id)
        assert success == True
        
        # Test emergency stop
        await wallet_security.emergency_stop("Security breach detected")
        status = wallet_security.get_wallet_status(wallet_id)
        assert status["locked"] == True
    
    @pytest.mark.asyncio
    async def test_audit_logging(self, audit_logger):
        """Test audit logging with tamper detection"""
        # Log security event
        event_id = await audit_logger.log_event(
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.HIGH,
            status="detected",
            source="security_module",
            action="unauthorized_access_attempt",
            resource="wallet_manager",
            details={"ip": "192.168.1.100", "attempts": 5}
        )
        
        assert event_id is not None
        
        # Search for events
        events = await audit_logger.search_events(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now(),
            filters={"event_type": AuditEventType.SECURITY}
        )
        
        assert len(events) > 0
        assert events[0]["event_id"] == event_id
        
        # Verify checksum (tamper detection)
        event = events[0]
        computed_checksum = audit_logger._compute_checksum(event)
        assert computed_checksum == event["checksum"]
        
        # Test tamper detection
        event["details"]["attempts"] = 10  # Modify event
        new_checksum = audit_logger._compute_checksum(event)
        assert new_checksum != event["checksum"]  # Should detect tampering
    
    @pytest.mark.asyncio
    async def test_sql_injection_protection(self, db_manager):
        """Test SQL injection protection"""
        # Attempt SQL injection
        malicious_token = "'; DROP TABLE trades; --"
        
        # This should be safely escaped
        result = await db_manager.get_token_data(malicious_token)
        
        # Table should still exist
        tables = await db_manager.get_tables()
        assert "trades" in tables
        
        # No data should be returned for malicious input
        assert result is None or len(result) == 0
    
    @pytest.mark.asyncio
    async def test_input_validation(self):
        """Test input validation and sanitization"""
        from config.validation import ConfigValidator
        
        validator = ConfigValidator()
        
        # Test valid Ethereum address
        valid_address = "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb8"
        assert validator.validate_ethereum_address(valid_address) == True
        
        # Test invalid addresses
        invalid_addresses = [
            "0x123",  # Too short
            "742d35Cc6634C0532925a3b844Bc9e7595f0bEb8",  # Missing 0x
            "0xGGGG35Cc6634C0532925a3b844Bc9e7595f0bEb8",  # Invalid characters
            "'; DROP TABLE users; --"  # SQL injection attempt
        ]
        
        for address in invalid_addresses:
            assert validator.validate_ethereum_address(address) == False
    
    @pytest.mark.asyncio
    async def test_private_key_protection(self, mock_config):
        """Test private key protection"""
        from security.wallet_security import WalletSecurityManager
        
        manager = WalletSecurityManager(mock_config)
        await manager.initialize()
        
        # Private keys should never be exposed in logs or errors
        with pytest.raises(Exception) as exc_info:
            # Simulate error that might expose private key
            await manager.sign_transaction("invalid_wallet", {}, "ethereum")
        
        error_message = str(exc_info.value)
        assert "private" not in error_message.lower()
        assert "key" not in error_message.lower()
        assert "0x" not in error_message  # No hex keys
    
    @pytest.mark.asyncio
    async def test_secure_random_generation(self):
        """Test secure random number generation"""
        # Generate secure random values
        random_bytes = secrets.token_bytes(32)
        assert len(random_bytes) == 32
        
        # Generate multiple values and ensure they're different
        values = set()
        for _ in range(100):
            values.add(secrets.token_hex(16))
        
        # All values should be unique
        assert len(values) == 100