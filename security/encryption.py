# security/encryption.py

import os
import logging
import hashlib
import secrets
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import json

logger = logging.getLogger(__name__)

class EncryptionManager:
    """Secure encryption for sensitive data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._master_key = self._get_or_create_master_key()
        self._fernet = Fernet(self._master_key)
        self._key_rotation_days = config.get('key_rotation_days', 30)
        self._last_rotation = datetime.utcnow()
        
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key"""
        # 1. First, check if encryption_key is provided directly in config (from .env)
        if 'encryption_key' in self.config:
            encryption_key = self.config['encryption_key']
            if isinstance(encryption_key, str):
                # If it's a string, it should be a base64-encoded Fernet key
                try:
                    # Validate it's a proper Fernet key
                    key_bytes = encryption_key.encode() if isinstance(encryption_key, str) else encryption_key
                    Fernet(key_bytes)  # This will raise if invalid
                    logger.info("Using encryption key from config/environment")
                    return key_bytes
                except Exception as e:
                    logger.warning(f"Invalid encryption_key in config: {e}. Falling back to file-based key.")

        # 2. Fall back to file-based key
        key_file = self.config.get('key_file', '.encryption_key')

        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                logger.info(f"Using encryption key from file: {key_file}")
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()

            # Save securely (consider using key management service in production)
            with open(key_file, 'wb') as f:
                f.write(key)

            # Set restrictive permissions
            os.chmod(key_file, 0o600)

            logger.info("New encryption key generated and saved to file")
            return key
            
    def encrypt_sensitive_data(self, data: Union[str, Dict, List]) -> str:
        """Encrypt sensitive data"""
        try:
            # Convert to JSON if needed
            if isinstance(data, (dict, list)):
                data = json.dumps(data)
            elif not isinstance(data, str):
                data = str(data)
                
            # Encrypt
            encrypted = self._fernet.encrypt(data.encode())
            
            # Return base64 encoded string
            return base64.b64encode(encrypted).decode()
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
            
    def decrypt_sensitive_data(self, encrypted_data: str) -> Union[str, Dict, List]:
        """Decrypt sensitive data"""
        try:
            # Decode from base64
            encrypted = base64.b64decode(encrypted_data.encode())

            # Decrypt
            decrypted = self._fernet.decrypt(encrypted).decode()

            # Try to parse as JSON
            try:
                return json.loads(decrypted)
            except json.JSONDecodeError:
                return decrypted

        except Exception as e:
            logger.error(f"Decryption failed: {type(e).__name__}: {e}")
            logger.error(f"This usually means the ENCRYPTION_KEY has changed since the data was encrypted.")
            logger.error(f"Solution: Re-run 'python scripts/import_env_secrets.py' to re-encrypt with current key")
            raise
            
    def encrypt_api_key(self, api_key: str) -> Dict[str, str]:
        """Encrypt API key with metadata"""
        encrypted = self.encrypt_sensitive_data(api_key)
        
        return {
            'encrypted_key': encrypted,
            'key_id': self._generate_key_id(api_key),
            'encrypted_at': datetime.utcnow().isoformat(),
            'algorithm': 'Fernet'
        }
        
    def decrypt_api_key(self, encrypted_data: Dict[str, str]) -> str:
        """Decrypt API key from metadata"""
        return self.decrypt_sensitive_data(encrypted_data['encrypted_key'])
        
    def encrypt_wallet_key(self, private_key: str) -> Dict[str, str]:
        """Encrypt wallet private key with extra security"""
        # Add extra entropy
        salt = secrets.token_hex(16)
        salted_key = f"{salt}:{private_key}"
        
        encrypted = self.encrypt_sensitive_data(salted_key)
        
        return {
            'encrypted_key': encrypted,
            'key_id': self._generate_key_id(private_key),
            'encrypted_at': datetime.utcnow().isoformat(),
            'algorithm': 'Fernet+Salt'
        }
        
    def decrypt_wallet_key(self, encrypted_data: Dict[str, str]) -> str:
        """Decrypt wallet private key"""
        decrypted = self.decrypt_sensitive_data(encrypted_data['encrypted_key'])
        
        # Remove salt
        if ':' in decrypted:
            _, private_key = decrypted.split(':', 1)
            return private_key
        return decrypted
        
    def rotate_encryption_key(self) -> bool:
        """Rotate encryption key periodically"""
        try:
            # Check if rotation needed
            days_since_rotation = (datetime.utcnow() - self._last_rotation).days
            if days_since_rotation < self._key_rotation_days:
                return False
                
            # Generate new key
            new_key = Fernet.generate_key()
            new_fernet = Fernet(new_key)
            
            # Re-encrypt all sensitive data (would need database access)
            # This is a placeholder - implement based on your storage
            
            # Update master key
            self._master_key = new_key
            self._fernet = new_fernet
            self._last_rotation = datetime.utcnow()
            
            # Save new key
            key_file = self.config.get('key_file', '.encryption_key')
            with open(key_file, 'wb') as f:
                f.write(new_key)
                
            logger.info("Encryption key rotated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            return False
            
    def _generate_key_id(self, key: str) -> str:
        """Generate unique ID for key (for tracking without exposing)"""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
        
    def secure_delete(self, data: Any) -> None:
        """Securely overwrite sensitive data in memory"""
        if isinstance(data, str):
            # Overwrite string memory
            data = 'x' * len(data)
        elif isinstance(data, bytes):
            # Overwrite bytes
            data = b'x' * len(data)


    # Add these methods to EncryptionManager class:

    def encrypt_data(self, data: str, key: str) -> str:
        """
        Encrypt data with provided key
        Wrapper for documented signature
        """
        # Create Fernet instance with provided key
        if len(key) != 44:  # Fernet keys are 44 chars when base64 encoded
            # Derive proper key from provided key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'stable_salt',  # In production, use random salt
                iterations=100000,
                backend=default_backend()
            )
            derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
            fernet = Fernet(derived_key)
        else:
            fernet = Fernet(key.encode())
        
        encrypted = fernet.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()

    def decrypt_data(self, encrypted: str, key: str) -> str:
        """
        Decrypt data with provided key
        Wrapper for documented signature
        """
        # Create Fernet instance with provided key
        if len(key) != 44:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'stable_salt',
                iterations=100000,
                backend=default_backend()
            )
            derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
            fernet = Fernet(derived_key)
        else:
            fernet = Fernet(key.encode())
        
        encrypted_bytes = base64.b64decode(encrypted.encode())
        decrypted = fernet.decrypt(encrypted_bytes)
        return decrypted.decode()

    def generate_key(self) -> str:
        """
        Generate a new encryption key
        """
        return Fernet.generate_key().decode()

    def hash_password(self, password: str) -> str:
        """
        Hash password using secure method
        """
        # Generate salt
        salt = secrets.token_bytes(16)
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password.encode())
        
        # Return salt + hash
        return base64.b64encode(salt + key).decode()

    def verify_password(self, password: str, hash: str) -> bool:
        """
        Verify password against hash
        """
        try:
            # Decode hash
            decoded = base64.b64decode(hash.encode())
            salt = decoded[:16]
            stored_key = decoded[16:]
            
            # Derive key with same salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            test_key = kdf.derive(password.encode())
            
            # Compare
            return test_key == stored_key
        except Exception:
            return False