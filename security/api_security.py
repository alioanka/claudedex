# security/api_security.py

import time
import hmac
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import jwt
import ipaddress

logger = logging.getLogger(__name__)

class APISecurityManager:
    """API security and rate limiting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.secret_key = config.get('api_secret', secrets.token_hex(32))
        
        # Rate limiting
        self.rate_limits = config.get('rate_limits', {
            'default': {'calls': 100, 'period': 60},
            'trading': {'calls': 10, 'period': 60},
            'data': {'calls': 1000, 'period': 60}
        })
        self._request_history = defaultdict(list)
        
        # IP whitelist/blacklist
        self.whitelisted_ips = set(config.get('whitelisted_ips', []))
        self.blacklisted_ips = set(config.get('blacklisted_ips', []))
        
        # API key management
        self._api_keys = {}
        self._load_api_keys()
        
    async def validate_request(
        self,
        request_data: Dict[str, Any],
        endpoint: str,
        method: str = 'GET'
    ) -> Tuple[bool, Optional[str]]:
        """Validate incoming API request"""
        try:
            # Check IP
            ip = request_data.get('ip_address')
            if ip:
                ip_check = self._check_ip_access(ip)
                if not ip_check[0]:
                    return False, ip_check[1]
                    
            # Check API key
            api_key = request_data.get('api_key')
            if not api_key:
                return False, "API key required"
                
            key_valid = self._validate_api_key(api_key)
            if not key_valid[0]:
                return False, key_valid[1]
                
            # Check signature
            signature = request_data.get('signature')
            if self.config.get('require_signature', True):
                sig_valid = self._validate_signature(
                    request_data,
                    signature,
                    api_key
                )
                if not sig_valid:
                    return False, "Invalid signature"
                    
            # Check rate limit
            rate_limit = await self.check_rate_limit(
                api_key,
                endpoint
            )
            if not rate_limit[0]:
                return False, rate_limit[1]
                
            # Check permissions
            permissions = self._check_permissions(
                api_key,
                endpoint,
                method
            )
            if not permissions[0]:
                return False, permissions[1]
                
            # Log successful validation
            await self._log_request(api_key, endpoint, True)
            
            return True, None
            
        except Exception as e:
            logger.error(f"Request validation error: {e}")
            return False, "Validation error"
            
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint_type: str = 'default'
    ) -> Tuple[bool, Optional[str]]:
        """Check if rate limit exceeded"""
        try:
            limits = self.rate_limits.get(endpoint_type, self.rate_limits['default'])
            max_calls = limits['calls']
            period = limits['period']
            
            now = time.time()
            
            # Clean old requests
            self._request_history[identifier] = [
                t for t in self._request_history[identifier]
                if now - t < period
            ]
            
            # Check limit
            request_count = len(self._request_history[identifier])
            
            if request_count >= max_calls:
                retry_after = period - (now - self._request_history[identifier][0])
                return False, f"Rate limit exceeded. Retry after {retry_after:.0f} seconds"
                
            # Record request
            self._request_history[identifier].append(now)
            
            return True, None
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return False, "Rate limit error"
            
    def generate_api_key(
        self,
        user_id: str,
        permissions: List[str],
        expires_days: int = 365
    ) -> Dict[str, Any]:
        """Generate new API key"""
        # Generate key
        api_key = secrets.token_urlsafe(32)
        
        # Generate secret
        api_secret = secrets.token_hex(32)
        
        # Create key data
        key_data = {
            'key': api_key,
            'secret': api_secret,
            'user_id': user_id,
            'permissions': permissions,
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(days=expires_days)).isoformat(),
            'is_active': True
        }
        
        # Store key
        self._api_keys[api_key] = key_data
        self._save_api_keys()
        
        return {
            'api_key': api_key,
            'api_secret': api_secret,
            'expires_at': key_data['expires_at']
        }
        
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key"""
        if api_key in self._api_keys:
            self._api_keys[api_key]['is_active'] = False
            self._api_keys[api_key]['revoked_at'] = datetime.utcnow().isoformat()
            self._save_api_keys()
            return True
        return False
        
    def generate_jwt_token(
        self,
        user_id: str,
        expires_minutes: int = 60
    ) -> str:
        """Generate JWT token for session"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(minutes=expires_minutes),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
        
    def validate_jwt_token(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=['HS256']
            )
            return True, payload
        except jwt.ExpiredSignatureError:
            return False, {'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return False, {'error': 'Invalid token'}
            
    def add_ip_whitelist(self, ip: str) -> bool:
        """Add IP to whitelist"""
        try:
            # Validate IP
            ipaddress.ip_address(ip)
            self.whitelisted_ips.add(ip)
            return True
        except ValueError:
            return False
            
    def add_ip_blacklist(self, ip: str) -> bool:
        """Add IP to blacklist"""
        try:
            ipaddress.ip_address(ip)
            self.blacklisted_ips.add(ip)
            self.whitelisted_ips.discard(ip)  # Remove from whitelist if present
            return True
        except ValueError:
            return False
            
    # Private methods
    
    def _check_ip_access(self, ip: str) -> Tuple[bool, Optional[str]]:
        """Check IP access permissions"""
        if ip in self.blacklisted_ips:
            return False, "IP blacklisted"
            
        if self.whitelisted_ips and ip not in self.whitelisted_ips:
            return False, "IP not whitelisted"
            
        return True, None
        
    def _validate_api_key(self, api_key: str) -> Tuple[bool, Optional[str]]:
        """Validate API key"""
        if api_key not in self._api_keys:
            return False, "Invalid API key"
            
        key_data = self._api_keys[api_key]
        
        if not key_data.get('is_active'):
            return False, "API key revoked"
            
        # Check expiration
        expires_at = datetime.fromisoformat(key_data['expires_at'])
        if datetime.utcnow() > expires_at:
            return False, "API key expired"
            
        return True, None
        
    def _validate_signature(
        self,
        request_data: Dict,
        signature: str,
        api_key: str
    ) -> bool:
        """Validate request signature"""
        if not signature:
            return False
            
        # Get secret for API key
        key_data = self._api_keys.get(api_key)
        if not key_data:
            return False
            
        secret = key_data['secret']
        
        # Create signature string
        timestamp = request_data.get('timestamp', '')
        nonce = request_data.get('nonce', '')
        body = request_data.get('body', '')
        
        message = f"{timestamp}{nonce}{body}"
        
        # Generate expected signature
        expected_signature = hmac.new(
            secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        return hmac.compare_digest(signature, expected_signature)
        
    def _check_permissions(
        self,
        api_key: str,
        endpoint: str,
        method: str
    ) -> Tuple[bool, Optional[str]]:
        """Check API key permissions for endpoint"""
        key_data = self._api_keys.get(api_key)
        if not key_data:
            return False, "Invalid API key"
            
        permissions = key_data.get('permissions', [])
        
        # Check specific permission
        required_permission = f"{method.lower()}:{endpoint}"
        
        if 'admin' in permissions or required_permission in permissions:
            return True, None
            
        # Check wildcard permissions
        endpoint_parts = endpoint.split('/')
        for i in range(len(endpoint_parts)):
            wildcard = '/'.join(endpoint_parts[:i+1]) + '/*'
            wildcard_permission = f"{method.lower()}:{wildcard}"
            if wildcard_permission in permissions:
                return True, None
                
        return False, "Insufficient permissions"
        
    def _load_api_keys(self) -> None:
        """Load API keys from storage"""
        # Implement based on your storage solution
        pass
        
    def _save_api_keys(self) -> None:
        """Save API keys to storage"""
        # Implement based on your storage solution
        pass
        
    async def _log_request(
        self,
        api_key: str,
        endpoint: str,
        success: bool
    ) -> None:
        """Log API request"""
        # Implement logging to database
        pass

    # Add these methods to APISecurityManager class:

    def validate_api_key(self, key: str) -> bool:
        """
        Public method to validate API key
        Wrapper for documented signature
        """
        result = self._validate_api_key(key)
        return result[0]

    async def rate_limit_check(self, ip: str) -> bool:
        """
        Check rate limit for IP address
        Simplified wrapper for documented signature
        """
        # Use IP as identifier
        result = await self.check_rate_limit(ip, 'default')
        return result[0]

    def generate_jwt(self, payload: Dict) -> str:
        """
        Generate JWT token
        Wrapper matching documented signature
        """
        # Add standard claims
        payload['iat'] = datetime.utcnow()
        if 'exp' not in payload:
            payload['exp'] = datetime.utcnow() + timedelta(hours=1)
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def verify_jwt(self, token: str) -> Dict:
        """
        Verify JWT token
        Wrapper matching documented signature
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=['HS256']
            )
            return payload
        except jwt.ExpiredSignatureError:
            return {'error': 'Token expired', 'valid': False}
        except jwt.InvalidTokenError:
            return {'error': 'Invalid token', 'valid': False}