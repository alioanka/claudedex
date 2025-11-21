"""
Honeypot Checker - Multi-API Token Verification
Comprehensive honeypot detection using multiple security APIs and contract analysis
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
import aiohttp
from web3 import Web3
from web3.exceptions import ContractLogicError
import json

from solders.pubkey import Pubkey  # For Solana address validation

from utils.helpers import retry_async, rate_limit, is_valid_address, format_token_amount
from utils.helpers import retry_async
from utils.constants import (
    HONEYPOT_CHECKS, HONEYPOT_THRESHOLDS, Chain, CHAIN_RPC_URLS,
    BLACKLISTED_TOKENS, BLACKLISTED_CONTRACTS, BLACKLISTED_WALLETS
)

logger = logging.getLogger(__name__)

SAFE_SOLANA_TOKENS = {
    # Major established tokens
    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263": "BONK",
    "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN": "JUP",
    "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R": "RAY (Raydium)",
    "So11111111111111111111111111111111111111112": "SOL (Wrapped)",
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC",
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "USDT",
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So": "mSOL (Marinade)",
    "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj": "stSOL (Lido)",
}

def is_valid_solana_address(address: str) -> bool:
    """
    Validate Solana address format
    
    Solana addresses are base58 encoded and typically 32-44 characters
    Must not contain confusing characters (0, O, I, l)
    
    Args:
        address: Token address to validate
        
    Returns:
        True if valid Solana address format
    """
    import re
    
    if not address or not isinstance(address, str):
        return False
    
    # Solana addresses are base58 encoded, 32-44 chars typical
    if len(address) < 32 or len(address) > 44:
        return False
    
    # Base58 alphabet (no 0, O, I, l to avoid confusion)
    base58_pattern = r'^[1-9A-HJ-NP-Za-km-z]+$'
    
    if not re.match(base58_pattern, address):
        return False
    
    return True


class HoneypotChecker:
    """Advanced multi-API honeypot detection system"""

    def __init__(self, config_manager, chain_rpc_urls: Dict[str, List[str]]):
        """Initialize honeypot checker with configuration and RPC URLs."""
        self.config_mgr = config_manager
        self.chain_rpc_urls = chain_rpc_urls
        self.session = None
        self.web3_connections = {}

        # --- FIX: Load API keys from the config manager ---
        api_config = self.config_mgr.get_api_config()
        self.api_keys = {
            "honeypot_is": api_config.honeypot_is_api_key if hasattr(api_config, 'honeypot_is_api_key') else "",
            "tokensniffer": api_config.tokensniffer_api_key if hasattr(api_config, 'tokensniffer_api_key') else "",
            "goplus": api_config.goplus_api_key if hasattr(api_config, 'goplus_api_key') else ""
        }
        
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

    async def initialize(self):
        """Initialize connections and resources"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        await self._setup_web3_connections()
        logger.info("✅ HoneypotChecker initialized (EVM + Solana support)")

    async def _setup_web3_connections(self):
        """Setup Web3 connections for each chain using the provided RPC URLs."""
        for chain_name, rpc_urls in self.chain_rpc_urls.items():
            if not rpc_urls:
                continue

            # Skip Solana - it's not an EVM chain and doesn't use Web3
            if chain_name.lower() == 'solana':
                continue

            # --- FIX: Use the correct chain enum and URLs from config ---
            try:
                chain_enum = Chain[chain_name.upper()]
                rpc_url = rpc_urls[0] # Use the first available RPC URL

                self.web3_connections[chain_enum] = Web3(Web3.HTTPProvider(rpc_url))
                if self.web3_connections[chain_enum].is_connected():
                    logger.info(f"✅ Web3 connected for {chain_name.upper()} via {rpc_url[:50]}...")
                else:
                    logger.error(f"❌ Failed to connect Web3 for {chain_name.upper()}")
            except (KeyError, IndexError):
                logger.warning(f"⚠️ No valid RPC URL or chain enum for '{chain_name}'")

    async def close(self):
        """Clean up resources"""
        if self.session and not self.session.closed:
            await self.session.close()
            
    async def check_token(self, address: str, chain: str) -> Dict:
        """
        Main method to check if token is a honeypot
        Returns comprehensive analysis from multiple sources
        """
        # ✅ Route Solana tokens to dedicated checker
        if chain.lower() in ['solana', 'sol']:
            return await self._check_solana_token(address)
        
        # Existing EVM validation continues below...
        if not is_valid_address(address):
            return {
                "is_honeypot": True,
                "confidence": 1.0,
                "reason": "Invalid token address",
                "checks": {}
            }
            
        # Check blacklists first
        if self._is_blacklisted(address):
            return {
                "is_honeypot": True,
                "confidence": 1.0,
                "reason": "Token is blacklisted",
                "checks": {"blacklist": True}
            }
            
        # Check cache
        cache_key = f"{chain}:{address}"
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if asyncio.get_event_loop().time() - timestamp < self.cache_ttl:
                logger.debug(f"✅ Using cached result for {address[:10]}...")
                return cached_result
                
        try:
            # Run all checks in parallel
            checks = await self.check_multiple_apis(address, chain)
            
            # ✅ ADD THIS - If contract analysis fails, don't fail the whole check
            try:
                contract_analysis = await self.analyze_contract_code(address, chain)
                checks["contract_analysis"] = contract_analysis
            except Exception as e:
                logger.warning(f"Contract analysis failed (non-critical): {e}")
                checks["contract_analysis"] = {"error": str(e), "skip": True}
            
            # Check liquidity locks
            liquidity_check = await self.check_liquidity_locks(address, chain)
            checks["liquidity"] = liquidity_check
            
            # Calculate final verdict
            result = self._calculate_verdict(checks)
            
            # Cache result
            self.cache[cache_key] = (result, asyncio.get_event_loop().time())
            
            return result
            
        except Exception as e:
            logger.error(f"Honeypot check failed for {address}: {e}")
            # ✅ RETURN UNKNOWN RISK, NOT HONEYPOT
            return {
                "is_honeypot": False,  # ✅ Changed from True
                "confidence": 0.0,
                "risk_level": "unknown",
                "reason": f"Check failed: {str(e)}",
                "checks": {},
                "error": str(e)
            }

    async def _check_solana_token(self, address: str) -> Dict:
        """
        Comprehensive Solana token verification using RugCheck.xyz v1 API
        
        Args:
            address: Solana token address (mint address)
            
        Returns:
            Dict with honeypot analysis:
            {
                "is_honeypot": bool,
                "confidence": float (0-1),
                "risk_level": str ("safe", "low", "medium", "high", "critical"),
                "reason": str,
                "checks": dict
            }
        """
        try:
            # ✅ PATCH: Validate address format first
            if not address or not isinstance(address, str):
                logger.warning(f"⚠️ Invalid token address type: {type(address)}")
                return {
                    "is_honeypot": True,
                    "confidence": 1.0,
                    "risk_level": "critical",
                    "reason": "Invalid token address",
                    "checks": {"error": "Invalid address type"}
                }
            
            # Strip whitespace
            address = address.strip()
            
            # ✅ PATCH: Validate Solana address format
            if not is_valid_solana_address(address):
                logger.warning(f"⚠️ Invalid Solana address format: {address[:16]}...")
                return {
                    "is_honeypot": True,
                    "confidence": 1.0,
                    "risk_level": "critical",
                    "reason": "Invalid Solana address format",
                    "checks": {"validation": "Failed Solana address format check"}
                }
            
            # Check blacklist first (fast fail)
            if self._is_blacklisted(address):
                logger.warning(f"⚠️ Blacklisted Solana token: {address[:8]}...")
                return {
                    "is_honeypot": True,
                    "confidence": 1.0,
                    "risk_level": "critical",
                    "reason": "Token is blacklisted",
                    "checks": {"blacklist": True}
                }
            
            # Check cache (5 min TTL)
            cache_key = f"solana:{address}"
            if cache_key in self.cache:
                cached_result, timestamp = self.cache[cache_key]
                if asyncio.get_event_loop().time() - timestamp < self.cache_ttl:
                    logger.debug(f"✅ Using cached Solana result for {address[:8]}...")
                    return cached_result
            
            # Fetch RugCheck summary report
            rugcheck_result = await self._check_rugcheck_summary(address)
            
            # Calculate verdict from RugCheck data
            result = self._calculate_solana_verdict(rugcheck_result, address)
            
            # Cache result
            self.cache[cache_key] = (result, asyncio.get_event_loop().time())
            
            return result
            
        except Exception as e:
            logger.error(f"Solana token check failed for {address[:16]}...: {e}", exc_info=True)
            # ✅ On error, allow through with unknown risk (don't block trading)
            return {
                "is_honeypot": False,
                "confidence": 0.5,
                "risk_level": "unknown",
                "reason": f"Verification failed: {str(e)}",
                "checks": {"error": str(e)}
            }



    @retry_async(max_retries=3, delay=2.0, exponential_backoff=True)
    @rate_limit(calls=12, period=60.0)
    async def _check_rugcheck_summary(self, address: str) -> Dict:
        """
        Check token using RugCheck.xyz v1 summary API
        
        Args:
            address: Solana token address (base58 encoded)
            
        Returns:
            Dict with status and RugCheck data
            
        Example response structure:
        {
            "tokenProgram": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
            "tokenType": "Fungible",
            "risks": [
                {
                "name": "Mutable Metadata",
                "description": "Token metadata can be changed by the owner",
                "score": 100,
                "level": "warn"
                }
            ],
            "score": 101,
            "score_normalised": 7,
            "lpLockedPct": 2.9768640982543357
        }
        """
        try:
            # ✅ PATCH: Validate Solana address format BEFORE API call
            if not is_valid_solana_address(address):
                logger.warning(f"⚠️ Invalid Solana address format: {address[:16]}...")
                return {
                    "status": "error",
                    "error": "Invalid Solana address format",
                    "address": address[:16] + "..."
                }
            
            url = f"https://api.rugcheck.xyz/v1/tokens/{address}/report/summary"
            
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                # Check rate limit headers
                rate_limit_remaining = response.headers.get('x-rate-limit-remaining')
                if rate_limit_remaining:
                    logger.debug(f"RugCheck rate limit remaining: {rate_limit_remaining}")
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract and normalize data
                    return {
                        "status": "success",
                        "token_program": data.get("tokenProgram", ""),
                        "token_type": data.get("tokenType", ""),
                        "risks": data.get("risks", []),
                        "score": data.get("score", 0),  # Raw score
                        "score_normalised": data.get("score_normalised", 0),  # 0-10 scale
                        "lp_locked_pct": data.get("lpLockedPct", 0),
                    }
                    
                elif response.status == 404:
                    # Token not found in RugCheck database
                    logger.warning(f"⚠️ Token {address[:8]}... not found in RugCheck")
                    return {
                        "status": "not_found",
                        "reason": "Token not indexed by RugCheck (may be very new)"
                    }
                    
                elif response.status == 400:
                    # ✅ PATCH: Enhanced 400 error logging with actual address
                    error_text = await response.text()
                    logger.error(
                        f"RugCheck API 400 Bad Request for {address[:16]}...\n"
                        f"  Full address: {address}\n"
                        f"  Response: {error_text[:200]}"
                    )
                    return {
                        "status": "error",
                        "error": f"Bad request - invalid address format",
                        "http_status": 400,
                        "address": address[:16] + "..."
                    }
                    
                elif response.status == 429:
                    # Rate limited
                    logger.error(f"🚫 RugCheck rate limit exceeded")
                    return {
                        "status": "rate_limited",
                        "error": "Rate limit exceeded"
                    }
                    
                else:
                    # ✅ PATCH: Log response text for other errors
                    error_text = await response.text()
                    logger.error(
                        f"RugCheck API error: HTTP {response.status} for {address[:16]}...\n"
                        f"  Response: {error_text[:200]}"
                    )
                    return {
                        "status": "error",
                        "error": f"API returned {response.status}",
                        "http_status": response.status
                    }
                    
        except asyncio.TimeoutError:
            logger.warning(f"RugCheck API timeout for {address[:8]}...")
            return {
                "status": "timeout",
                "error": "API request timed out"
            }
        except Exception as e:
            logger.error(f"RugCheck check failed for {address[:8]}...: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

    # ==============================================================================
    # 6. ADD NEW METHOD - Calculate Verdict from RugCheck Data
    # ==============================================================================
    def _calculate_solana_verdict(self, rugcheck_data: Dict, address: str) -> Dict:
        """
        Calculate honeypot verdict from RugCheck summary data
        
        RugCheck Scoring:
        - score: Raw risk score (higher = riskier)
        - score_normalised: 0-10 scale (0=safest, 10=most dangerous)
        - risks: Array of risk objects with level (warn/danger/critical)
        - lpLockedPct: % of LP locked (higher = safer)
        
        Risk Levels:
        - warn: Minor issues (e.g., mutable metadata)
        - danger: Moderate risks
        - critical: Severe risks (scam/rug likely)
        """
        status = rugcheck_data.get("status")
        
        # Handle API failures gracefully
        if status == "error" or status == "timeout" or status == "rate_limited":
            logger.warning(f"⚠️ RugCheck unavailable for {address[:8]}... - allowing with caution")
            return {
                "is_honeypot": False,
                "confidence": 0.4,
                "risk_level": "medium",
                "reason": f"Token verification unavailable ({status}) - proceed with extreme caution",
                "checks": rugcheck_data,
                "note": "RugCheck data unavailable"
            }
        
        # Handle new/unknown tokens
        if status == "not_found":
            logger.info(f"🆕 Token {address[:8]}... not indexed - treating as high risk")
            return {
                "is_honeypot": False,  # Don't block, but warn
                "confidence": 0.5,
                "risk_level": "high",
                "reason": "Token not verified by RugCheck - very new or obscure",
                "checks": rugcheck_data,
                "note": "Unverified token - trade at your own risk"
            }
        
        # Extract risk data
        score_normalised = rugcheck_data.get("score_normalised", 5)  # 0-10 scale
        risks = rugcheck_data.get("risks", [])
        lp_locked_pct = rugcheck_data.get("lp_locked_pct", 0)
        
        # Count risks by severity
        critical_risks = [r for r in risks if r.get("level") == "critical"]
        danger_risks = [r for r in risks if r.get("level") == "danger"]
        warn_risks = [r for r in risks if r.get("level") == "warn"]
        
        # Build risk indicators list
        risk_names = [r.get("name", "Unknown") for r in risks[:5]]  # Top 5 risks
        
        # ✅ DECISION LOGIC - Based on RugCheck normalized score (0-10)
        
        # CRITICAL: Block immediately (score 9-10)
        if score_normalised >= 9 or len(critical_risks) > 0:
            return {
                "is_honeypot": True,
                "confidence": 0.95,
                "risk_level": "critical",
                "reason": f"Critical risks detected (score: {score_normalised}/10, {len(critical_risks)} critical)",
                "checks": rugcheck_data,
                "indicators": risk_names
            }
        
        # HIGH RISK: Block (score 7-8)
        if score_normalised >= 7 or len(danger_risks) >= 2:
            return {
                "is_honeypot": True,
                "confidence": 0.85,
                "risk_level": "high",
                "reason": f"High risk token (score: {score_normalised}/10, {len(danger_risks)} danger risks)",
                "checks": rugcheck_data,
                "indicators": risk_names
            }
        
        # MEDIUM RISK: Allow with warning (score 5-6)
        if score_normalised >= 5 or len(danger_risks) >= 1:
            return {
                "is_honeypot": False,  # ✅ Allow through
                "confidence": 0.65,
                "risk_level": "medium",
                "reason": f"Medium risk (score: {score_normalised}/10) - proceed with caution",
                "checks": rugcheck_data,
                "indicators": risk_names,
                "warning": "Trade at your own risk"
            }
        
        # LOW RISK: Allow (score 3-4)
        if score_normalised >= 3 or len(warn_risks) >= 2:
            return {
                "is_honeypot": False,
                "confidence": 0.75,
                "risk_level": "low",
                "reason": f"Low risk token (score: {score_normalised}/10, minor warnings only)",
                "checks": rugcheck_data,
                "indicators": risk_names if warn_risks else []
            }
        
        # MINIMAL RISK: Safe (score 0-2)
        return {
            "is_honeypot": False,
            "confidence": 0.9,
            "risk_level": "minimal",
            "reason": f"Token passed safety checks (score: {score_normalised}/10)",
            "checks": rugcheck_data,
            "indicators": [],
            "lp_locked": f"{lp_locked_pct:.1f}%" if lp_locked_pct > 0 else "unknown"
        }


    async def check_multiple_apis(self, address: str, chain: str) -> Dict:
        """Check token across multiple security APIs"""
        results = {}
        
        # Run API checks concurrently
        tasks = []
        
        if self.api_keys.get("honeypot_is"):
            tasks.append(self._check_honeypot_is(address, chain))
        if self.api_keys.get("tokensniffer"):
            tasks.append(self._check_tokensniffer(address, chain))
        if self.api_keys.get("goplus"):
            tasks.append(self._check_goplus(address, chain))
            
        # Always run free checks
        tasks.append(self._check_dextools(address, chain))
        tasks.append(self._check_contract_verification(address, chain))
        
        api_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        api_names = ["honeypot_is", "tokensniffer", "goplus", "dextools", "contract_verification"]
        for i, result in enumerate(api_results):
            if i < len(api_names):
                if isinstance(result, Exception):
                    logger.error(f"API check failed for {api_names[i]}: {result}")
                    results[api_names[i]] = {"error": str(result)}
                else:
                    results[api_names[i]] = result
                    
        return results
        
    @retry_async(max_retries=3, delay=1.0)
    @rate_limit(calls=5, period=1.0)
    async def _check_honeypot_is(self, address: str, chain: str) -> Dict:
        """Check token using Honeypot.is API"""
        try:
            chain_id = self._get_chain_id(chain)
            url = f"https://api.honeypot.is/v2/IsHoneypot"
            params = {
                "address": address,
                "chainID": chain_id
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "is_honeypot": data.get("honeypotResult", {}).get("isHoneypot", False),
                        "buy_tax": float(data.get("simulationResult", {}).get("buyTax", 0)),
                        "sell_tax": float(data.get("simulationResult", {}).get("sellTax", 0)),
                        "transfer_tax": float(data.get("simulationResult", {}).get("transferTax", 0)),
                        "liquidity": float(data.get("pair", {}).get("liquidity", 0)),
                        "holders": data.get("token", {}).get("totalHolders", 0)
                    }
                    
        except Exception as e:
            logger.error(f"Honeypot.is check failed: {e}")
            
        return {"error": "API check failed"}
        
    @retry_async(max_retries=3, delay=1.0)
    @rate_limit(calls=5, period=1.0)
    async def _check_tokensniffer(self, address: str, chain: str) -> Dict:
        """Check token using TokenSniffer API"""
        try:
            chain_id = self._get_chain_id(chain)
            url = f"https://tokensniffer.com/api/v2/tokens/{chain_id}/{address}"
            headers = {"Authorization": f"Bearer {self.api_keys['tokensniffer']}"}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "score": data.get("score", 0),
                        "is_flagged": data.get("is_flagged", False),
                        "buy_tax": data.get("buy_tax", 0),
                        "sell_tax": data.get("sell_tax", 0),
                        "is_honeypot": data.get("is_honeypot", False),
                        "is_mintable": data.get("is_mintable", False),
                        "owner_percentage": data.get("owner_percentage", 0),
                        "creator_percentage": data.get("creator_percentage", 0)
                    }
                    
        except Exception as e:
            logger.error(f"TokenSniffer check failed: {e}")
            
        return {"error": "API check failed"}
        
    @retry_async(max_retries=3, delay=1.0)
    @rate_limit(calls=5, period=1.0)
    async def _check_goplus(self, address: str, chain: str) -> Dict:
        """Check token using GoPlus Security API"""
        try:
            chain_id = self._get_chain_id(chain)
            url = f"https://api.gopluslabs.io/api/v1/token_security/{chain_id}"
            params = {"contract_addresses": address}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    token_data = data.get("result", {}).get(address.lower(), {})
                    
                    return {
                        "is_honeypot": token_data.get("is_honeypot", "0") == "1",
                        "is_mintable": token_data.get("is_mintable", "0") == "1",
                        "is_proxy": token_data.get("is_proxy", "0") == "1",
                        "can_take_back_ownership": token_data.get("can_take_back_ownership", "0") == "1",
                        "owner_change_balance": token_data.get("owner_change_balance", "0") == "1",
                        "hidden_owner": token_data.get("hidden_owner", "0") == "1",
                        "selfdestruct": token_data.get("selfdestruct", "0") == "1",
                        "buy_tax": float(token_data.get("buy_tax", "0")),
                        "sell_tax": float(token_data.get("sell_tax", "0")),
                        "holder_count": int(token_data.get("holder_count", "0")),
                        "total_supply": token_data.get("total_supply", "0")
                    }
                    
        except Exception as e:
            logger.error(f"GoPlus check failed: {e}")
            
        return {"error": "API check failed"}
        
    async def _check_dextools(self, address: str, chain: str) -> Dict:
        """Check token information from DexTools (free tier)"""
        try:
            # This would integrate with DexTools API if available
            # For now, returning placeholder
            return {
                "score": 50,
                "liquidity_locked": False,
                "verified": False
            }
        except Exception as e:
            logger.error(f"DexTools check failed: {e}")
            return {"error": "API check failed"}
            
    async def _check_contract_verification(self, address: str, chain: str) -> Dict:
        """Check if contract is verified on block explorer"""
        try:
            # Check contract verification status
            # This would integrate with Etherscan/BSCscan APIs
            return {
                "is_verified": False,
                "has_source": False
            }
        except Exception as e:
            logger.error(f"Contract verification check failed: {e}")
            return {"error": "Check failed"}
            
    # ✅ ADD THIS to analyze_contract_code around line 280:
    @retry_async(max_retries=3, delay=1.0, exponential_backoff=True)
    async def analyze_contract_code(self, address: str, chain: str) -> Dict:
        """Analyze smart contract code for honeypot patterns"""
        try:
            chain_id = self._get_chain_id(chain)
            if chain_id not in self.web3_connections:
                logger.warning(f"Chain {chain_id} not in web3_connections")
                return {"error": "Chain not supported", "skip": True}
            
            w3 = self.web3_connections[chain_id]
            
            # ✅ ADD THIS CHECK:
            if not w3.is_connected():
                logger.error(f"Web3 not connected for chain {chain_id}")
                return {"error": "Web3 not connected", "skip": True}
            
            # ... rest of method
            
            # Get contract code
            code = w3.eth.get_code(address)
            if not code:
                return {"is_contract": False}
                
            # Check for common honeypot patterns in bytecode
            code_hex = code.hex()
            honeypot_patterns = [
                "6c6f636b",  # "lock" in hex
                "626c61636b6c697374",  # "blacklist" in hex
                "77686974656c697374",  # "whitelist" in hex
                "6f6e6c794f776e6572",  # "onlyOwner" pattern
                "70617573",  # "paus" (pause) pattern
            ]
            
            suspicious_patterns_found = []
            for pattern in honeypot_patterns:
                if pattern in code_hex.lower():
                    suspicious_patterns_found.append(pattern)
                    
            # Try to detect standard interfaces
            is_standard_erc20 = self._check_erc20_interface(w3, address)
            
            return {
                "is_contract": True,
                "code_size": len(code),
                "suspicious_patterns": suspicious_patterns_found,
                "pattern_count": len(suspicious_patterns_found),
                "is_standard_erc20": is_standard_erc20,
                "risk_score": min(len(suspicious_patterns_found) * 20, 100)
            }
            
        except Exception as e:
            logger.error(f"Contract analysis failed: {e}")
            return {"error": str(e)}
            
    def _check_erc20_interface(self, w3: Web3, address: str) -> bool:
        """Check if contract implements standard ERC20 interface"""
        try:
            # Minimal ERC20 ABI for checking
            erc20_abi = [
                {"inputs": [], "name": "totalSupply", "outputs": [{"type": "uint256"}], "type": "function"},
                {"inputs": [{"type": "address"}], "name": "balanceOf", "outputs": [{"type": "uint256"}], "type": "function"},
                {"inputs": [{"type": "address"}, {"type": "uint256"}], "name": "transfer", "outputs": [{"type": "bool"}], "type": "function"},
            ]
            
            contract = w3.eth.contract(address=address, abi=erc20_abi)
            
            # Try to call basic functions
            _ = contract.functions.totalSupply().call()
            return True
            
        except (ContractLogicError, Exception):
            return False
            
    async def check_liquidity_locks(self, address: str, chain: str) -> Dict:
        """Check if liquidity is locked"""
        try:
            # Check major liquidity lockers
            # This would integrate with services like Unicrypt, Team.Finance, etc.
            
            return {
                "is_locked": False,
                "lock_period": 0,
                "locked_amount": 0,
                "locked_percentage": 0,
                "locker_address": None,
                "unlock_date": None
            }
            
        except Exception as e:
            logger.error(f"Liquidity lock check failed: {e}")
            return {"error": str(e)}
            
    def _is_blacklisted(self, address: str) -> bool:
        """Check if token/contract is blacklisted"""
        address_lower = address.lower()
        return (
            address_lower in BLACKLISTED_TOKENS or
            address_lower in BLACKLISTED_CONTRACTS
        )
        
    def _get_chain_id(self, chain: str) -> int:
        """Convert chain string to chain ID"""
        chain_map = {
            "ethereum": Chain.ETHEREUM,
            "eth": Chain.ETHEREUM,
            "bsc": Chain.BSC,
            "binance": Chain.BSC,
            "polygon": Chain.POLYGON,
            "matic": Chain.POLYGON,
            "arbitrum": Chain.ARBITRUM,
            "base": Chain.BASE,
            "solana": 999,  # Solana uses different addressing, not EVM chain ID
            "sol": 999
        }
        return chain_map.get(chain.lower(), Chain.ETHEREUM)
        
    def _calculate_verdict(self, checks: Dict) -> Dict:
        """Calculate final honeypot verdict based on all checks"""

        # ✅ HIGH ACTIVITY BYPASS - If token has high volume/liquidity, be lenient
        has_high_activity = False
        for api_name, result in checks.items():
            if isinstance(result, dict):
                volume = result.get("volume_24h", 0)
                liquidity = result.get("liquidity", 0)
                # $100K+ volume OR $50K+ liquidity = likely legitimate
                if volume > 100000 or liquidity > 50000:
                    has_high_activity = True
                    logger.info(f"🔥 High activity detected: vol=${volume:,.0f}, liq=${liquidity:,.0f}")
                    break

        honeypot_indicators = []
        confidence_scores = []
        reasons = []
        
        # Process API results
        for api_name, result in checks.items():
            if isinstance(result, dict) and "error" not in result:
                # Check for honeypot flags
                if result.get("is_honeypot"):
                    honeypot_indicators.append(f"{api_name}_honeypot")
                    confidence_scores.append(0.9)
                    reasons.append(f"{api_name} detected honeypot")
                    
                # Check buy/sell taxes
                buy_tax = result.get("buy_tax", 0)
                sell_tax = result.get("sell_tax", 0)
                
                if buy_tax > float(HONEYPOT_THRESHOLDS["max_buy_tax"]):
                    honeypot_indicators.append(f"{api_name}_high_buy_tax")
                    confidence_scores.append(0.8)
                    reasons.append(f"High buy tax: {buy_tax:.2%}")
                    
                if sell_tax > float(HONEYPOT_THRESHOLDS["max_sell_tax"]):
                    honeypot_indicators.append(f"{api_name}_high_sell_tax")
                    confidence_scores.append(0.9)
                    reasons.append(f"High sell tax: {sell_tax:.2%}")
                    
                # Check ownership concentration
                owner_percentage = result.get("owner_percentage", 0)
                if owner_percentage > float(HONEYPOT_THRESHOLDS["max_ownership"]) * 100:
                    honeypot_indicators.append(f"{api_name}_high_ownership")
                    confidence_scores.append(0.7)
                    reasons.append(f"High ownership concentration: {owner_percentage:.1f}%")
                    
                # Check holder count
                holders = result.get("holders", result.get("holder_count", 0))
                if holders and holders < HONEYPOT_THRESHOLDS["min_holders"]:
                    honeypot_indicators.append(f"{api_name}_low_holders")
                    confidence_scores.append(0.6)
                    reasons.append(f"Low holder count: {holders}")
                    
                # Check mintable flag
                if result.get("is_mintable"):
                    honeypot_indicators.append(f"{api_name}_mintable")
                    confidence_scores.append(0.8)
                    reasons.append("Token is mintable")
                    
                # Check other risk flags
                if result.get("can_take_back_ownership"):
                    honeypot_indicators.append(f"{api_name}_ownership_risk")
                    confidence_scores.append(0.85)
                    reasons.append("Ownership can be taken back")
                    
                if result.get("hidden_owner"):
                    honeypot_indicators.append(f"{api_name}_hidden_owner")
                    confidence_scores.append(0.75)
                    reasons.append("Hidden owner detected")
                    
                if result.get("selfdestruct"):
                    honeypot_indicators.append(f"{api_name}_selfdestruct")
                    confidence_scores.append(0.95)
                    reasons.append("Self-destruct function present")
                    
        # Process contract analysis
        if "contract_analysis" in checks:
            analysis = checks["contract_analysis"]
            if not analysis.get("error"):
                if analysis.get("suspicious_patterns"):
                    honeypot_indicators.append("suspicious_code_patterns")
                    confidence_scores.append(0.7)
                    reasons.append(f"Suspicious code patterns: {analysis['pattern_count']}")
                    
                if not analysis.get("is_standard_erc20"):
                    honeypot_indicators.append("non_standard_token")
                    confidence_scores.append(0.5)
                    reasons.append("Non-standard token implementation")
                    
                risk_score = analysis.get("risk_score", 0)
                if risk_score > 60:
                    honeypot_indicators.append("high_risk_score")
                    confidence_scores.append(risk_score / 100)
                    reasons.append(f"High risk score: {risk_score}")
                    
        # Process liquidity checks
        if "liquidity" in checks:
            liquidity = checks["liquidity"]
            if not liquidity.get("error"):
                if not liquidity.get("is_locked"):
                    honeypot_indicators.append("unlocked_liquidity")
                    confidence_scores.append(0.6)
                    reasons.append("Liquidity not locked")
                    
                # Check liquidity amount
                liquidity_usd = liquidity.get("liquidity", 0)
                if liquidity_usd < HONEYPOT_THRESHOLDS["min_liquidity"]:
                    honeypot_indicators.append("low_liquidity")
                    confidence_scores.append(0.7)
                    reasons.append(f"Low liquidity: ${liquidity_usd:,.0f}")

        # Then BEFORE the final verdict calculation (around line 483), add:
        if has_high_activity and avg_confidence < 0.8:
            logger.info(f"✅ Bypassing strict check due to high activity")
            return {
                "is_honeypot": False,
                "confidence": 0.5,
                "risk_level": "low",
                "reason": "High trading activity suggests legitimate token",
                "checks": checks,
                "indicators": honeypot_indicators,
                "bypass_reason": "high_activity"
            } 

        # Calculate final verdict
        # Calculate final verdict
        if not honeypot_indicators:
            return {
                "is_honeypot": False,
                "confidence": 0.9,
                "risk_level": "low",
                "reason": "No honeypot indicators detected",
                "checks": checks,
                "indicators": []
            }

        # Calculate weighted confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        # ✅ FIXED THRESHOLDS - Much more reasonable:
        if avg_confidence >= 0.9 and len(honeypot_indicators) >= 4:
            risk_level = "critical"
            is_honeypot = True
        elif avg_confidence >= 0.75 and len(honeypot_indicators) >= 3:
            risk_level = "high"
            is_honeypot = True
        elif avg_confidence >= 0.6 or len(honeypot_indicators) >= 2:
            risk_level = "medium"
            is_honeypot = False  # ✅ LET MEDIUM RISK THROUGH
        elif avg_confidence >= 0.4 or len(honeypot_indicators) >= 1:
            risk_level = "low"
            is_honeypot = False
        else:
            risk_level = "minimal"
            is_honeypot = False

        return {
            "is_honeypot": is_honeypot,
            "confidence": avg_confidence,
            "risk_level": risk_level,
            "reason": "; ".join(reasons[:3]) if reasons else "Multiple risk factors detected",
            "checks": checks,
            "indicators": honeypot_indicators,
            "detailed_reasons": reasons
        }
        
    async def batch_check(self, tokens: List[Dict[str, str]]) -> List[Dict]:
        """Check multiple tokens in batch"""
        tasks = []
        for token in tokens:
            tasks.append(self.check_token(token["address"], token["chain"]))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch check failed for token {i}: {result}")
                processed_results.append({
                    "is_honeypot": True,
                    "confidence": 1.0,
                    "reason": "Check failed",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
                
        return processed_results
        
    def get_risk_score(self, check_result: Dict) -> float:
        """Calculate risk score from check result (0-100)"""
        if check_result.get("is_honeypot"):
            base_score = 80
        else:
            base_score = 20
            
        # Adjust based on confidence
        confidence = check_result.get("confidence", 0.5)
        risk_score = base_score * confidence
        
        # Adjust based on indicators count
        indicators = check_result.get("indicators", [])
        risk_score += len(indicators) * 5
        
        return min(risk_score, 100)
        
    async def monitor_token(self, address: str, chain: str, interval: int = 60):
        """Monitor token continuously for changes"""
        while True:
            try:
                result = await self.check_token(address, chain)
                
                # Check for significant changes
                if result.get("is_honeypot"):
                    logger.warning(f"Honeypot alert for {address}: {result['reason']}")
                    yield result
                    
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error for {address}: {e}")
                await asyncio.sleep(interval)
                
    def update_blacklist(self, address: str, reason: str):
        """Add token to blacklist"""
        address_lower = address.lower()
        BLACKLISTED_TOKENS.add(address_lower)
        logger.info(f"Added {address} to blacklist: {reason}")
        
    def remove_from_blacklist(self, address: str):
        """Remove token from blacklist"""
        address_lower = address.lower()
        BLACKLISTED_TOKENS.discard(address_lower)
        logger.info(f"Removed {address} from blacklist")
        
    def get_statistics(self) -> Dict:
        """Get checker statistics"""
        return {
            "total_checks": len(self.cache),
            "blacklisted_tokens": len(BLACKLISTED_TOKENS),
            "blacklisted_contracts": len(BLACKLISTED_CONTRACTS),
            "cache_size": len(self.cache),
            "active_apis": sum(1 for v in self.api_keys.values() if v)
        }