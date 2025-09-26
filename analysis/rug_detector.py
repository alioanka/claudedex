# analysis/rug_detector.py

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import aiohttp
from web3 import Web3
from eth_account import Account
import json

logger = logging.getLogger(__name__)


class RugDetector:
    """
    Comprehensive rug pull detection and analysis system.
    Combines smart contract analysis, liquidity checks, and holder analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.web3_providers = {}
        self.contract_abis = {}
        
        # Initialize Web3 connections for different chains
        self._initialize_web3()
        
        # Detection thresholds
        self.thresholds = {
            'min_liquidity_usd': 10000,
            'min_holders': 50,
            'max_owner_percentage': 10,
            'max_top10_percentage': 50,
            'min_liquidity_lock_days': 30,
            'max_mint_risk': 0.3,
            'max_honeypot_risk': 0.2,
            'min_contract_age_hours': 24
        }
        
        # Risk weights
        self.risk_weights = {
            'liquidity': 0.20,
            'holders': 0.15,
            'contract': 0.25,
            'developer': 0.20,
            'honeypot': 0.20
        }
    
    def _initialize_web3(self):
        """Initialize Web3 providers for different chains."""
        chains = {
            'ethereum': self.config.get('ETH_RPC_URL'),
            'bsc': self.config.get('BSC_RPC_URL'),
            'polygon': self.config.get('POLYGON_RPC_URL'),
            'arbitrum': self.config.get('ARBITRUM_RPC_URL'),
            'base': self.config.get('BASE_RPC_URL')
        }
        
        for chain, rpc_url in chains.items():
            if rpc_url:
                self.web3_providers[chain] = Web3(Web3.HTTPProvider(rpc_url))
                logger.info(f"Initialized Web3 for {chain}")
    
    async def analyze_contract(
        self,
        address: str,
        chain: str = 'ethereum'
    ) -> Dict[str, Any]:
        """
        Comprehensive smart contract analysis.
        """
        web3 = self.web3_providers.get(chain)
        if not web3:
            return {'error': f'Web3 provider not configured for {chain}'}
        
        analysis = {
            'address': address,
            'chain': chain,
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'risks': [],
            'score': 0
        }
        
        # Check if address is valid
        if not web3.is_address(address):
            analysis['error'] = 'Invalid contract address'
            return analysis
        
        # Get contract code
        code = web3.eth.get_code(address)
        if code == b'':
            analysis['error'] = 'No contract found at address'
            return analysis
        
        # Perform various checks
        analysis['checks']['has_contract'] = True
        analysis['checks']['contract_size'] = len(code)
        
        # Check for risky functions
        risky_functions = await self._check_risky_functions(address, chain, code)
        analysis['checks'].update(risky_functions)
        
        # Check ownership
        ownership = await self._check_ownership(address, chain)
        analysis['checks'].update(ownership)
        
        # Check for proxy pattern
        is_proxy = await self._check_proxy_pattern(address, chain)
        analysis['checks']['is_proxy'] = is_proxy
        
        # Check contract verification
        is_verified = await self._check_verification(address, chain)
        analysis['checks']['is_verified'] = is_verified
        
        # Calculate risk score
        analysis['score'] = self._calculate_contract_risk_score(analysis['checks'])
        
        # Generate risks list
        if analysis['checks'].get('has_mint_function'):
            analysis['risks'].append('Contract can mint new tokens')
        
        if analysis['checks'].get('has_pause_function'):
            analysis['risks'].append('Contract can be paused')
        
        if analysis['checks'].get('has_blacklist'):
            analysis['risks'].append('Contract has blacklist functionality')
        
        if analysis['checks'].get('hidden_owner'):
            analysis['risks'].append('Contract ownership is hidden or complex')
        
        if not is_verified:
            analysis['risks'].append('Contract source code not verified')
        
        return analysis
    
    async def _check_risky_functions(
        self,
        address: str,
        chain: str,
        code: bytes
    ) -> Dict[str, bool]:
        """Check for risky functions in contract bytecode."""
        checks = {
            'has_mint_function': False,
            'has_pause_function': False,
            'has_blacklist': False,
            'has_proxy_delegate': False,
            'has_selfdestruct': False,
            'modifiable_fees': False,
            'hidden_owner': False
        }
        
        # Convert bytecode to hex string for analysis
        code_hex = code.hex()
        
        # Common function signatures (first 4 bytes of keccak256)
        signatures = {
            'mint': ['40c10f19', 'a0712d68', '449a52f8'],  # mint variations
            'pause': ['8456cb59', '5c975abb'],  # pause, paused
            'blacklist': ['f9f92be4', '9b19251a', 'f3bdc228'],  # blacklist variations
            'delegate': ['5c19a95c', '587cde1e'],  # delegate variations
            'selfdestruct': ['ff'],  # SELFDESTRUCT opcode
            'owner': ['8da5cb5b', '893d20e8'],  # owner variations
            'setFee': ['69fe0e2d', 'c0d78655', 'f2fde38b']  # fee setters
        }
        
        # Check for function signatures
        for func_type, sigs in signatures.items():
            for sig in sigs:
                if sig in code_hex:
                    if func_type == 'mint':
                        checks['has_mint_function'] = True
                    elif func_type == 'pause':
                        checks['has_pause_function'] = True
                    elif func_type == 'blacklist':
                        checks['has_blacklist'] = True
                    elif func_type == 'delegate':
                        checks['has_proxy_delegate'] = True
                    elif func_type == 'selfdestruct':
                        checks['has_selfdestruct'] = True
                    elif func_type == 'setFee':
                        checks['modifiable_fees'] = True
        
        # Check for hidden ownership patterns
        if '8da5cb5b' not in code_hex and '893d20e8' not in code_hex:
            checks['hidden_owner'] = True
        
        return checks
    
    async def _check_ownership(
        self,
        address: str,
        chain: str
    ) -> Dict[str, Any]:
        """Check contract ownership details."""
        web3 = self.web3_providers[chain]
        ownership = {
            'owner': None,
            'is_renounced': False,
            'is_multisig': False,
            'owner_balance': 0
        }
        
        try:
            # Try to call owner() function
            contract_abi = [
                {
                    "inputs": [],
                    "name": "owner",
                    "outputs": [{"type": "address"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
            
            contract = web3.eth.contract(
                address=Web3.to_checksum_address(address),
                abi=contract_abi
            )
            
            owner = contract.functions.owner().call()
            ownership['owner'] = owner
            
            # Check if renounced (owner is 0x0 or dead address)
            if owner in ['0x0000000000000000000000000000000000000000',
                        '0x000000000000000000000000000000000000dEaD']:
                ownership['is_renounced'] = True
            
            # Check if owner is a contract (potential multisig)
            owner_code = web3.eth.get_code(owner)
            if len(owner_code) > 0:
                ownership['is_multisig'] = True
            
            # Get owner balance
            ownership['owner_balance'] = web3.eth.get_balance(owner)
            
        except Exception as e:
            logger.debug(f"Could not check ownership: {e}")
        
        return ownership
    
    async def _check_proxy_pattern(
        self,
        address: str,
        chain: str
    ) -> bool:
        """Check if contract uses proxy pattern."""
        web3 = self.web3_providers[chain]
        
        try:
            # Check for proxy storage slots
            # EIP-1967 proxy implementation slot
            impl_slot = '0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc'
            implementation = web3.eth.get_storage_at(address, impl_slot)
            
            if implementation != b'\x00' * 32:
                return True
            
            # Check for other common proxy patterns
            # EIP-1967 admin slot
            admin_slot = '0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103'
            admin = web3.eth.get_storage_at(address, admin_slot)
            
            if admin != b'\x00' * 32:
                return True
                
        except Exception as e:
            logger.debug(f"Proxy check error: {e}")
        
        return False
    
    async def _check_verification(
        self,
        address: str,
        chain: str
    ) -> bool:
        """Check if contract is verified on block explorer."""
        # This would typically call block explorer APIs
        # Simplified for example
        explorers = {
            'ethereum': 'https://api.etherscan.io/api',
            'bsc': 'https://api.bscscan.com/api',
            'polygon': 'https://api.polygonscan.com/api'
        }
        
        api_url = explorers.get(chain)
        if not api_url:
            return False
        
        api_key = self.config.get(f'{chain.upper()}_SCAN_API_KEY')
        if not api_key:
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'module': 'contract',
                    'action': 'getsourcecode',
                    'address': address,
                    'apikey': api_key
                }
                
                async with session.get(api_url, params=params) as response:
                    data = await response.json()
                    
                    if data['status'] == '1' and data['result']:
                        source = data['result'][0]
                        return source.get('SourceCode', '') != ''
                        
        except Exception as e:
            logger.error(f"Verification check error: {e}")
        
        return False
    
    async def check_liquidity_lock(
        self,
        token: str,
        chain: str = 'ethereum'
    ) -> Dict[str, Any]:
        """
        Check liquidity lock status for a token.
        """
        lock_info = {
            'is_locked': False,
            'lock_percentage': 0,
            'unlock_date': None,
            'lock_provider': None,
            'risks': []
        }
        
        # Check major lock providers
        providers = [
            'unicrypt',
            'teamfinance',
            'pinksale',
            'dxsale',
            'mudra'
        ]
        
        for provider in providers:
            locked = await self._check_lock_provider(token, chain, provider)
            if locked['is_locked']:
                lock_info.update(locked)
                break
        
        # Analyze lock quality
        if lock_info['is_locked']:
            # Check lock duration
            if lock_info['unlock_date']:
                days_until_unlock = (lock_info['unlock_date'] - datetime.now()).days
                
                if days_until_unlock < 30:
                    lock_info['risks'].append('Lock expires soon')
                elif days_until_unlock < 90:
                    lock_info['risks'].append('Medium-term lock only')
            
            # Check lock percentage
            if lock_info['lock_percentage'] < 50:
                lock_info['risks'].append('Less than 50% liquidity locked')
            elif lock_info['lock_percentage'] < 80:
                lock_info['risks'].append('Not all liquidity locked')
        else:
            lock_info['risks'].append('No liquidity lock found')
        
        return lock_info
    
    async def _check_lock_provider(
        self,
        token: str,
        chain: str,
        provider: str
    ) -> Dict[str, Any]:
        """Check specific lock provider for token liquidity."""
        # This would integrate with various lock provider APIs
        # Simplified for example
        
        # Mock check - in production, call actual provider APIs
        return {
            'is_locked': False,
            'lock_percentage': 0,
            'unlock_date': None,
            'lock_provider': provider
        }
    
    async def analyze_holder_distribution(
        self,
        token: str,
        chain: str = 'ethereum'
    ) -> Dict[str, Any]:
        """
        Analyze token holder distribution for concentration risks.
        """
        distribution = {
            'total_holders': 0,
            'top_holders': [],
            'concentration_metrics': {},
            'risks': [],
            'score': 0
        }
        
        # Get holder data (would use block explorer API or graph protocol)
        holders = await self._get_token_holders(token, chain)
        
        if not holders:
            distribution['error'] = 'Could not fetch holder data'
            return distribution
        
        distribution['total_holders'] = len(holders)
        
        # Calculate concentration metrics
        total_supply = sum(h['balance'] for h in holders)
        
        # Top 10 concentration
        top10 = sorted(holders, key=lambda x: x['balance'], reverse=True)[:10]
        top10_balance = sum(h['balance'] for h in top10)
        top10_percentage = (top10_balance / total_supply * 100) if total_supply > 0 else 0
        
        distribution['concentration_metrics']['top10_percentage'] = top10_percentage
        
        # Top 50 concentration
        top50 = sorted(holders, key=lambda x: x['balance'], reverse=True)[:50]
        top50_balance = sum(h['balance'] for h in top50)
        top50_percentage = (top50_balance / total_supply * 100) if total_supply > 0 else 0
        
        distribution['concentration_metrics']['top50_percentage'] = top50_percentage
        
        # Whale detection
        whale_threshold = total_supply * 0.01  # 1% = whale
        whales = [h for h in holders if h['balance'] > whale_threshold]
        distribution['concentration_metrics']['whale_count'] = len(whales)
        
        # Store top holders
        for holder in top10:
            distribution['top_holders'].append({
                'address': holder['address'],
                'balance': holder['balance'],
                'percentage': (holder['balance'] / total_supply * 100) if total_supply > 0 else 0,
                'is_contract': holder.get('is_contract', False),
                'label': holder.get('label', 'Unknown')
            })
        
        # Risk assessment
        if len(holders) < self.thresholds['min_holders']:
            distribution['risks'].append(f'Only {len(holders)} holders')
        
        if top10_percentage > self.thresholds['max_top10_percentage']:
            distribution['risks'].append(f'Top 10 holders own {top10_percentage:.1f}%')
        
        if distribution['concentration_metrics']['whale_count'] > 10:
            distribution['risks'].append(f'{len(whales)} whale wallets detected')
        
        # Calculate score (0 = good, 1 = bad)
        distribution['score'] = self._calculate_holder_risk_score(distribution)
        
        return distribution
    
    async def _get_token_holders(
        self,
        token: str,
        chain: str
    ) -> List[Dict[str, Any]]:
        """Get token holder list."""
        # This would use block explorer API or The Graph
        # Simplified mock for example
        holders = []
        
        # Mock data - in production, fetch real holder data
        # Would typically paginate through all holders
        
        return holders
    
    def _calculate_contract_risk_score(self, checks: Dict[str, Any]) -> float:
        """Calculate contract risk score (0-1, higher is riskier)."""
        score = 0.0
        
        # Penalize risky functions
        if checks.get('has_mint_function'):
            score += 0.2
        
        if checks.get('has_pause_function'):
            score += 0.1
        
        if checks.get('has_blacklist'):
            score += 0.15
        
        if checks.get('has_selfdestruct'):
            score += 0.25
        
        if checks.get('hidden_owner'):
            score += 0.15
        
        if not checks.get('is_verified'):
            score += 0.1
        
        if checks.get('is_proxy'):
            score += 0.05
        
        return min(score, 1.0)
    
    def _calculate_holder_risk_score(self, distribution: Dict[str, Any]) -> float:
        """Calculate holder distribution risk score."""
        score = 0.0
        
        # Few holders
        holders = distribution['total_holders']
        if holders < 50:
            score += 0.3
        elif holders < 100:
            score += 0.2
        elif holders < 500:
            score += 0.1
        
        # High concentration
        top10 = distribution['concentration_metrics'].get('top10_percentage', 0)
        if top10 > 70:
            score += 0.3
        elif top10 > 50:
            score += 0.2
        elif top10 > 30:
            score += 0.1
        
        # Many whales
        whales = distribution['concentration_metrics'].get('whale_count', 0)
        if whales > 20:
            score += 0.2
        elif whales > 10:
            score += 0.15
        elif whales > 5:
            score += 0.1
        
        return min(score, 1.0)
    
    async def comprehensive_analysis(
        self,
        token: str,
        chain: str = 'ethereum'
    ) -> Dict[str, Any]:
        """
        Perform comprehensive rug pull analysis on a token.
        """
        logger.info(f"Starting comprehensive rug analysis for {token} on {chain}")
        
        analysis = {
            'token': token,
            'chain': chain,
            'timestamp': datetime.now().isoformat(),
            'contract_analysis': {},
            'liquidity_analysis': {},
            'holder_analysis': {},
            'overall_risk_score': 0,
            'risk_level': 'UNKNOWN',
            'recommendations': [],
            'red_flags': []
        }
        
        # Parallel analysis tasks
        tasks = [
            self.analyze_contract(token, chain),
            self.check_liquidity_lock(token, chain),
            self.analyze_holder_distribution(token, chain)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        if not isinstance(results[0], Exception):
            analysis['contract_analysis'] = results[0]
        
        if not isinstance(results[1], Exception):
            analysis['liquidity_analysis'] = results[1]
        
        if not isinstance(results[2], Exception):
            analysis['holder_analysis'] = results[2]
        
        # Calculate overall risk score
        scores = []
        
        if analysis['contract_analysis']:
            scores.append(analysis['contract_analysis'].get('score', 0.5))
            analysis['red_flags'].extend(analysis['contract_analysis'].get('risks', []))
        
        if analysis['liquidity_analysis']:
            liq_score = 1.0 if not analysis['liquidity_analysis']['is_locked'] else 0.2
            scores.append(liq_score)
            analysis['red_flags'].extend(analysis['liquidity_analysis'].get('risks', []))
        
        if analysis['holder_analysis']:
            scores.append(analysis['holder_analysis'].get('score', 0.5))
            analysis['red_flags'].extend(analysis['holder_analysis'].get('risks', []))
        
        # Calculate weighted average
        if scores:
            analysis['overall_risk_score'] = sum(scores) / len(scores)
        
        # Determine risk level
        if analysis['overall_risk_score'] < 0.3:
            analysis['risk_level'] = 'LOW'
            analysis['recommendations'].append('Token appears relatively safe')
        elif analysis['overall_risk_score'] < 0.6:
            analysis['risk_level'] = 'MEDIUM'
            analysis['recommendations'].append('Trade with caution and small positions')
        elif analysis['overall_risk_score'] < 0.8:
            analysis['risk_level'] = 'HIGH'
            analysis['recommendations'].append('High risk - not recommended')
        else:
            analysis['risk_level'] = 'CRITICAL'
            analysis['recommendations'].append('AVOID - Extremely high rug risk')
        
        logger.info(f"Rug analysis complete. Risk level: {analysis['risk_level']}")
        
        return analysis

    # ============================================================================
    # PATCH FOR: rug_detector.py
    # Add these wrapper methods to the RugDetector class
    # ============================================================================

    async def analyze_token(self, token: str, chain: str = 'ethereum') -> Dict:
        """
        Analyze token for rug pull risk (wrapper for comprehensive_analysis)
        
        Args:
            token: Token address
            chain: Blockchain network
            
        Returns:
            Rug analysis results
        """
        # Use existing comprehensive_analysis
        analysis = await self.comprehensive_analysis(token, chain)
        
        # Format to expected structure
        return {
            'token': token,
            'chain': chain,
            'risk_score': analysis['overall_risk_score'],
            'risk_level': analysis['risk_level'],
            'contract_analysis': analysis['contract_analysis'],
            'liquidity_analysis': analysis['liquidity_analysis'],
            'holder_analysis': analysis['holder_analysis'],
            'red_flags': analysis['red_flags'],
            'recommendations': analysis['recommendations'],
            'timestamp': analysis['timestamp']
        }

    def check_ownership_concentration(self, holder_data: Dict) -> float:
        """
        Check ownership concentration risk
        
        Args:
            holder_data: Holder distribution data
            
        Returns:
            Concentration risk score (0-1)
        """
        if not holder_data:
            return 0.5  # Unknown risk
        
        # Extract concentration metrics
        top_10_percentage = holder_data.get('concentration_metrics', {}).get('top10_percentage', 0)
        top_50_percentage = holder_data.get('concentration_metrics', {}).get('top50_percentage', 0)
        whale_count = holder_data.get('concentration_metrics', {}).get('whale_count', 0)
        total_holders = holder_data.get('total_holders', 0)
        
        risk_score = 0.0
        
        # Top 10 concentration
        if top_10_percentage > 70:
            risk_score += 0.4
        elif top_10_percentage > 50:
            risk_score += 0.25
        elif top_10_percentage > 30:
            risk_score += 0.1
        
        # Total holders
        if total_holders < 50:
            risk_score += 0.3
        elif total_holders < 100:
            risk_score += 0.15
        elif total_holders < 500:
            risk_score += 0.05
        
        # Whale presence
        if whale_count > 20:
            risk_score += 0.2
        elif whale_count > 10:
            risk_score += 0.1
        elif whale_count > 5:
            risk_score += 0.05
        
        return min(1.0, risk_score)

    def check_contract_vulnerabilities(self, contract_code: str) -> List[str]:
        """
        Check contract code for vulnerabilities
        
        Args:
            contract_code: Contract source code or bytecode
            
        Returns:
            List of detected vulnerabilities
        """
        vulnerabilities = []
        
        if not contract_code:
            return ['Contract code not available']
        
        # Convert to lowercase for checking
        code_lower = contract_code.lower()
        
        # Check for dangerous functions
        if 'selfdestruct' in code_lower or 'suicide' in code_lower:
            vulnerabilities.append('Self-destruct capability')
        
        if 'delegatecall' in code_lower:
            vulnerabilities.append('Delegatecall usage (potential proxy)')
        
        # Check for mint functions
        if 'mint' in code_lower and 'onlyowner' in code_lower:
            vulnerabilities.append('Owner can mint tokens')
        
        # Check for pause capability
        if 'pause' in code_lower or 'whennotpaused' in code_lower:
            vulnerabilities.append('Contract can be paused')
        
        # Check for blacklist
        if 'blacklist' in code_lower or 'blocked' in code_lower:
            vulnerabilities.append('Blacklist functionality present')
        
        # Check for fee manipulation
        if 'setfee' in code_lower or 'updatefee' in code_lower:
            vulnerabilities.append('Fees can be modified')
        
        # Check for hidden functions
        if 'internal' in code_lower and 'onlyowner' in code_lower:
            if code_lower.count('function') > 20:
                vulnerabilities.append('Many restricted functions')
        
        # Check for upgrade capability
        if 'upgrade' in code_lower or 'implementation' in code_lower:
            vulnerabilities.append('Contract is upgradeable')
        
        return vulnerabilities if vulnerabilities else ['No major vulnerabilities detected']

    def calculate_rug_score(self, factors: Dict) -> float:
        """
        Calculate overall rug pull risk score
        
        Args:
            factors: Dictionary of risk factors
            
        Returns:
            Rug risk score (0-1, higher is riskier)
        """
        score = 0.0
        
        # Liquidity factors
        if not factors.get('liquidity_locked', True):
            score += 0.25
        
        liquidity_amount = factors.get('liquidity_amount', 0)
        if liquidity_amount < 10000:
            score += 0.15
        elif liquidity_amount < 50000:
            score += 0.05
        
        # Contract factors
        if not factors.get('contract_verified', True):
            score += 0.1
        
        if factors.get('has_mint_function', False):
            score += 0.15
        
        if factors.get('has_pause_function', False):
            score += 0.1
        
        # Ownership factors
        ownership_concentration = factors.get('ownership_concentration', 0)
        score += ownership_concentration * 0.2
        
        # Holder factors
        holder_count = factors.get('holder_count', 0)
        if holder_count < 50:
            score += 0.2
        elif holder_count < 100:
            score += 0.1
        elif holder_count < 500:
            score += 0.05
        
        # Developer factors
        if factors.get('anonymous_team', False):
            score += 0.1
        
        if factors.get('previous_rugs', 0) > 0:
            score += 0.3
        
        return min(1.0, score)

    # ============================================================================
    # FIXES FOR: rug_detector.py
    # ============================================================================

    # Add missing method: check_liquidity_removal_risk
    def check_liquidity_removal_risk(self, liquidity_data: Dict) -> float:
        """
        Check liquidity removal risk based on liquidity data
        
        Args:
            liquidity_data: Dictionary containing liquidity information
            
        Returns:
            Risk score (0-1, higher is riskier)
        """
        risk_score = 0.0
        
        # Check if liquidity is locked
        if not liquidity_data.get('is_locked', False):
            risk_score += 0.3
        
        # Check liquidity amount
        liquidity_usd = liquidity_data.get('liquidity_usd', 0)
        if liquidity_usd < self.thresholds['min_liquidity_usd']:
            risk_score += 0.3
        elif liquidity_usd < self.thresholds['min_liquidity_usd'] * 5:
            risk_score += 0.15
        
        # Check lock duration
        unlock_date = liquidity_data.get('unlock_date')
        if unlock_date:
            from datetime import datetime
            days_until_unlock = (unlock_date - datetime.now()).days
            
            if days_until_unlock < 7:
                risk_score += 0.3  # Unlocking very soon
            elif days_until_unlock < 30:
                risk_score += 0.2  # Unlocking soon
            elif days_until_unlock < 90:
                risk_score += 0.1  # Medium-term lock
        else:
            risk_score += 0.1  # No unlock date info
        
        # Check lock percentage
        lock_percentage = liquidity_data.get('lock_percentage', 0)
        if lock_percentage < 50:
            risk_score += 0.2
        elif lock_percentage < 80:
            risk_score += 0.1
        elif lock_percentage < 95:
            risk_score += 0.05
        
        # Check for recent removals
        recent_removals = liquidity_data.get('recent_removals', [])
        if len(recent_removals) > 0:
            total_removed = sum(r.get('amount_usd', 0) for r in recent_removals)
            if total_removed > liquidity_usd * 0.2:  # More than 20% removed
                risk_score += 0.25
            elif total_removed > liquidity_usd * 0.1:  # More than 10% removed
                risk_score += 0.15
        
        # Check removal velocity
        removal_velocity = liquidity_data.get('removal_velocity', 0)
        if removal_velocity > 2:  # More than 2 removals per hour
            risk_score += 0.15
        elif removal_velocity > 1:
            risk_score += 0.05
        
        return min(1.0, risk_score)
