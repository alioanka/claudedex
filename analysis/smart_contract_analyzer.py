"""
Smart Contract Analyzer - Automated contract security analysis for ClaudeDex Trading Bot

This module provides comprehensive smart contract analysis including:
- Vulnerability detection
- Code pattern analysis
- Ownership and permission checks
- Upgrade mechanism detection
- Fee and tax analysis
- Mint function detection
- Proxy pattern analysis
"""

import asyncio
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import json

from web3 import Web3
from web3.contract import Contract
from eth_utils import is_address, to_checksum_address
import aiohttp
from loguru import logger

from utils.helpers import retry_async, measure_time
from utils.constants import CHAIN_RPC_URLS, BLOCK_EXPLORERS
from utils.errors import NetworkError, ABIError, DecodeError, ContractError


class VulnerabilityLevel(Enum):
    """Contract vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ContractType(Enum):
    """Smart contract types"""
    TOKEN = "token"
    DEX_PAIR = "dex_pair"
    FARM = "farm"
    VAULT = "vault"
    PROXY = "proxy"
    MULTISIG = "multisig"
    TIMELOCK = "timelock"
    UNKNOWN = "unknown"


@dataclass
class Vulnerability:
    """Vulnerability finding"""
    name: str
    level: VulnerabilityLevel
    description: str
    location: Optional[str] = None
    recommendation: Optional[str] = None
    confidence: float = 1.0


@dataclass
class ContractAnalysis:
    """Complete contract analysis result"""
    address: str
    chain: str
    contract_type: ContractType
    is_verified: bool
    is_proxy: bool
    implementation_address: Optional[str]
    owner: Optional[str]
    vulnerabilities: List[Vulnerability]
    permissions: Dict[str, Any]
    fees: Dict[str, Decimal]
    functions: Dict[str, Dict]
    modifiers: List[str]
    events: List[str]
    storage_layout: Dict[str, Any]
    upgrade_mechanism: Optional[str]
    risk_score: float
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class SmartContractAnalyzer:
    """Analyzes smart contracts for security vulnerabilities and risks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.web3_connections: Dict[str, Web3] = {}
        self.etherscan_apis: Dict[str, str] = {}
        self.vulnerability_patterns: Dict[str, Dict] = self._load_vulnerability_patterns()
        self.known_signatures: Dict[str, str] = self._load_function_signatures()
        self.malicious_contracts: Set[str] = set()
        self.analysis_cache: Dict[str, ContractAnalysis] = {}
        self.cache_ttl = config.get("cache_ttl", 3600)
        
    async def initialize(self) -> None:
        """Initialize analyzer components"""
        logger.info("Initializing Smart Contract Analyzer...")
        
        # Setup Web3 connections
        for chain, rpc_url in CHAIN_RPC_URLS.items():
            try:
                self.web3_connections[chain] = Web3(Web3.HTTPProvider(rpc_url))
                logger.info(f"Connected to {chain} RPC")
            except Exception as e:
                logger.error(f"Failed to connect to {chain}: {e}")
        
        # Load Etherscan API keys
        self.etherscan_apis = self.config.get("etherscan_apis", {})
        
        # Load known malicious contracts
        await self._load_malicious_contracts()
        
        logger.info("Smart Contract Analyzer initialized")
    
    @retry_async(max_retries=3, delay=1.0)
    @measure_time
    async def analyze_contract(
        self,
        address: str,
        chain: str,
        deep_analysis: bool = True
    ) -> ContractAnalysis:
        """
        Perform comprehensive contract analysis
        
        Args:
            address: Contract address
            chain: Blockchain network
            deep_analysis: Whether to perform deep code analysis
            
        Returns:
            Complete contract analysis
        """
        try:
            # Check cache
            cache_key = f"{chain}:{address}"
            if cache_key in self.analysis_cache:
                cached = self.analysis_cache[cache_key]
                if (datetime.now() - cached.timestamp).seconds < self.cache_ttl:
                    logger.debug(f"Using cached analysis for {address}")
                    return cached
            
            # Validate address
            if not is_address(address):
                raise ValueError(f"Invalid address: {address}")
            
            address = to_checksum_address(address)
            
            # Check if malicious
            if address.lower() in self.malicious_contracts:
                logger.warning(f"Known malicious contract detected: {address}")
                return self._create_malicious_analysis(address, chain)
            
            # Get contract data
            contract_data = await self._fetch_contract_data(address, chain)
            
            # Determine contract type
            contract_type = await self._identify_contract_type(contract_data, chain)
            
            # Check if verified
            is_verified = await self._check_verification(address, chain)
            
            # Analyze proxy pattern
            is_proxy, impl_address = await self._analyze_proxy_pattern(
                address, contract_data, chain
            )
            
            # Get owner
            owner = await self._get_owner(address, contract_data, chain)
            
            # Analyze vulnerabilities
            vulnerabilities = []
            if deep_analysis and contract_data.get("source_code"):
                vulnerabilities = await self._analyze_vulnerabilities(
                    contract_data["source_code"]
                )
            
            # Analyze permissions
            permissions = await self._analyze_permissions(contract_data, chain)
            
            # Analyze fees
            fees = await self._analyze_fees(contract_data, chain)
            
            # Extract functions
            functions = self._extract_functions(contract_data)
            
            # Extract modifiers
            modifiers = self._extract_modifiers(contract_data)
            
            # Extract events
            events = self._extract_events(contract_data)
            
            # Analyze storage
            storage_layout = await self._analyze_storage(address, chain)
            
            # Check upgrade mechanism
            upgrade_mechanism = await self._check_upgrade_mechanism(
                contract_data, is_proxy
            )
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(
                vulnerabilities=vulnerabilities,
                is_verified=is_verified,
                is_proxy=is_proxy,
                owner=owner,
                permissions=permissions,
                fees=fees,
                upgrade_mechanism=upgrade_mechanism
            )
            
            # Create analysis
            analysis = ContractAnalysis(
                address=address,
                chain=chain,
                contract_type=contract_type,
                is_verified=is_verified,
                is_proxy=is_proxy,
                implementation_address=impl_address,
                owner=owner,
                vulnerabilities=vulnerabilities,
                permissions=permissions,
                fees=fees,
                functions=functions,
                modifiers=modifiers,
                events=events,
                storage_layout=storage_layout,
                upgrade_mechanism=upgrade_mechanism,
                risk_score=risk_score,
                metadata={
                    "compiler_version": contract_data.get("compiler_version"),
                    "optimization": contract_data.get("optimization_enabled"),
                    "runs": contract_data.get("runs"),
                    "evm_version": contract_data.get("evm_version"),
                    "license": contract_data.get("license"),
                    "creation_tx": contract_data.get("creation_tx"),
                    "creation_block": contract_data.get("creation_block")
                }
            )
            
            # Cache result
            self.analysis_cache[cache_key] = analysis
            
            # Log findings
            self._log_analysis_summary(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Contract analysis failed for {address}: {e}")
            raise
    
    async def _fetch_contract_data(
        self,
        address: str,
        chain: str
    ) -> Dict[str, Any]:
        """Fetch contract data from blockchain and explorers"""
        data = {}
        
        # Get bytecode from blockchain
        w3 = self.web3_connections.get(chain)
        if w3:
            code = w3.eth.get_code(address)
            data["bytecode"] = code.hex() if code else ""
            
            # Try to get contract ABI from Etherscan
            if chain in self.etherscan_apis:
                contract_info = await self._fetch_from_etherscan(address, chain)
                data.update(contract_info)
        
        return data
    
    async def _fetch_from_etherscan(
        self,
        address: str,
        chain: str
    ) -> Dict[str, Any]:
        """Fetch contract data from Etherscan-like explorers"""
        api_key = self.etherscan_apis.get(chain)
        if not api_key:
            return {}
        
        base_url = BLOCK_EXPLORERS.get(chain, {}).get("api_url")
        if not base_url:
            return {}
        
        async with aiohttp.ClientSession() as session:
            # Get contract source code
            params = {
                "module": "contract",
                "action": "getsourcecode",
                "address": address,
                "apikey": api_key
            }
            
            async with session.get(base_url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data["status"] == "1" and data["result"]:
                        result = data["result"][0]
                        return {
                            "source_code": result.get("SourceCode", ""),
                            "abi": json.loads(result.get("ABI", "[]")),
                            "contract_name": result.get("ContractName"),
                            "compiler_version": result.get("CompilerVersion"),
                            "optimization_enabled": result.get("OptimizationUsed") == "1",
                            "runs": int(result.get("Runs", 0)),
                            "evm_version": result.get("EVMVersion"),
                            "license": result.get("LicenseType"),
                            "is_proxy": result.get("Proxy") == "1",
                            "implementation": result.get("Implementation")
                        }
        
        return {}
    
    async def _identify_contract_type(
        self,
        contract_data: Dict,
        chain: str
    ) -> ContractType:
        """Identify the type of smart contract"""
        source = contract_data.get("source_code", "")
        abi = contract_data.get("abi", [])
        
        # Check function signatures
        functions = {item["name"] for item in abi if item.get("type") == "function"}
        
        # Token detection
        token_functions = {"transfer", "approve", "balanceOf", "totalSupply"}
        if token_functions.issubset(functions):
            return ContractType.TOKEN
        
        # DEX pair detection
        pair_functions = {"swap", "getReserves", "token0", "token1"}
        if pair_functions.issubset(functions):
            return ContractType.DEX_PAIR
        
        # Farm detection
        farm_functions = {"stake", "withdraw", "getReward", "earned"}
        if len(farm_functions.intersection(functions)) >= 3:
            return ContractType.FARM
        
        # Vault detection
        vault_functions = {"deposit", "withdraw", "getPricePerFullShare"}
        if vault_functions.issubset(functions):
            return ContractType.VAULT
        
        # Proxy detection
        if "implementation" in functions or "upgradeTo" in functions:
            return ContractType.PROXY
        
        # Multisig detection
        multisig_functions = {"submitTransaction", "confirmTransaction", "executeTransaction"}
        if multisig_functions.issubset(functions):
            return ContractType.MULTISIG
        
        # Timelock detection
        if "queueTransaction" in functions and "executeTransaction" in functions:
            return ContractType.TIMELOCK
        
        return ContractType.UNKNOWN
    
    async def _check_verification(self, address: str, chain: str) -> bool:
        """Check if contract is verified on explorer"""
        contract_data = await self._fetch_from_etherscan(address, chain)
        return bool(contract_data.get("source_code"))
    
    async def _analyze_proxy_pattern(
        self,
        address: str,
        contract_data: Dict,
        chain: str
    ) -> Tuple[bool, Optional[str]]:
        """Analyze if contract uses proxy pattern"""
        # Check for proxy signatures in bytecode
        bytecode = contract_data.get("bytecode", "")
        
        # EIP-1967 proxy slots
        proxy_patterns = [
            "360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc",  # Implementation
            "b53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103",  # Admin
            "a3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50"   # Beacon
        ]
        
        for pattern in proxy_patterns:
            if pattern in bytecode:
                # Get implementation address
                w3 = self.web3_connections.get(chain)
                if w3:
                    try:
                        impl_slot = "0x" + pattern
                        impl_address = w3.eth.get_storage_at(address, impl_slot)
                        if impl_address != b'\x00' * 32:
                            return True, Web3.to_checksum_address(impl_address[-20:])
                    except (NetworkError, ABIError, DecodeError, ValueError, Exception) as e:
                        logger.debug(f"Error reading proxy storage slot {impl_slot}: {e}")
                        pass
        
        # Check ABI for proxy functions
        abi = contract_data.get("abi", [])
        proxy_functions = {"implementation", "upgradeTo", "upgradeToAndCall"}
        functions = {item["name"] for item in abi if item.get("type") == "function"}
        
        if proxy_functions.intersection(functions):
            return True, None
        
        return False, None
    
    async def _get_owner(
        self,
        address: str,
        contract_data: Dict,
        chain: str
    ) -> Optional[str]:
        """Get contract owner if exists"""
        w3 = self.web3_connections.get(chain)
        if not w3:
            return None
        
        abi = contract_data.get("abi", [])
        
        # Common owner function names
        owner_functions = ["owner", "getOwner", "admin", "administrator"]
        
        for func_name in owner_functions:
            # Check if function exists in ABI
            for item in abi:
                if (item.get("type") == "function" and 
                    item.get("name") == func_name and
                    len(item.get("inputs", [])) == 0):
                    try:
                        contract = w3.eth.contract(address=address, abi=abi)
                        owner = getattr(contract.functions, func_name)().call()
                        if owner and owner != "0x0000000000000000000000000000000000000000":
                            return owner
                    except (NetworkError, ABIError, AttributeError, Exception) as e:
                        logger.debug(f"Error calling {func_name}() on contract: {e}")
                        pass
        
        return None
    
    async def _analyze_vulnerabilities(self, source_code: str) -> List[Vulnerability]:
        """Analyze source code for vulnerabilities"""
        vulnerabilities = []
        
        if not source_code:
            return vulnerabilities
        
        # Reentrancy vulnerability
        if re.search(r'\.call\{value:', source_code) and not re.search(r'nonReentrant', source_code):
            vulnerabilities.append(Vulnerability(
                name="Potential Reentrancy",
                level=VulnerabilityLevel.HIGH,
                description="External call without reentrancy guard",
                recommendation="Use OpenZeppelin's ReentrancyGuard"
            ))
        
        # Unchecked external calls
        if re.search(r'\.call\(|\.delegatecall\(|\.staticcall\(', source_code):
            if not re.search(r'require\(.*\.call|if\s*\(.*\.call', source_code):
                vulnerabilities.append(Vulnerability(
                    name="Unchecked External Call",
                    level=VulnerabilityLevel.MEDIUM,
                    description="External call return value not checked",
                    recommendation="Always check return values of external calls"
                ))
        
        # Integer overflow (for older Solidity versions)
        if re.search(r'pragma solidity \^0\.[4-7]\.', source_code):
            if not re.search(r'using SafeMath', source_code):
                vulnerabilities.append(Vulnerability(
                    name="Integer Overflow Risk",
                    level=VulnerabilityLevel.MEDIUM,
                    description="No SafeMath library for older Solidity version",
                    recommendation="Use SafeMath library or upgrade to Solidity 0.8+"
                ))
        
        # Tx.origin authentication
        if re.search(r'tx\.origin\s*==', source_code):
            vulnerabilities.append(Vulnerability(
                name="tx.origin Authentication",
                level=VulnerabilityLevel.HIGH,
                description="Using tx.origin for authentication",
                recommendation="Use msg.sender instead of tx.origin"
            ))
        
        # Block timestamp dependency
        if re.search(r'block\.timestamp|now', source_code):
            vulnerabilities.append(Vulnerability(
                name="Timestamp Dependency",
                level=VulnerabilityLevel.LOW,
                description="Contract depends on block timestamp",
                recommendation="Be aware that miners can manipulate timestamp"
            ))
        
        # Hardcoded addresses
        addresses = re.findall(r'0x[a-fA-F0-9]{40}', source_code)
        if len(addresses) > 3:
            vulnerabilities.append(Vulnerability(
                name="Hardcoded Addresses",
                level=VulnerabilityLevel.INFO,
                description=f"Found {len(addresses)} hardcoded addresses",
                recommendation="Consider using configurable addresses"
            ))
        
        # Selfdestruct
        if re.search(r'selfdestruct\(|suicide\(', source_code):
            vulnerabilities.append(Vulnerability(
                name="Selfdestruct Function",
                level=VulnerabilityLevel.HIGH,
                description="Contract can be destroyed",
                recommendation="Avoid selfdestruct unless absolutely necessary"
            ))
        
        # Floating pragma
        if re.search(r'pragma solidity \^', source_code):
            vulnerabilities.append(Vulnerability(
                name="Floating Pragma",
                level=VulnerabilityLevel.LOW,
                description="Compiler version not locked",
                recommendation="Lock pragma to specific version"
            ))
        
        # Missing events
        state_changes = re.findall(r'(\w+)\s*=\s*', source_code)
        events = re.findall(r'emit\s+(\w+)', source_code)
        if len(state_changes) > len(events) * 2:
            vulnerabilities.append(Vulnerability(
                name="Missing Events",
                level=VulnerabilityLevel.LOW,
                description="State changes without events",
                recommendation="Emit events for important state changes"
            ))
        
        # Centralization risk
        admin_functions = re.findall(r'onlyOwner|onlyAdmin|onlyGovernance', source_code)
        if len(admin_functions) > 10:
            vulnerabilities.append(Vulnerability(
                name="High Centralization",
                level=VulnerabilityLevel.MEDIUM,
                description=f"Found {len(admin_functions)} admin-only functions",
                recommendation="Consider decentralizing control"
            ))
        
        return vulnerabilities
    
    async def _analyze_permissions(
        self,
        contract_data: Dict,
        chain: str
    ) -> Dict[str, Any]:
        """Analyze contract permissions and access control"""
        permissions = {
            "owner_functions": [],
            "public_functions": [],
            "external_functions": [],
            "modifiers": [],
            "roles": []
        }
        
        abi = contract_data.get("abi", [])
        source = contract_data.get("source_code", "")
        
        for item in abi:
            if item.get("type") == "function":
                name = item.get("name")
                state_mutability = item.get("stateMutability", "nonpayable")
                
                # Check for owner functions
                if source and re.search(f"{name}.*onlyOwner|onlyAdmin", source):
                    permissions["owner_functions"].append(name)
                elif state_mutability in ["view", "pure"]:
                    permissions["public_functions"].append(name)
                else:
                    permissions["external_functions"].append(name)
        
        # Extract modifiers from source
        if source:
            modifiers = re.findall(r'modifier\s+(\w+)', source)
            permissions["modifiers"] = list(set(modifiers))
            
            # Check for role-based access
            if "AccessControl" in source or "hasRole" in source:
                roles = re.findall(r'bytes32.*constant\s+(\w+_ROLE)', source)
                permissions["roles"] = list(set(roles))
        
        return permissions
    
    async def _analyze_fees(
        self,
        contract_data: Dict,
        chain: str
    ) -> Dict[str, Decimal]:
        """Analyze contract fees and taxes"""
        fees = {}
        source = contract_data.get("source_code", "")
        
        if not source:
            return fees
        
        # Look for fee variables
        fee_patterns = [
            r'uint.*\s+(?:buy|sell|transfer|)(?:Fee|Tax)\s*=\s*(\d+)',
            r'uint.*\s+(?:liquidity|marketing|dev|team)Fee\s*=\s*(\d+)',
            r'uint.*\s+_?(?:fee|tax)(?:Buy|Sell|Transfer)?\s*=\s*(\d+)'
        ]
        
        for pattern in fee_patterns:
            matches = re.findall(pattern, source, re.IGNORECASE)
            for match in matches:
                try:
                    fee_value = Decimal(match) / Decimal(100)  # Assume percentage
                    if fee_value > 0:
                        # Extract fee name from pattern
                        fee_name = re.search(r'(\w+)(?:Fee|Tax)', pattern).group(1)
                        fees[fee_name.lower()] = fee_value
                except (NetworkError, ContractError, Exception) as e:
                    logger.debug(f"Extract fee value error: {e}")
                    pass
        
        # Check for fee limits
        max_fee_pattern = r'require\(.*fee.*<=?\s*(\d+)'
        max_fees = re.findall(max_fee_pattern, source, re.IGNORECASE)
        if max_fees:
            try:
                max_fee = max(Decimal(f) for f in max_fees) / Decimal(100)
                fees["max_fee"] = max_fee
            except (NetworkError, ABIError, DecodeError) as e:
                logger.warning(f"Check max fee error: {e}")
                pass
        
        return fees
    
    def _extract_functions(self, contract_data: Dict) -> Dict[str, Dict]:
        """Extract function information from contract"""
        functions = {}
        abi = contract_data.get("abi", [])
        
        for item in abi:
            if item.get("type") == "function":
                name = item.get("name")
                functions[name] = {
                    "inputs": item.get("inputs", []),
                    "outputs": item.get("outputs", []),
                    "stateMutability": item.get("stateMutability"),
                    "payable": item.get("payable", False)
                }
        
        return functions
    
    def _extract_modifiers(self, contract_data: Dict) -> List[str]:
        """Extract modifiers from contract source"""
        source = contract_data.get("source_code", "")
        if not source:
            return []
        
        modifiers = re.findall(r'modifier\s+(\w+)', source)
        return list(set(modifiers))
    
    def _extract_events(self, contract_data: Dict) -> List[str]:
        """Extract events from contract"""
        events = []
        abi = contract_data.get("abi", [])
        
        for item in abi:
            if item.get("type") == "event":
                events.append(item.get("name"))
        
        return events
    
    async def _analyze_storage(
        self,
        address: str,
        chain: str
    ) -> Dict[str, Any]:
        """Analyze contract storage layout"""
        storage = {}
        
        # This would require more complex analysis
        # For now, return basic structure
        storage["slots_used"] = 0
        storage["mappings"] = []
        storage["arrays"] = []
        
        return storage
    
    async def _check_upgrade_mechanism(
        self,
        contract_data: Dict,
        is_proxy: bool
    ) -> Optional[str]:
        """Check contract upgrade mechanism"""
        if not is_proxy:
            return None
        
        source = contract_data.get("source_code", "")
        abi = contract_data.get("abi", [])
        
        # Check for upgrade functions
        upgrade_functions = ["upgradeTo", "upgradeToAndCall", "changeImplementation"]
        for func in upgrade_functions:
            for item in abi:
                if item.get("type") == "function" and item.get("name") == func:
                    # Check if it has protection
                    if source and "onlyOwner" in source:
                        return f"upgradeable_with_owner"
                    elif source and "onlyRole" in source:
                        return f"upgradeable_with_roles"
                    else:
                        return f"upgradeable_unprotected"
        
        return "upgradeable" if is_proxy else None
    
    def _calculate_risk_score(
        self,
        vulnerabilities: List[Vulnerability],
        is_verified: bool,
        is_proxy: bool,
        owner: Optional[str],
        permissions: Dict,
        fees: Dict,
        upgrade_mechanism: Optional[str]
    ) -> float:
        """Calculate overall risk score (0-100)"""
        score = 0.0
        
        # Vulnerability scoring
        vuln_weights = {
            VulnerabilityLevel.CRITICAL: 20,
            VulnerabilityLevel.HIGH: 10,
            VulnerabilityLevel.MEDIUM: 5,
            VulnerabilityLevel.LOW: 2,
            VulnerabilityLevel.INFO: 0.5
        }
        
        for vuln in vulnerabilities:
            score += vuln_weights.get(vuln.level, 0) * vuln.confidence
        
        # Verification
        if not is_verified:
            score += 15
        
        # Proxy without protection
        if is_proxy and upgrade_mechanism == "upgradeable_unprotected":
            score += 20
        elif is_proxy:
            score += 5
        
        # Centralization
        if owner and owner != "0x0000000000000000000000000000000000000000":
            owner_func_count = len(permissions.get("owner_functions", []))
            if owner_func_count > 10:
                score += 10
            elif owner_func_count > 5:
                score += 5
        
        # High fees
        max_fee = fees.get("max_fee", Decimal(0))
        if max_fee > Decimal(25):
            score += 15
        elif max_fee > Decimal(10):
            score += 8
        elif max_fee > Decimal(5):
            score += 3
        
        # Cap at 100
        return min(score, 100.0)
    
    def _create_malicious_analysis(self, address: str, chain: str) -> ContractAnalysis:
        """Create analysis for known malicious contract"""
        return ContractAnalysis(
            address=address,
            chain=chain,
            contract_type=ContractType.UNKNOWN,
            is_verified=False,
            is_proxy=False,
            implementation_address=None,
            owner=None,
            vulnerabilities=[
                Vulnerability(
                    name="Known Malicious Contract",
                    level=VulnerabilityLevel.CRITICAL,
                    description="This contract is flagged as malicious",
                    recommendation="Do not interact with this contract"
                )
            ],
            permissions={},
            fees={},
            functions={},
            modifiers=[],
            events=[],
            storage_layout={},
            upgrade_mechanism=None,
            risk_score=100.0,
            metadata={"malicious": True}
        )
    
    async def _load_malicious_contracts(self) -> None:
        """Load list of known malicious contracts"""
        # This would load from a database or external service
        # For now, using a placeholder
        self.malicious_contracts = set()
    
    def _load_vulnerability_patterns(self) -> Dict[str, Dict]:
        """Load vulnerability detection patterns"""
        return {
            "reentrancy": {
                "pattern": r'\.call\{value:.*\}.*\n.*state\s*=',
                "level": VulnerabilityLevel.HIGH
            },
            "unchecked_call": {
                "pattern": r'\.call\([^\)]*\)[^;]*;(?!\s*require)',
                "level": VulnerabilityLevel.MEDIUM
            }
        }
    
    def _load_function_signatures(self) -> Dict[str, str]:
        """Load known function signatures"""
        return {
            "0xa9059cbb": "transfer(address,uint256)",
            "0x095ea7b3": "approve(address,uint256)",
            "0x70a08231": "balanceOf(address)",
            "0x18160ddd": "totalSupply()",
            "0x23b872dd": "transferFrom(address,address,uint256)"
        }
    
    def _log_analysis_summary(self, analysis: ContractAnalysis) -> None:
        """Log analysis summary"""
        vuln_count = len(analysis.vulnerabilities)
        critical_count = sum(1 for v in analysis.vulnerabilities 
                           if v.level == VulnerabilityLevel.CRITICAL)
        high_count = sum(1 for v in analysis.vulnerabilities 
                        if v.level == VulnerabilityLevel.HIGH)
        
        logger.info(
            f"Contract Analysis Complete: {analysis.address[:10]}... "
            f"[Risk: {analysis.risk_score:.1f}/100] "
            f"[Vulnerabilities: {vuln_count} "
            f"(Critical: {critical_count}, High: {high_count})] "
            f"[Type: {analysis.contract_type.value}] "
            f"[Verified: {analysis.is_verified}]"
        )
    
    async def batch_analyze(
        self,
        contracts: List[Dict[str, str]],
        deep_analysis: bool = False
    ) -> List[ContractAnalysis]:
        """
        Analyze multiple contracts in batch
        
        Args:
            contracts: List of dicts with 'address' and 'chain'
            deep_analysis: Whether to perform deep analysis
            
        Returns:
            List of contract analyses
        """
        tasks = []
        for contract in contracts:
            task = self.analyze_contract(
                contract["address"],
                contract["chain"],
                deep_analysis
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        analyses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to analyze {contracts[i]['address']}: {result}")
            else:
                analyses.append(result)
        
        return analyses
    
    async def monitor_contract(
        self,
        address: str,
        chain: str,
        interval: int = 3600,
        callback: Optional[callable] = None
    ) -> None:
        """
        Monitor contract for changes
        
        Args:
            address: Contract address
            chain: Blockchain network
            interval: Check interval in seconds
            callback: Function to call on changes
        """
        logger.info(f"Starting contract monitoring for {address} on {chain}")
        
        last_analysis = None
        
        while True:
            try:
                # Analyze contract
                analysis = await self.analyze_contract(address, chain)
                
                # Check for changes
                if last_analysis:
                    changes = self._detect_changes(last_analysis, analysis)
                    if changes:
                        logger.warning(f"Contract changes detected: {changes}")
                        if callback:
                            await callback(analysis, changes)
                
                last_analysis = analysis
                
                # Wait for next check
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Contract monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def _detect_changes(
        self,
        old_analysis: ContractAnalysis,
        new_analysis: ContractAnalysis
    ) -> Dict[str, Any]:
        """Detect changes between analyses"""
        changes = {}
        
        # Check owner change
        if old_analysis.owner != new_analysis.owner:
            changes["owner"] = {
                "old": old_analysis.owner,
                "new": new_analysis.owner
            }
        
        # Check new vulnerabilities
        old_vulns = {v.name for v in old_analysis.vulnerabilities}
        new_vulns = {v.name for v in new_analysis.vulnerabilities}
        
        if new_vulns - old_vulns:
            changes["new_vulnerabilities"] = list(new_vulns - old_vulns)
        
        # Check risk score change
        if abs(old_analysis.risk_score - new_analysis.risk_score) > 5:
            changes["risk_score"] = {
                "old": old_analysis.risk_score,
                "new": new_analysis.risk_score
            }
        
        # Check upgrade
        if old_analysis.implementation_address != new_analysis.implementation_address:
            changes["implementation"] = {
                "old": old_analysis.implementation_address,
                "new": new_analysis.implementation_address
            }
        
        return changes
    
    async def verify_token_safety(
        self,
        token_address: str,
        chain: str
    ) -> Dict[str, Any]:
        """
        Comprehensive token safety verification
        
        Args:
            token_address: Token contract address
            chain: Blockchain network
            
        Returns:
            Safety assessment with recommendations
        """
        # Analyze contract
        analysis = await self.analyze_contract(token_address, chain, deep_analysis=True)
        
        # Safety checks
        safety_checks = {
            "is_verified": analysis.is_verified,
            "has_critical_vulnerabilities": any(
                v.level == VulnerabilityLevel.CRITICAL 
                for v in analysis.vulnerabilities
            ),
            "has_high_vulnerabilities": any(
                v.level == VulnerabilityLevel.HIGH 
                for v in analysis.vulnerabilities
            ),
            "is_upgradeable": analysis.upgrade_mechanism is not None,
            "has_owner": analysis.owner is not None,
            "high_centralization": len(analysis.permissions.get("owner_functions", [])) > 10,
            "excessive_fees": any(
                fee > Decimal(10) for fee in analysis.fees.values()
            ),
            "is_token": analysis.contract_type == ContractType.TOKEN
        }
        
        # Calculate safety score
        safety_score = 100
        
        if not safety_checks["is_verified"]:
            safety_score -= 20
        if safety_checks["has_critical_vulnerabilities"]:
            safety_score -= 40
        if safety_checks["has_high_vulnerabilities"]:
            safety_score -= 20
        if safety_checks["is_upgradeable"]:
            safety_score -= 10
        if safety_checks["high_centralization"]:
            safety_score -= 15
        if safety_checks["excessive_fees"]:
            safety_score -= 25
        if not safety_checks["is_token"]:
            safety_score -= 30
        
        safety_score = max(0, safety_score)
        
        # Generate recommendation
        if safety_score >= 80:
            recommendation = "SAFE"
            message = "Token appears safe for trading"
        elif safety_score >= 60:
            recommendation = "CAUTION"
            message = "Token has some risks, trade with caution"
        elif safety_score >= 40:
            recommendation = "HIGH_RISK"
            message = "Token has significant risks"
        else:
            recommendation = "AVOID"
            message = "Token is highly risky, avoid trading"
        
        return {
            "token_address": token_address,
            "chain": chain,
            "safety_score": safety_score,
            "recommendation": recommendation,
            "message": message,
            "safety_checks": safety_checks,
            "risk_score": analysis.risk_score,
            "vulnerabilities": [
                {
                    "name": v.name,
                    "level": v.level.value,
                    "description": v.description
                }
                for v in analysis.vulnerabilities
            ],
            "analysis": analysis
        }
    
    async def get_contract_creation_info(
        self,
        address: str,
        chain: str
    ) -> Dict[str, Any]:
        """Get contract creation information"""
        info = {}
        
        w3 = self.web3_connections.get(chain)
        if not w3:
            return info
        
        # Get contract creation transaction
        # This would require querying the explorer API
        # For now, return placeholder
        info["creation_tx"] = None
        info["creator"] = None
        info["creation_block"] = None
        info["creation_timestamp"] = None
        
        return info
    
    async def check_renounced_ownership(
        self,
        address: str,
        chain: str
    ) -> bool:
        """Check if contract ownership is renounced"""
        analysis = await self.analyze_contract(address, chain, deep_analysis=False)
        
        return (
            analysis.owner == "0x0000000000000000000000000000000000000000" or
            analysis.owner == "0x000000000000000000000000000000000000dEaD"
        )
    
    async def estimate_gas_usage(
        self,
        address: str,
        chain: str,
        function_name: str,
        params: List[Any]
    ) -> Optional[int]:
        """Estimate gas usage for contract function"""
        w3 = self.web3_connections.get(chain)
        if not w3:
            return None
        
        try:
            # Get contract ABI
            contract_data = await self._fetch_contract_data(address, chain)
            abi = contract_data.get("abi", [])
            
            if not abi:
                return None
            
            # Create contract instance
            contract = w3.eth.contract(address=address, abi=abi)
            
            # Get function
            func = getattr(contract.functions, function_name)
            
            # Estimate gas
            gas_estimate = func(*params).estimate_gas()
            
            return gas_estimate
            
        except Exception as e:
            logger.error(f"Gas estimation failed: {e}")
            return None
    
    def get_risk_summary(self, analysis: ContractAnalysis) -> str:
        """Generate human-readable risk summary"""
        risk_level = "LOW"
        if analysis.risk_score >= 70:
            risk_level = "CRITICAL"
        elif analysis.risk_score >= 50:
            risk_level = "HIGH"
        elif analysis.risk_score >= 30:
            risk_level = "MEDIUM"
        
        summary_parts = [
            f"Risk Level: {risk_level} ({analysis.risk_score:.1f}/100)"
        ]
        
        if not analysis.is_verified:
            summary_parts.append("⚠️ Unverified contract")
        
        if analysis.is_proxy:
            summary_parts.append("🔄 Upgradeable proxy contract")
        
        critical_vulns = sum(1 for v in analysis.vulnerabilities 
                           if v.level == VulnerabilityLevel.CRITICAL)
        if critical_vulns > 0:
            summary_parts.append(f"🚨 {critical_vulns} critical vulnerabilities")
        
        high_fees = any(fee > Decimal(10) for fee in analysis.fees.values())
        if high_fees:
            summary_parts.append("💸 High transaction fees detected")
        
        if analysis.owner:
            summary_parts.append(f"👤 Has owner: {analysis.owner[:10]}...")
        
        return " | ".join(summary_parts)