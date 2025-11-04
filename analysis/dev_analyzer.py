"""
Developer Analyzer - Team reputation and project analysis for ClaudeDex Trading Bot

This module analyzes developer teams and their history to assess project risk.
"""
from __future__ import annotations
from typing import Set

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import json

import aiohttp
from web3 import Web3
from eth_utils import is_address, to_checksum_address
from loguru import logger

from utils.helpers import retry_async, measure_time
from utils.constants import BLOCK_EXPLORERS


class DeveloperRisk(Enum):
    """Developer risk levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNKNOWN = "unknown"


class ProjectStatus(Enum):
    """Project status types"""
    ACTIVE = "active"
    ABANDONED = "abandoned"
    RUGGED = "rugged"
    SUCCESSFUL = "successful"
    UNKNOWN = "unknown"


@dataclass
class DeveloperProfile:
    """Developer profile information"""
    address: str
    total_projects: int
    successful_projects: int
    failed_projects: int
    rugged_projects: int
    total_raised: Decimal
    total_liquidity_provided: Decimal
    average_project_lifespan: timedelta
    social_presence: Dict[str, Any]
    reputation_score: float
    risk_level: DeveloperRisk
    known_aliases: List[str]
    associated_wallets: List[str]
    metadata: Dict[str, Any]
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ProjectAnalysis:
    """Project analysis result"""
    project_address: str
    developer_address: str
    chain: str
    launch_date: datetime
    status: ProjectStatus
    liquidity_locked: bool
    liquidity_amount: Decimal
    holders_count: int
    transaction_count: int
    code_quality_score: float
    community_score: float
    transparency_score: float
    overall_score: float
    red_flags: List[str]
    green_flags: List[str]
    similar_projects: List[str]
    metadata: Dict[str, Any]


class DeveloperAnalyzer:
    """Analyzes developer teams and their project history"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.web3_connections: Dict[str, Web3] = {}
        self.etherscan_apis: Dict[str, str] = {}
        self.known_developers: Dict[str, DeveloperProfile] = {}
        self.known_ruggers: Set[str] = set()
        self.trusted_developers: Set[str] = set()
        self.project_cache: Dict[str, ProjectAnalysis] = {}
        self.cache_ttl = config.get("cache_ttl", 3600)
        
    async def initialize(self) -> None:
        """Initialize analyzer components"""
        logger.info("Initializing Developer Analyzer...")
        
        # Setup Web3 connections
        if hasattr(self.config, 'get_rpc_urls'):
            for chain_name in self.config.get('chains', {}).get('enabled', []):
                rpc_urls = self.config.get_rpc_urls(chain_name)
                if rpc_urls:
                    for rpc_url in rpc_urls:
                        try:
                            w3 = Web3(Web3.HTTPProvider(rpc_url))
                            if w3.is_connected():
                                self.web3_connections[chain_name] = w3
                                logger.info(f"DeveloperAnalyzer connected to {chain_name}: {rpc_url[:50]}...")
                                break
                        except Exception as e:
                            logger.warning(f"DeveloperAnalyzer failed to connect to {rpc_url} for {chain_name}: {e}")

        # Load API keys
        self.etherscan_apis = self.config.get("etherscan_apis", {})
        
        # Load known developers database
        await self._load_developer_database()
        
        logger.info("Developer Analyzer initialized")
    
    @retry_async(max_retries=3, delay=1.0)
    @measure_time
    async def analyze_developer(
        self,
        address: str,
        chain: str
    ) -> DeveloperProfile:
        """
        Analyze a developer's history and reputation
        
        Args:
            address: Developer wallet address
            chain: Blockchain network
            
        Returns:
            Developer profile with risk assessment
        """
        try:
            # Validate address
            if not is_address(address):
                raise ValueError(f"Invalid address: {address}")
            
            address = to_checksum_address(address)
            
            # Check cache
            if address in self.known_developers:
                profile = self.known_developers[address]
                if (datetime.now() - profile.last_updated).seconds < self.cache_ttl:
                    return profile
            
            # Check if known rugger
            if address.lower() in self.known_ruggers:
                return self._create_rugger_profile(address)
            
            # Check if trusted developer
            if address.lower() in self.trusted_developers:
                return self._create_trusted_profile(address)
            
            # Get developer history
            history = await self._get_developer_history(address, chain)
            
            # Analyze previous projects
            projects = await self._analyze_previous_projects(history)
            
            # Check social presence
            social = await self._check_social_presence(address)
            
            # Get associated wallets
            associated = await self._find_associated_wallets(address, chain)
            
            # Calculate statistics
            total_projects = len(projects)
            successful = sum(1 for p in projects.values() 
                           if p.get("status") == ProjectStatus.SUCCESSFUL)
            failed = sum(1 for p in projects.values() 
                       if p.get("status") == ProjectStatus.ABANDONED)
            rugged = sum(1 for p in projects.values() 
                       if p.get("status") == ProjectStatus.RUGGED)
            
            total_raised = sum(p.get("raised", 0) for p in projects.values())
            total_liquidity = sum(p.get("liquidity", 0) for p in projects.values())
            
            # Calculate average project lifespan
            lifespans = []
            for project in projects.values():
                if project.get("launch_date") and project.get("end_date"):
                    lifespan = project["end_date"] - project["launch_date"]
                    lifespans.append(lifespan)
            
            avg_lifespan = (sum(lifespans, timedelta()) / len(lifespans) 
                          if lifespans else timedelta(days=0))
            
            # Calculate reputation score
            reputation = self._calculate_reputation(projects, social, history)
            
            # Determine risk level
            risk_level = self._determine_risk_level(reputation, projects)
            
            # Find known aliases
            aliases = await self._find_aliases(address, associated)
            
            # Create profile
            profile = DeveloperProfile(
                address=address,
                total_projects=total_projects,
                successful_projects=successful,
                failed_projects=failed,
                rugged_projects=rugged,
                total_raised=Decimal(str(total_raised)),
                total_liquidity_provided=Decimal(str(total_liquidity)),
                average_project_lifespan=avg_lifespan,
                social_presence=social,
                reputation_score=reputation,
                risk_level=risk_level,
                known_aliases=aliases,
                associated_wallets=associated,
                metadata={
                    "chain": chain,
                    "projects": projects,
                    "history": history
                }
            )
            
            # Cache profile
            self.known_developers[address] = profile
            
            # Log summary
            self._log_developer_summary(profile)
            
            return profile
            
        except Exception as e:
            logger.error(f"Developer analysis failed for {address}: {e}")
            raise
    
    @retry_async(max_retries=3, delay=1.0)
    @measure_time
    async def analyze_project(
        self,
        project_address: str,
        chain: str
    ) -> ProjectAnalysis:
        """
        Analyze a specific project
        
        Args:
            project_address: Project contract address
            chain: Blockchain network
            
        Returns:
            Project analysis with risk assessment
        """
        try:
            # Check cache
            cache_key = f"{chain}:{project_address}"
            if cache_key in self.project_cache:
                cached = self.project_cache[cache_key]
                if (datetime.now() - cached.metadata.get("timestamp", datetime.min)).seconds < self.cache_ttl:
                    return cached
            
            # Get project data
            project_data = await self._get_project_data(project_address, chain)
            
            # Get developer address
            developer = await self._get_project_developer(project_address, chain)
            
            # Check liquidity status
            liquidity = await self._check_liquidity_status(project_address, chain)
            
            # Get holder statistics
            holders = await self._get_holder_statistics(project_address, chain)
            
            # Analyze code quality
            code_quality = await self._analyze_code_quality(project_address, chain)
            
            # Check community metrics
            community = await self._analyze_community_metrics(project_address)
            
            # Calculate transparency score
            transparency = self._calculate_transparency_score(project_data, liquidity)
            
            # Identify red flags
            red_flags = self._identify_red_flags(
                project_data, liquidity, holders, developer
            )
            
            # Identify green flags
            green_flags = self._identify_green_flags(
                project_data, liquidity, holders, code_quality
            )
            
            # Find similar projects
            similar = await self._find_similar_projects(project_address, chain)
            
            # Calculate overall score
            overall_score = self._calculate_project_score(
                code_quality, community, transparency, 
                len(red_flags), len(green_flags)
            )
            
            # Determine status
            status = self._determine_project_status(
                project_data, liquidity, holders
            )
            
            # Create analysis
            analysis = ProjectAnalysis(
                project_address=project_address,
                developer_address=developer,
                chain=chain,
                launch_date=project_data.get("launch_date", datetime.now()),
                status=status,
                liquidity_locked=liquidity.get("is_locked", False),
                liquidity_amount=Decimal(str(liquidity.get("amount", 0))),
                holders_count=holders.get("count", 0),
                transaction_count=project_data.get("tx_count", 0),
                code_quality_score=code_quality,
                community_score=community.get("score", 0),
                transparency_score=transparency,
                overall_score=overall_score,
                red_flags=red_flags,
                green_flags=green_flags,
                similar_projects=similar,
                metadata={
                    "timestamp": datetime.now(),
                    "project_data": project_data,
                    "liquidity": liquidity,
                    "holders": holders
                }
            )
            
            # Cache result
            self.project_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Project analysis failed for {project_address}: {e}")
            raise
    
    async def _get_developer_history(
        self,
        address: str,
        chain: str
    ) -> Dict[str, Any]:
        """Get developer's transaction and contract deployment history"""
        history = {
            "deployments": [],
            "transactions": [],
            "first_activity": None,
            "last_activity": None,
            "total_gas_spent": 0
        }
        
        # Get transaction history from explorer
        if chain in self.etherscan_apis:
            api_key = self.etherscan_apis[chain]
            base_url = BLOCK_EXPLORERS.get(chain, {}).get("api_url")
            
            if base_url:
                async with aiohttp.ClientSession() as session:
                    # Get normal transactions
                    params = {
                        "module": "account",
                        "action": "txlist",
                        "address": address,
                        "startblock": 0,
                        "endblock": 99999999,
                        "sort": "asc",
                        "apikey": api_key
                    }
                    
                    async with session.get(base_url, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data["status"] == "1":
                                transactions = data["result"]
                                history["transactions"] = transactions
                                
                                # Find contract deployments
                                for tx in transactions:
                                    if tx.get("to") == "":
                                        history["deployments"].append({
                                            "contract": tx.get("contractAddress"),
                                            "block": int(tx.get("blockNumber", 0)),
                                            "timestamp": int(tx.get("timeStamp", 0)),
                                            "gas_used": int(tx.get("gasUsed", 0))
                                        })
                                
                                # Calculate statistics
                                if transactions:
                                    history["first_activity"] = datetime.fromtimestamp(
                                        int(transactions[0].get("timeStamp", 0))
                                    )
                                    history["last_activity"] = datetime.fromtimestamp(
                                        int(transactions[-1].get("timeStamp", 0))
                                    )
                                    history["total_gas_spent"] = sum(
                                        int(tx.get("gasUsed", 0)) * int(tx.get("gasPrice", 0))
                                        for tx in transactions
                                    )
        
        return history
    
    async def _analyze_previous_projects(
        self,
        history: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze developer's previous projects"""
        projects = {}
        
        for deployment in history.get("deployments", []):
            contract_address = deployment.get("contract")
            if contract_address:
                # Analyze each deployed contract
                try:
                    # Get contract data
                    project_data = {
                        "address": contract_address,
                        "launch_date": datetime.fromtimestamp(deployment.get("timestamp", 0)),
                        "status": ProjectStatus.UNKNOWN,
                        "raised": 0,
                        "liquidity": 0
                    }
                    
                    # Determine project status based on activity
                    # This would require more complex analysis
                    # For now, using placeholder logic
                    current_time = datetime.now()
                    launch_time = project_data["launch_date"]
                    age = current_time - launch_time
                    
                    if age.days < 30:
                        project_data["status"] = ProjectStatus.ACTIVE
                    elif age.days < 180:
                        project_data["status"] = ProjectStatus.SUCCESSFUL
                    else:
                        project_data["status"] = ProjectStatus.ABANDONED
                    
                    projects[contract_address] = project_data
                    
                except Exception as e:
                    logger.debug(f"Failed to analyze project {contract_address}: {e}")
        
        return projects
    
    async def _check_social_presence(self, address: str) -> Dict[str, Any]:
        """Check developer's social media presence"""
        social = {
            "twitter": None,
            "telegram": None,
            "discord": None,
            "github": None,
            "website": None,
            "verified": False,
            "followers": 0,
            "reputation": 0
        }
        
        # This would integrate with social media APIs
        # For now, returning placeholder data
        # In production, would check:
        # - Twitter API for verified accounts
        # - GitHub for code contributions
        # - Telegram/Discord for community management
        
        return social
    
    async def _find_associated_wallets(
        self,
        address: str,
        chain: str
    ) -> List[str]:
        """Find wallets associated with the developer"""
        associated = []
        
        # Analyze transaction patterns to find related wallets
        # This would look for:
        # - Wallets that frequently interact
        # - Wallets that receive funds from same sources
        # - Wallets with similar transaction patterns
        
        w3 = self.web3_connections.get(chain)
        if w3:
            # Get recent transactions
            # Analyze patterns
            # For now, returning empty list
            pass
        
        return associated
    
    async def _find_aliases(
        self,
        address: str,
        associated_wallets: List[str]
    ) -> List[str]:
        """Find known aliases for the developer"""
        aliases = []
        
        # Check ENS names
        # Check social media handles
        # Check known developer databases
        
        return aliases
    
    def _calculate_reputation(
        self,
        projects: Dict[str, Any],
        social: Dict[str, Any],
        history: Dict[str, Any]
    ) -> float:
        """Calculate developer reputation score (0-100)"""
        score = 50.0  # Start neutral
        
        # Project history scoring
        total_projects = len(projects)
        if total_projects > 0:
            successful = sum(1 for p in projects.values() 
                           if p.get("status") == ProjectStatus.SUCCESSFUL)
            rugged = sum(1 for p in projects.values() 
                       if p.get("status") == ProjectStatus.RUGGED)
            
            success_rate = successful / total_projects
            rug_rate = rugged / total_projects
            
            score += success_rate * 30  # Up to +30 for successes
            score -= rug_rate * 50  # Up to -50 for rugs
        
        # Social presence scoring
        if social.get("verified"):
            score += 10
        
        if social.get("followers", 0) > 10000:
            score += 5
        elif social.get("followers", 0) > 1000:
            score += 2
        
        # Activity history scoring
        if history.get("first_activity"):
            account_age = datetime.now() - history["first_activity"]
            if account_age.days > 365:
                score += 5  # Established account
            elif account_age.days < 30:
                score -= 10  # New account
        
        # Cap between 0 and 100
        return max(0, min(100, score))
    
    def _determine_risk_level(
        self,
        reputation: float,
        projects: Dict[str, Any]
    ) -> DeveloperRisk:
        """Determine developer risk level based on reputation and history"""
        # Check for immediate red flags
        rugged = sum(1 for p in projects.values() 
                    if p.get("status") == ProjectStatus.RUGGED)
        
        if rugged > 0:
            return DeveloperRisk.VERY_HIGH
        
        # Reputation-based risk
        if reputation >= 80:
            return DeveloperRisk.VERY_LOW
        elif reputation >= 60:
            return DeveloperRisk.LOW
        elif reputation >= 40:
            return DeveloperRisk.MEDIUM
        elif reputation >= 20:
            return DeveloperRisk.HIGH
        else:
            return DeveloperRisk.VERY_HIGH
    
    async def _analyze_code_quality(
        self,
        project_address: str,
        chain: str
    ) -> float:
        """Analyze smart contract code quality"""
        score = 50.0  # Base score
        
        # Check if contract is verified
        # Analyze code complexity
        # Check for best practices
        # Look for known vulnerabilities
        
        # This would integrate with contract analyzer
        # For now, returning base score
        
        return score
    
    async def _get_project_data(
        self,
        project_address: str,
        chain: str
    ) -> Dict[str, Any]:
        """Get project data from blockchain"""
        data = {
            "launch_date": datetime.now(),
            "tx_count": 0,
            "unique_users": 0,
            "total_volume": 0
        }
        
        # Get transaction count and other metrics
        # This would query the blockchain
        # For now, returning placeholder data
        
        return data
    
    async def _get_project_developer(
        self,
        project_address: str,
        chain: str
    ) -> str:
        """Get project developer/deployer address"""
        # Get contract creation transaction
        # Return deployer address
        # For now, returning placeholder
        return "0x0000000000000000000000000000000000000000"
    
    async def _check_liquidity_status(
        self,
        project_address: str,
        chain: str
    ) -> Dict[str, Any]:
        """Check project liquidity status"""
        liquidity = {
            "is_locked": False,
            "lock_duration": 0,
            "amount": 0,
            "dex": None,
            "pair_address": None
        }
        
        # Check DEX pairs for liquidity
        # Check if liquidity is locked
        # Calculate total liquidity value
        
        return liquidity
    
    async def _get_holder_statistics(
        self,
        project_address: str,
        chain: str
    ) -> Dict[str, Any]:
        """Get token holder statistics"""
        holders = {
            "count": 0,
            "top_10_percentage": 0,
            "whale_count": 0,
            "average_balance": 0
        }
        
        # Query blockchain for holder data
        # Calculate distribution metrics
        
        return holders
    
    async def _analyze_community_metrics(
        self,
        project_address: str
    ) -> Dict[str, Any]:
        """Analyze project community metrics"""
        community = {
            "score": 50.0,
            "telegram_members": 0,
            "discord_members": 0,
            "twitter_followers": 0,
            "activity_level": "medium"
        }
        
        # Check social media metrics
        # Analyze community engagement
        
        return community
    
    def _calculate_transparency_score(
        self,
        project_data: Dict[str, Any],
        liquidity: Dict[str, Any]
    ) -> float:
        """Calculate project transparency score"""
        score = 0.0
        
        # Verified contract: +20
        # Locked liquidity: +30
        # Active communication: +20
        # Clear roadmap: +15
        # Team doxxed: +15
        
        if liquidity.get("is_locked"):
            score += 30
        
        # Add other transparency factors
        
        return min(100, score)
    
    def _identify_red_flags(
        self,
        project_data: Dict[str, Any],
        liquidity: Dict[str, Any],
        holders: Dict[str, Any],
        developer: str
    ) -> List[str]:
        """Identify project red flags"""
        red_flags = []
        
        # Check for common red flags
        if not liquidity.get("is_locked"):
            red_flags.append("Liquidity not locked")
        
        if holders.get("top_10_percentage", 0) > 50:
            red_flags.append("High concentration in top holders")
        
        if developer in self.known_ruggers:
            red_flags.append("Developer has history of rugs")
        
        if holders.get("count", 0) < 50:
            red_flags.append("Very few holders")
        
        if liquidity.get("amount", 0) < 10000:  # $10k minimum
            red_flags.append("Low liquidity")
        
        return red_flags
    
    def _identify_green_flags(
        self,
        project_data: Dict[str, Any],
        liquidity: Dict[str, Any],
        holders: Dict[str, Any],
        code_quality: float
    ) -> List[str]:
        """Identify project green flags"""
        green_flags = []
        
        # Check for positive indicators
        if liquidity.get("is_locked") and liquidity.get("lock_duration", 0) > 365:
            green_flags.append("Liquidity locked for >1 year")
        
        if holders.get("count", 0) > 1000:
            green_flags.append("Large holder base")
        
        if holders.get("top_10_percentage", 0) < 20:
            green_flags.append("Well distributed tokens")
        
        if code_quality > 80:
            green_flags.append("High quality code")
        
        if liquidity.get("amount", 0) > 100000:  # $100k+
            green_flags.append("Strong liquidity")
        
        return green_flags
    
    async def _find_similar_projects(
        self,
        project_address: str,
        chain: str
    ) -> List[str]:
        """Find similar projects by the same developer"""
        similar = []
        
        # Get developer address
        # Find other projects by same developer
        # Check for code similarity
        
        return similar
    
    def _calculate_project_score(
        self,
        code_quality: float,
        community_score: float,
        transparency: float,
        red_flags_count: int,
        green_flags_count: int
    ) -> float:
        """Calculate overall project score"""
        # Weighted average of factors
        score = (
            code_quality * 0.25 +
            community_score * 0.20 +
            transparency * 0.25 +
            (green_flags_count * 5) -  # Bonus for green flags
            (red_flags_count * 10)  # Penalty for red flags
        ) * 0.3
        
        # Ensure score is between 0 and 100
        return max(0, min(100, score))
    
    def _determine_project_status(
        self,
        project_data: Dict[str, Any],
        liquidity: Dict[str, Any],
        holders: Dict[str, Any]
    ) -> ProjectStatus:
        """Determine current project status"""
        # Check various factors to determine status
        
        if holders.get("count", 0) == 0:
            return ProjectStatus.ABANDONED
        
        if liquidity.get("amount", 0) == 0:
            return ProjectStatus.RUGGED
        
        # Check last activity
        # For now, default to active
        return ProjectStatus.ACTIVE
    
    async def _load_developer_database(self) -> None:
        """Load known developer database"""
        # Load from database or external API
        # Known ruggers list
        self.known_ruggers = set()
        
        # Trusted developers list
        self.trusted_developers = set()
        
        # This would be loaded from a real database
        # For now, using empty sets
    
    def _create_rugger_profile(self, address: str) -> DeveloperProfile:
        """Create profile for known rugger"""
        return DeveloperProfile(
            address=address,
            total_projects=0,
            successful_projects=0,
            failed_projects=0,
            rugged_projects=1,  # At least one known rug
            total_raised=Decimal("0"),
            total_liquidity_provided=Decimal("0"),
            average_project_lifespan=timedelta(days=0),
            social_presence={},
            reputation_score=0.0,
            risk_level=DeveloperRisk.VERY_HIGH,
            known_aliases=[],
            associated_wallets=[],
            metadata={"known_rugger": True}
        )
    
    def _create_trusted_profile(self, address: str) -> DeveloperProfile:
        """Create profile for trusted developer"""
        return DeveloperProfile(
            address=address,
            total_projects=0,
            successful_projects=0,
            failed_projects=0,
            rugged_projects=0,
            total_raised=Decimal("0"),
            total_liquidity_provided=Decimal("0"),
            average_project_lifespan=timedelta(days=0),
            social_presence={},
            reputation_score=100.0,
            risk_level=DeveloperRisk.VERY_LOW,
            known_aliases=[],
            associated_wallets=[],
            metadata={"trusted": True}
        )
    
    def _log_developer_summary(self, profile: DeveloperProfile) -> None:
        """Log developer analysis summary"""
        logger.info(
            f"Developer Analysis: {profile.address[:10]}... "
            f"[Risk: {profile.risk_level.value}] "
            f"[Reputation: {profile.reputation_score:.1f}/100] "
            f"[Projects: {profile.total_projects} "
            f"(Success: {profile.successful_projects}, "
            f"Failed: {profile.failed_projects}, "
            f"Rugged: {profile.rugged_projects})]"
        )
    
    async def check_developer_reputation(
        self,
        address: str,
        chain: str
    ) -> Dict[str, Any]:
        """
        Quick reputation check for a developer
        
        Args:
            address: Developer address
            chain: Blockchain network
            
        Returns:
            Reputation assessment
        """
        profile = await self.analyze_developer(address, chain)
        
        return {
            "address": address,
            "reputation_score": profile.reputation_score,
            "risk_level": profile.risk_level.value,
            "total_projects": profile.total_projects,
            "successful_projects": profile.successful_projects,
            "rugged_projects": profile.rugged_projects,
            "recommendation": self._get_recommendation(profile)
        }
    
    def _get_recommendation(self, profile: DeveloperProfile) -> str:
        """Get trading recommendation based on developer profile"""
        if profile.risk_level == DeveloperRisk.VERY_HIGH:
            return "AVOID - Very high risk developer"
        elif profile.risk_level == DeveloperRisk.HIGH:
            return "EXTREME CAUTION - High risk developer"
        elif profile.risk_level == DeveloperRisk.MEDIUM:
            return "CAUTION - Moderate risk, careful analysis needed"
        elif profile.risk_level == DeveloperRisk.LOW:
            return "ACCEPTABLE - Low risk developer"
        elif profile.risk_level == DeveloperRisk.VERY_LOW:
            return "TRUSTED - Very low risk developer"
        else:
            return "UNKNOWN - Insufficient data"
    
    async def monitor_developer(
        self,
        address: str,
        chain: str,
        interval: int = 3600,
        callback: Optional[callable] = None
    ) -> None:
        """
        Monitor a developer for new activities
        
        Args:
            address: Developer address to monitor
            chain: Blockchain network
            interval: Check interval in seconds
            callback: Function to call on new activity
        """
        logger.info(f"Starting developer monitoring for {address}")
        
        last_profile = None
        
        while True:
            try:
                # Analyze developer
                profile = await self.analyze_developer(address, chain)
                
                # Check for changes
                if last_profile:
                    # Check for new projects
                    if profile.total_projects > last_profile.total_projects:
                        logger.warning(f"Developer {address} launched new project")
                        if callback:
                            await callback(profile, "new_project")
                    
                    # Check for rugs
                    if profile.rugged_projects > last_profile.rugged_projects:
                        logger.critical(f"Developer {address} rugged a project!")
                        if callback:
                            await callback(profile, "rug_detected")
                
                last_profile = profile
                
                # Wait for next check
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Developer monitoring error: {e}")
                await asyncio.sleep(interval)
    
    async def batch_analyze_developers(
        self,
        developers: List[Dict[str, str]]
    ) -> List[DeveloperProfile]:
        """
        Analyze multiple developers in batch
        
        Args:
            developers: List of dicts with 'address' and 'chain'
            
        Returns:
            List of developer profiles
        """
        tasks = []
        for dev in developers:
            task = self.analyze_developer(dev["address"], dev["chain"])
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        profiles = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to analyze {developers[i]['address']}: {result}")
            else:
                profiles.append(result)
        
        return profiles
    
    def get_risk_summary(self, profile: DeveloperProfile) -> str:
        """Generate human-readable risk summary"""
        summary_parts = [
            f"Developer: {profile.address[:10]}...",
            f"Risk: {profile.risk_level.value.upper()}",
            f"Reputation: {profile.reputation_score:.0f}/100"
        ]
        
        if profile.rugged_projects > 0:
            summary_parts.append(f"⚠️ {profile.rugged_projects} RUGGED PROJECTS")
        
        if profile.successful_projects > 0:
            summary_parts.append(f"✅ {profile.successful_projects} successful projects")
        
        if profile.total_projects > 5:
            summary_parts.append(f"📊 {profile.total_projects} total projects")
        
        if profile.social_presence.get("verified"):
            summary_parts.append("✓ Verified social presence")
        
        return " | ".join(summary_parts)