"""
Token Sniffer Integration - Additional token verification for ClaudeDex Trading Bot
Provides multi-source token analysis and verification
"""

import asyncio
import aiohttp
import hashlib
from typing import Dict, List, Optional, Set, Any, Tuple
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import re
from loguru import logger

from utils.helpers import retry_async, measure_time
from utils.constants import CHAIN_NAMES, HONEYPOT_THRESHOLDS


class TokenRisk(Enum):
    """Token risk levels"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    SCAM = "scam"


class TokenFlag(Enum):
    """Token warning flags"""
    HONEYPOT = "honeypot"
    RUG_PULL = "rug_pull"
    MINT_FUNCTION = "mint_function"
    HIDDEN_OWNER = "hidden_owner"
    UNLOCKED_LIQUIDITY = "unlocked_liquidity"
    HIGH_TAX = "high_tax"
    BLACKLIST_FUNCTION = "blacklist_function"
    PROXY_CONTRACT = "proxy_contract"
    NO_AUDIT = "no_audit"
    LOW_HOLDERS = "low_holders"
    CONCENTRATED_HOLDINGS = "concentrated_holdings"
    SUSPICIOUS_TRANSFERS = "suspicious_transfers"


@dataclass
class TokenAnalysis:
    """Complete token analysis result"""
    token_address: str
    chain: str
    name: str
    symbol: str
    risk_level: TokenRisk
    risk_score: float  # 0-100
    flags: List[TokenFlag]
    honeypot_result: Dict[str, Any]
    contract_analysis: Dict[str, Any]
    holder_analysis: Dict[str, Any]
    liquidity_analysis: Dict[str, Any]
    audit_results: Dict[str, Any]
    similar_scams: List[Dict[str, str]]
    recommendations: List[str]
    confidence: float
    sources_checked: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TokenMetrics:
    """Token metrics and statistics"""
    total_supply: Decimal
    circulating_supply: Decimal
    holder_count: int
    top_10_holdings_percent: float
    liquidity_usd: Decimal
    liquidity_locked_percent: float
    buy_tax: float
    sell_tax: float
    transfer_tax: float
    contract_verified: bool
    renounced_ownership: bool
    has_mint_function: bool
    has_pause_function: bool
    has_blacklist: bool
    creation_date: datetime


class TokenSniffer:
    """
    Multi-source token verification and analysis
    Integrates with TokenSniffer API, GoPlus, Honeypot.is, and custom analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # API endpoints
        self.token_sniffer_api = "https://tokensniffer.com/api/v2"
        self.goplus_api = "https://api.gopluslabs.io/api/v1"
        self.honeypot_api = "https://api.honeypot.is/v2"
        self.dextools_api = "https://api.dextools.io/v1"
        
        # API keys
        self.token_sniffer_key = config.get("token_sniffer_api_key")
        self.goplus_key = config.get("goplus_api_key")
        self.dextools_key = config.get("dextools_api_key")
        
        # Risk thresholds
        self.high_tax_threshold = config.get("high_tax_threshold", 10)  # 10%
        self.min_holders = config.get("min_holders", 50)
        self.max_concentration = config.get("max_concentration", 50)  # 50% in top 10
        self.min_liquidity = config.get("min_liquidity", 10000)  # $10k
        
        # Caches
        self.analysis_cache: Dict[str, TokenAnalysis] = {}
        self.blacklist: Set[str] = set()
        self.whitelist: Set[str] = set()
        self.known_scams: Dict[str, Dict] = {}
        
        # Session
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info("TokenSniffer initialized")
    
    async def initialize(self) -> None:
        """Initialize TokenSniffer components"""
        self.session = aiohttp.ClientSession()
        await self._load_known_scams()
        await self._load_lists()
        logger.info("TokenSniffer ready")
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    @retry_async(max_retries=3, delay=1.0)
    @measure_time
    async def analyze_token(
        self,
        token_address: str,
        chain: str,
        deep_scan: bool = True
    ) -> TokenAnalysis:
        """
        Perform comprehensive token analysis using multiple sources
        
        Args:
            token_address: Token contract address
            chain: Blockchain network
            deep_scan: Whether to perform deep analysis
            
        Returns:
            Complete token analysis with risk assessment
        """
        try:
            # Check cache
            cache_key = f"{chain}:{token_address}"
            if cache_key in self.analysis_cache:
                cached = self.analysis_cache[cache_key]
                if (datetime.now() - cached.timestamp).seconds < 600:  # 10 min cache
                    return cached
            
            # Check blacklist/whitelist
            if token_address.lower() in self.blacklist:
                return self._create_blacklisted_result(token_address, chain)
            
            if token_address.lower() in self.whitelist:
                logger.info(f"Token {token_address} is whitelisted")
            
            # Gather data from multiple sources
            sources_checked = []
            
            # 1. TokenSniffer API
            token_sniffer_data = await self._check_token_sniffer(token_address, chain)
            if token_sniffer_data:
                sources_checked.append("TokenSniffer")
            
            # 2. GoPlus Security API
            goplus_data = await self._check_goplus(token_address, chain)
            if goplus_data:
                sources_checked.append("GoPlus")
            
            # 3. Honeypot.is
            honeypot_data = await self._check_honeypot(token_address, chain)
            if honeypot_data:
                sources_checked.append("Honeypot.is")
            
            # 4. DexTools data
            dextools_data = await self._check_dextools(token_address, chain)
            if dextools_data:
                sources_checked.append("DexTools")
            
            # 5. On-chain analysis
            onchain_data = await self._analyze_onchain(token_address, chain)
            if onchain_data:
                sources_checked.append("OnChain")
            
            # Combine all data
            combined_data = self._combine_analysis_data(
                token_sniffer_data,
                goplus_data,
                honeypot_data,
                dextools_data,
                onchain_data
            )
            
            # Calculate risk score and level
            risk_score, risk_level, flags = self._calculate_risk(combined_data)
            
            # Check for similar scams
            similar_scams = await self._find_similar_scams(
                combined_data.get("name", ""),
                combined_data.get("symbol", "")
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(risk_level, flags)
            
            # Build analysis result
            analysis = TokenAnalysis(
                token_address=token_address,
                chain=chain,
                name=combined_data.get("name", "Unknown"),
                symbol=combined_data.get("symbol", "???"),
                risk_level=risk_level,
                risk_score=risk_score,
                flags=flags,
                honeypot_result=honeypot_data or {},
                contract_analysis=combined_data.get("contract", {}),
                holder_analysis=combined_data.get("holders", {}),
                liquidity_analysis=combined_data.get("liquidity", {}),
                audit_results=combined_data.get("audit", {}),
                similar_scams=similar_scams,
                recommendations=recommendations,
                confidence=self._calculate_confidence(sources_checked),
                sources_checked=sources_checked
            )
            
            # Update cache
            self.analysis_cache[cache_key] = analysis
            
            # Update blacklist if critical
            if risk_level in [TokenRisk.CRITICAL, TokenRisk.SCAM]:
                self.blacklist.add(token_address.lower())
                logger.warning(f"Token {token_address} added to blacklist: {risk_level}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Token analysis failed for {token_address}: {e}")
            return self._create_error_result(token_address, chain, str(e))
    
    async def _check_token_sniffer(self, token_address: str, chain: str) -> Optional[Dict]:
        """Check TokenSniffer API"""
        if not self.token_sniffer_key:
            return None
        
        try:
            url = f"{self.token_sniffer_api}/tokens/{chain}/{token_address}"
            headers = {"Authorization": f"Bearer {self.token_sniffer_key}"}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_token_sniffer_response(data)
                    
        except Exception as e:
            logger.error(f"TokenSniffer API error: {e}")
        
        return None
    
    async def _check_goplus(self, token_address: str, chain: str) -> Optional[Dict]:
        """Check GoPlus Security API"""
        try:
            chain_id = self._get_chain_id(chain)
            url = f"{self.goplus_api}/token_security/{chain_id}"
            params = {"contract_addresses": token_address}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_goplus_response(data, token_address)
                    
        except Exception as e:
            logger.error(f"GoPlus API error: {e}")
        
        return None
    
    async def _check_honeypot(self, token_address: str, chain: str) -> Optional[Dict]:
        """Check Honeypot.is API"""
        try:
            url = f"{self.honeypot_api}/token/{chain}/{token_address}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_honeypot_response(data)
                    
        except Exception as e:
            logger.error(f"Honeypot API error: {e}")
        
        return None
    
    async def _check_dextools(self, token_address: str, chain: str) -> Optional[Dict]:
        """Check DexTools API"""
        if not self.dextools_key:
            return None
        
        try:
            chain_name = self._get_dextools_chain(chain)
            url = f"{self.dextools_api}/token/{chain_name}/{token_address}"
            headers = {"X-API-KEY": self.dextools_key}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_dextools_response(data)
                    
        except Exception as e:
            logger.error(f"DexTools API error: {e}")
        
        return None
    
    async def _analyze_onchain(self, token_address: str, chain: str) -> Optional[Dict]:
        """Perform on-chain analysis"""
        # This would integrate with web3 to analyze the contract directly
        # Placeholder for now
        return {
            "verified": False,
            "has_mint": False,
            "has_pause": False,
            "has_blacklist": False,
            "ownership_renounced": False
        }
    
    def _parse_token_sniffer_response(self, data: Dict) -> Dict:
        """Parse TokenSniffer API response"""
        result = data.get("result", {})
        
        return {
            "score": result.get("score", 0),
            "risks": result.get("risks", []),
            "warnings": result.get("warnings", []),
            "tests": result.get("tests", {}),
            "similar_tokens": result.get("similar_tokens", [])
        }
    
    def _parse_goplus_response(self, data: Dict, token_address: str) -> Dict:
        """Parse GoPlus API response"""
        result = data.get("result", {}).get(token_address.lower(), {})
        
        return {
            "is_honeypot": result.get("is_honeypot") == "1",
            "buy_tax": float(result.get("buy_tax", 0)),
            "sell_tax": float(result.get("sell_tax", 0)),
            "is_mintable": result.get("is_mintable") == "1",
            "can_take_back_ownership": result.get("can_take_back_ownership") == "1",
            "is_blacklisted": result.get("is_blacklisted") == "1",
            "is_proxy": result.get("is_proxy") == "1",
            "is_open_source": result.get("is_open_source") == "1",
            "holder_count": int(result.get("holder_count", 0)),
            "total_supply": result.get("total_supply", "0")
        }
    
    def _parse_honeypot_response(self, data: Dict) -> Dict:
        """Parse Honeypot.is API response"""
        return {
            "is_honeypot": data.get("honeypot", False),
            "buy_tax": data.get("buy_tax", 0),
            "sell_tax": data.get("sell_tax", 0),
            "transfer_tax": data.get("transfer_tax", 0),
            "liquidity": data.get("liquidity", 0),
            "liquidity_locked": data.get("liquidity_locked", False),
            "ownership_renounced": data.get("ownership_renounced", False),
            "simulation_success": data.get("simulation_success", False),
            "simulation_error": data.get("simulation_error", "")
        }
    
    def _parse_dextools_response(self, data: Dict) -> Dict:
        """Parse DexTools API response"""
        token_data = data.get("data", {})
        
        return {
            "score": token_data.get("score", 0),
            "votes": token_data.get("votes", 0),
            "creation_time": token_data.get("creationTime", ""),
            "holder_count": token_data.get("holders", 0),
            "liquidity": token_data.get("liquidity", 0),
            "audit": token_data.get("audit", {})
        }
    
    def _combine_analysis_data(self, *data_sources) -> Dict:
        """Combine data from multiple sources"""
        combined = {
            "name": "",
            "symbol": "",
            "contract": {},
            "holders": {},
            "liquidity": {},
            "audit": {},
            "taxes": {},
            "risks": []
        }
        
        for source in data_sources:
            if not source:
                continue
            
            # Merge contract data
            if "is_honeypot" in source:
                combined["contract"]["is_honeypot"] = source["is_honeypot"]
            if "is_mintable" in source:
                combined["contract"]["is_mintable"] = source["is_mintable"]
            if "is_proxy" in source:
                combined["contract"]["is_proxy"] = source["is_proxy"]
            if "ownership_renounced" in source:
                combined["contract"]["ownership_renounced"] = source["ownership_renounced"]
            
            # Merge holder data
            if "holder_count" in source:
                combined["holders"]["count"] = max(
                    combined["holders"].get("count", 0),
                    source["holder_count"]
                )
            
            # Merge liquidity data
            if "liquidity" in source:
                combined["liquidity"]["amount"] = source["liquidity"]
            if "liquidity_locked" in source:
                combined["liquidity"]["locked"] = source["liquidity_locked"]
            
            # Merge tax data
            if "buy_tax" in source:
                combined["taxes"]["buy"] = source["buy_tax"]
            if "sell_tax" in source:
                combined["taxes"]["sell"] = source["sell_tax"]
            
            # Merge risks
            if "risks" in source:
                combined["risks"].extend(source["risks"])
        
        return combined
    
    def _calculate_risk(self, data: Dict) -> Tuple[float, TokenRisk, List[TokenFlag]]:
        """Calculate risk score and determine risk level"""
        score = 0
        flags = []
        
        # Check honeypot
        if data.get("contract", {}).get("is_honeypot"):
            score += 40
            flags.append(TokenFlag.HONEYPOT)
        
        # Check taxes
        taxes = data.get("taxes", {})
        if taxes.get("buy", 0) > self.high_tax_threshold:
            score += 20
            flags.append(TokenFlag.HIGH_TAX)
        if taxes.get("sell", 0) > self.high_tax_threshold:
            score += 20
            flags.append(TokenFlag.HIGH_TAX)
        
        # Check mintable
        if data.get("contract", {}).get("is_mintable"):
            score += 15
            flags.append(TokenFlag.MINT_FUNCTION)
        
        # Check ownership
        if not data.get("contract", {}).get("ownership_renounced"):
            score += 10
            flags.append(TokenFlag.HIDDEN_OWNER)
        
        # Check liquidity
        liquidity = data.get("liquidity", {})
        if not liquidity.get("locked"):
            score += 15
            flags.append(TokenFlag.UNLOCKED_LIQUIDITY)
        
        # Check holders
        holder_count = data.get("holders", {}).get("count", 0)
        if holder_count < self.min_holders:
            score += 10
            flags.append(TokenFlag.LOW_HOLDERS)
        
        # Check proxy
        if data.get("contract", {}).get("is_proxy"):
            score += 10
            flags.append(TokenFlag.PROXY_CONTRACT)
        
        # Determine risk level
        if score >= 80:
            risk_level = TokenRisk.SCAM
        elif score >= 60:
            risk_level = TokenRisk.CRITICAL
        elif score >= 40:
            risk_level = TokenRisk.HIGH
        elif score >= 25:
            risk_level = TokenRisk.MEDIUM
        elif score >= 10:
            risk_level = TokenRisk.LOW
        else:
            risk_level = TokenRisk.SAFE
        
        return score, risk_level, flags
    
    async def _find_similar_scams(self, name: str, symbol: str) -> List[Dict]:
        """Find similar known scams"""
        similar = []
        
        if not name and not symbol:
            return similar
        
        for scam_address, scam_data in self.known_scams.items():
            similarity = 0
            
            # Check name similarity
            if name and scam_data.get("name"):
                if name.lower() == scam_data["name"].lower():
                    similarity += 50
                elif name.lower() in scam_data["name"].lower() or scam_data["name"].lower() in name.lower():
                    similarity += 30
            
            # Check symbol similarity
            if symbol and scam_data.get("symbol"):
                if symbol.lower() == scam_data["symbol"].lower():
                    similarity += 50
                elif symbol.lower() in scam_data["symbol"].lower():
                    similarity += 20
            
            if similarity >= 50:
                similar.append({
                    "address": scam_address,
                    "name": scam_data.get("name", ""),
                    "symbol": scam_data.get("symbol", ""),
                    "similarity": similarity,
                    "report_date": scam_data.get("report_date", "")
                })
        
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)[:5]
    
    def _generate_recommendations(self, risk_level: TokenRisk, flags: List[TokenFlag]) -> List[str]:
        """Generate trading recommendations based on risk assessment"""
        recommendations = []
        
        if risk_level == TokenRisk.SCAM:
            recommendations.append("DO NOT TRADE - Token identified as scam")
            recommendations.append("Report to authorities if you've been affected")
            
        elif risk_level == TokenRisk.CRITICAL:
            recommendations.append("AVOID - Extremely high risk detected")
            recommendations.append("Multiple critical security issues found")
            
        elif risk_level == TokenRisk.HIGH:
            recommendations.append("HIGH RISK - Not recommended for trading")
            if TokenFlag.HONEYPOT in flags:
                recommendations.append("Honeypot detected - You may not be able to sell")
            if TokenFlag.MINT_FUNCTION in flags:
                recommendations.append("Mint function present - Supply can be inflated")
            
        elif risk_level == TokenRisk.MEDIUM:
            recommendations.append("CAUTION - Moderate risk detected")
            recommendations.append("Consider small position size if trading")
            if TokenFlag.HIGH_TAX in flags:
                recommendations.append("High taxes detected - Check buy/sell fees")
            
        elif risk_level == TokenRisk.LOW:
            recommendations.append("Low risk - Standard precautions apply")
            recommendations.append("Monitor liquidity and holder distribution")
            
        else:  # SAFE
            recommendations.append("Token appears safe for trading")
            recommendations.append("Always DYOR and manage risk appropriately")
        
        return recommendations
    
    def _calculate_confidence(self, sources_checked: List[str]) -> float:
        """Calculate confidence score based on sources checked"""
        max_sources = 5
        return len(sources_checked) / max_sources
    
    def _get_chain_id(self, chain: str) -> int:
        """Get chain ID for GoPlus API"""
        chain_ids = {
            "ethereum": 1,
            "bsc": 56,
            "polygon": 137,
            "arbitrum": 42161,
            "base": 8453
        }
        return chain_ids.get(chain, 1)
    
    def _get_dextools_chain(self, chain: str) -> str:
        """Get chain name for DexTools API"""
        chain_names = {
            "ethereum": "ether",
            "bsc": "bsc",
            "polygon": "polygon",
            "arbitrum": "arbitrum",
            "base": "base"
        }
        return chain_names.get(chain, "ether")
    
    async def _load_known_scams(self) -> None:
        """Load database of known scams"""
        # This would load from a database or file
        # Placeholder for now
        pass
    
    async def _load_lists(self) -> None:
        """Load blacklist and whitelist"""
        # This would load from configuration or database
        # Placeholder for now
        pass
    
    def _create_blacklisted_result(self, token_address: str, chain: str) -> TokenAnalysis:
        """Create result for blacklisted token"""
        return TokenAnalysis(
            token_address=token_address,
            chain=chain,
            name="BLACKLISTED",
            symbol="SCAM",
            risk_level=TokenRisk.SCAM,
            risk_score=100,
            flags=[TokenFlag.HONEYPOT, TokenFlag.RUG_PULL],
            honeypot_result={"blacklisted": True},
            contract_analysis={},
            holder_analysis={},
            liquidity_analysis={},
            audit_results={},
            similar_scams=[],
            recommendations=["DO NOT TRADE - Token is blacklisted"],
            confidence=1.0,
            sources_checked=["Blacklist"]
        )
    
    def _create_error_result(self, token_address: str, chain: str, error: str) -> TokenAnalysis:
        """Create result when analysis fails"""
        return TokenAnalysis(
            token_address=token_address,
            chain=chain,
            name="Unknown",
            symbol="???",
            risk_level=TokenRisk.HIGH,
            risk_score=75,
            flags=[],
            honeypot_result={},
            contract_analysis={},
            holder_analysis={},
            liquidity_analysis={},
            audit_results={},
            similar_scams=[],
            recommendations=[f"Analysis failed: {error}", "Treat as high risk"],
            confidence=0.0,
            sources_checked=[]
        )
    
    async def batch_analyze(
        self,
        tokens: List[Dict[str, str]],
        deep_scan: bool = False
        ) -> List[TokenAnalysis]:
        """Analyze multiple tokens in parallel"""
        tasks = []
        for token in tokens:
            task = self.analyze_token(
                token["address"],
                token["chain"],
                deep_scan
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        analyses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to analyze token {tokens[i]}: {result}")
                analyses.append(
                    self._create_error_result(
                        tokens[i]["address"],
                        tokens[i]["chain"],
                        str(result)
                    )
                )
            else:
                analyses.append(result)
        
        return analyses
    
    # PATCH for token_sniffer.py - Fix the monitor_token method and complete the file

    async def monitor_token(
        self,
        token_address: str,
        chain: str,
        interval: int = 300,
        callback: Optional[Any] = None
        ) -> None:
        """
        Monitor token continuously for risk changes
        
        Args:
            token_address: Token contract address
            chain: Blockchain network
            interval: Check interval in seconds (default 5 minutes)
            callback: Optional async callback function for updates
        """
        previous_risk_level = None
        
        while True:
            try:
                analysis = await self.analyze_token(token_address, chain)
                
                # Check for risk level changes
                if previous_risk_level and analysis.risk_level != previous_risk_level:
                    logger.info(
                        f"Risk level changed for {token_address}: "
                        f"{previous_risk_level.value} -> {analysis.risk_level.value}"
                    )
                
                # Execute callback if provided
                if callback:
                    await callback(analysis)
                
                # Alert on high risk
                if analysis.risk_level in [TokenRisk.HIGH, TokenRisk.CRITICAL, TokenRisk.SCAM]:
                    logger.warning(
                        f"High risk detected for {token_address}: "
                        f"{analysis.risk_level.value} (score: {analysis.risk_score})"
                    )
                    
                    # Add to blacklist if critical
                    if analysis.risk_level == TokenRisk.SCAM:
                        self.blacklist.add(token_address.lower())
                
                previous_risk_level = analysis.risk_level
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                logger.info(f"Monitoring cancelled for {token_address}")
                break
            except Exception as e:
                logger.error(f"Token monitoring error for {token_address}: {e}")
                await asyncio.sleep(60)  # Short retry delay on error

    def get_risk_summary(self, analysis: TokenAnalysis) -> str:
        """
        Get human-readable risk summary
        
        Args:
            analysis: Token analysis result
            
        Returns:
            Formatted risk summary string
        """
        summary = f"Token: {analysis.symbol} ({analysis.token_address[:8]}...)\n"
        summary += f"Risk Level: {analysis.risk_level.value.upper()}\n"
        summary += f"Risk Score: {analysis.risk_score:.1f}/100\n"
        
        if analysis.flags:
            summary += f"Flags: {', '.join([f.value for f in analysis.flags])}\n"
        
        if analysis.recommendations:
            summary += "Recommendations:\n"
            for rec in analysis.recommendations[:3]:
                summary += f"  - {rec}\n"
        
        return summary

    async def export_analysis(
        self,
        analysis: TokenAnalysis,
        format: str = "json"
    ) -> str:
        """
        Export analysis in specified format
        
        Args:
            analysis: Token analysis to export
            format: Export format (json, csv, text)
            
        Returns:
            Formatted analysis string
        """
        if format == "json":
            return json.dumps({
                "token_address": analysis.token_address,
                "chain": analysis.chain,
                "name": analysis.name,
                "symbol": analysis.symbol,
                "risk_level": analysis.risk_level.value,
                "risk_score": analysis.risk_score,
                "flags": [f.value for f in analysis.flags],
                "recommendations": analysis.recommendations,
                "confidence": analysis.confidence,
                "timestamp": analysis.timestamp.isoformat()
            }, indent=2)
        
        elif format == "csv":
            return (
                f"{analysis.token_address},{analysis.chain},{analysis.symbol},"
                f"{analysis.risk_level.value},{analysis.risk_score:.2f},"
                f"{len(analysis.flags)},{analysis.confidence:.2f},"
                f"{analysis.timestamp.isoformat()}"
            )
        
        else:  # text format
            return self.get_risk_summary(analysis)

    def update_blacklist(self, token_address: str, reason: str = "") -> None:
        """
        Add token to blacklist
        
        Args:
            token_address: Token contract address
            reason: Reason for blacklisting
        """
        self.blacklist.add(token_address.lower())
        logger.warning(f"Token {token_address} blacklisted: {reason}")

    def update_whitelist(self, token_address: str) -> None:
        """
        Add token to whitelist
        
        Args:
            token_address: Token contract address
        """
        self.whitelist.add(token_address.lower())
        # Remove from blacklist if present
        self.blacklist.discard(token_address.lower())
        logger.info(f"Token {token_address} whitelisted")

    def clear_cache(self) -> None:
        """Clear analysis cache"""
        self.analysis_cache.clear()
        logger.info("Analysis cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cached_tokens": len(self.analysis_cache),
            "blacklisted_tokens": len(self.blacklist),
            "whitelisted_tokens": len(self.whitelist),
            "known_scams": len(self.known_scams)
        }