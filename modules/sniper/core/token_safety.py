"""
Token Safety Checker for Sniper Module
Performs comprehensive safety analysis before sniping a token.

Checks performed:
- Honeypot detection (can you sell?)
- Tax analysis (buy/sell tax percentages)
- Liquidity verification (locked? amount?)
- Contract analysis (verified? renounced?)
- Holder analysis (distribution, whales)
"""

import asyncio
import logging
import aiohttp
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("TokenSafety")


class SafetyRating(Enum):
    """Safety rating levels"""
    SAFE = "safe"           # Low risk, good to snipe
    CAUTION = "caution"     # Medium risk, proceed carefully
    DANGER = "danger"       # High risk, likely scam
    HONEYPOT = "honeypot"   # Cannot sell - avoid completely


@dataclass
class SafetyReport:
    """Token safety analysis report"""
    token_address: str
    chain: str
    rating: SafetyRating
    score: int  # 0-100, higher is safer

    # Detailed checks
    is_honeypot: bool
    buy_tax: float
    sell_tax: float
    is_contract_verified: bool
    is_ownership_renounced: bool
    liquidity_usd: float
    liquidity_locked: bool
    holder_count: int
    top_holder_percentage: float

    # Warnings
    warnings: list

    def is_safe_to_snipe(self, max_tax: float = 15.0, min_liquidity: float = 1000.0) -> bool:
        """Check if token passes minimum safety requirements"""
        if self.is_honeypot:
            return False
        if self.sell_tax > max_tax:
            return False
        if self.liquidity_usd < min_liquidity:
            return False
        if self.top_holder_percentage > 50:
            return False
        return self.rating in [SafetyRating.SAFE, SafetyRating.CAUTION]


class TokenSafetyChecker:
    """
    Comprehensive token safety checker.
    Uses multiple APIs:
    - GoPlus Security API (free, rate limited)
    - Honeypot.is API (EVM only)
    - RugCheck.xyz (Solana only)
    """

    # API endpoints
    GOPLUS_API = "https://api.gopluslabs.io/api/v1"
    HONEYPOT_API = "https://api.honeypot.is/v2"
    RUGCHECK_API = "https://api.rugcheck.xyz/v1"

    # Chain IDs for GoPlus
    CHAIN_IDS = {
        'ethereum': '1',
        'bsc': '56',
        'polygon': '137',
        'arbitrum': '42161',
        'base': '8453',
        'avalanche': '43114',
        'fantom': '250',
        'optimism': '10'
    }

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.session: Optional[aiohttp.ClientSession] = None

        # Cache to avoid redundant checks
        self._cache: Dict[str, SafetyReport] = {}
        self._cache_ttl = 300  # 5 minutes

    async def initialize(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info("🛡️ Token Safety Checker initialized")

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def check_token(self, token_address: str, chain: str) -> SafetyReport:
        """
        Main entry point - check token safety.
        Returns a SafetyReport with comprehensive analysis.
        """
        cache_key = f"{chain}:{token_address.lower()}"

        # Check cache
        if cache_key in self._cache:
            logger.debug(f"Using cached safety report for {token_address}")
            return self._cache[cache_key]

        logger.info(f"🔍 Checking token safety: {token_address} on {chain}")

        if chain == 'solana':
            report = await self._check_solana_token(token_address)
        else:
            report = await self._check_evm_token(token_address, chain)

        # Cache result
        self._cache[cache_key] = report

        # Log result
        self._log_report(report)

        return report

    async def _check_evm_token(self, token_address: str, chain: str) -> SafetyReport:
        """Check EVM token using GoPlus and Honeypot.is APIs"""
        warnings = []

        # Default values
        is_honeypot = False
        buy_tax = 0.0
        sell_tax = 0.0
        is_verified = False
        is_renounced = False
        liquidity_usd = 0.0
        liquidity_locked = False
        holder_count = 0
        top_holder_pct = 100.0

        # 1. GoPlus Security Check
        goplus_data = await self._call_goplus(token_address, chain)
        if goplus_data:
            # CRITICAL: use (x or {}) to handle explicit null values from API
            token_info = goplus_data.get(token_address.lower()) or {}
            if not isinstance(token_info, dict):
                token_info = {}

            # Honeypot check
            if token_info.get('is_honeypot') == '1':
                is_honeypot = True
                warnings.append("🍯 HONEYPOT: Cannot sell this token")

            # Tax checks - handle None/null values
            try:
                buy_tax = float(token_info.get('buy_tax') or 0) * 100
                sell_tax = float(token_info.get('sell_tax') or 0) * 100
            except (TypeError, ValueError):
                buy_tax = 0.0
                sell_tax = 0.0

            if buy_tax > 10:
                warnings.append(f"⚠️ High buy tax: {buy_tax:.1f}%")
            if sell_tax > 10:
                warnings.append(f"⚠️ High sell tax: {sell_tax:.1f}%")

            # Contract checks
            is_verified = token_info.get('is_open_source') == '1'
            is_renounced = token_info.get('owner_address') in ['0x0000000000000000000000000000000000000000', None, '']

            if not is_verified:
                warnings.append("⚠️ Contract not verified")

            if token_info.get('can_take_back_ownership') == '1':
                warnings.append("🚨 Owner can reclaim ownership")

            if token_info.get('owner_change_balance') == '1':
                warnings.append("🚨 Owner can modify balances")

            if token_info.get('hidden_owner') == '1':
                warnings.append("🚨 Hidden owner detected")

            if token_info.get('selfdestruct') == '1':
                warnings.append("🚨 Contract has selfdestruct")

            if token_info.get('external_call') == '1':
                warnings.append("⚠️ External calls detected")

            if token_info.get('is_anti_whale') == '1':
                warnings.append("ℹ️ Anti-whale mechanism")

            # Holder analysis - handle None/null values
            try:
                holder_count = int(token_info.get('holder_count') or 0)
            except (TypeError, ValueError):
                holder_count = 0
            if holder_count < 50:
                warnings.append(f"⚠️ Low holder count: {holder_count}")

            holders = token_info.get('holders') or []
            if holders and isinstance(holders, list) and len(holders) > 0:
                first_holder = holders[0]
                if isinstance(first_holder, dict):
                    try:
                        top_holder_pct = float(first_holder.get('percent') or 100) * 100
                    except (TypeError, ValueError):
                        top_holder_pct = 100.0
                    if top_holder_pct > 30:
                        warnings.append(f"⚠️ Top holder owns {top_holder_pct:.1f}%")

            # Liquidity
            lp_holders = token_info.get('lp_holders') or []
            if isinstance(lp_holders, list):
                for lp in lp_holders:
                    if isinstance(lp, dict) and lp.get('is_locked') == 1:
                        liquidity_locked = True
                        break

            try:
                total_supply = float(token_info.get('total_supply') or 0)
            except (TypeError, ValueError):
                total_supply = 0.0

        # 2. Honeypot.is additional check (more accurate for honeypot detection)
        honeypot_data = await self._call_honeypot_is(token_address, chain)
        if honeypot_data and isinstance(honeypot_data, dict):
            # CRITICAL: use (x or {}) to handle explicit null values
            hp_result = honeypot_data.get('honeypotResult') or {}
            if isinstance(hp_result, dict) and hp_result.get('isHoneypot'):
                is_honeypot = True
                warnings.append(f"🍯 HONEYPOT: {hp_result.get('honeypotReason') or 'Unknown reason'}")

            simulation = honeypot_data.get('simulationResult') or {}
            if isinstance(simulation, dict):
                try:
                    sim_buy = simulation.get('buyTax')
                    sim_sell = simulation.get('sellTax')
                    if sim_buy is not None:
                        buy_tax = float(sim_buy)
                    if sim_sell is not None:
                        sell_tax = float(sim_sell)
                except (TypeError, ValueError):
                    pass

            pair = honeypot_data.get('pair') or {}
            if isinstance(pair, dict):
                try:
                    liquidity_usd = float(pair.get('liquidity') or 0)
                except (TypeError, ValueError):
                    liquidity_usd = 0.0

        # Calculate safety score
        score = self._calculate_score(
            is_honeypot=is_honeypot,
            buy_tax=buy_tax,
            sell_tax=sell_tax,
            is_verified=is_verified,
            is_renounced=is_renounced,
            liquidity_usd=liquidity_usd,
            liquidity_locked=liquidity_locked,
            holder_count=holder_count,
            top_holder_pct=top_holder_pct,
            warning_count=len(warnings)
        )

        # Determine rating
        rating = self._get_rating(score, is_honeypot)

        return SafetyReport(
            token_address=token_address,
            chain=chain,
            rating=rating,
            score=score,
            is_honeypot=is_honeypot,
            buy_tax=buy_tax,
            sell_tax=sell_tax,
            is_contract_verified=is_verified,
            is_ownership_renounced=is_renounced,
            liquidity_usd=liquidity_usd,
            liquidity_locked=liquidity_locked,
            holder_count=holder_count,
            top_holder_percentage=top_holder_pct,
            warnings=warnings
        )

    async def _check_solana_token(self, token_address: str) -> SafetyReport:
        """Check Solana token using RugCheck.xyz API"""
        warnings = []

        # Default values
        is_honeypot = False
        buy_tax = 0.0
        sell_tax = 0.0
        is_verified = False
        is_renounced = False
        liquidity_usd = 0.0
        liquidity_locked = False
        holder_count = 0
        top_holder_pct = 100.0

        # Call RugCheck API
        rugcheck_data = await self._call_rugcheck(token_address)
        if rugcheck_data:
            # Risk score (0-100, lower is safer in RugCheck)
            risk_score = rugcheck_data.get('score', 100) or 100

            # Risks array - use (x or []) pattern to handle explicit null
            risks = rugcheck_data.get('risks') or []
            if isinstance(risks, list):
                for risk in risks:
                    if not isinstance(risk, dict):
                        continue
                    risk_name = risk.get('name', '') or ''
                    risk_level = risk.get('level', '') or ''
                    risk_desc = risk.get('description', '') or ''

                    if risk_level in ['critical', 'high']:
                        warnings.append(f"🚨 {risk_name}: {risk_desc}")
                    elif risk_level == 'medium':
                        warnings.append(f"⚠️ {risk_name}: {risk_desc}")

                    # Check for specific risks
                    if 'freeze' in risk_name.lower():
                        is_honeypot = True
                        warnings.append("🍯 Token can be frozen (honeypot risk)")

            # Token info - CRITICAL: use (x or {}) to handle explicit null
            token_info = rugcheck_data.get('token') or {}
            if isinstance(token_info, dict):
                holder_count = token_info.get('holderCount', 0) or 0

            # Top holder percentage - CRITICAL: handle null values
            top_holders = rugcheck_data.get('topHolders') or []
            if top_holders and isinstance(top_holders, list) and len(top_holders) > 0:
                first_holder = top_holders[0]
                if isinstance(first_holder, dict):
                    top_holder_pct = first_holder.get('percentage', 100) or 100
                    if top_holder_pct > 30:
                        warnings.append(f"⚠️ Top holder owns {top_holder_pct:.1f}%")

            # Liquidity - CRITICAL: use (x or {}) to handle explicit null
            liquidity = rugcheck_data.get('liquidity') or {}
            if isinstance(liquidity, dict):
                liquidity_usd = liquidity.get('usd', 0) or 0
                liquidity_locked = liquidity.get('locked', False) or False

            # Mint authority (renounced = safer)
            mint_authority = rugcheck_data.get('mintAuthority')
            is_renounced = mint_authority is None
            if not is_renounced:
                warnings.append("⚠️ Mint authority not renounced")

            # Freeze authority
            freeze_authority = rugcheck_data.get('freezeAuthority')
            if freeze_authority:
                warnings.append("🚨 Freeze authority exists - can freeze wallets")
                is_honeypot = True
        else:
            # Fallback: Try Helius API for basic token info
            helius_data = await self._call_helius_token_info(token_address)
            if helius_data and isinstance(helius_data, dict):
                # Basic info from Helius
                holder_count = helius_data.get('holderCount', 0) or 0

        # Calculate score (invert for Solana since RugCheck gives risk, not safety)
        score = self._calculate_score(
            is_honeypot=is_honeypot,
            buy_tax=buy_tax,
            sell_tax=sell_tax,
            is_verified=True,  # Solana programs are always "open"
            is_renounced=is_renounced,
            liquidity_usd=liquidity_usd,
            liquidity_locked=liquidity_locked,
            holder_count=holder_count,
            top_holder_pct=top_holder_pct,
            warning_count=len(warnings)
        )

        rating = self._get_rating(score, is_honeypot)

        return SafetyReport(
            token_address=token_address,
            chain='solana',
            rating=rating,
            score=score,
            is_honeypot=is_honeypot,
            buy_tax=buy_tax,
            sell_tax=sell_tax,
            is_contract_verified=True,
            is_ownership_renounced=is_renounced,
            liquidity_usd=liquidity_usd,
            liquidity_locked=liquidity_locked,
            holder_count=holder_count,
            top_holder_percentage=top_holder_pct,
            warnings=warnings
        )

    async def _call_goplus(self, token_address: str, chain: str) -> Optional[Dict]:
        """Call GoPlus Security API"""
        chain_id = self.CHAIN_IDS.get(chain, '1')
        url = f"{self.GOPLUS_API}/token_security/{chain_id}?contract_addresses={token_address}"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('code') == 1:
                        return data.get('result', {})
        except Exception as e:
            logger.debug(f"GoPlus API error: {e}")
        return None

    async def _call_honeypot_is(self, token_address: str, chain: str) -> Optional[Dict]:
        """Call Honeypot.is API"""
        chain_id = self.CHAIN_IDS.get(chain, '1')
        url = f"{self.HONEYPOT_API}/IsHoneypot?address={token_address}&chainId={chain_id}"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.debug(f"Honeypot.is API error: {e}")
        return None

    async def _call_rugcheck(self, token_address: str) -> Optional[Dict]:
        """Call RugCheck.xyz API for Solana tokens"""
        url = f"{self.RUGCHECK_API}/tokens/{token_address}/report"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.debug(f"RugCheck API error: {e}")
        return None

    async def _call_helius_token_info(self, token_address: str) -> Optional[Dict]:
        """Call Helius API for Solana token info (fallback)"""
        api_key = os.getenv('HELIUS_API_KEY')
        if not api_key:
            return None

        url = f"https://api.helius.xyz/v0/token-metadata?api-key={api_key}"
        payload = {"mintAccounts": [token_address]}

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        return data[0]
        except Exception as e:
            logger.debug(f"Helius API error: {e}")
        return None

    def _calculate_score(
        self,
        is_honeypot: bool,
        buy_tax: float,
        sell_tax: float,
        is_verified: bool,
        is_renounced: bool,
        liquidity_usd: float,
        liquidity_locked: bool,
        holder_count: int,
        top_holder_pct: float,
        warning_count: int
    ) -> int:
        """Calculate safety score 0-100"""
        if is_honeypot:
            return 0

        score = 100

        # Tax penalties
        score -= min(buy_tax, 30)
        score -= min(sell_tax, 30)

        # Verification bonus
        if not is_verified:
            score -= 15

        # Ownership bonus
        if is_renounced:
            score += 10
        else:
            score -= 10

        # Liquidity scoring
        if liquidity_usd < 1000:
            score -= 25
        elif liquidity_usd < 5000:
            score -= 15
        elif liquidity_usd < 10000:
            score -= 5
        else:
            score += 5

        if liquidity_locked:
            score += 15

        # Holder distribution
        if holder_count < 10:
            score -= 20
        elif holder_count < 50:
            score -= 10
        elif holder_count > 500:
            score += 5

        # Top holder concentration
        if top_holder_pct > 80:
            score -= 30
        elif top_holder_pct > 50:
            score -= 20
        elif top_holder_pct > 30:
            score -= 10

        # Warning penalty
        score -= warning_count * 3

        return max(0, min(100, int(score)))

    def _get_rating(self, score: int, is_honeypot: bool) -> SafetyRating:
        """Determine safety rating from score"""
        if is_honeypot:
            return SafetyRating.HONEYPOT
        if score >= 70:
            return SafetyRating.SAFE
        if score >= 40:
            return SafetyRating.CAUTION
        return SafetyRating.DANGER

    def _log_report(self, report: SafetyReport):
        """Log safety report - only log SAFE/CAUTION at INFO, others at DEBUG"""
        emoji = {
            SafetyRating.SAFE: "✅",
            SafetyRating.CAUTION: "⚠️",
            SafetyRating.DANGER: "🚨",
            SafetyRating.HONEYPOT: "🍯"
        }.get(report.rating, "❓")

        log_msg = (f"{emoji} Safety: {report.token_address[:12]}... "
                   f"| {report.rating.value.upper()} | Score: {report.score} | "
                   f"Tax: {report.buy_tax:.0f}%/{report.sell_tax:.0f}% | "
                   f"Liq: ${report.liquidity_usd:,.0f}")

        # Only log SAFE/CAUTION tokens at INFO level to reduce spam
        if report.rating in [SafetyRating.SAFE, SafetyRating.CAUTION]:
            logger.info(log_msg)
            # Only show warnings for tokens that might be traded
            for warning in report.warnings[:3]:
                logger.debug(f"  {warning}")
        else:
            # Log DANGER/HONEYPOT at DEBUG to reduce spam
            logger.debug(log_msg)


# Convenience function for quick checks
async def check_token_safety(token_address: str, chain: str, config: Dict = None) -> SafetyReport:
    """
    Quick token safety check.

    Usage:
        report = await check_token_safety("0x...", "ethereum")
        if report.is_safe_to_snipe():
            # Execute trade
    """
    checker = TokenSafetyChecker(config)
    await checker.initialize()
    try:
        return await checker.check_token(token_address, chain)
    finally:
        await checker.close()
