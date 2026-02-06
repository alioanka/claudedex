"""
Pump.fun Scam Token Blacklist Engine

Maintains a persistent blacklist of scam tokens detected through:
1. Rapid crash detection (>30% drop in short time)
2. Rug pull patterns (developer dump, liquidity removal)
3. Honeypot behavior (can't sell)
4. Known scam token names/patterns

Features:
- Persistent storage in database
- In-memory cache for fast lookups
- Auto-blacklisting on scam detection
- Token name pattern matching
- Manual blacklist management
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger('SolanaTradingEngine')


@dataclass
class BlacklistEntry:
    """Entry for a blacklisted token"""
    mint: str
    symbol: str
    reason: str
    detected_at: datetime
    loss_pct: Optional[float] = None  # Loss percentage if applicable
    metadata: Dict = field(default_factory=dict)


# Known scam token name patterns - AGGRESSIVE matching
SCAM_NAME_PATTERNS = [
    # Celebrity names - match ANYWHERE in name (not just exact)
    r'TRUMP',  # Any token with TRUMP in name
    r'BIDEN',
    r'ELON',
    r'MUSK',
    r'OBAMA',
    r'MELANIA',
    r'BARRON',
    r'IVANKA',

    # Common rug pull patterns
    r'SAFE\s*MOON',
    r'BABY\s*(DOGE|SHIB|PEPE)',
    r'MOON\s*SHOT',
    r'100X',
    r'1000X',
    r'GEM\s*FINDER',
    r'RUG\s*PULL',
    r'WOJAK',

    # Urgency patterns
    r'LAST\s*CHANCE',
    r'DONT\s*MISS',
    r'PRESALE\s*LIVE',
    r'BUY\s*NOW',
    r'SEND\s*IT',

    # Suspicious patterns
    r'^[A-Z]{1,2}\d+$',  # Very short name with numbers (A1, X99)
    r'FREE\s*MONEY',
    r'GUARANTEED',
    r'NO\s*TAX',
    r'STEALTH\s*LAUNCH',
    r'AIRDROP',
    r'GIVEAWAY',

    # Animal + modifier scams
    r'FLOKI',
    r'SHIBA',
    r'DOGE.*INU',

    # Crypto celebrity scams
    r'VITALIK',
    r'CZ\s*BNB',
    r'SBF',

    # Unknown/suspicious tokens
    r'^UNKNOWN$',  # Literal "UNKNOWN" name
    r'^TEST',
    r'^SCAM',
]

# Compile patterns for efficiency
COMPILED_SCAM_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SCAM_NAME_PATTERNS]


class ScamBlacklist:
    """
    Manages blacklist of scam tokens with persistence.

    Uses in-memory cache backed by database for persistence across restarts.
    """

    def __init__(self, db_pool=None):
        self.db_pool = db_pool

        # In-memory cache for fast lookups
        self._blacklisted_mints: Set[str] = set()
        self._blacklist_entries: Dict[str, BlacklistEntry] = {}

        # Statistics
        self.total_blacklisted = 0
        self.rapid_crashes_blocked = 0
        self.pattern_matches_blocked = 0

        # Rate limiting for database updates
        self._last_db_sync = datetime.now()
        self._db_sync_interval = timedelta(minutes=5)
        self._pending_writes: List[BlacklistEntry] = []

        logger.info("🚫 Scam Blacklist Engine initialized")

    async def initialize(self):
        """Load blacklist from database on startup"""
        if not self.db_pool:
            logger.warning("No database pool - blacklist will not persist")
            return

        try:
            async with self.db_pool.acquire() as conn:
                # Create table if not exists
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS scam_token_blacklist (
                        mint VARCHAR(64) PRIMARY KEY,
                        symbol VARCHAR(64),
                        reason TEXT,
                        detected_at TIMESTAMP DEFAULT NOW(),
                        loss_pct FLOAT,
                        metadata JSONB DEFAULT '{}'
                    )
                """)

                # Load existing entries
                rows = await conn.fetch("SELECT * FROM scam_token_blacklist")
                for row in rows:
                    entry = BlacklistEntry(
                        mint=row['mint'],
                        symbol=row['symbol'],
                        reason=row['reason'],
                        detected_at=row['detected_at'],
                        loss_pct=row['loss_pct'],
                        metadata=dict(row['metadata']) if row['metadata'] else {}
                    )
                    self._blacklisted_mints.add(entry.mint)
                    self._blacklist_entries[entry.mint] = entry

                self.total_blacklisted = len(self._blacklisted_mints)
                logger.info(f"   Loaded {self.total_blacklisted} blacklisted tokens from database")

        except Exception as e:
            logger.error(f"Error loading blacklist from database: {e}")

    def is_blacklisted(self, mint: str) -> bool:
        """Check if a token is blacklisted (fast lookup)"""
        return mint in self._blacklisted_mints

    def get_blacklist_reason(self, mint: str) -> Optional[str]:
        """Get the reason a token was blacklisted"""
        entry = self._blacklist_entries.get(mint)
        return entry.reason if entry else None

    def check_name_pattern(self, symbol: str, name: str = None) -> Tuple[bool, Optional[str]]:
        """
        Check if token name matches known scam patterns.

        Args:
            symbol: Token symbol
            name: Token full name (optional)

        Returns:
            Tuple of (is_scam, pattern_matched)
        """
        text_to_check = f"{symbol} {name}" if name else symbol

        for pattern in COMPILED_SCAM_PATTERNS:
            if pattern.search(text_to_check):
                return True, pattern.pattern

        return False, None

    async def add_to_blacklist(
        self,
        mint: str,
        symbol: str,
        reason: str,
        loss_pct: Optional[float] = None,
        metadata: Dict = None
    ) -> bool:
        """
        Add a token to the blacklist.

        Args:
            mint: Token mint address
            symbol: Token symbol
            reason: Reason for blacklisting
            loss_pct: Loss percentage if applicable
            metadata: Additional data about the scam

        Returns:
            True if added, False if already exists
        """
        if mint in self._blacklisted_mints:
            return False

        entry = BlacklistEntry(
            mint=mint,
            symbol=symbol,
            reason=reason,
            detected_at=datetime.now(),
            loss_pct=loss_pct,
            metadata=metadata or {}
        )

        # Add to cache
        self._blacklisted_mints.add(mint)
        self._blacklist_entries[mint] = entry
        self.total_blacklisted += 1

        # Track statistics
        if 'rapid' in reason.lower() or 'crash' in reason.lower():
            self.rapid_crashes_blocked += 1
        if 'pattern' in reason.lower():
            self.pattern_matches_blocked += 1

        # Queue for database write
        self._pending_writes.append(entry)

        logger.warning(f"🚫 BLACKLISTED: {symbol} ({mint[:10]}...) - {reason}")
        if loss_pct:
            logger.warning(f"   Loss: {loss_pct:.1f}%")

        # Sync to database if interval passed
        await self._maybe_sync_to_db()

        return True

    async def _maybe_sync_to_db(self):
        """Sync pending writes to database if interval passed"""
        if not self.db_pool or not self._pending_writes:
            return

        now = datetime.now()
        if now - self._last_db_sync < self._db_sync_interval:
            return

        await self._sync_to_db()

    async def _sync_to_db(self):
        """Force sync pending writes to database"""
        if not self.db_pool or not self._pending_writes:
            return

        try:
            async with self.db_pool.acquire() as conn:
                for entry in self._pending_writes:
                    # Serialize metadata to JSON string for JSONB column
                    metadata_json = json.dumps(entry.metadata if entry.metadata else {})
                    await conn.execute("""
                        INSERT INTO scam_token_blacklist (mint, symbol, reason, detected_at, loss_pct, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                        ON CONFLICT (mint) DO UPDATE SET
                            reason = EXCLUDED.reason,
                            loss_pct = EXCLUDED.loss_pct,
                            metadata = EXCLUDED.metadata
                    """, entry.mint, entry.symbol, entry.reason,
                        entry.detected_at, entry.loss_pct, metadata_json)

            logger.info(f"💾 Synced {len(self._pending_writes)} blacklist entries to database")
            self._pending_writes = []
            self._last_db_sync = datetime.now()

        except Exception as e:
            logger.error(f"Error syncing blacklist to database: {e}")

    async def remove_from_blacklist(self, mint: str) -> bool:
        """Remove a token from the blacklist (manual override)"""
        if mint not in self._blacklisted_mints:
            return False

        self._blacklisted_mints.discard(mint)
        self._blacklist_entries.pop(mint, None)
        self.total_blacklisted -= 1

        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        "DELETE FROM scam_token_blacklist WHERE mint = $1",
                        mint
                    )
            except Exception as e:
                logger.error(f"Error removing from blacklist: {e}")

        logger.info(f"✅ Removed {mint[:10]}... from blacklist")
        return True

    def get_statistics(self) -> Dict:
        """Get blacklist statistics"""
        return {
            'total_blacklisted': self.total_blacklisted,
            'rapid_crashes_blocked': self.rapid_crashes_blocked,
            'pattern_matches_blocked': self.pattern_matches_blocked,
            'in_memory_count': len(self._blacklisted_mints),
            'pending_writes': len(self._pending_writes)
        }

    async def on_rapid_crash_detected(
        self,
        mint: str,
        symbol: str,
        drop_pct: float,
        time_seconds: int
    ):
        """
        Called when a rapid price crash is detected.
        Automatically blacklists the token.

        Args:
            mint: Token mint address
            symbol: Token symbol
            drop_pct: Percentage drop (positive number)
            time_seconds: Time period of the drop in seconds
        """
        reason = f"Rapid crash: -{drop_pct:.1f}% in {time_seconds}s"
        await self.add_to_blacklist(
            mint=mint,
            symbol=symbol,
            reason=reason,
            loss_pct=drop_pct,
            metadata={
                'crash_pct': drop_pct,
                'crash_duration_seconds': time_seconds,
                'type': 'rapid_crash'
            }
        )

    async def on_rug_pull_detected(
        self,
        mint: str,
        symbol: str,
        rug_type: str,
        details: str = None
    ):
        """
        Called when a rug pull pattern is detected.

        Args:
            mint: Token mint address
            symbol: Token symbol
            rug_type: Type of rug (dev_dump, liquidity_removal, honeypot)
            details: Additional details
        """
        reason = f"Rug pull detected: {rug_type}"
        if details:
            reason += f" - {details}"

        await self.add_to_blacklist(
            mint=mint,
            symbol=symbol,
            reason=reason,
            metadata={
                'rug_type': rug_type,
                'details': details,
                'type': 'rug_pull'
            }
        )

    async def close(self):
        """Flush pending writes on shutdown"""
        if self._pending_writes:
            await self._sync_to_db()
        logger.info("🚫 Scam Blacklist Engine stopped")


# Global instance for easy access
_blacklist_instance: Optional[ScamBlacklist] = None


def get_blacklist() -> Optional[ScamBlacklist]:
    """Get the global blacklist instance"""
    return _blacklist_instance


async def initialize_blacklist(db_pool=None) -> ScamBlacklist:
    """Initialize and return the global blacklist instance"""
    global _blacklist_instance
    if _blacklist_instance is None:
        _blacklist_instance = ScamBlacklist(db_pool)
        await _blacklist_instance.initialize()
    return _blacklist_instance
