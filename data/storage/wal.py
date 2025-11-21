# data/storage/wal.py
"""
Write-Ahead Logging (WAL) for crash recovery

Ensures that critical operations can be recovered or rolled back if the system crashes.
Records all critical state changes BEFORE they happen, enabling recovery on restart.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from decimal import Decimal

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations tracked by WAL"""
    BALANCE_DEDUCT = "balance_deduct"
    BALANCE_ADD = "balance_add"
    POSITION_OPEN = "position_open"
    POSITION_CLOSE = "position_close"
    POSITION_UPDATE = "position_update"
    TRADE_EXECUTE = "trade_execute"


class OperationStatus(Enum):
    """Status of WAL entries"""
    PENDING = "pending"      # Operation logged but not started
    IN_PROGRESS = "in_progress"  # Operation started
    COMPLETED = "completed"  # Operation completed successfully
    FAILED = "failed"        # Operation failed
    ROLLED_BACK = "rolled_back"  # Operation rolled back


@dataclass
class WALEntry:
    """A single WAL log entry"""
    entry_id: str
    operation_type: OperationType
    timestamp: datetime
    status: OperationStatus
    data: Dict[str, Any]
    rollback_data: Optional[Dict[str, Any]] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    def to_json(self) -> str:
        """Convert to JSON string for storage"""
        return json.dumps({
            'entry_id': self.entry_id,
            'operation_type': self.operation_type.value,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'data': self._serialize_data(self.data),
            'rollback_data': self._serialize_data(self.rollback_data) if self.rollback_data else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error': self.error
        })

    @staticmethod
    def _serialize_data(data: Any) -> Any:
        """Serialize Decimal and other non-JSON types"""
        if isinstance(data, dict):
            return {k: WALEntry._serialize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [WALEntry._serialize_data(v) for v in data]
        elif isinstance(data, Decimal):
            return str(data)
        elif isinstance(data, datetime):
            return data.isoformat()
        return data

    @classmethod
    def from_json(cls, json_str: str) -> 'WALEntry':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls(
            entry_id=data['entry_id'],
            operation_type=OperationType(data['operation_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            status=OperationStatus(data['status']),
            data=data['data'],
            rollback_data=data.get('rollback_data'),
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            error=data.get('error')
        )


class WriteAheadLog:
    """
    Write-Ahead Logging system for crash recovery

    CRITICAL FIX (P1): Ensures critical operations can be recovered or rolled back

    Usage:
        wal = WriteAheadLog('/path/to/wal.log')

        # Before critical operation:
        entry_id = await wal.log_operation(
            OperationType.BALANCE_DEDUCT,
            data={'amount': 100, 'token': 'SOL'},
            rollback_data={'old_balance': 500}
        )

        try:
            # Perform operation
            perform_balance_deduction()

            # Mark complete
            await wal.complete_operation(entry_id)
        except Exception as e:
            # Mark failed (can be rolled back on recovery)
            await wal.fail_operation(entry_id, str(e))
            raise
    """

    def __init__(self, wal_file_path: str):
        """
        Initialize WAL

        Args:
            wal_file_path: Path to WAL log file
        """
        self.wal_file_path = Path(wal_file_path)
        self.wal_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()

        # Ensure WAL file exists
        if not self.wal_file_path.exists():
            self.wal_file_path.touch()
            logger.info(f"Created WAL file: {self.wal_file_path}")

        logger.info(f"WAL initialized: {self.wal_file_path}")

    async def log_operation(
        self,
        operation_type: OperationType,
        data: Dict[str, Any],
        rollback_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an operation BEFORE executing it

        Args:
            operation_type: Type of operation
            data: Operation data
            rollback_data: Data needed to rollback the operation

        Returns:
            entry_id: Unique ID for this operation
        """
        import uuid

        entry_id = f"{operation_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        entry = WALEntry(
            entry_id=entry_id,
            operation_type=operation_type,
            timestamp=datetime.utcnow(),
            status=OperationStatus.PENDING,
            data=data,
            rollback_data=rollback_data
        )

        async with self.lock:
            # Append to WAL file
            with open(self.wal_file_path, 'a') as f:
                f.write(entry.to_json() + '\n')

        logger.debug(f"WAL: Logged {operation_type.value} operation {entry_id}")
        return entry_id

    async def start_operation(self, entry_id: str) -> None:
        """Mark operation as started"""
        await self._update_status(entry_id, OperationStatus.IN_PROGRESS)

    async def complete_operation(self, entry_id: str) -> None:
        """Mark operation as completed successfully"""
        await self._update_status(entry_id, OperationStatus.COMPLETED, completed_at=datetime.utcnow())
        logger.debug(f"WAL: Completed operation {entry_id}")

    async def fail_operation(self, entry_id: str, error: str) -> None:
        """Mark operation as failed"""
        await self._update_status(entry_id, OperationStatus.FAILED, error=error)
        logger.warning(f"WAL: Failed operation {entry_id}: {error}")

    async def rollback_operation(self, entry_id: str) -> None:
        """Mark operation as rolled back"""
        await self._update_status(entry_id, OperationStatus.ROLLED_BACK)
        logger.info(f"WAL: Rolled back operation {entry_id}")

    async def _update_status(
        self,
        entry_id: str,
        status: OperationStatus,
        completed_at: Optional[datetime] = None,
        error: Optional[str] = None
    ) -> None:
        """Update the status of a WAL entry"""
        async with self.lock:
            # Read all entries
            entries = await self._read_all_entries()

            # Update the entry
            updated = False
            for entry in entries:
                if entry.entry_id == entry_id:
                    entry.status = status
                    if completed_at:
                        entry.completed_at = completed_at
                    if error:
                        entry.error = error
                    updated = True
                    break

            if not updated:
                logger.warning(f"WAL entry {entry_id} not found for update")
                return

            # Write all entries back
            with open(self.wal_file_path, 'w') as f:
                for entry in entries:
                    f.write(entry.to_json() + '\n')

    async def _read_all_entries(self) -> List[WALEntry]:
        """Read all WAL entries from file"""
        entries = []

        if not self.wal_file_path.exists():
            return entries

        with open(self.wal_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = WALEntry.from_json(line)
                        entries.append(entry)
                    except Exception as e:
                        logger.error(f"Error parsing WAL entry: {e}")

        return entries

    async def get_pending_operations(self) -> List[WALEntry]:
        """Get all pending or in-progress operations"""
        async with self.lock:
            entries = await self._read_all_entries()
            return [
                e for e in entries
                if e.status in [OperationStatus.PENDING, OperationStatus.IN_PROGRESS]
            ]

    async def get_failed_operations(self) -> List[WALEntry]:
        """Get all failed operations that need attention"""
        async with self.lock:
            entries = await self._read_all_entries()
            return [e for e in entries if e.status == OperationStatus.FAILED]

    async def cleanup_completed(self, older_than_hours: int = 24) -> int:
        """
        Remove completed entries older than specified hours

        Args:
            older_than_hours: Remove entries completed more than this many hours ago

        Returns:
            Number of entries removed
        """
        from datetime import timedelta

        async with self.lock:
            entries = await self._read_all_entries()
            cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)

            # Keep entries that are:
            # - Not completed, OR
            # - Completed recently
            kept_entries = [
                e for e in entries
                if e.status != OperationStatus.COMPLETED or
                (e.completed_at and e.completed_at > cutoff_time)
            ]

            removed_count = len(entries) - len(kept_entries)

            if removed_count > 0:
                # Write back kept entries
                with open(self.wal_file_path, 'w') as f:
                    for entry in kept_entries:
                        f.write(entry.to_json() + '\n')

                logger.info(f"WAL: Cleaned up {removed_count} completed entries")

            return removed_count

    async def recover(self, recovery_handler) -> Dict[str, int]:
        """
        Recover from pending/failed operations

        Args:
            recovery_handler: Async function that handles recovery for each entry
                             Should accept (entry: WALEntry) and return True if recovered

        Returns:
            Dict with recovery statistics
        """
        logger.info("WAL: Starting crash recovery...")

        pending = await self.get_pending_operations()
        failed = await self.get_failed_operations()

        stats = {
            'pending_found': len(pending),
            'failed_found': len(failed),
            'recovered': 0,
            'rolled_back': 0,
            'errors': 0
        }

        # Process pending and failed operations
        for entry in pending + failed:
            try:
                logger.info(
                    f"WAL: Recovering {entry.operation_type.value} "
                    f"from {entry.timestamp.isoformat()}"
                )

                # Call recovery handler
                recovered = await recovery_handler(entry)

                if recovered:
                    await self.complete_operation(entry.entry_id)
                    stats['recovered'] += 1
                else:
                    await self.rollback_operation(entry.entry_id)
                    stats['rolled_back'] += 1

            except Exception as e:
                logger.error(f"WAL: Recovery error for {entry.entry_id}: {e}", exc_info=True)
                await self.fail_operation(entry.entry_id, str(e))
                stats['errors'] += 1

        logger.info(
            f"WAL: Recovery complete. "
            f"Recovered: {stats['recovered']}, "
            f"Rolled back: {stats['rolled_back']}, "
            f"Errors: {stats['errors']}"
        )

        return stats


# Global WAL instance (initialized in main.py or engine.py)
_wal_instance: Optional[WriteAheadLog] = None


def get_wal() -> WriteAheadLog:
    """Get the global WAL instance"""
    global _wal_instance
    if _wal_instance is None:
        # Default path - should be overridden by calling init_wal()
        wal_dir = os.getenv('WAL_DIR', './data/wal')
        _wal_instance = WriteAheadLog(f'{wal_dir}/operations.wal')
    return _wal_instance


def init_wal(wal_file_path: str) -> WriteAheadLog:
    """Initialize the global WAL instance"""
    global _wal_instance
    _wal_instance = WriteAheadLog(wal_file_path)
    return _wal_instance
