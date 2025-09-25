"""
Utility Helper Functions for ClaudeDex Trading Bot
Core utilities for data manipulation, formatting, calculations, and common operations
"""

import asyncio
import hashlib
import hmac
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import wraps, lru_cache
import aiohttp
import numpy as np
from web3 import Web3

logger = logging.getLogger(__name__)

# ============= Decorators =============

def retry_async(max_retries: int = 3, delay: float = 1.0, exponential_backoff: bool = True):
    """Async retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait_time = delay * (2 ** attempt) if exponential_backoff else delay
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
            raise last_exception
        return wrapper
    return decorator

def measure_time(func):
    """Measure execution time decorator"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__name__} took {elapsed:.4f} seconds")
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__name__} took {elapsed:.4f} seconds")
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def rate_limit(calls: int = 10, period: float = 1.0):
    """Rate limiting decorator"""
    def decorator(func):
        calls_made = []
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            # Remove old calls outside the period
            nonlocal calls_made
            calls_made = [t for t in calls_made if now - t < period]
            
            if len(calls_made) >= calls:
                sleep_time = period - (now - calls_made[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    calls_made = []
            
            calls_made.append(time.time())
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# ============= Web3 Utilities =============

def is_valid_address(address: str) -> bool:
    """Validate Ethereum address"""
    try:
        Web3.to_checksum_address(address)
        return True
    except:
        return False

def normalize_address(address: str) -> str:
    """Normalize Ethereum address to checksum format"""
    try:
        return Web3.to_checksum_address(address.lower())
    except:
        return address

def wei_to_ether(wei: Union[int, str, Decimal]) -> Decimal:
    """Convert Wei to Ether"""
    return Decimal(str(wei)) / Decimal(10**18)

def ether_to_wei(ether: Union[float, str, Decimal]) -> int:
    """Convert Ether to Wei"""
    return int(Decimal(str(ether)) * Decimal(10**18))

def format_token_amount(amount: Union[int, str, Decimal], decimals: int) -> Decimal:
    """Format token amount based on decimals"""
    return Decimal(str(amount)) / Decimal(10**decimals)

def to_base_unit(amount: Union[float, str, Decimal], decimals: int) -> int:
    """Convert to base unit (smallest denomination)"""
    return int(Decimal(str(amount)) * Decimal(10**decimals))

# ============= Math & Financial Utilities =============

def calculate_percentage_change(old_value: Decimal, new_value: Decimal) -> Decimal:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return Decimal(0)
    return ((new_value - old_value) / old_value) * 100

def calculate_slippage(expected_price: Decimal, actual_price: Decimal) -> Decimal:
    """Calculate slippage percentage"""
    if expected_price == 0:
        return Decimal(0)
    return abs((actual_price - expected_price) / expected_price) * 100

def round_to_significant_digits(value: Decimal, sig_digits: int = 6) -> Decimal:
    """Round to significant digits"""
    if value == 0:
        return Decimal(0)
    
    from math import log10, floor
    digits = sig_digits - int(floor(log10(abs(float(value))))) - 1
    return value.quantize(Decimal(10) ** -digits)

def calculate_profit_loss(entry_price: Decimal, exit_price: Decimal, 
                         amount: Decimal, fees: Decimal = Decimal(0)) -> Dict[str, Decimal]:
    """Calculate profit/loss for a trade"""
    gross_value = amount * (exit_price - entry_price)
    net_value = gross_value - fees
    percentage = calculate_percentage_change(entry_price, exit_price)
    
    return {
        'gross_pnl': gross_value,
        'net_pnl': net_value,
        'fees': fees,
        'percentage': percentage,
        'is_profit': net_value > 0
    }

@lru_cache(maxsize=128)
def calculate_moving_average(values: tuple, window: int) -> float:
    """Calculate simple moving average"""
    if len(values) < window:
        return float(np.mean(values))
    return float(np.mean(values[-window:]))

def calculate_ema(values: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average"""
    if not values or period <= 0:
        return []
    
    ema = []
    multiplier = 2 / (period + 1)
    
    # Start with SMA for first value
    if len(values) >= period:
        ema.append(sum(values[:period]) / period)
    else:
        ema.append(sum(values) / len(values))
    
    # Calculate EMA for rest
    for i in range(1, len(values)):
        if i < period:
            ema.append(sum(values[:i+1]) / (i+1))
        else:
            ema.append((values[i] * multiplier) + (ema[-1] * (1 - multiplier)))
    
    return ema

# ============= Time Utilities =============

def get_timestamp() -> int:
    """Get current Unix timestamp"""
    return int(time.time())

def get_timestamp_ms() -> int:
    """Get current Unix timestamp in milliseconds"""
    return int(time.time() * 1000)

def format_timestamp(timestamp: Union[int, float], fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format Unix timestamp to readable string"""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(fmt)

def parse_timeframe(timeframe: str) -> timedelta:
    """Parse timeframe string to timedelta (e.g., '1h', '5m', '1d')"""
    patterns = {
        r'(\d+)s': lambda x: timedelta(seconds=int(x)),
        r'(\d+)m': lambda x: timedelta(minutes=int(x)),
        r'(\d+)h': lambda x: timedelta(hours=int(x)),
        r'(\d+)d': lambda x: timedelta(days=int(x)),
        r'(\d+)w': lambda x: timedelta(weeks=int(x)),
    }
    
    for pattern, func in patterns.items():
        match = re.match(pattern, timeframe.lower())
        if match:
            return func(match.group(1))
    
    raise ValueError(f"Invalid timeframe format: {timeframe}")

def is_market_hours(timezone_str: str = "UTC") -> bool:
    """Check if current time is within typical crypto market active hours"""
    # Crypto markets are 24/7, but we can define "active" hours
    now = datetime.now(timezone.utc)
    hour = now.hour
    
    # Most active: 13:00-23:00 UTC (covers US and Asian markets)
    return 13 <= hour <= 23

# ============= Data Formatting =============

def format_number(value: Union[int, float, Decimal], decimals: int = 2) -> str:
    """Format number with thousands separator and decimal places"""
    if isinstance(value, (int, float)):
        value = Decimal(str(value))
    
    formatted = f"{value:,.{decimals}f}"
    return formatted

def format_currency(value: Union[int, float, Decimal], symbol: str = "$", decimals: int = 2) -> str:
    """Format value as currency"""
    formatted_number = format_number(value, decimals)
    return f"{symbol}{formatted_number}"

def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate string to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely load JSON with default fallback"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

# ============= Validation Utilities =============

def validate_token_symbol(symbol: str) -> bool:
    """Validate token symbol format"""
    pattern = r'^[A-Z0-9]{2,10}$'
    return bool(re.match(pattern, symbol.upper()))

def validate_chain_id(chain_id: Union[int, str]) -> bool:
    """Validate blockchain chain ID"""
    valid_chain_ids = {1, 56, 137, 42161, 8453}  # ETH, BSC, Polygon, Arbitrum, Base
    try:
        return int(chain_id) in valid_chain_ids
    except:
        return False

def sanitize_input(value: str, max_length: int = 100) -> str:
    """Sanitize user input to prevent injection attacks"""
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>\"\'`;(){}]', '', value)
    # Truncate to max length
    return sanitized[:max_length]

# ============= Network Utilities =============

@retry_async(max_retries=3, delay=1.0)
async def fetch_json(url: str, headers: Optional[Dict] = None, 
                     timeout: int = 30) -> Dict:
    """Fetch JSON from URL with retry logic"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, timeout=timeout) as response:
            response.raise_for_status()
            return await response.json()

async def batch_request(urls: List[str], max_concurrent: int = 10) -> List[Dict]:
    """Execute batch requests with concurrency limit"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_with_semaphore(url):
        async with semaphore:
            try:
                return await fetch_json(url)
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
                return None
    
    results = await asyncio.gather(*[fetch_with_semaphore(url) for url in urls])
    return [r for r in results if r is not None]

# ============= Security Utilities =============

def generate_signature(message: str, secret: str) -> str:
    """Generate HMAC signature for message"""
    return hmac.new(
        secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def hash_data(data: str) -> str:
    """Generate SHA256 hash of data"""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
    """Mask sensitive data showing only first and last few characters"""
    if len(data) <= visible_chars * 2:
        return '*' * len(data)
    
    return f"{data[:visible_chars]}{'*' * (len(data) - visible_chars * 2)}{data[-visible_chars:]}"

# ============= Chunk Processing =============

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

async def process_in_chunks(items: List, processor, chunk_size: int = 100):
    """Process items in chunks asynchronously"""
    results = []
    for chunk in chunk_list(items, chunk_size):
        chunk_results = await asyncio.gather(*[processor(item) for item in chunk])
        results.extend(chunk_results)
    return results

# ============= Cache Utilities =============

class TTLCache:
    """Simple TTL cache implementation"""
    
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.timestamps.clear()

# ============= Export All Utilities =============

__all__ = [
    # Decorators
    'retry_async', 'measure_time', 'rate_limit',
    
    # Web3
    'is_valid_address', 'normalize_address', 'wei_to_ether', 
    'ether_to_wei', 'format_token_amount', 'to_base_unit',
    
    # Math & Financial
    'calculate_percentage_change', 'calculate_slippage',
    'round_to_significant_digits', 'calculate_profit_loss',
    'calculate_moving_average', 'calculate_ema',
    
    # Time
    'get_timestamp', 'get_timestamp_ms', 'format_timestamp',
    'parse_timeframe', 'is_market_hours',
    
    # Formatting
    'format_number', 'format_currency', 'truncate_string',
    'safe_json_loads', 'deep_merge_dicts',
    
    # Validation
    'validate_token_symbol', 'validate_chain_id', 'sanitize_input',
    
    # Network
    'fetch_json', 'batch_request',
    
    # Security
    'generate_signature', 'hash_data', 'mask_sensitive_data',
    
    # Processing
    'chunk_list', 'process_in_chunks',
    
    # Cache
    'TTLCache'
]