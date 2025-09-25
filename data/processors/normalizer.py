"""
Data Normalizer - Standardizes data from multiple sources for ClaudeDex Trading Bot
Ensures consistent data format across all components
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from decimal import Decimal
from datetime import datetime, timezone
import re
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class DataType(Enum):
    """Types of data to normalize"""
    PRICE = "price"
    VOLUME = "volume"
    MARKET_CAP = "market_cap"
    PERCENTAGE = "percentage"
    TIMESTAMP = "timestamp"
    ADDRESS = "address"
    SYMBOL = "symbol"
    CHAIN = "chain"
    DEX = "dex"
    TRANSACTION = "transaction"


@dataclass
class NormalizationConfig:
    """Configuration for data normalization"""
    decimal_places: Dict[DataType, int]
    strip_whitespace: bool = True
    lowercase_addresses: bool = True
    uppercase_symbols: bool = True
    convert_to_utc: bool = True
    remove_special_chars: bool = True
    validate_addresses: bool = True


class DataNormalizer:
    """
    Normalizes and standardizes data from various sources
    Ensures consistency across the entire system
    """
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or self._default_config()
        
        # Chain mappings
        self.chain_mappings = {
            "eth": "ethereum",
            "ether": "ethereum",
            "mainnet": "ethereum",
            "bsc": "bsc",
            "binance": "bsc",
            "bnb": "bsc",
            "polygon": "polygon",
            "matic": "polygon",
            "arb": "arbitrum",
            "arbitrum": "arbitrum",
            "base": "base"
        }
        
        # DEX mappings
        self.dex_mappings = {
            "uni": "uniswap",
            "uniswapv2": "uniswap_v2",
            "uniswapv3": "uniswap_v3",
            "pancake": "pancakeswap",
            "pancakeswapv2": "pancakeswap_v2",
            "pancakeswapv3": "pancakeswap_v3",
            "sushi": "sushiswap",
            "1inch": "1inch",
            "paraswap": "paraswap"
        }
        
        # Common token mappings
        self.token_mappings = {
            "weth": "WETH",
            "wrapped eth": "WETH",
            "wbtc": "WBTC",
            "wrapped btc": "WBTC",
            "usdt": "USDT",
            "tether": "USDT",
            "usdc": "USDC",
            "usd coin": "USDC",
            "dai": "DAI",
            "busd": "BUSD"
        }
        
        logger.info("DataNormalizer initialized")
    
    def _default_config(self) -> NormalizationConfig:
        """Default normalization configuration"""
        return NormalizationConfig(
            decimal_places={
                DataType.PRICE: 18,
                DataType.VOLUME: 2,
                DataType.MARKET_CAP: 0,
                DataType.PERCENTAGE: 4
            }
        )
    
    def normalize_batch(self, data: List[Dict[str, Any]], schema: Dict[str, DataType]) -> List[Dict[str, Any]]:
        """
        Normalize a batch of data records
        
        Args:
            data: List of data records
            schema: Mapping of field names to data types
            
        Returns:
            List of normalized records
        """
        normalized_data = []
        
        for record in data:
            normalized_record = self.normalize_record(record, schema)
            normalized_data.append(normalized_record)
        
        return normalized_data
    
    def normalize_record(self, record: Dict[str, Any], schema: Dict[str, DataType]) -> Dict[str, Any]:
        """
        Normalize a single data record
        
        Args:
            record: Data record to normalize
            schema: Mapping of field names to data types
            
        Returns:
            Normalized record
        """
        normalized = {}
        
        for field, value in record.items():
            if field in schema:
                data_type = schema[field]
                normalized_value = self.normalize_value(value, data_type)
                normalized[field] = normalized_value
            else:
                # Keep unknown fields as-is
                normalized[field] = value
        
        return normalized
    
    def normalize_value(self, value: Any, data_type: DataType) -> Any:
        """
        Normalize a single value based on its type
        
        Args:
            value: Value to normalize
            data_type: Type of the data
            
        Returns:
            Normalized value
        """
        if value is None:
            return None
        
        try:
            if data_type == DataType.PRICE:
                return self.normalize_price(value)
            elif data_type == DataType.VOLUME:
                return self.normalize_volume(value)
            elif data_type == DataType.MARKET_CAP:
                return self.normalize_market_cap(value)
            elif data_type == DataType.PERCENTAGE:
                return self.normalize_percentage(value)
            elif data_type == DataType.TIMESTAMP:
                return self.normalize_timestamp(value)
            elif data_type == DataType.ADDRESS:
                return self.normalize_address(value)
            elif data_type == DataType.SYMBOL:
                return self.normalize_symbol(value)
            elif data_type == DataType.CHAIN:
                return self.normalize_chain(value)
            elif data_type == DataType.DEX:
                return self.normalize_dex(value)
            elif data_type == DataType.TRANSACTION:
                return self.normalize_transaction(value)
            else:
                return value
                
        except Exception as e:
            logger.error(f"Error normalizing value {value} of type {data_type}: {e}")
            return value
    
    def normalize_price(self, price: Union[str, float, int, Decimal]) -> Decimal:
        """Normalize price to Decimal with proper precision"""
        if isinstance(price, str):
            # Remove currency symbols and commas
            price = re.sub(r'[$,€£¥₹]', '', price).strip()
        
        decimal_price = Decimal(str(price))
        
        # Round to configured decimal places
        decimal_places = self.config.decimal_places.get(DataType.PRICE, 18)
        return decimal_price.quantize(Decimal(10) ** -decimal_places)
    
    def normalize_volume(self, volume: Union[str, float, int, Decimal]) -> Decimal:
        """Normalize volume to Decimal"""
        if isinstance(volume, str):
            # Handle suffixes like K, M, B
            volume = volume.upper().strip()
            multiplier = 1
            
            if volume.endswith('K'):
                multiplier = 1_000
                volume = volume[:-1]
            elif volume.endswith('M'):
                multiplier = 1_000_000
                volume = volume[:-1]
            elif volume.endswith('B'):
                multiplier = 1_000_000_000
                volume = volume[:-1]
            
            # Remove currency symbols and commas
            volume = re.sub(r'[$,€£¥₹]', '', volume).strip()
            volume = float(volume) * multiplier
        
        decimal_volume = Decimal(str(volume))
        
        # Round to configured decimal places
        decimal_places = self.config.decimal_places.get(DataType.VOLUME, 2)
        return decimal_volume.quantize(Decimal(10) ** -decimal_places)
    
    def normalize_market_cap(self, market_cap: Union[str, float, int, Decimal]) -> Decimal:
        """Normalize market cap to Decimal"""
        return self.normalize_volume(market_cap)  # Same logic as volume
    
    def normalize_percentage(self, percentage: Union[str, float, int]) -> float:
        """Normalize percentage to float between 0 and 100"""
        if isinstance(percentage, str):
            # Remove % sign
            percentage = percentage.replace('%', '').strip()
        
        percentage = float(percentage)
        
        # Convert to 0-100 range if in 0-1 range
        if -1 <= percentage <= 1 and percentage != 0:
            percentage = percentage * 100
        
        # Round to configured decimal places
        decimal_places = self.config.decimal_places.get(DataType.PERCENTAGE, 4)
        return round(percentage, decimal_places)
    
    def normalize_timestamp(self, timestamp: Union[str, int, float, datetime]) -> datetime:
        """Normalize timestamp to UTC datetime"""
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            elif self.config.convert_to_utc:
                timestamp = timestamp.astimezone(timezone.utc)
            return timestamp
        
        if isinstance(timestamp, str):
            # Try to parse ISO format
            try:
                dt = datetime.fromisoformat(timestamp)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except:
                # Try to parse as Unix timestamp
                timestamp = float(timestamp)
        
        if isinstance(timestamp, (int, float)):
            # Unix timestamp
            if timestamp > 10**10:  # Milliseconds
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        
        raise ValueError(f"Cannot normalize timestamp: {timestamp}")
    
    def normalize_address(self, address: str) -> str:
        """Normalize blockchain address"""
        if not address:
            return address
        
        address = str(address).strip()
        
        if self.config.strip_whitespace:
            address = address.strip()
        
        # Check if it's an Ethereum-style address
        if address.startswith('0x') or address.startswith('0X'):
            # Ensure consistent 0x prefix
            address = '0x' + address[2:]
            
            if self.config.lowercase_addresses:
                # Ethereum addresses are case-insensitive
                address = address.lower()
            
            if self.config.validate_addresses:
                # Basic validation: should be 42 characters (0x + 40 hex chars)
                if len(address) != 42:
                    logger.warning(f"Invalid address length: {address}")
                if not all(c in '0123456789abcdefABCDEF' for c in address[2:]):
                    logger.warning(f"Invalid address characters: {address}")
        
        return address
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize token symbol"""
        if not symbol:
            return symbol
        
        symbol = str(symbol).strip()
        
        if self.config.strip_whitespace:
            symbol = symbol.strip()
        
        if self.config.remove_special_chars:
            # Remove special characters except hyphens and underscores
            symbol = re.sub(r'[^a-zA-Z0-9\-_]', '', symbol)
        
        # Check for common mappings
        symbol_lower = symbol.lower()
        if symbol_lower in self.token_mappings:
            symbol = self.token_mappings[symbol_lower]
        elif self.config.uppercase_symbols:
            symbol = symbol.upper()
        
        return symbol
    
    def normalize_chain(self, chain: str) -> str:
        """Normalize blockchain name"""
        if not chain:
            return chain
        
        chain = str(chain).strip().lower()
        
        # Apply mappings
        if chain in self.chain_mappings:
            chain = self.chain_mappings[chain]
        
        return chain
    
    def normalize_dex(self, dex: str) -> str:
        """Normalize DEX name"""
        if not dex:
            return dex
        
        dex = str(dex).strip().lower()
        
        # Remove spaces and special characters
        dex = re.sub(r'[^a-z0-9_]', '', dex)
        
        # Apply mappings
        if dex in self.dex_mappings:
            dex = self.dex_mappings[dex]
        
        return dex
    
    def normalize_transaction(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize transaction data"""
        normalized_tx = {}
        
        # Standard transaction fields
        if 'hash' in tx:
            normalized_tx['hash'] = self.normalize_address(tx['hash'])
        
        if 'from' in tx:
            normalized_tx['from'] = self.normalize_address(tx['from'])
        
        if 'to' in tx:
            normalized_tx['to'] = self.normalize_address(tx['to'])
        
        if 'value' in tx:
            normalized_tx['value'] = self.normalize_price(tx['value'])
        
        if 'gas' in tx:
            normalized_tx['gas'] = int(tx['gas']) if tx['gas'] else 0
        
        if 'gasPrice' in tx:
            normalized_tx['gas_price'] = self.normalize_price(tx['gasPrice'])
        elif 'gas_price' in tx:
            normalized_tx['gas_price'] = self.normalize_price(tx['gas_price'])
        
        if 'timestamp' in tx:
            normalized_tx['timestamp'] = self.normalize_timestamp(tx['timestamp'])
        
        if 'blockNumber' in tx:
            normalized_tx['block_number'] = int(tx['blockNumber'])
        elif 'block_number' in tx:
            normalized_tx['block_number'] = int(tx['block_number'])
        
        # Copy other fields
        for key, value in tx.items():
            if key not in normalized_tx:
                normalized_tx[key] = value
        
        return normalized_tx
    
    def normalize_dataframe(self, df: pd.DataFrame, schema: Dict[str, DataType]) -> pd.DataFrame:
        """
        Normalize a pandas DataFrame
        
        Args:
            df: DataFrame to normalize
            schema: Mapping of column names to data types
            
        Returns:
            Normalized DataFrame
        """
        normalized_df = df.copy()
        
        for column, data_type in schema.items():
            if column in normalized_df.columns:
                normalized_df[column] = normalized_df[column].apply(
                    lambda x: self.normalize_value(x, data_type)
                )
        
        return normalized_df
    
    def validate_normalized_data(self, data: Dict[str, Any], schema: Dict[str, DataType]) -> Tuple[bool, List[str]]:
        """
        Validate normalized data against schema
        
        Args:
            data: Normalized data to validate
            schema: Expected schema
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for field, data_type in schema.items():
            if field not in data:
                errors.append(f"Missing required field: {field}")
                continue
            
            value = data[field]
            
            # Type checking
            if data_type == DataType.PRICE and not isinstance(value, Decimal):
                errors.append(f"Field {field} should be Decimal, got {type(value)}")
            elif data_type == DataType.VOLUME and not isinstance(value, Decimal):
                errors.append(f"Field {field} should be Decimal, got {type(value)}")
            elif data_type == DataType.TIMESTAMP and not isinstance(value, datetime):
                errors.append(f"Field {field} should be datetime, got {type(value)}")
            elif data_type in [DataType.ADDRESS, DataType.SYMBOL, DataType.CHAIN, DataType.DEX] and not isinstance(value, str):
                errors.append(f"Field {field} should be string, got {type(value)}")
            elif data_type == DataType.PERCENTAGE and not isinstance(value, (int, float)):
                errors.append(f"Field {field} should be numeric, got {type(value)}")
        
        return len(errors) == 0, errors
    
    def merge_normalized_data(self, *data_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge multiple normalized data sources
        
        Args:
            data_sources: Variable number of data source lists
            
        Returns:
            Merged and deduplicated data
        """
        merged_data = []
        seen_keys = set()
        
        for source in data_sources:
            for record in source:
                # Create a unique key for deduplication
                key = self._create_record_key(record)
                
                if key not in seen_keys:
                    seen_keys.add(key)
                    merged_data.append(record)
        
        return merged_data
    
    def _create_record_key(self, record: Dict[str, Any]) -> str:
        """Create a unique key for a record for deduplication"""
        # Use combination of important fields
        key_fields = []
        
        if 'token_address' in record:
            key_fields.append(str(record['token_address']))
        if 'chain' in record:
            key_fields.append(str(record['chain']))
        if 'timestamp' in record:
            key_fields.append(str(record['timestamp']))
        if 'hash' in record:
            key_fields.append(str(record['hash']))
        
        return '|'.join(key_fields)
    
    def export_schema(self, schema: Dict[str, DataType]) -> Dict[str, str]:
        """
        Export schema definition for documentation
        
        Args:
            schema: Schema to export
            
        Returns:
            Schema as string dictionary
        """
        return {field: data_type.value for field, data_type in schema.items()}