"""
Data Validator - Validates data quality and integrity for ClaudeDex Trading Bot
Ensures data meets quality standards before processing
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class ValidationLevel(Enum):
    """Validation severity levels"""
    CRITICAL = "critical"  # Must fix
    WARNING = "warning"    # Should fix
    INFO = "info"          # Nice to fix


class DataField(Enum):
    """Common data fields to validate"""
    PRICE = "price"
    VOLUME = "volume"
    LIQUIDITY = "liquidity"
    MARKET_CAP = "market_cap"
    HOLDERS = "holders"
    ADDRESS = "address"
    SYMBOL = "symbol"
    TIMESTAMP = "timestamp"
    PERCENTAGE = "percentage"
    TRANSACTION = "transaction"


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    info: List[Dict[str, Any]]
    quality_score: float
    validated_data: Optional[Dict[str, Any]]
    suggestions: List[str]


@dataclass
class ValidationRule:
    """Validation rule definition"""
    field: str
    rule_type: str  # 'range', 'regex', 'type', 'custom'
    params: Dict[str, Any]
    level: ValidationLevel
    message: str


class DataValidator:
    """
    Validates data quality and integrity
    Ensures data meets standards before processing
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Validation rules
        self.rules = self._initialize_rules()
        
        # Custom validators
        self.custom_validators = {}
        
        # Validation statistics
        self.validation_stats = {
            'total_validated': 0,
            'total_passed': 0,
            'total_failed': 0,
            'common_errors': {}
        }
        
        # Field-specific validators
        self.field_validators = {
            DataField.PRICE: self._validate_price,
            DataField.VOLUME: self._validate_volume,
            DataField.LIQUIDITY: self._validate_liquidity,
            DataField.ADDRESS: self._validate_address,
            DataField.SYMBOL: self._validate_symbol,
            DataField.TIMESTAMP: self._validate_timestamp,
            DataField.PERCENTAGE: self._validate_percentage
        }
        
        logger.info("DataValidator initialized")
    
    def _initialize_rules(self) -> List[ValidationRule]:
        """Initialize default validation rules"""
        rules = [
            # Price rules
            ValidationRule(
                field="price",
                rule_type="range",
                params={"min": 0, "max": 1e12},
                level=ValidationLevel.CRITICAL,
                message="Price must be positive and reasonable"
            ),
            
            # Volume rules
            ValidationRule(
                field="volume_24h",
                rule_type="range",
                params={"min": 0, "max": 1e15},
                level=ValidationLevel.WARNING,
                message="Volume must be non-negative"
            ),
            
            # Liquidity rules
            ValidationRule(
                field="liquidity",
                rule_type="range",
                params={"min": 0, "max": 1e15},
                level=ValidationLevel.WARNING,
                message="Liquidity must be non-negative"
            ),
            
            # Holder rules
            ValidationRule(
                field="holders",
                rule_type="range",
                params={"min": 0, "max": 1e9},
                level=ValidationLevel.WARNING,
                message="Holder count must be non-negative"
            ),
            
            # Address rules
            ValidationRule(
                field="token_address",
                rule_type="regex",
                params={"pattern": r"^0x[a-fA-F0-9]{40}$"},
                level=ValidationLevel.CRITICAL,
                message="Invalid token address format"
            ),
            
            # Symbol rules
            ValidationRule(
                field="symbol",
                rule_type="regex",
                params={"pattern": r"^[A-Z0-9]{1,20}$"},
                level=ValidationLevel.INFO,
                message="Symbol should be uppercase alphanumeric"
            ),
            
            # Percentage rules
            ValidationRule(
                field="*_percent",
                rule_type="range",
                params={"min": 0, "max": 100},
                level=ValidationLevel.WARNING,
                message="Percentage must be between 0 and 100"
            )
        ]
        
        return rules
    
    def validate(self, data: Dict[str, Any], schema: Optional[Dict] = None) -> ValidationResult:
        """
        Validate data against rules and schema
        
        Args:
            data: Data to validate
            schema: Optional schema definition
            
        Returns:
            ValidationResult with details
        """
        errors = []
        warnings = []
        info = []
        validated_data = data.copy()
        
        self.validation_stats['total_validated'] += 1
        
        # Apply validation rules
        for rule in self.rules:
            field_pattern = rule.field.replace('*', '.*')
            matching_fields = [f for f in data.keys() if re.match(field_pattern, f)]
            
            for field in matching_fields:
                result = self._apply_rule(rule, field, data.get(field))
                
                if result:
                    if rule.level == ValidationLevel.CRITICAL:
                        errors.append(result)
                    elif rule.level == ValidationLevel.WARNING:
                        warnings.append(result)
                    else:
                        info.append(result)
        
        # Schema validation if provided
        if schema:
            schema_errors = self._validate_schema(data, schema)
            errors.extend(schema_errors)
        
        # Field-specific validation
        for field, value in data.items():
            field_type = self._determine_field_type(field)
            if field_type in self.field_validators:
                field_result = self.field_validators[field_type](value)
                if field_result:
                    warnings.append(field_result)
        
        # Check data consistency
        consistency_issues = self._check_consistency(data)
        warnings.extend(consistency_issues)
        
        # Check for missing critical fields
        missing = self._check_required_fields(data)
        errors.extend(missing)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(errors, warnings, info)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(errors, warnings)
        
        # Determine overall validity
        is_valid = len(errors) == 0
        
        if is_valid:
            self.validation_stats['total_passed'] += 1
        else:
            self.validation_stats['total_failed'] += 1
            # Track common errors
            for error in errors:
                error_type = error.get('field', 'unknown')
                self.validation_stats['common_errors'][error_type] = \
                    self.validation_stats['common_errors'].get(error_type, 0) + 1
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info,
            quality_score=quality_score,
            validated_data=validated_data if is_valid else None,
            suggestions=suggestions
        )
    
    def _apply_rule(self, rule: ValidationRule, field: str, value: Any) -> Optional[Dict]:
        """Apply a single validation rule"""
        if value is None:
            return None
        
        try:
            if rule.rule_type == "range":
                min_val = rule.params.get("min")
                max_val = rule.params.get("max")
                
                numeric_value = float(value)
                
                if min_val is not None and numeric_value < min_val:
                    return {
                        "field": field,
                        "value": value,
                        "message": f"{field}: {rule.message} (value: {value} < {min_val})",
                        "rule": rule.rule_type
                    }
                
                if max_val is not None and numeric_value > max_val:
                    return {
                        "field": field,
                        "value": value,
                        "message": f"{field}: {rule.message} (value: {value} > {max_val})",
                        "rule": rule.rule_type
                    }
            
            elif rule.rule_type == "regex":
                pattern = rule.params.get("pattern")
                if pattern and not re.match(pattern, str(value)):
                    return {
                        "field": field,
                        "value": value,
                        "message": f"{field}: {rule.message}",
                        "rule": rule.rule_type
                    }
            
            elif rule.rule_type == "type":
                expected_type = rule.params.get("type")
                if not isinstance(value, expected_type):
                    return {
                        "field": field,
                        "value": value,
                        "message": f"{field}: Expected {expected_type.__name__}, got {type(value).__name__}",
                        "rule": rule.rule_type
                    }
            
            elif rule.rule_type == "custom":
                validator_name = rule.params.get("validator")
                if validator_name in self.custom_validators:
                    if not self.custom_validators[validator_name](value):
                        return {
                            "field": field,
                            "value": value,
                            "message": f"{field}: {rule.message}",
                            "rule": rule.rule_type
                        }
        
        except Exception as e:
            logger.debug(f"Rule application error for {field}: {e}")
        
        return None
    
    def _validate_schema(self, data: Dict, schema: Dict) -> List[Dict]:
        """Validate data against schema"""
        errors = []
        
        for field, field_def in schema.items():
            # Check required fields
            if field_def.get("required", False) and field not in data:
                errors.append({
                    "field": field,
                    "message": f"Required field '{field}' is missing",
                    "rule": "schema"
                })
                continue
            
            if field in data:
                value = data[field]
                
                # Type validation
                expected_type = field_def.get("type")
                if expected_type and not isinstance(value, expected_type):
                    errors.append({
                        "field": field,
                        "value": value,
                        "message": f"Type mismatch: expected {expected_type.__name__}",
                        "rule": "schema"
                    })
                
                # Enum validation
                if "enum" in field_def and value not in field_def["enum"]:
                    errors.append({
                        "field": field,
                        "value": value,
                        "message": f"Value must be one of {field_def['enum']}",
                        "rule": "schema"
                    })
        
        return errors
    
    def _check_consistency(self, data: Dict) -> List[Dict]:
        """Check data consistency"""
        issues = []
        
        # Check price/market cap consistency
        if all(k in data for k in ['price', 'total_supply', 'market_cap']):
            calculated_mcap = data['price'] * data['total_supply']
            if abs(calculated_mcap - data['market_cap']) / data['market_cap'] > 0.1:
                issues.append({
                    "field": "market_cap",
                    "message": "Market cap inconsistent with price and supply",
                    "rule": "consistency"
                })
        
        # Check volume/liquidity ratio
        if all(k in data for k in ['volume_24h', 'liquidity']):
            if data['liquidity'] > 0:
                volume_liquidity_ratio = data['volume_24h'] / data['liquidity']
                if volume_liquidity_ratio > 10:
                    issues.append({
                        "field": "volume",
                        "message": "Volume unusually high relative to liquidity",
                        "rule": "consistency"
                    })
        
        # Check holder/supply ratio
        if all(k in data for k in ['holders', 'total_supply']):
            if data['holders'] > 0:
                avg_holding = data['total_supply'] / data['holders']
                if avg_holding < 1:
                    issues.append({
                        "field": "holders",
                        "message": "Average holding per wallet is suspiciously low",
                        "rule": "consistency"
                    })
        
        return issues
    
    def _check_required_fields(self, data: Dict) -> List[Dict]:
        """Check for required fields"""
        errors = []
        required_fields = ['token_address', 'chain', 'price']
        
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append({
                    "field": field,
                    "message": f"Required field '{field}' is missing or null",
                    "rule": "required"
                })
        
        return errors
    
    def _determine_field_type(self, field: str) -> Optional[DataField]:
        """Determine the type of a field"""
        field_lower = field.lower()
        
        if 'price' in field_lower:
            return DataField.PRICE
        elif 'volume' in field_lower:
            return DataField.VOLUME
        elif 'liquidity' in field_lower or 'liq' in field_lower:
            return DataField.LIQUIDITY
        elif 'address' in field_lower:
            return DataField.ADDRESS
        elif 'symbol' in field_lower or 'ticker' in field_lower:
            return DataField.SYMBOL
        elif 'time' in field_lower or 'date' in field_lower:
            return DataField.TIMESTAMP
        elif 'percent' in field_lower or 'pct' in field_lower:
            return DataField.PERCENTAGE
        
        return None
    
    def _validate_price(self, value: Any) -> Optional[Dict]:
        """Validate price field"""
        try:
            price = float(value)
            
            if price < 0:
                return {"field": "price", "message": "Price cannot be negative"}
            
            if price > 1e12:
                return {"field": "price", "message": "Price unrealistically high"}
            
            # Check for suspicious precision (possible float errors)
            if len(str(price).split('.')[-1]) > 18:
                return {"field": "price", "message": "Price has excessive decimal places"}
            
        except (ValueError, TypeError):
            return {"field": "price", "message": "Price must be numeric"}
        
        return None
    
    def _validate_volume(self, value: Any) -> Optional[Dict]:
        """Validate volume field"""
        try:
            volume = float(value)
            
            if volume < 0:
                return {"field": "volume", "message": "Volume cannot be negative"}
            
            if volume > 1e15:
                return {"field": "volume", "message": "Volume unrealistically high"}
            
        except (ValueError, TypeError):
            return {"field": "volume", "message": "Volume must be numeric"}
        
        return None
    
    def _validate_liquidity(self, value: Any) -> Optional[Dict]:
        """Validate liquidity field"""
        try:
            liquidity = float(value)
            
            if liquidity < 0:
                return {"field": "liquidity", "message": "Liquidity cannot be negative"}
            
            if liquidity < 1000:
                return {"field": "liquidity", "message": "Liquidity dangerously low"}
            
        except (ValueError, TypeError):
            return {"field": "liquidity", "message": "Liquidity must be numeric"}
        
        return None
    
    def _validate_address(self, value: Any) -> Optional[Dict]:
        """Validate blockchain address"""
        if not isinstance(value, str):
            return {"field": "address", "message": "Address must be string"}
        
        # Ethereum-style address validation
        if value.startswith('0x'):
            if len(value) != 42:
                return {"field": "address", "message": "Invalid address length"}
            
            if not all(c in '0123456789abcdefABCDEF' for c in value[2:]):
                return {"field": "address", "message": "Invalid address characters"}
        
        return None
    
    def _validate_symbol(self, value: Any) -> Optional[Dict]:
        """Validate token symbol"""
        if not isinstance(value, str):
            return {"field": "symbol", "message": "Symbol must be string"}
        
        if len(value) > 20:
            return {"field": "symbol", "message": "Symbol too long"}
        
        if not value.replace('-', '').replace('_', '').isalnum():
            return {"field": "symbol", "message": "Symbol contains invalid characters"}
        
        return None
    
    def _validate_timestamp(self, value: Any) -> Optional[Dict]:
        """Validate timestamp"""
        try:
            if isinstance(value, (int, float)):
                # Unix timestamp validation
                if value < 0:
                    return {"field": "timestamp", "message": "Timestamp cannot be negative"}
                
                # Check if timestamp is reasonable (not too far in past or future)
                current_time = datetime.now().timestamp()
                if value > current_time + 86400:  # More than 1 day in future
                    return {"field": "timestamp", "message": "Timestamp is in the future"}
                
                if value < current_time - 31536000 * 10:  # More than 10 years old
                    return {"field": "timestamp", "message": "Timestamp is too old"}
                    
            elif isinstance(value, str):
                # Try to parse string timestamp
                datetime.fromisoformat(value)
            
            elif not isinstance(value, datetime):
                return {"field": "timestamp", "message": "Invalid timestamp format"}
                
        except Exception:
            return {"field": "timestamp", "message": "Invalid timestamp"}
        
        return None
    
    def _validate_percentage(self, value: Any) -> Optional[Dict]:
        """Validate percentage field"""
        try:
            percentage = float(value)
            
            if percentage < 0:
                return {"field": "percentage", "message": "Percentage cannot be negative"}
            
            if percentage > 100:
                return {"field": "percentage", "message": "Percentage cannot exceed 100"}
            
        except (ValueError, TypeError):
            return {"field": "percentage", "message": "Percentage must be numeric"}
        
        return None
    
    def _calculate_quality_score(
        self,
        errors: List[Dict],
        warnings: List[Dict],
        info: List[Dict]
    ) -> float:
        """Calculate overall data quality score"""
        # Start with perfect score
        score = 100.0
        
        # Deduct for errors (critical issues)
        score -= len(errors) * 20
        
        # Deduct for warnings
        score -= len(warnings) * 5
        
        # Deduct for info issues
        score -= len(info) * 1
        
        # Ensure score is between 0 and 100
        return max(0, min(100, score))
    
    def _generate_suggestions(
        self,
        errors: List[Dict],
        warnings: List[Dict]
    ) -> List[str]:
        """Generate suggestions for fixing issues"""
        suggestions = []
        
        # Analyze errors
        error_fields = set(e.get('field') for e in errors)
        warning_fields = set(w.get('field') for w in warnings)
        
        if 'token_address' in error_fields:
            suggestions.append("Verify token address format (0x + 40 hex characters)")
        
        if 'price' in error_fields:
            suggestions.append("Check price data source and ensure positive numeric value")
        
        if 'liquidity' in warning_fields:
            suggestions.append("Low liquidity detected - consider minimum liquidity thresholds")
        
        if 'volume' in warning_fields:
            suggestions.append("Verify volume data - check for wash trading")
        
        # General suggestions based on patterns
        if len(errors) > 5:
            suggestions.append("Multiple critical errors - verify data source reliability")
        
        if len(warnings) > 10:
            suggestions.append("Many warnings detected - consider data cleaning pipeline")
        
        return suggestions
    
    def validate_batch(
        self,
        data_list: List[Dict[str, Any]],
        schema: Optional[Dict] = None
    ) -> Tuple[List[ValidationResult], Dict[str, Any]]:
        """
        Validate a batch of data records
        
        Args:
            data_list: List of data records
            schema: Optional schema definition
            
        Returns:
            Tuple of (list of results, summary statistics)
        """
        results = []
        
        for data in data_list:
            result = self.validate(data, schema)
            results.append(result)
        
        # Calculate summary statistics
        summary = {
            'total': len(results),
            'valid': sum(1 for r in results if r.is_valid),
            'invalid': sum(1 for r in results if not r.is_valid),
            'average_quality': sum(r.quality_score for r in results) / len(results) if results else 0,
            'common_errors': self._summarize_errors(results),
            'common_warnings': self._summarize_warnings(results)
        }
        
        return results, summary
    
    def _summarize_errors(self, results: List[ValidationResult]) -> Dict[str, int]:
        """Summarize common errors across results"""
        error_counts = {}
        
        for result in results:
            for error in result.errors:
                field = error.get('field', 'unknown')
                error_counts[field] = error_counts.get(field, 0) + 1
        
        return error_counts
    
    def _summarize_warnings(self, results: List[ValidationResult]) -> Dict[str, int]:
        """Summarize common warnings across results"""
        warning_counts = {}
        
        for result in results:
            for warning in result.warnings:
                field = warning.get('field', 'unknown')
                warning_counts[field] = warning_counts.get(field, 0) + 1
        
        return warning_counts
    
    def register_custom_validator(
        self,
        name: str,
        validator_func: callable
    ) -> None:
        """
        Register a custom validation function
        
        Args:
            name: Validator name
            validator_func: Function that returns True if valid
        """
        self.custom_validators[name] = validator_func
        logger.info(f"Registered custom validator: {name}")
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a new validation rule"""
        self.rules.append(rule)
        logger.info(f"Added validation rule for field: {rule.field}")
    
    def remove_rule(self, field: str, rule_type: str) -> None:
        """Remove a validation rule"""
        self.rules = [
            r for r in self.rules
            if not (r.field == field and r.rule_type == rule_type)
        ]
        logger.info(f"Removed validation rule: {field} - {rule_type}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            'total_validated': self.validation_stats['total_validated'],
            'total_passed': self.validation_stats['total_passed'],
            'total_failed': self.validation_stats['total_failed'],
            'pass_rate': self.validation_stats['total_passed'] / max(1, self.validation_stats['total_validated']),
            'common_errors': dict(sorted(
                self.validation_stats['common_errors'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])  # Top 10 common errors
        }
    
    def reset_stats(self) -> None:
        """Reset validation statistics"""
        self.validation_stats = {
            'total_validated': 0,
            'total_passed': 0,
            'total_failed': 0,
            'common_errors': {}
        }
        logger.info("Validation statistics reset")
    
    def export_rules(self) -> List[Dict[str, Any]]:
        """Export validation rules for documentation"""
        return [
            {
                'field': rule.field,
                'type': rule.rule_type,
                'params': rule.params,
                'level': rule.level.value,
                'message': rule.message
            }
            for rule in self.rules
        ]
    
    def import_rules(self, rules_data: List[Dict[str, Any]]) -> None:
        """Import validation rules from configuration"""
        for rule_data in rules_data:
            rule = ValidationRule(
                field=rule_data['field'],
                rule_type=rule_data['type'],
                params=rule_data['params'],
                level=ValidationLevel(rule_data['level']),
                message=rule_data['message']
            )
            self.add_rule(rule)
        
        logger.info(f"Imported {len(rules_data)} validation rules")