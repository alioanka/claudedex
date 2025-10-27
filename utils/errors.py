"""
Typed Exception Classes for ClaudeDex Trading Bot

This module provides specific exception types for better error handling,
debugging, and circuit breaker logic throughout the trading system.
"""


# ============================================================================
# Network & API Exceptions
# ============================================================================

class NetworkError(Exception):
    """Base exception for network-related errors"""
    pass


class RPCError(NetworkError):
    """RPC endpoint failures"""
    pass


class QuoteError(NetworkError):
    """DEX quote retrieval failures"""
    pass


class APIRateLimitError(NetworkError):
    """API rate limit exceeded"""
    pass


# ============================================================================
# Trading & Execution Exceptions
# ============================================================================

class ExecutionError(Exception):
    """Base exception for trade execution errors"""
    pass


class InsufficientBalanceError(ExecutionError):
    """Wallet has insufficient balance"""
    pass


class SlippageExceededError(ExecutionError):
    """Trade slippage exceeded maximum threshold"""
    pass


class ConfirmationTimeout(ExecutionError):
    """Transaction confirmation timeout"""
    pass


class NonceError(ExecutionError):
    """Nonce management issues"""
    pass


# ============================================================================
# Contract & ABI Exceptions
# ============================================================================

class ContractError(Exception):
    """Base exception for smart contract errors"""
    pass


class ABIError(ContractError):
    """ABI parsing or encoding errors"""
    pass


class DecodeError(ContractError):
    """Data decoding errors"""
    pass


class HoneypotDetected(ContractError):
    """Contract identified as honeypot"""
    pass


# ============================================================================
# Configuration & Validation Exceptions
# ============================================================================

class ConfigurationError(Exception):
    """Configuration validation errors"""
    pass


class ValidationError(Exception):
    """Data validation errors"""
    pass


# ============================================================================
# Risk Management Exceptions
# ============================================================================

class RiskLimitError(Exception):
    """Risk limit exceeded"""
    pass


class CircuitBreakerTripped(Exception):
    """Circuit breaker activated"""
    pass


class PositionLimitError(RiskLimitError):
    """Position size or count limit exceeded"""
    pass


# ============================================================================
# Database Exceptions
# ============================================================================

class DatabaseError(Exception):
    """Database operation errors"""
    pass


class DataIntegrityError(DatabaseError):
    """Data integrity constraint violation"""
    pass


# ============================================================================
# ML & Analysis Exceptions
# ============================================================================

class AnalysisError(Exception):
    """Analysis or prediction errors"""
    pass


class ModelError(AnalysisError):
    """ML model inference errors"""
    pass