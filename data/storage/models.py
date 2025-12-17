# data/storage/models.py

from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, Numeric, DateTime, Boolean,
    Text, ForeignKey, Index, UniqueConstraint, CheckConstraint,
    JSON, ARRAY, Float
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP
import uuid

Base = declarative_base()


class TradeSide(Enum):
    """Trade side enumeration."""
    BUY = "buy"
    SELL = "sell"


class TradeStatus(Enum):
    """Trade status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class PositionStatus(Enum):
    """Position status enumeration."""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Trade(Base):
    """Trade record model."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(100), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Token information
    token_address = Column(String(100), nullable=False, index=True)
    chain = Column(String(50), nullable=False)
    token_symbol = Column(String(20))
    token_name = Column(String(100))
    
    # Trade details
    side = Column(String(10), nullable=False)
    entry_price = Column(Numeric(30, 18), nullable=False)
    exit_price = Column(Numeric(30, 18))
    amount = Column(Numeric(30, 18), nullable=False)
    usd_value = Column(Numeric(20, 2), nullable=False)
    
    # Costs and fees
    gas_fee = Column(Numeric(20, 8))
    slippage = Column(Numeric(5, 4))
    total_fees = Column(Numeric(20, 8))
    
    # Profit/Loss
    profit_loss = Column(Numeric(20, 8))
    profit_loss_percentage = Column(Numeric(10, 4))
    realized_pnl = Column(Numeric(20, 8))
    
    # Strategy and scoring
    strategy = Column(String(50), nullable=False, index=True)
    risk_score = Column(Numeric(5, 4))
    ml_confidence = Column(Numeric(5, 4))
    entry_score = Column(Numeric(5, 4))
    
    # Timing
    entry_timestamp = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    exit_timestamp = Column(TIMESTAMP(timezone=True))
    hold_duration_seconds = Column(Integer)
    
    # Status
    status = Column(String(20), nullable=False, default='open', index=True)
    exit_reason = Column(String(100))
    
    # Additional data
    metadata = Column(JSONB)
    notes = Column(Text)
    
    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_trades_token_chain', 'token_address', 'chain'),
        Index('idx_trades_entry_exit', 'entry_timestamp', 'exit_timestamp'),
        Index('idx_trades_pnl', 'profit_loss_percentage'),
        CheckConstraint('amount > 0', name='check_positive_amount'),
        CheckConstraint('usd_value > 0', name='check_positive_usd_value'),
    )
    
    @validates('side')
    def validate_side(self, key, value):
        if value not in ['buy', 'sell']:
            raise ValueError(f"Invalid trade side: {value}")
        return value
    
    @validates('status')
    def validate_status(self, key, value):
        valid_statuses = ['pending', 'open', 'closed', 'cancelled', 'failed']
        if value not in valid_statuses:
            raise ValueError(f"Invalid trade status: {value}")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'trade_id': self.trade_id,
            'token_address': self.token_address,
            'chain': self.chain,
            'side': self.side,
            'entry_price': float(self.entry_price) if self.entry_price else None,
            'exit_price': float(self.exit_price) if self.exit_price else None,
            'amount': float(self.amount) if self.amount else None,
            'usd_value': float(self.usd_value) if self.usd_value else None,
            'profit_loss': float(self.profit_loss) if self.profit_loss else None,
            'profit_loss_percentage': float(self.profit_loss_percentage) if self.profit_loss_percentage else None,
            'strategy': self.strategy,
            'status': self.status,
            'entry_timestamp': self.entry_timestamp.isoformat() if self.entry_timestamp else None,
            'exit_timestamp': self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            'metadata': self.metadata
        }


class Position(Base):
    """Active position model."""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(String(100), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Token information
    token_address = Column(String(100), nullable=False, index=True)
    chain = Column(String(50), nullable=False)
    token_symbol = Column(String(20))
    token_name = Column(String(100))
    
    # Position details
    entry_price = Column(Numeric(30, 18), nullable=False)
    current_price = Column(Numeric(30, 18))
    amount = Column(Numeric(30, 18), nullable=False)
    usd_value = Column(Numeric(20, 2), nullable=False)
    
    # Risk management
    stop_loss = Column(Numeric(30, 18))
    take_profit = Column(JSONB)  # Array of take profit levels
    trailing_stop = Column(Numeric(5, 4))
    max_position_size = Column(Numeric(20, 8))
    
    # P&L tracking
    unrealized_pnl = Column(Numeric(20, 8))
    unrealized_pnl_percentage = Column(Numeric(10, 4))
    peak_pnl = Column(Numeric(20, 8))
    trough_pnl = Column(Numeric(20, 8))
    
    # Risk scores
    risk_score = Column(Numeric(5, 4))
    correlation_score = Column(Numeric(5, 4))
    volatility_score = Column(Numeric(5, 4))
    liquidity_score = Column(Numeric(5, 4))
    
    # Timing
    opened_at = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    last_checked = Column(TIMESTAMP(timezone=True))
    last_updated = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    # Status
    status = Column(String(20), nullable=False, default='open', index=True)
    health_status = Column(String(20), default='healthy')
    
    # Strategy
    strategy = Column(String(50), nullable=False)
    entry_signals = Column(JSONB)
    
    # Additional data
    metadata = Column(JSONB)
    alerts_triggered = Column(JSONB)
    
    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    trades = relationship("Trade", backref="position", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_positions_token_chain', 'token_address', 'chain'),
        Index('idx_positions_pnl', 'unrealized_pnl_percentage'),
        CheckConstraint('amount > 0', name='check_positive_position_amount'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'position_id': self.position_id,
            'token_address': self.token_address,
            'chain': self.chain,
            'entry_price': float(self.entry_price) if self.entry_price else None,
            'current_price': float(self.current_price) if self.current_price else None,
            'amount': float(self.amount) if self.amount else None,
            'unrealized_pnl': float(self.unrealized_pnl) if self.unrealized_pnl else None,
            'unrealized_pnl_percentage': float(self.unrealized_pnl_percentage) if self.unrealized_pnl_percentage else None,
            'stop_loss': float(self.stop_loss) if self.stop_loss else None,
            'take_profit': self.take_profit,
            'status': self.status,
            'opened_at': self.opened_at.isoformat() if self.opened_at else None,
            'metadata': self.metadata
        }


class MarketData(Base):
    """Market data time-series model."""
    __tablename__ = 'market_data'
    
    # Composite primary key for time-series data
    time = Column(TIMESTAMP(timezone=True), nullable=False, primary_key=True)
    token_address = Column(String(100), nullable=False, primary_key=True)
    chain = Column(String(50), nullable=False, primary_key=True)
    
    # Price data
    price = Column(Numeric(30, 18), nullable=False)
    price_btc = Column(Numeric(30, 18))
    price_eth = Column(Numeric(30, 18))
    
    # Volume data
    volume_24h = Column(Numeric(30, 18))
    volume_5m = Column(Numeric(30, 18))
    volume_1h = Column(Numeric(30, 18))
    buy_volume_5m = Column(Numeric(30, 18))
    sell_volume_5m = Column(Numeric(30, 18))
    
    # Liquidity
    liquidity_usd = Column(Numeric(20, 2))
    liquidity_base = Column(Numeric(30, 18))
    liquidity_quote = Column(Numeric(30, 18))
    
    # Market metrics
    market_cap = Column(Numeric(20, 2))
    fdv = Column(Numeric(20, 2))  # Fully diluted valuation
    circulating_supply = Column(Numeric(30, 0))
    total_supply = Column(Numeric(30, 0))
    
    # Holder data
    holders = Column(Integer)
    unique_buyers_5m = Column(Integer)
    unique_sellers_5m = Column(Integer)
    buy_count_5m = Column(Integer)
    sell_count_5m = Column(Integer)
    
    # Price changes
    price_change_5m = Column(Numeric(10, 4))
    price_change_15m = Column(Numeric(10, 4))
    price_change_30m = Column(Numeric(10, 4))
    price_change_1h = Column(Numeric(10, 4))
    price_change_4h = Column(Numeric(10, 4))
    price_change_24h = Column(Numeric(10, 4))
    
    # Technical indicators (optional, calculated)
    rsi_14 = Column(Numeric(5, 2))
    volatility_24h = Column(Numeric(10, 4))
    
    # Additional data
    metadata = Column(JSONB)
    
    __table_args__ = (
        Index('idx_market_data_token', 'token_address'),
        Index('idx_market_data_time', 'time'),
        Index('idx_market_data_chain_time', 'chain', 'time'),
    )


class Alert(Base):
    """Alert/notification model."""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(100), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Alert details
    alert_type = Column(String(50), nullable=False, index=True)
    priority = Column(String(20), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    message = Column(Text)
    
    # Related entities
    trade_id = Column(String(100))
    position_id = Column(String(100))
    token_address = Column(String(100))
    
    # Alert data
    data = Column(JSONB)
    metrics = Column(JSONB)
    
    # Delivery
    channels = Column(JSONB)  # List of channels alert was sent to
    sent_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    delivered = Column(Boolean, default=False)
    delivery_errors = Column(JSONB)
    
    # Acknowledgment
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(TIMESTAMP(timezone=True))
    acknowledged_by = Column(String(100))
    
    # Action taken
    action_required = Column(Boolean, default=False)
    action_taken = Column(Text)
    action_timestamp = Column(TIMESTAMP(timezone=True))
    
    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_alerts_sent_at', 'sent_at'),
        Index('idx_alerts_acknowledged', 'acknowledged'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'priority': self.priority,
            'title': self.title,
            'message': self.message,
            'data': self.data,
            'channels': self.channels,
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'acknowledged': self.acknowledged
        }


class TokenAnalysis(Base):
    """Token analysis results model."""
    __tablename__ = 'token_analysis'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Token identification
    token_address = Column(String(100), nullable=False, index=True)
    chain = Column(String(50), nullable=False)
    token_symbol = Column(String(20))
    token_name = Column(String(100))
    
    # Analysis timestamp
    analysis_timestamp = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    analysis_version = Column(String(20))
    
    # Risk scores (0-1 scale, higher is riskier)
    risk_score = Column(Numeric(5, 4))
    honeypot_risk = Column(Numeric(5, 4))
    rug_probability = Column(Numeric(5, 4))
    pump_probability = Column(Numeric(5, 4))
    
    # Component scores (0-1 scale, higher is better)
    liquidity_score = Column(Numeric(5, 4))
    holder_score = Column(Numeric(5, 4))
    contract_score = Column(Numeric(5, 4))
    developer_score = Column(Numeric(5, 4))
    social_score = Column(Numeric(5, 4))
    technical_score = Column(Numeric(5, 4))
    volume_score = Column(Numeric(5, 4))
    market_score = Column(Numeric(5, 4))
    
    # Detailed analysis
    honeypot_checks = Column(JSONB)  # Results from multiple honeypot APIs
    contract_analysis = Column(JSONB)  # Contract verification results
    holder_analysis = Column(JSONB)  # Holder distribution analysis
    liquidity_analysis = Column(JSONB)  # Liquidity pool analysis
    developer_analysis = Column(JSONB)  # Developer wallet analysis
    social_metrics = Column(JSONB)  # Social media metrics
    technical_indicators = Column(JSONB)  # Technical analysis results
    
    # Flags and warnings
    is_honeypot = Column(Boolean, default=False)
    is_rugpull = Column(Boolean, default=False)
    has_mint_function = Column(Boolean)
    has_blacklist = Column(Boolean)
    has_hidden_owner = Column(Boolean)
    liquidity_locked = Column(Boolean)
    contract_verified = Column(Boolean)
    
    # ML predictions
    ml_pump_confidence = Column(Numeric(5, 4))
    ml_dump_confidence = Column(Numeric(5, 4))
    ml_rug_confidence = Column(Numeric(5, 4))
    ml_signals = Column(JSONB)
    
    # Metadata
    metadata = Column(JSONB)
    warnings = Column(JSONB)
    recommendations = Column(JSONB)
    
    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    __table_args__ = (
        UniqueConstraint('token_address', 'chain', 'analysis_timestamp', 
                        name='uix_token_analysis'),
        Index('idx_token_analysis_risk', 'risk_score'),
        Index('idx_token_analysis_ml', 'ml_pump_confidence', 'ml_rug_confidence'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'token_address': self.token_address,
            'chain': self.chain,
            'analysis_timestamp': self.analysis_timestamp.isoformat() if self.analysis_timestamp else None,
            'risk_score': float(self.risk_score) if self.risk_score else None,
            'honeypot_risk': float(self.honeypot_risk) if self.honeypot_risk else None,
            'rug_probability': float(self.rug_probability) if self.rug_probability else None,
            'pump_probability': float(self.pump_probability) if self.pump_probability else None,
            'liquidity_score': float(self.liquidity_score) if self.liquidity_score else None,
            'holder_score': float(self.holder_score) if self.holder_score else None,
            'contract_score': float(self.contract_score) if self.contract_score else None,
            'warnings': self.warnings,
            'metadata': self.metadata
        }


class PerformanceMetrics(Base):
    """Performance metrics snapshot model."""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Period information
    period = Column(String(20), nullable=False, index=True)  # hourly, daily, weekly, monthly
    start_date = Column(TIMESTAMP(timezone=True), nullable=False)
    end_date = Column(TIMESTAMP(timezone=True), nullable=False)
    
    # Trade statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    breakeven_trades = Column(Integer, default=0)
    
    # Volume statistics
    total_volume = Column(Numeric(20, 2))
    buy_volume = Column(Numeric(20, 2))
    sell_volume = Column(Numeric(20, 2))
    
    # P&L metrics
    total_pnl = Column(Numeric(20, 8))
    total_pnl_percentage = Column(Numeric(10, 4))
    gross_profit = Column(Numeric(20, 8))
    gross_loss = Column(Numeric(20, 8))
    total_fees = Column(Numeric(20, 8))
    net_pnl = Column(Numeric(20, 8))
    
    # Risk metrics
    sharpe_ratio = Column(Numeric(10, 4))
    sortino_ratio = Column(Numeric(10, 4))
    calmar_ratio = Column(Numeric(10, 4))
    max_drawdown = Column(Numeric(10, 4))
    max_drawdown_duration_hours = Column(Integer)
    var_95 = Column(Numeric(20, 8))  # Value at Risk 95%
    cvar_95 = Column(Numeric(20, 8))  # Conditional VaR 95%
    
    # Trade quality metrics
    win_rate = Column(Numeric(5, 4))
    profit_factor = Column(Numeric(10, 4))
    avg_win = Column(Numeric(20, 8))
    avg_loss = Column(Numeric(20, 8))
    avg_win_percentage = Column(Numeric(10, 4))
    avg_loss_percentage = Column(Numeric(10, 4))
    best_trade = Column(Numeric(20, 8))
    worst_trade = Column(Numeric(20, 8))
    avg_trade_duration_minutes = Column(Integer)
    
    # Strategy breakdown
    strategy_performance = Column(JSONB)  # Performance by strategy
    token_performance = Column(JSONB)  # Performance by token
    chain_performance = Column(JSONB)  # Performance by chain
    
    # Risk exposure
    avg_position_size = Column(Numeric(20, 8))
    max_position_size = Column(Numeric(20, 8))
    avg_risk_score = Column(Numeric(5, 4))
    correlation_score = Column(Numeric(5, 4))
    
    # Additional metrics
    metadata = Column(JSONB)
    anomalies = Column(JSONB)
    
    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_performance_period_dates', 'period', 'start_date', 'end_date'),
        UniqueConstraint('period', 'start_date', 'end_date', 
                        name='uix_performance_period'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'period': self.period,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'total_pnl': float(self.total_pnl) if self.total_pnl else None,
            'win_rate': float(self.win_rate) if self.win_rate else None,
            'sharpe_ratio': float(self.sharpe_ratio) if self.sharpe_ratio else None,
            'max_drawdown': float(self.max_drawdown) if self.max_drawdown else None
        }


class WhaleWallet(Base):
    """Whale wallet tracking model."""
    __tablename__ = 'whale_wallets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Wallet identification
    wallet_address = Column(String(100), nullable=False, unique=True)
    chain = Column(String(50), nullable=False)
    wallet_label = Column(String(100))  # Known wallet labels
    wallet_type = Column(String(50))  # whale, smart_money, institution, etc.
    
    # Wallet metrics
    total_value_usd = Column(Numeric(20, 2))
    pnl_30d = Column(Numeric(20, 8))
    win_rate = Column(Numeric(5, 4))
    avg_holding_time_hours = Column(Integer)
    total_trades_30d = Column(Integer)
    
    # Tracking status
    is_tracked = Column(Boolean, default=True)
    first_seen = Column(TIMESTAMP(timezone=True))
    last_activity = Column(TIMESTAMP(timezone=True))
    
    # Activity patterns
    activity_patterns = Column(JSONB)
    trading_patterns = Column(JSONB)
    favorite_tokens = Column(JSONB)
    
    # Metadata
    metadata = Column(JSONB)
    
    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_whale_wallets_chain', 'chain'),
        Index('idx_whale_wallets_value', 'total_value_usd'),
    )


class MEVTransaction(Base):
    """MEV transaction monitoring model."""
    __tablename__ = 'mev_transactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Transaction details
    tx_hash = Column(String(100), nullable=False, unique=True)
    block_number = Column(Integer, nullable=False)
    chain = Column(String(50), nullable=False)
    
    # MEV details
    mev_type = Column(String(50))  # sandwich, frontrun, backrun, arbitrage
    profit_eth = Column(Numeric(20, 8))
    profit_usd = Column(Numeric(20, 2))
    gas_used = Column(Integer)
    gas_price_gwei = Column(Numeric(10, 2))
    
    # Involved addresses
    bot_address = Column(String(100))
    victim_address = Column(String(100))
    token_address = Column(String(100))
    
    # Risk assessment
    sandwich_risk = Column(Numeric(5, 4))
    frontrun_risk = Column(Numeric(5, 4))
    
    # Metadata
    metadata = Column(JSONB)
    
    # Timestamps
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_mev_block', 'block_number'),
        Index('idx_mev_timestamp', 'timestamp'),
        Index('idx_mev_bot', 'bot_address'),
    )


class SentimentLog(Base):
    """AI Sentiment analysis logs."""
    __tablename__ = 'sentiment_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    score = Column(Float, nullable=False)
    source = Column(String(50))
    timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True)
    metadata = Column(JSONB)


class SystemLog(Base):
    """System event logging model."""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Log details
    log_level = Column(String(20), nullable=False)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    component = Column(String(50), nullable=False)  # Component that generated the log
    event_type = Column(String(50))
    message = Column(Text, nullable=False)
    
    # Related entities
    trade_id = Column(String(100))
    position_id = Column(String(100))
    token_address = Column(String(100))
    
    # Error tracking
    error_type = Column(String(100))
    error_message = Column(Text)
    stack_trace = Column(Text)
    
    # Context
    context = Column(JSONB)
    metrics = Column(JSONB)
    
    # Timestamps
    timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_logs_timestamp', 'timestamp'),
        Index('idx_logs_level', 'log_level'),
        Index('idx_logs_component', 'component'),
    )


# Utility functions for models
def create_all_tables(engine):
    """Create all database tables."""
    Base.metadata.create_all(engine)


def drop_all_tables(engine):
    """Drop all database tables."""
    Base.metadata.drop_all(engine)

Performance = PerformanceMetrics