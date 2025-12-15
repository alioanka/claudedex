from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    trade_id = Column(String, unique=True, nullable=False)
    token_address = Column(String, nullable=False)
    chain = Column(String, nullable=False)
    side = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    amount = Column(Float, nullable=False)
    usd_value = Column(Float, nullable=False)
    gas_fee = Column(Float)
    slippage = Column(Float)
    profit_loss = Column(Float)
    profit_loss_percentage = Column(Float)
    strategy = Column(String)
    risk_score = Column(Float)
    ml_confidence = Column(Float)
    entry_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    exit_timestamp = Column(DateTime(timezone=True))
    status = Column(String, default='open')
    json_metadata = Column(JSON)

class FuturesTrade(Base):
    __tablename__ = 'futures_trades'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    size = Column(Float, nullable=False)
    notional_value = Column(Float)
    leverage = Column(Integer)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    fees = Column(Float)
    net_pnl = Column(Float)
    exit_reason = Column(String)
    entry_time = Column(DateTime(timezone=True), server_default=func.now())
    exit_time = Column(DateTime(timezone=True))
    duration_seconds = Column(Integer)
    is_simulated = Column(Boolean, default=True)
    exchange = Column(String)
    network = Column(String)

class SolanaTrade(Base):
    __tablename__ = 'solana_trades'

    id = Column(Integer, primary_key=True)
    trade_id = Column(String, unique=True)
    token_symbol = Column(String, nullable=False)
    token_mint = Column(String, nullable=False)
    strategy = Column(String, nullable=False)
    side = Column(String, default='long', nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    amount_sol = Column(Float, nullable=False)
    amount_tokens = Column(Float)
    pnl_sol = Column(Float)
    pnl_usd = Column(Float)
    pnl_pct = Column(Float)
    fees_sol = Column(Float, default=0)
    exit_reason = Column(String)
    entry_time = Column(DateTime(timezone=True), nullable=False)
    exit_time = Column(DateTime(timezone=True))
    duration_seconds = Column(Integer, default=0)
    is_simulated = Column(Boolean, default=True, nullable=False)
    sol_price_usd = Column(Float)
    json_metadata = Column(JSON)
    status = Column(String, default='closed')
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ConfigSetting(Base):
    __tablename__ = 'config_settings'

    id = Column(Integer, primary_key=True)
    config_type = Column(String, nullable=False)
    key = Column(String, nullable=False)
    value = Column(String, nullable=False)
    value_type = Column(String, default='string', nullable=False)
    is_editable = Column(Boolean, default=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    email = Column(String, unique=True)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
