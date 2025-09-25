# tests/fixtures/mock_data.py
"""
Mock data generators for testing
"""
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List
import random
import string

class MockDataGenerator:
    """Generate mock data for testing"""
    
    @staticmethod
    def generate_token_address() -> str:
        """Generate random token address"""
        return "0x" + "".join(random.choices("0123456789abcdef", k=40))
    
    @staticmethod
    def generate_token(
        is_rug: bool = False,
        is_honeypot: bool = False,
        liquidity: Decimal = None
    ) -> Dict:
        """Generate mock token data"""
        address = MockDataGenerator.generate_token_address()
        
        if liquidity is None:
            liquidity = Decimal(random.randint(1000, 1000000))
        
        return {
            "address": address,
            "name": f"Token{random.randint(1, 9999)}",
            "symbol": "".join(random.choices(string.ascii_uppercase, k=4)),
            "decimals": 18,
            "total_supply": str(random.randint(1000000, 1000000000)),
            "liquidity": str(liquidity),
            "volume24h": str(liquidity * Decimal(random.uniform(0.1, 2))),
            "price": str(Decimal(random.uniform(0.0001, 10))),
            "holders": 10 if is_rug else random.randint(50, 5000),
            "contract_verified": not is_rug,
            "liquidity_locked": not is_rug and not is_honeypot,
            "dev_holdings_pct": 80 if is_rug else random.randint(1, 30),
            "buy_tax": 50 if is_honeypot else random.randint(0, 5),
            "sell_tax": 99 if is_honeypot else random.randint(0, 5),
            "is_honeypot": is_honeypot,
            "is_rug": is_rug
        }
    
    @staticmethod
    def generate_price_history(
        days: int = 30,
        interval: str = "1h",
        include_pump: bool = False
    ) -> List[Dict]:
        """Generate price history data"""
        history = []
        base_price = Decimal("0.001")
        current_time = datetime.now() - timedelta(days=days)
        
        intervals = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1)
        }
        
        delta = intervals.get(interval, timedelta(hours=1))
        num_points = int(timedelta(days=days) / delta)
        
        for i in range(num_points):
            # Add pump pattern if requested
            if include_pump and num_points // 3 <= i <= num_points // 2:
                price_change = Decimal(1 + (i - num_points // 3) * 0.05)
            else:
                price_change = Decimal(1 + random.uniform(-0.05, 0.05))
            
            price = base_price * price_change
            volume = Decimal(random.randint(1000, 100000))
            
            history.append({
                "timestamp": current_time,
                "open": price * Decimal(0.99),
                "high": price * Decimal(1.02),
                "low": price * Decimal(0.98),
                "close": price,
                "volume": volume
            })
            
            current_time += delta
        
        return history
    
    @staticmethod
    def generate_trades(count: int = 100) -> List[Dict]:
        """Generate trade history"""
        trades = []
        base_time = datetime.now()
        
        for i in range(count):
            trades.append({
                "id": f"trade_{i}",
                "timestamp": base_time - timedelta(minutes=i),
                "type": random.choice(["buy", "sell"]),
                "token": MockDataGenerator.generate_token_address(),
                "amount": Decimal(random.randint(100, 10000)),
                "price": Decimal(random.uniform(0.0001, 1)),
                "gas_price": random.randint(20, 200),
                "gas_used": random.randint(100000, 300000),
                "status": "completed",
                "tx_hash": "0x" + "".join(random.choices("0123456789abcdef", k=64))
            })
        
        return trades
    
    @staticmethod
    def generate_whale_movement() -> Dict:
        """Generate whale movement data"""
        return {
            "wallet": "0x" + "".join(random.choices("0123456789abcdef", k=40)),
            "token": MockDataGenerator.generate_token_address(),
            "type": random.choice(["buy", "sell", "transfer"]),
            "amount": Decimal(random.randint(100000, 10000000)),
            "timestamp": datetime.now(),
            "impact": random.choice(["high", "medium", "low"]),
            "exchange": random.choice(["uniswap", "sushiswap", "pancakeswap"])
        }
    
    @staticmethod
    def generate_market_conditions() -> Dict:
        """Generate market condition data"""
        return {
            "timestamp": datetime.now(),
            "market_trend": random.choice(["bullish", "bearish", "neutral"]),
            "volatility": random.choice(["low", "medium", "high"]),
            "volume_trend": random.choice(["increasing", "decreasing", "stable"]),
            "fear_greed_index": random.randint(0, 100),
            "btc_dominance": Decimal(random.uniform(40, 60)),
            "alt_season_index": random.randint(0, 100)
        }
