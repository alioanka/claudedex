# ClaudeDex Trading Bot - API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Core API Endpoints](#core-api-endpoints)
4. [WebSocket Connections](#websocket-connections)
5. [Data Models](#data-models)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Code Examples](#code-examples)

## Overview

The ClaudeDex Trading Bot provides a comprehensive REST API and WebSocket interface for automated trading operations, market analysis, and bot management.

### Base URLs
- **Production**: `https://api.claudedex.io/v1`
- **Development**: `http://localhost:8080/v1`
- **WebSocket**: `wss://ws.claudedex.io/v1`

### Response Format
All responses are in JSON format with the following structure:
```json
{
  "success": true,
  "data": {},
  "message": "Success",
  "timestamp": "2025-01-01T00:00:00Z"
}
```

## Authentication

### API Key Authentication
Include your API key in the request header:
```
X-API-Key: your_api_key_here
```

### JWT Authentication
For enhanced security, use JWT tokens:
```
Authorization: Bearer your_jwt_token_here
```

### Generating API Keys
```python
POST /auth/api-key
{
  "name": "My Trading Bot",
  "permissions": ["read", "trade", "manage"]
}
```

## Core API Endpoints

### 1. Trading Operations

#### Start Trading Bot
```http
POST /bot/start
```
**Request Body:**
```json
{
  "mode": "production",
  "chains": ["ethereum", "bsc"],
  "strategies": ["momentum", "scalping"],
  "risk_level": "medium"
}
```

#### Stop Trading Bot
```http
POST /bot/stop
```
**Request Body:**
```json
{
  "emergency": false,
  "close_positions": true
}
```

#### Get Bot Status
```http
GET /bot/status
```
**Response:**
```json
{
  "status": "running",
  "uptime": 3600,
  "active_positions": 5,
  "total_trades": 150,
  "profit_loss": 1250.50
}
```

### 2. Position Management

#### Get Active Positions
```http
GET /positions/active
```
**Query Parameters:**
- `chain`: Filter by blockchain (optional)
- `token`: Filter by token address (optional)
- `limit`: Number of results (default: 50)

**Response:**
```json
{
  "positions": [
    {
      "id": "pos_123",
      "token_address": "0x...",
      "chain": "ethereum",
      "entry_price": "0.0012",
      "current_price": "0.0015",
      "amount": "10000",
      "pnl": 250.00,
      "pnl_percentage": 25.0,
      "opened_at": "2025-01-01T00:00:00Z"
    }
  ]
}
```

#### Close Position
```http
POST /positions/{position_id}/close
```
**Request Body:**
```json
{
  "reason": "manual",
  "slippage_tolerance": 0.05
}
```

#### Set Stop Loss
```http
PUT /positions/{position_id}/stop-loss
```
**Request Body:**
```json
{
  "stop_loss_price": "0.0010",
  "trailing": true,
  "trailing_distance": 0.05
}
```

### 3. Market Analysis

#### Analyze Token
```http
POST /analysis/token
```
**Request Body:**
```json
{
  "token_address": "0x...",
  "chain": "ethereum",
  "deep_analysis": true
}
```

**Response:**
```json
{
  "token_info": {
    "name": "Example Token",
    "symbol": "EXT",
    "price": "0.0012",
    "market_cap": 1000000,
    "liquidity": 500000
  },
  "risk_analysis": {
    "overall_score": 75,
    "rug_risk": 10,
    "honeypot_risk": 0,
    "liquidity_risk": 15
  },
  "ml_predictions": {
    "pump_probability": 0.65,
    "price_24h": 0.0015,
    "confidence": 0.78
  },
  "recommendations": ["Buy", "Set tight stop loss"]
}
```

#### Get Market Overview
```http
GET /analysis/market
```
**Query Parameters:**
- `chain`: Blockchain network
- `timeframe`: 1h, 24h, 7d, 30d

**Response:**
```json
{
  "total_volume": 150000000,
  "active_pairs": 12500,
  "trending_tokens": [],
  "market_sentiment": "bullish",
  "fear_greed_index": 65
}
```

### 4. Historical Data

#### Get Trade History
```http
GET /trades/history
```
**Query Parameters:**
- `start_date`: ISO format date
- `end_date`: ISO format date
- `token`: Token address (optional)
- `status`: open, closed, all

**Response:**
```json
{
  "trades": [
    {
      "id": "trade_456",
      "token_address": "0x...",
      "side": "buy",
      "price": "0.0012",
      "amount": "10000",
      "timestamp": "2025-01-01T00:00:00Z",
      "profit_loss": 50.25
    }
  ],
  "total": 100,
  "page": 1
}
```

#### Get Performance Metrics
```http
GET /performance/metrics
```
**Query Parameters:**
- `period`: 24h, 7d, 30d, all

**Response:**
```json
{
  "total_trades": 500,
  "winning_trades": 300,
  "win_rate": 0.60,
  "total_profit": 5000.00,
  "total_loss": 2000.00,
  "net_profit": 3000.00,
  "sharpe_ratio": 1.85,
  "max_drawdown": 0.15,
  "roi": 0.30
}
```

### 5. Configuration

#### Get Configuration
```http
GET /config
```

#### Update Configuration
```http
PUT /config
```
**Request Body:**
```json
{
  "risk_management": {
    "max_position_size": 0.05,
    "stop_loss_default": 0.05,
    "max_daily_loss": 0.10
  },
  "trading": {
    "min_liquidity": 50000,
    "max_slippage": 0.05
  }
}
```

### 6. ML Models

#### Get Model Status
```http
GET /ml/models/status
```

#### Trigger Model Retraining
```http
POST /ml/models/retrain
```
**Request Body:**
```json
{
  "model": "rug_classifier",
  "use_latest_data": true
}
```

#### Get Predictions
```http
POST /ml/predict
```
**Request Body:**
```json
{
  "token_address": "0x...",
  "chain": "ethereum",
  "models": ["rug_classifier", "pump_predictor"]
}
```

## WebSocket Connections

### Connection
```javascript
const ws = new WebSocket('wss://ws.claudedex.io/v1');
ws.send(JSON.stringify({
  type: 'auth',
  api_key: 'your_api_key'
}));
```

### Subscriptions

#### Price Updates
```json
{
  "type": "subscribe",
  "channel": "prices",
  "params": {
    "tokens": ["0x..."],
    "chains": ["ethereum"]
  }
}
```

#### Trade Executions
```json
{
  "type": "subscribe",
  "channel": "trades"
}
```

#### Bot Status
```json
{
  "type": "subscribe",
  "channel": "bot_status"
}
```

### Message Format
```json
{
  "channel": "prices",
  "data": {
    "token_address": "0x...",
    "price": "0.0012",
    "change_24h": 0.15,
    "volume_24h": 150000
  },
  "timestamp": "2025-01-01T00:00:00Z"
}
```

## Data Models

### Position
```typescript
interface Position {
  id: string;
  token_address: string;
  chain: string;
  side: 'long' | 'short';
  entry_price: string;
  current_price: string;
  amount: string;
  value_usd: number;
  pnl: number;
  pnl_percentage: number;
  stop_loss?: string;
  take_profit?: string;
  opened_at: string;
  updated_at: string;
}
```

### Trade
```typescript
interface Trade {
  id: string;
  position_id?: string;
  token_address: string;
  chain: string;
  side: 'buy' | 'sell';
  price: string;
  amount: string;
  value_usd: number;
  gas_fee: string;
  slippage: number;
  status: 'pending' | 'completed' | 'failed';
  timestamp: string;
}
```

### Token Analysis
```typescript
interface TokenAnalysis {
  token_info: {
    address: string;
    name: string;
    symbol: string;
    decimals: number;
    total_supply: string;
  };
  market_data: {
    price: string;
    price_change_24h: number;
    volume_24h: string;
    market_cap: string;
    liquidity: string;
    holders: number;
  };
  risk_scores: {
    overall: number;
    rug_risk: number;
    honeypot_risk: number;
    liquidity_risk: number;
    developer_risk: number;
  };
  ml_predictions: {
    pump_probability: number;
    dump_probability: number;
    price_prediction_24h: string;
    confidence: number;
  };
}
```

## Error Handling

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "INSUFFICIENT_FUNDS",
    "message": "Insufficient balance for trade",
    "details": {
      "required": 1000,
      "available": 500
    }
  },
  "timestamp": "2025-01-01T00:00:00Z"
}
```

### Common Error Codes
- `AUTH_FAILED`: Authentication failed
- `INVALID_REQUEST`: Invalid request parameters
- `INSUFFICIENT_FUNDS`: Not enough balance
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `POSITION_NOT_FOUND`: Position doesn't exist
- `TOKEN_NOT_SUPPORTED`: Token not supported
- `SLIPPAGE_TOO_HIGH`: Slippage exceeds tolerance
- `HONEYPOT_DETECTED`: Token is a honeypot
- `MIN_LIQUIDITY_NOT_MET`: Liquidity too low
- `INTERNAL_ERROR`: Server error

## Rate Limiting

### Limits
- **Public Endpoints**: 100 requests/minute
- **Authenticated Endpoints**: 1000 requests/minute
- **Trading Endpoints**: 100 requests/minute
- **WebSocket Messages**: 100 messages/second

### Headers
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1640995200
```

## Code Examples

### Python
```python
import requests
import json

class ClaudeDexAPI:
    def __init__(self, api_key, base_url="https://api.claudedex.io/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}
    
    def start_bot(self, config):
        response = requests.post(
            f"{self.base_url}/bot/start",
            headers=self.headers,
            json=config
        )
        return response.json()
    
    def get_positions(self):
        response = requests.get(
            f"{self.base_url}/positions/active",
            headers=self.headers
        )
        return response.json()
    
    def analyze_token(self, token_address, chain):
        response = requests.post(
            f"{self.base_url}/analysis/token",
            headers=self.headers,
            json={
                "token_address": token_address,
                "chain": chain,
                "deep_analysis": True
            }
        )
        return response.json()

# Usage
api = ClaudeDexAPI("your_api_key")
api.start_bot({"mode": "production"})
positions = api.get_positions()
```

### JavaScript/TypeScript
```typescript
class ClaudeDexAPI {
  private apiKey: string;
  private baseUrl: string;

  constructor(apiKey: string, baseUrl = "https://api.claudedex.io/v1") {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  async startBot(config: any): Promise<any> {
    const response = await fetch(`${this.baseUrl}/bot/start`, {
      method: 'POST',
      headers: {
        'X-API-Key': this.apiKey,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(config)
    });
    return response.json();
  }

  async getPositions(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/positions/active`, {
      headers: {
        'X-API-Key': this.apiKey
      }
    });
    return response.json();
  }
}

// WebSocket connection
const ws = new WebSocket('wss://ws.claudedex.io/v1');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'auth',
    api_key: 'your_api_key'
  }));
  
  ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'trades'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### cURL
```bash
# Start bot
curl -X POST https://api.claudedex.io/v1/bot/start \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"mode": "production", "chains": ["ethereum"]}'

# Get positions
curl -X GET https://api.claudedex.io/v1/positions/active \
  -H "X-API-Key: your_api_key"

# Analyze token
curl -X POST https://api.claudedex.io/v1/analysis/token \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"token_address": "0x...", "chain": "ethereum"}'
```

## API Versioning

The API uses URL versioning. Current version: `v1`

Future versions will be available at:
- `https://api.claudedex.io/v2`
- `https://api.claudedex.io/v3`

## Support

For API support and questions:
- Documentation: https://docs.claudedex.io
- Email: api-support@claudedex.io
- Discord: https://discord.gg/claudedex
- GitHub Issues: https://github.com/claudedex/trading-bot/issues