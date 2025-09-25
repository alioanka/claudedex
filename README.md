# DexScreener Trading Bot

## 📊 Advanced Automated Cryptocurrency Trading System

A sophisticated trading bot with ML ensemble models, multi-DEX support, MEV protection, and self-learning capabilities.

## 🚀 Features

### Core Capabilities
- **Multi-Chain Support**: Ethereum, BSC, Polygon, Arbitrum, Base
- **DEX Integration**: Uniswap, PancakeSwap, SushiSwap, and more
- **ML-Powered Analysis**: Rug detection, pump prediction, market analysis
- **Risk Management**: VaR, correlation limits, position sizing, stop-loss
- **MEV Protection**: Flashbots integration, private mempool usage
- **Real-time Monitoring**: Dashboard, alerts, performance tracking

### Security Features
- **Wallet Security**: Hardware wallet support, multisig, key rotation
- **API Security**: Rate limiting, JWT auth, IP whitelisting  
- **Audit Logging**: Tamper-proof logs with HMAC checksums
- **Encryption**: AES-256-GCM for sensitive data

### Trading Strategies
- **Momentum Trading**: Breakout, trend following, reversal, mean reversion
- **Scalping**: Quick trades, range trading, news-based, volume spike
- **AI Strategy**: ML-driven decisions based on ensemble models

## 📋 Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 7+
- Docker & Docker Compose
- Node.js 18+ (for dashboard)

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/tradingbot.git
cd tradingbot
```

### 2. Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r test-requirements.txt  # For testing
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 5. Setup Database
```bash
# Start PostgreSQL with TimescaleDB
docker-compose up -d postgres

# Run migrations
python scripts/setup_database.py
```

### 6. Start Redis
```bash
docker-compose up -d redis
```

### 7. Initialize Configuration
```bash
python scripts/init_config.py
```

## ⚙️ Configuration

### Basic Configuration
Edit `config/settings.py` for global settings:

```python
ENVIRONMENT = Environment.DEVELOPMENT  # or PRODUCTION
SUPPORTED_CHAINS = {...}  # Chain configurations
FEATURE_FLAGS = {...}     # Enable/disable features
```

### Trading Configuration
Create `config/trading.json`:

```json
{
  "max_position_size": 1000,
  "max_slippage": 0.05,
  "min_liquidity": 50000,
  "strategies": {
    "momentum": {"enabled": true},
    "scalping": {"enabled": true}
  }
}
```

### Security Configuration
Create `config/security.json`:

```json
{
  "require_2fa": true,
  "api_rate_limits": {
    "default": {"requests": 100, "window": 60}
  },
  "wallet_security": {
    "require_multisig": true,
    "min_signatures": 2
  }
}
```

## 🚀 Usage

### Starting the Bot

#### Development Mode
```bash
python main.py --mode development
```

#### Production Mode
```bash
python main.py --mode production --config config/production.json
```

#### Docker Deployment
```bash
docker-compose up -d
```

### CLI Commands

```bash
# Start trading
python main.py start

# Stop trading
python main.py stop

# Check status
python main.py status

# Run specific strategy
python main.py --strategy momentum

# Dry run mode
python main.py --dry-run

# Backtest
python main.py backtest --from 2024-01-01 --to 2024-12-31
```

## 📊 Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                       │
│                    (Dashboard / CLI / API)                   │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                        Trading Engine                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Event   │  │   Risk   │  │ Decision │  │Portfolio │   │
│  │   Bus    │  │ Manager  │  │  Maker   │  │ Manager  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                      Analysis Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   Rug    │  │  Pump    │  │Liquidity │  │  Market  │   │
│  │ Detector │  │Predictor │  │ Monitor  │  │ Analyzer │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                        ML Models                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Ensemble │  │   Rug    │  │   Pump   │  │  Volume  │   │
│  │  Model   │  │Classifier│  │ Predictor│  │Validator │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   DEX    │  │  Chain   │  │ Honeypot │  │  Whale   │   │
│  │Collector │  │   Data   │  │ Checker  │  │ Tracker  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                      Storage Layer                           │
│         PostgreSQL / TimescaleDB      Redis Cache           │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Collection**: Real-time data from DEXs, blockchain, and social sources
2. **Analysis**: ML models analyze opportunities and risks
3. **Decision Making**: Risk-adjusted decisions based on multiple signals
4. **Execution**: Orders executed with MEV protection and optimal routing
5. **Monitoring**: Real-time tracking of positions and performance

## 🧪 Testing

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Types
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/ --benchmark-only

# Security tests
pytest tests/security/

# Smoke tests
pytest -m smoke tests/
```

### Coverage Report
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

## 📈 Performance Metrics

- **Database Throughput**: >1000 writes/second
- **Cache Performance**: >10000 reads/second
- **ML Inference**: >100 predictions/second
- **Order Execution**: <100ms latency
- **Concurrent Requests**: >100 requests/second

## 🔒 Security Considerations

- Never commit `.env` files or private keys
- Use hardware wallets for production
- Enable 2FA for all admin operations
- Regularly rotate API keys and passwords
- Monitor audit logs for suspicious activity
- Keep dependencies updated

## 🚨 Monitoring & Alerts

### Available Metrics
- Portfolio value and P&L
- Win rate and risk metrics
- System performance (CPU, memory, latency)
- Error rates and failed transactions
- Gas usage and costs

### Alert Channels
- Telegram
- Discord
- Email
- Slack
- Webhook

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This bot is for educational purposes. Cryptocurrency trading carries significant risk. Never trade with funds you cannot afford to lose. Past performance does not guarantee future results.

## 📞 Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/yourusername/tradingbot/issues)
- Discord: [Join our server](https://discord.gg/yourserver)

## 🙏 Acknowledgments

- OpenZeppelin for security patterns
- Flashbots for MEV protection
- DexScreener for API access
- The open-source community