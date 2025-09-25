# ClaudeDex Trading Bot - Production Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Database Setup](#database-setup)
6. [Security Configuration](#security-configuration)
7. [Monitoring Setup](#monitoring-setup)
8. [Backup & Recovery](#backup--recovery)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance](#maintenance)

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ / CentOS 8+ / macOS 11+
- **CPU**: 8+ cores (16+ recommended)
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 500GB SSD (1TB recommended)
- **Network**: 100Mbps+ dedicated bandwidth

### Software Requirements
```bash
# Required software versions
- Docker: 20.10+
- Docker Compose: 2.0+
- Kubernetes: 1.24+ (optional)
- PostgreSQL: 14+
- Redis: 7+
- Python: 3.11+
- Node.js: 18+ (for dashboard)
- Git: 2.30+
```

### API Keys Required
```yaml
# Required API keys
- DexScreener API
- Etherscan API (for each chain)
- Infura/Alchemy RPC endpoints
- GoPlus Security API
- TokenSniffer API (optional)
- Twitter API (optional)
- Telegram Bot Token
- Discord Webhook URL
```

## Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/claudedex/trading-bot.git
cd trading-bot
```

### 2. Create Environment File
```bash
cp .env.example .env
```

Edit `.env` with your configuration:
```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=claudedex_trading
DB_USER=claudedex
DB_PASSWORD=your_secure_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# API Keys
DEXSCREENER_API_KEY=your_key
ETHERSCAN_API_KEY=your_key
INFURA_PROJECT_ID=your_project_id
GOPLUS_API_KEY=your_key

# Trading Configuration
MAX_POSITION_SIZE=0.05
DEFAULT_STOP_LOSS=0.05
MIN_LIQUIDITY_USD=50000

# Notification Services
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
DISCORD_WEBHOOK_URL=your_webhook_url

# Security
JWT_SECRET=your_jwt_secret_min_32_chars
ENCRYPTION_KEY=your_encryption_key_32_chars
WALLET_PRIVATE_KEY=encrypted_private_key
```

### 3. Install Dependencies
```bash
# Python dependencies
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install package
pip install -e .

# Verify installation
python -c "import claudedex; print('ClaudeDex installed successfully')"
```

## Docker Deployment

### 1. Build Docker Image
```bash
# Production build
docker build -t claudedex:latest \
  --build-arg ENV=production \
  --target production \
  .

# Verify image
docker images | grep claudedex
```

### 2. Docker Compose Setup
```yaml
# docker-compose.production.yml
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg14
    environment:
      POSTGRES_DB: ${DB_NAME}
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  trading-bot:
    image: claudedex:latest
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    env_file: .env
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    ports:
      - "8080:8080"  # API
      - "8081:8081"  # Dashboard
      - "9090:9090"  # Prometheus metrics
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "scripts/health_check.py"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./observability/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9091:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    depends_on:
      - prometheus
    volumes:
      - ./observability/grafana:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 3. Start Services
```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f trading-bot
```

## Kubernetes Deployment

### 1. Create Namespace
```bash
kubectl create namespace claudedex
```

### 2. Create Secrets
```bash
# Create secret from .env file
kubectl create secret generic claudedex-secrets \
  --from-env-file=.env \
  -n claudedex
```

### 3. Apply Configurations
```bash
# Apply all Kubernetes manifests
kubectl apply -f kubernetes/ -n claudedex

# Verify deployment
kubectl get all -n claudedex
```

### 4. Setup Ingress
```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: claudedex-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
    - hosts:
        - api.claudedex.io
      secretName: claudedex-tls
  rules:
    - host: api.claudedex.io
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: claudedex-api
                port:
                  number: 8080
```

### 5. Scale Deployment
```bash
# Scale trading bot replicas
kubectl scale deployment claudedex-bot -n claudedex --replicas=3

# Setup autoscaling
kubectl autoscale deployment claudedex-bot \
  -n claudedex \
  --min=2 \
  --max=10 \
  --cpu-percent=70
```

## Database Setup

### 1. Initialize Database
```bash
# Run database setup script
python scripts/setup_database.py --env production

# Verify tables created
psql -h localhost -U claudedex -d claudedex_trading -c "\dt"
```

### 2. Enable TimescaleDB
```sql
-- Connect to database
psql -h localhost -U claudedex -d claudedex_trading

-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create hypertables for time-series data
SELECT create_hypertable('market_data', 'timestamp');
SELECT create_hypertable('trades', 'entry_timestamp');
```

### 3. Setup Replication (Optional)
```bash
# On primary server
echo "host replication replica 10.0.0.2/32 md5" >> /etc/postgresql/14/main/pg_hba.conf

# Create replication user
psql -c "CREATE USER replica REPLICATION LOGIN ENCRYPTED PASSWORD 'replica_password';"

# Restart PostgreSQL
systemctl restart postgresql
```

## Security Configuration

### 1. SSL/TLS Setup
```bash
# Generate SSL certificates with Let's Encrypt
certbot certonly --standalone -d api.claudedex.io

# Update nginx config
server {
    listen 443 ssl http2;
    server_name api.claudedex.io;
    
    ssl_certificate /etc/letsencrypt/live/api.claudedex.io/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.claudedex.io/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 2. Firewall Configuration
```bash
# UFW setup
ufw allow 22/tcp
ufw allow 443/tcp
ufw allow 8080/tcp
ufw allow 5432/tcp from 10.0.0.0/24
ufw enable

# iptables rules
iptables -A INPUT -p tcp --dport 5432 -s 10.0.0.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 5432 -j DROP
```

### 3. Wallet Security
```bash
# Encrypt private keys
python scripts/encrypt_wallet.py --key-file wallet.key

# Setup hardware wallet (Ledger)
python scripts/setup_hardware_wallet.py --device ledger

# Enable multi-signature
python scripts/enable_multisig.py --signers 3 --threshold 2
```

### 4. API Security
```python
# config/security.py
SECURITY_CONFIG = {
    "rate_limiting": {
        "enabled": True,
        "default_limit": 1000,
        "trading_limit": 100
    },
    "jwt": {
        "expiry": 3600,
        "refresh_expiry": 86400
    },
    "api_keys": {
        "rotation_days": 30,
        "max_keys_per_user": 5
    },
    "ip_whitelist": [
        "10.0.0.0/24",
        "192.168.1.0/24"
    ]
}
```

## Monitoring Setup

### 1. Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'claudedex'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
```

### 2. Grafana Dashboards
```bash
# Import dashboards
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -H "Content-Type: application/json" \
  -d @observability/grafana/dashboard.json
```

### 3. Alert Configuration
```yaml
# alerts.yml
groups:
  - name: trading_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(errors_total[5m]) > 0.05
        for: 10m
        annotations:
          summary: "High error rate detected"
          
      - alert: LowLiquidity
        expr: liquidity_usd < 10000
        for: 5m
        annotations:
          summary: "Low liquidity warning"
```

### 4. Log Aggregation
```bash
# Setup ELK stack (optional)
docker run -d --name elasticsearch \
  -e "discovery.type=single-node" \
  -p 9200:9200 \
  elasticsearch:8.5.0

docker run -d --name logstash \
  -v $(pwd)/logstash.conf:/config/logstash.conf \
  -p 5000:5000 \
  logstash:8.5.0

docker run -d --name kibana \
  -e "ELASTICSEARCH_HOSTS=http://elasticsearch:9200" \
  -p 5601:5601 \
  kibana:8.5.0
```

## Backup & Recovery

### 1. Database Backup
```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/postgres"

# Backup database
pg_dump -h localhost -U claudedex -d claudedex_trading \
  -f "$BACKUP_DIR/backup_$DATE.sql"

# Compress backup
gzip "$BACKUP_DIR/backup_$DATE.sql"

# Upload to S3
aws s3 cp "$BACKUP_DIR/backup_$DATE.sql.gz" \
  s3://claudedex-backups/postgres/

# Keep only last 30 days locally
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete
```

### 2. Redis Backup
```bash
# Redis backup
redis-cli -a $REDIS_PASSWORD BGSAVE

# Copy dump file
cp /var/lib/redis/dump.rdb /backup/redis/dump_$(date +%Y%m%d).rdb
```

### 3. Recovery Procedure
```bash
# Restore PostgreSQL
gunzip < backup_20250101_120000.sql.gz | psql -h localhost -U claudedex -d claudedex_trading

# Restore Redis
systemctl stop redis
cp /backup/redis/dump_20250101.rdb /var/lib/redis/dump.rdb
systemctl start redis
```

## Troubleshooting

### Common Issues

#### 1. Bot Won't Start
```bash
# Check logs
docker logs claudedex_trading-bot_1

# Verify database connection
psql -h localhost -U claudedex -d claudedex_trading -c "SELECT 1;"

# Check Redis connection
redis-cli -a $REDIS_PASSWORD ping

# Verify API keys
python scripts/verify_api_keys.py
```

#### 2. High Memory Usage
```bash
# Check memory usage
docker stats

# Increase memory limits
docker update --memory=4g claudedex_trading-bot_1

# Optimize Python memory
export PYTHONMALLOC=malloc
```

#### 3. Slow Performance
```bash
# Analyze slow queries
psql -c "SELECT * FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"

# Check indexes
psql -c "\di"

# Optimize Redis
redis-cli -a $REDIS_PASSWORD INFO memory
```

#### 4. Network Issues
```bash
# Test RPC endpoints
curl -X POST https://mainnet.infura.io/v3/$INFURA_PROJECT_ID \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'

# Check API rate limits
python scripts/check_rate_limits.py
```

## Maintenance

### Daily Tasks
```bash
# Health check
python scripts/health_check.py

# Check disk space
df -h

# Review error logs
grep ERROR logs/trading_bot.log | tail -100
```

### Weekly Tasks
```bash
# Update dependencies
pip list --outdated
pip install --upgrade -r requirements.txt

# Database maintenance
psql -c "VACUUM ANALYZE;"
psql -c "REINDEX DATABASE claudedex_trading;"

# Security scan
pip-audit
safety check
```

### Monthly Tasks
```bash
# Rotate API keys
python scripts/rotate_api_keys.py

# Update ML models
python scripts/retrain_models.py

# Performance review
python scripts/generate_performance_report.py --period monthly
```

## Production Checklist

### Pre-Deployment
- [ ] All environment variables configured
- [ ] Database initialized and backed up
- [ ] SSL certificates installed
- [ ] Firewall rules configured
- [ ] API keys validated
- [ ] Wallet secured (hardware/multisig)
- [ ] Monitoring alerts configured
- [ ] Backup scripts scheduled

### Post-Deployment
- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs aggregated and searchable
- [ ] Alerts functioning
- [ ] API endpoints responding
- [ ] WebSocket connections stable
- [ ] Trading executing properly
- [ ] Performance within targets

### Security
- [ ] Private keys encrypted
- [ ] Database passwords secure
- [ ] API authentication enabled
- [ ] Rate limiting active
- [ ] IP whitelisting configured
- [ ] Audit logging enabled
- [ ] Security scan completed
- [ ] Penetration testing done

## Support Resources

- **Documentation**: https://docs.claudedex.io
- **GitHub Issues**: https://github.com/claudedex/trading-bot/issues
- **Discord Community**: https://discord.gg/claudedex
- **Email Support**: support@claudedex.io
- **Emergency Hotline**: +1-xxx-xxx-xxxx (24/7 critical issues)

## License & Compliance

Ensure compliance with:
- Financial regulations in your jurisdiction
- Cryptocurrency trading laws
- Data protection regulations (GDPR, CCPA)
- API terms of service
- Exchange trading rules

---

**Note**: This bot is for educational purposes. Always test thoroughly in development before deploying real funds. Start with small amounts and gradually increase as confidence grows.