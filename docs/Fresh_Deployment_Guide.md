# 🚀 ClaudeDex - Fresh VPS Deployment from GitHub

## 📋 Prerequisites
- **Fresh Ubuntu 22.04 LTS** (recommended) or 24.04
- Root access
- Minimum: 4GB RAM, 2 CPU cores, 50GB storage
- Your GitHub access configured

---

## ⚡ Quick Start (Copy-Paste Commands)

### **Step 1: Initial System Setup (2 minutes)**

```bash
# SSH into fresh VPS
ssh root@YOUR_VPS_IP

# Update system
apt update && apt upgrade -y

# Install essential tools
apt install -y curl wget git vim screen htop build-essential

# Set timezone
timedatectl set-timezone Europe/Istanbul

# Verify system
free -h
df -h
uname -a
```

---

### **Step 2: Install Docker (Native - NOT Snap!) (3 minutes)**

```bash
# Remove any existing Docker/Snap
apt remove -y docker docker-engine docker.io containerd runc
snap remove docker 2>/dev/null || true

# Install Docker using official script
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh

# Install Docker Compose plugin
apt install -y docker-compose-plugin

# Enable and start Docker
systemctl enable docker
systemctl start docker

# Verify Docker (MUST show native Docker, not snap)
docker --version
# Should show: Docker version 27.x.x
docker compose version
# Should show: Docker Compose version v2.x.x

# Test Docker
docker run hello-world
# Should print "Hello from Docker!"

# Verify Docker is NOT snap
which docker
# Should show: /usr/bin/docker (NOT /snap/bin/docker)
```

**⚠️ CRITICAL:** If `which docker` shows `/snap/bin/docker`, Docker is still using snap. Remove it:
```bash
snap remove docker
systemctl restart docker
```

---

### **Step 3: Install Python 3.11 (2 minutes)**

```bash
# Install Python 3.11 and dependencies
apt install -y python3.11 python3.11-venv python3-pip python3.11-dev

# Verify Python version
python3.11 --version
# Should show: Python 3.11.x

# Install system dependencies for Python packages
apt install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    gcc \
    g++ \
    make \
    wget \
    pkg-config
```

---

### **Step 4: Install TA-Lib (CRITICAL - 5 minutes)**

TA-Lib is tricky. Here's the CORRECT way:

```bash
# Download TA-Lib source
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

# Extract
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib

# Configure and install (this takes 2-3 minutes)
./configure --prefix=/usr
make
make install

# Update library cache
ldconfig

# Verify TA-Lib C library is installed
ls -la /usr/lib/libta_lib.*
# Should show: /usr/lib/libta_lib.so.0.0.0

# Clean up
cd ~
rm -rf /tmp/ta-lib*

echo "✅ TA-Lib C library installed"
```

---

### **Step 5: Clone Repository (1 minute)**

```bash
# Clone your GitHub repo
cd ~
git clone https://github.com/alioanka/claudedex.git
cd claudedex

# Verify files
ls -la
# Should show: Dockerfile, docker-compose.yml, requirements.txt, main.py, etc.

# Check git status
git status
git log --oneline -5
```

---

### **Step 6: Setup Python Environment (3 minutes)**

```bash
cd ~/claudedex

# Create virtual environment
python3.11 -m venv venv

# Activate venv
source venv/bin/activate
# Your prompt should now show (venv)

# Upgrade pip
pip install --upgrade pip

# Install Python TA-Lib wrapper (this needs the C library from Step 4)
pip install TA-Lib==0.6.7

# Test TA-Lib import
python -c "import talib; print('✅ TA-Lib version:', talib.__version__)"
# Should show: ✅ TA-Lib version: 0.6.7

# Install remaining requirements
pip install -r requirements.txt

# This takes 3-5 minutes. Watch for errors!
```

**If TA-Lib installation fails:**
```bash
# Check if C library exists
ldconfig -p | grep ta_lib
# Should show: libta_lib.so.0

# If not found, reinstall C library:
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make && make install
ldconfig
cd ~/claudedex
pip install TA-Lib==0.6.7
```

---

### **Step 7: Configure Environment (2 minutes)**

```bash
cd ~/claudedex

# Backup original .env if exists
cp .env .env.backup 2>/dev/null || true

# Edit .env file
nano .env
```

**Minimum required changes in .env:**

```bash
# ⚠️ MUST CHANGE THESE:

# Solana wallet
SOLANA_PRIVATE_KEY=YOUR_SOLANA_PRIVATE_KEY_HERE

# EVM wallet
PRIVATE_KEY=0xYOUR_EVM_PRIVATE_KEY_HERE

# RPC endpoints (get free from Alchemy/Infura)
WEB3_PROVIDER_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY

# Telegram (optional but recommended)
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_CHAT_ID

# API Keys (optional)
GOPLUS_API_KEY=YOUR_KEY
1INCH_API_KEY=YOUR_KEY
```

**Keep these as-is:**
```bash
# Database (Docker will handle these)
DATABASE_URL=postgresql://bot_user:bot_password@postgres:5432/tradingbot
REDIS_URL=redis://redis:6379/0

# Mode
DRY_RUN=true  # Keep true for testing!
MODE=production
```

Save: `Ctrl+X`, `Y`, `Enter`

**Set file permissions:**
```bash
chmod 600 .env
```

---

### **Step 8: Create Required Directories (30 seconds)**

```bash
cd ~/claudedex

# Create directories that Docker needs
mkdir -p logs
mkdir -p data
mkdir -p config

# Set permissions
chmod 755 logs data config
```

---

### **Step 9: Build Docker Images (5-10 minutes)**

```bash
cd ~/claudedex

# Build the Docker image (this takes time on first build)
docker compose build --no-cache

# This will:
# - Download Python 3.11 base image
# - Install TA-Lib in container (using pip, not source)
# - Install all Python packages
# - Copy your code

# Watch for errors, especially TA-Lib related!
```

**Expected output at the end:**
```
Successfully built sha256:xxxxxxxxxxxx
Successfully tagged claudedex_trading-bot:latest
```

**If build fails on TA-Lib:**

Your Dockerfile uses `pip install TA-Lib` which needs the C library. But the Python slim image doesn't have it!

**Fix:** Update Dockerfile:

```bash
nano Dockerfile
```

Replace content with:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

# Install system dependencies FIRST
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library in container
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy requirements
COPY requirements.txt .

# Install Python packages (including TA-Lib wrapper)
RUN pip install --no-cache-dir -r requirements.txt

# Verify TA-Lib works
RUN python -c "import talib; print('✅ TA-Lib version:', talib.__version__)"

# Copy application code
COPY . .

# Run bot
CMD ["python", "main.py", "--mode", "production"]
```

Save and rebuild:
```bash
docker compose build --no-cache
```

---

### **Step 10: Start Services (2 minutes)**

```bash
cd ~/claudedex

# Start all services
docker compose up -d

# Check status
docker compose ps

# Should show:
# NAME              IMAGE                  STATUS
# trading-bot       claudedex_trading-bot  Up
# trading-postgres  timescale/timescaledb  Up (healthy)
# trading-redis     redis:7-alpine         Up

# If trading-bot shows "Exited (1)", check logs:
docker compose logs trading-bot
```

---

### **Step 11: Verify Everything (2 minutes)**

```bash
# Check all containers running
docker ps

# Check PostgreSQL
docker exec -it trading-postgres psql -U bot_user -d tradingbot -c "SELECT 1;"
# Should return: 1

# Check Redis
docker exec -it trading-redis redis-cli ping
# Should return: PONG

# Check bot logs (MOST IMPORTANT)
docker compose logs -f trading-bot

# Expected output:
# ✅ Database connected
# ✅ Solana Jupiter Executor initialized
# 🌐 Multi-chain mode: 6 chains enabled
# 🔍 Starting new pairs monitoring loop...
```

**If you see errors:**

```bash
# Common issues:

# 1. ImportError: BaseExecutor
# Fix: Update base_executor.py (we'll do this next)

# 2. Database connection failed
docker compose restart postgres
sleep 5
docker compose restart trading-bot

# 3. Solana errors
# Check SOLANA_PRIVATE_KEY in .env
```

---

### **Step 12: Fix BaseExecutor Issue (if needed)**

If logs show: `ImportError: cannot import name 'BaseExecutor'`

```bash
cd ~/claudedex

# Edit base_executor.py
nano trading/executors/base_executor.py
```

Add at line 14 (after imports, before `@dataclass`):

```python
from abc import ABC, abstractmethod

class BaseExecutor(ABC):
    """Abstract base for all executors"""
    def __init__(self, config: Dict):
        self.config = config
        self.stats = {'total_trades': 0, 'successful_trades': 0, 'failed_trades': 0}
    async def initialize(self): pass
    async def execute_trade(self, order, quote=None) -> Dict: pass
    def validate_order(self, order) -> bool: pass
    async def cleanup(self): pass
    async def get_execution_stats(self) -> Dict: return self.stats.copy()
```

Change line ~58:
```python
# FROM:
class TradeExecutor:

# TO:
class TradeExecutor(BaseExecutor):
```

Add methods to TradeExecutor (after `__init__`):
```python
async def initialize(self):
    if not self.w3.is_connected():
        raise ConnectionError("Web3 connection failed")

async def cleanup(self):
    pass
```

Save, commit, and rebuild:
```bash
git add .
git commit -m "Add BaseExecutor class"
docker compose down
docker compose build
docker compose up -d
docker compose logs -f trading-bot
```

---

### **Step 13: Access Dashboard (30 seconds)**

```bash
# Get your VPS IP
curl ifconfig.me

# Open in browser:
http://YOUR_VPS_IP:8080/dashboard

# Or test with curl:
curl http://localhost:8080/api/dashboard
```

---

## 🔐 Step 14: Secure Your VPS (Important!)

```bash
# Enable firewall
ufw allow 22/tcp     # SSH
ufw allow 8080/tcp   # Dashboard
ufw --force enable

# Check firewall status
ufw status

# Set proper permissions
cd ~/claudedex
chmod 600 .env
chmod 600 config/*.yaml 2>/dev/null || true

# (Optional) Create non-root user
adduser trader
usermod -aG docker trader
chown -R trader:trader ~/claudedex
```

---

## 📝 Daily Operations

### **View Logs**
```bash
cd ~/claudedex
docker compose logs -f trading-bot
```

### **Restart Bot**
```bash
docker compose restart trading-bot
```

### **Stop Everything**
```bash
docker compose down
```

### **Start Everything**
```bash
docker compose up -d
```

### **Update from GitHub**
```bash
cd ~/claudedex
docker compose down
git pull origin main
docker compose build
docker compose up -d
docker compose logs -f trading-bot
```

### **Check Status**
```bash
docker compose ps
docker stats --no-stream
```

---

## 🆘 Common Issues & Fixes

### **1. TA-Lib Import Error**

```bash
# In Docker container
docker exec -it trading-bot python -c "import talib"

# If fails, rebuild with correct Dockerfile (see Step 9)
```

### **2. Port 5432 Already in Use**

```bash
# Check what's using it
sudo lsof -i :5432

# If system PostgreSQL:
sudo systemctl stop postgresql
sudo systemctl disable postgresql
```

### **3. Permission Denied (Docker)**

```bash
# Check Docker is NOT snap
which docker
# Should be: /usr/bin/docker

# If /snap/bin/docker, remove snap Docker:
sudo snap remove docker
sudo systemctl restart docker
```

### **4. Solana Import Errors**

```bash
# Verify solders installed
docker exec -it trading-bot pip list | grep solders

# Should show: solders 0.18.1

# If missing, add to requirements.txt and rebuild
```

### **5. Database Connection Fails**

```bash
# Check PostgreSQL health
docker compose ps postgres

# Restart database
docker compose restart postgres
sleep 10
docker compose restart trading-bot
```

### **6. Container Keeps Restarting**

```bash
# Check logs
docker compose logs trading-bot | tail -100

# Usually shows the error
# Fix the error, then:
docker compose down
docker compose up -d
```

---

## 📊 Performance Monitoring

```bash
# Check resource usage
docker stats

# Check disk space
df -h

# Check logs size
du -sh ~/claudedex/logs/*

# Rotate logs if too large
cd ~/claudedex/logs
tar -czf old_logs_$(date +%Y%m%d).tar.gz *.log
rm *.log
docker compose restart trading-bot
```

---

## 🔄 Backup & Restore

### **Backup**
```bash
cd ~
tar -czf claudedex_backup_$(date +%Y%m%d).tar.gz claudedex/
scp claudedex_backup_*.tar.gz user@backup-server:/backups/
```

### **Backup Database**
```bash
docker exec trading-postgres pg_dump -U bot_user tradingbot > backup.sql
```

### **Restore Database**
```bash
cat backup.sql | docker exec -i trading-postgres psql -U bot_user tradingbot
```

---

## ✅ Final Verification Checklist

- [ ] Ubuntu 22.04/24.04 fresh install
- [ ] Docker installed (native, verified with `which docker`)
- [ ] Python 3.11 installed
- [ ] TA-Lib C library installed (`ldconfig -p | grep ta_lib`)
- [ ] Repository cloned from GitHub
- [ ] Virtual environment created
- [ ] Python packages installed (including TA-Lib wrapper)
- [ ] .env file configured with real keys
- [ ] Dockerfile updated with TA-Lib installation
- [ ] Docker images built successfully
- [ ] All 3 containers running (`docker compose ps`)
- [ ] PostgreSQL healthy (`docker exec -it trading-postgres psql -U bot_user -d tradingbot -c "SELECT 1;"`)
- [ ] Redis responding (`docker exec -it trading-redis redis-cli ping`)
- [ ] Bot logs show no errors
- [ ] Dashboard accessible at http://VPS_IP:8080
- [ ] BaseExecutor issue fixed (if applicable)
- [ ] Firewall configured
- [ ] .env permissions set (chmod 600)

---

## 🎯 Expected Final State

```bash
# All containers running
$ docker compose ps
NAME              STATUS
trading-bot       Up
trading-postgres  Up (healthy)
trading-redis     Up

# Bot logs showing success
$ docker compose logs trading-bot | tail -20
✅ Database connected
✅ Solana Jupiter Executor initialized  
🌐 Multi-chain mode: 6 chains enabled
  Chains: ethereum, bsc, base, arbitrum, polygon, solana
🔍 Starting new pairs monitoring loop...
```

---

## 🚀 Total Time: ~30 minutes

- System setup: 2 min
- Docker install: 3 min
- Python install: 2 min
- TA-Lib install: 5 min
- Clone repo: 1 min
- Python packages: 3 min
- Configure .env: 2 min
- Build Docker: 10 min
- Start & verify: 2 min

**You're now running ClaudeDex on a clean VPS!** 🎉

---

## 📞 Need Help?

Check logs: `docker compose logs -f trading-bot`  
GitHub Issues: https://github.com/alioanka/claudedex/issues  
Review this guide's troubleshooting section above

**Happy Trading!** 📈🚀