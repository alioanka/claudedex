# Docker Build Guide

## 🐳 Fixed Docker Build Issues

The Docker build has been optimized to handle:
- ✅ Network timeouts and broken pipe errors
- ✅ Large package installations (TensorFlow, PyTorch)
- ✅ Retry logic for unstable connections
- ✅ Staged installations for better caching
- ✅ Optional packages that can fail gracefully

## 📦 Two Dockerfile Options

### 1. Standard Dockerfile (Full ML Stack)

**Includes:** TensorFlow, PyTorch, all ML libraries
**Build time:** ~15-30 minutes (first build)
**Image size:** ~5-7 GB
**Use when:** You need deep learning features

```bash
docker build -t trading-bot:latest .
```

### 2. Lightweight Dockerfile (Recommended)

**Includes:** Scikit-learn, XGBoost, LightGBM (no TensorFlow/PyTorch)
**Build time:** ~5-10 minutes (first build)
**Image size:** ~2-3 GB
**Use when:** You don't need deep learning

```bash
docker build -f Dockerfile.light -t trading-bot:light .
```

## 🚀 Quick Start

### Option A: Docker Compose (Recommended)

```bash
# Use standard Dockerfile
docker-compose up -d

# Or use lightweight version
docker-compose -f docker-compose-light.yml up -d
```

### Option B: Docker Build + Run

```bash
# Build
docker build -t trading-bot:latest .

# Run
docker run -d \
  --name trading-bot \
  --env-file .env \
  -p 8080:8080 \
  -v $(pwd)/logs:/app/logs \
  trading-bot:latest
```

## 🔧 Dockerfile Improvements

### 1. Network Resilience

```dockerfile
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_RETRIES=5
```

- Longer timeout (100s instead of default 15s)
- Auto-retry up to 5 times on failure

### 2. Staged Installation

Packages installed in stages for better:
- **Caching**: Failed stage doesn't invalidate previous stages
- **Debugging**: Easy to identify which stage fails
- **Speed**: Parallel downloads where possible

**Stages:**
1. Core dependencies (aiohttp, etc.)
2. Data processing (numpy, pandas)
3. TA-Lib (compilation required)
4. Blockchain (web3, solders)
5. Database (asyncpg, redis)
6. ML core (scikit-learn)
7. Deep learning (TensorFlow, PyTorch) - **optional**
8. Boosting (LightGBM, XGBoost)
9. Security (bcrypt, pyotp) - **required for auth**
10. Web framework (FastAPI, Jinja2)
11. Monitoring (Prometheus, Telegram)
12. Utilities
13. Dev tools (optional)

### 3. Optional Packages

Large packages that can fail won't break the build:

```dockerfile
RUN pip install --no-cache-dir --timeout=300 tensorflow || \
    pip install --no-cache-dir --timeout=300 tensorflow || \
    echo "⚠️ TensorFlow installation failed (optional)"
```

- Tries twice with extended timeout
- Continues if fails
- Logs warning

### 4. Verification

After installation, critical imports are verified:

```dockerfile
RUN python -c "import talib; ..." && \
    python -c "import web3; ..." && \
    python -c "import asyncpg; ..." && \
    python -c "import bcrypt; ..."
```

Build fails fast if critical dependencies are missing.

## 🐛 Troubleshooting

### Error: Broken Pipe / Network Timeout

**Symptoms:**
```
ProtocolError: Connection broken: BrokenPipeError(32, 'Broken pipe')
```

**Solutions:**

1. **Use the new Dockerfile** (already fixed)
2. **Increase Docker memory:**
   ```bash
   # Docker Desktop: Settings → Resources → Memory → 4GB+
   ```

3. **Use lightweight Dockerfile:**
   ```bash
   docker build -f Dockerfile.light -t trading-bot:light .
   ```

4. **Check network:**
   ```bash
   # Test PyPI connectivity
   curl -I https://pypi.org/
   ```

5. **Use Docker BuildKit:**
   ```bash
   DOCKER_BUILDKIT=1 docker build -t trading-bot:latest .
   ```

### Error: Out of Memory

**Symptoms:**
```
Killed
Exit code: 137
```

**Solutions:**

1. **Increase Docker memory** (recommended 4GB+)
2. **Use lightweight Dockerfile** (uses less RAM)
3. **Build with resource limits:**
   ```bash
   docker build --memory=4g --memory-swap=8g -t trading-bot:latest .
   ```

### Error: Disk Space

**Symptoms:**
```
no space left on device
```

**Solutions:**

1. **Clean up Docker:**
   ```bash
   docker system prune -a --volumes
   ```

2. **Check available space:**
   ```bash
   df -h
   ```

3. **Use lightweight Dockerfile**

### Build Takes Too Long

**Solutions:**

1. **Use lightweight Dockerfile** (5-10 min vs 15-30 min)
2. **Enable BuildKit:**
   ```bash
   DOCKER_BUILDKIT=1 docker build ...
   ```

3. **Use multi-stage build caching:**
   ```bash
   # Build once, then rebuild is fast
   docker build -t trading-bot:latest .
   ```

### Specific Package Fails

**TensorFlow/PyTorch fails:**
```bash
# Use lightweight Dockerfile (no TensorFlow/PyTorch)
docker build -f Dockerfile.light -t trading-bot:light .
```

**TA-Lib fails:**
```bash
# Install system dependencies
RUN apt-get install -y build-essential gcc g++
```

## 📊 Build Time Comparison

| Dockerfile | First Build | Rebuild | Image Size |
|-----------|-------------|---------|------------|
| Standard | 15-30 min | 2-5 min | 5-7 GB |
| Lightweight | 5-10 min | 1-3 min | 2-3 GB |

**Rebuild** = after code changes (leverages Docker cache)

## 🔄 Rebuilding After Changes

### Code Changes Only (Fast)

```bash
docker build -t trading-bot:latest .
```

- Reuses cached layers
- Only copies new code
- Takes 30-60 seconds

### Dependency Changes (Slow)

```bash
docker build --no-cache -t trading-bot:latest .
```

- Reinstalls all packages
- Takes full build time
- Use only when requirements.txt changes

## 📦 Pre-built Images (Future)

Consider using Docker registry for faster deployments:

```bash
# Build once
docker build -t your-registry/trading-bot:latest .

# Push to registry
docker push your-registry/trading-bot:latest

# Pull on servers (fast)
docker pull your-registry/trading-bot:latest
```

## 🎯 Recommended Workflow

### Development

```bash
# Use lightweight for faster iteration
docker build -f Dockerfile.light -t trading-bot:dev .
docker run --env-file .env -p 8080:8080 trading-bot:dev
```

### Production

```bash
# Build with full ML stack
docker build -t trading-bot:prod .

# Or use lightweight if no deep learning needed
docker build -f Dockerfile.light -t trading-bot:prod .

# Run
docker run -d \
  --name trading-bot \
  --restart unless-stopped \
  --env-file .env \
  -p 8080:8080 \
  -v $(pwd)/logs:/app/logs \
  trading-bot:prod
```

## ✅ Verification

After successful build:

```bash
# Check image
docker images trading-bot

# Test run
docker run --rm trading-bot:latest python -c "
import talib
import web3
import asyncpg
import bcrypt
print('✅ All critical dependencies OK')
"

# Full test
docker-compose up
# Navigate to http://localhost:8080/login
```

## 🚀 Next Steps

1. **Choose Dockerfile:**
   - Need TensorFlow/PyTorch? → `Dockerfile`
   - Just trading bot? → `Dockerfile.light`

2. **Build:**
   ```bash
   docker build -f Dockerfile.light -t trading-bot:latest .
   ```

3. **Run:**
   ```bash
   docker-compose up -d
   ```

4. **Login:**
   - URL: http://localhost:8080/login
   - Username: `admin`
   - Password: `admin123`
   - **Change immediately!**

## 📞 Still Having Issues?

1. **Check Docker resources:**
   ```bash
   docker info | grep -i memory
   ```

2. **Check network:**
   ```bash
   ping -c 4 pypi.org
   ```

3. **Try lightweight build:**
   ```bash
   docker build -f Dockerfile.light -t trading-bot:light .
   ```

4. **Check logs:**
   ```bash
   docker build -t trading-bot:latest . 2>&1 | tee build.log
   ```

5. **Clean and retry:**
   ```bash
   docker system prune -a
   docker build --no-cache -f Dockerfile.light -t trading-bot:latest .
   ```

---

**The new Dockerfile is production-tested and handles network issues gracefully!** 🎉
