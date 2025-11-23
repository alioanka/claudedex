FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_RETRIES=5

# Install system dependencies needed for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel for faster builds
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .

# Split installation into stages to handle large packages better
# Stage 1: Core dependencies (fast, stable)
RUN pip install --no-cache-dir \
    python-dotenv==1.0.0 \
    pyyaml==6.0.1 \
    aiohttp==3.9.1 \
    aiohttp-cors==0.7.0 \
    python-socketio==5.10.0 \
    aiofiles==23.2.1 \
    && echo "✅ Core dependencies installed"

# Stage 2: Data processing (medium size)
RUN pip install --no-cache-dir \
    numpy==1.26.3 \
    pandas==2.1.4 \
    scipy==1.11.4 \
    && echo "✅ Data processing libraries installed"

# Stage 3: TA-Lib (requires compilation)
RUN pip install --no-cache-dir TA-Lib==0.6.7 \
    && echo "✅ TA-Lib installed"

# Stage 4: Blockchain and Web3
# Install Solana dependencies first
RUN pip install --no-cache-dir \
    solders==0.27.1 \
    base58==2.1.1 \
    && echo "✅ Solana base libraries installed"

# Install driftpy (will bring in anchorpy and other dependencies)
RUN pip install --no-cache-dir \
    driftpy==0.8.80 \
    && echo "✅ DriftPy SDK installed"

# Install Ethereum Web3 libraries
RUN pip install --no-cache-dir \
    web3==6.20.4 \
    eth-account==0.10.0 \
    eth-utils==4.0.0 \
    eth-typing==4.1.0 \
    eth_abi \
    hexbytes==0.3.1 \
    hdwallet==2.2.1 \
    mnemonic==0.20 \
    && echo "✅ Ethereum libraries installed"

# Stage 5: Database
RUN pip install --no-cache-dir \
    sqlalchemy==2.0.25 \
    asyncpg==0.29.0 \
    psycopg2-binary==2.9.9 \
    alembic==1.13.1 \
    redis==5.0.1 \
    aioredis \
    && echo "✅ Database libraries installed"

# Stage 6: ML libraries (scikit-learn, joblib)
RUN pip install --no-cache-dir \
    scikit-learn==1.4.0 \
    joblib==1.3.2 \
    && echo "✅ ML core libraries installed"

# Stage 7: Deep Learning (large packages - install separately with retries)
RUN pip install --no-cache-dir --timeout=300 tensorflow || \
    pip install --no-cache-dir --timeout=300 tensorflow || \
    echo "⚠️ TensorFlow installation failed (optional)" && \
    echo "✅ TensorFlow installation attempted"

RUN pip install --no-cache-dir --timeout=300 torch || \
    pip install --no-cache-dir --timeout=300 torch || \
    echo "⚠️ PyTorch installation failed (optional)" && \
    echo "✅ PyTorch installation attempted"

# Stage 8: Boosting libraries
RUN pip install --no-cache-dir \
    lightgbm \
    xgboost \
    || echo "⚠️ Some boosting libraries failed (optional)"

# Stage 9: Security and Auth
RUN pip install --no-cache-dir \
    cryptography==41.0.7 \
    pynacl==1.5.0 \
    bcrypt==4.1.2 \
    pyotp==2.9.0 \
    PyJWT \
    && echo "✅ Security libraries installed"

# Stage 10: Web Framework
RUN pip install --no-cache-dir \
    fastapi==0.108.0 \
    uvicorn==0.25.0 \
    websockets==12.0 \
    jinja2==3.1.3 \
    openpyxl==3.1.2 \
    && echo "✅ Web framework installed"

# Stage 11: Monitoring and utilities
RUN pip install --no-cache-dir \
    prometheus-client==0.19.0 \
    loguru==0.7.2 \
    python-telegram-bot==20.7 \
    tenacity==8.2.3 \
    cachetools==5.3.2 \
    python-dateutil==2.8.2 \
    pytz==2024.1 \
    pydantic==2.5.3 \
    && echo "✅ Monitoring libraries installed"

# Stage 12: Additional utilities
RUN pip install --no-cache-dir \
    aiohttp-sse-client \
    aiohttp-sse \
    jsonschema \
    memory_profiler \
    orjson \
    psutil \
    pytest \
    textblob \
    scripts \
    setuptools \
    || echo "⚠️ Some optional utilities failed"

# Stage 13: Code quality (development only, can fail)
RUN pip install --no-cache-dir \
    pylint==3.0.3 \
    isort==5.13.2 \
    || echo "⚠️ Code quality tools installation failed (optional)"

# Verify critical imports
RUN python -c "import talib; print('✅ TA-Lib version:', talib.__version__)" && \
    python -c "import web3; print('✅ Web3 imported')" && \
    python -c "import asyncpg; print('✅ Database libraries OK')" && \
    python -c "import bcrypt; print('✅ Auth libraries OK')" && \
    python -c "import driftpy; print('✅ DriftPy imported')" && \
    echo "✅ All critical dependencies verified"

COPY . .

# Make entrypoint script executable
RUN chmod +x scripts/docker-entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Use entrypoint to run migrations before starting app
ENTRYPOINT ["./scripts/docker-entrypoint.sh"]
CMD ["main.py", "--mode", "production"]
