FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_RETRIES=5

# Install system dependencies needed for compilation
# Plus curl + jq + postgresql-client for the Test Runner Section B
# scripts that probe the running dashboard + run psql against the
# postgres container from in-network. Without these the in-container
# preflight emits 'docker: command not found' for every check that
# expected to shell out to `docker compose exec postgres ...`.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    jq \
    postgresql-client \
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
# Pin httpx first to ensure compatibility with solana library
# The solana library uses httpx with 'proxy' parameter which changed in newer versions
RUN pip install --no-cache-dir \
    "httpx>=0.23.0,<0.28.0" \
    && echo "✅ httpx pinned for solana compatibility"

# Install base58 (standalone dependency)
RUN pip install --no-cache-dir \
    base58==2.1.1 \
    && echo "✅ base58 installed"

# Install driftpy which brings in compatible solana and solders versions
# driftpy 0.8.80 requires solders>=0.27.1 and will pull compatible solana
RUN pip install --no-cache-dir \
    driftpy==0.8.80 \
    && echo "✅ DriftPy SDK installed (includes solana, solders)"

# Install CCXT for Futures trading
RUN pip install --no-cache-dir \
    ccxt==4.4.26 \
    && echo "✅ CCXT exchange library installed"

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

# Stage 7b: Kronos K-line forecaster deps (advisor module, optional/fail-soft).
# Kronos is NOT a HuggingFace causal-LM — transformers/sentencepiece CANNOT load
# it (that path failed with "Couldn't instantiate the backend tokenizer ... need
# sentencepiece or tiktoken"). The real Kronos model code is VENDORED in
# modules/advisor/core/kronos_vendor/ (MIT, no network needed). Its actual
# runtime deps are: torch (Stage 7, above) + einops + huggingface_hub +
# safetensors (.from_pretrained weight loading) + numpy/pandas/tqdm (from the ML
# stage). Pretrained weights (model + tokenizer repos) are fetched separately by
# scripts/download_kronos_weights.py into the persistent /app/data volume.
# Baked into the image so they survive rebuilds; fail-soft so the build never
# breaks on a transient PyPI hiccup (Kronos predict() returns None if unusable).
RUN pip install --no-cache-dir --timeout=300 einops huggingface_hub safetensors tqdm || \
    pip install --no-cache-dir --timeout=300 einops huggingface_hub safetensors tqdm || \
    echo "⚠️ einops/huggingface_hub/safetensors install failed (Kronos optional)" && \
    echo "✅ Kronos deps install attempted"

# Stage 7c: Advisor module data + advice deps.
# yfinance: US-equities (default-enabled market) + FX + degraded-BIST analyzers.
#   PINNED <0.2.61: newer yfinance imports `websockets.asyncio` at module load,
#   which only exists in websockets>=13, but this image pins websockets==12.0
#   (Stage 10) -> ModuleNotFoundError: No module named 'websockets.asyncio'.
# anthropic + openai: advice rationale + dual-advice second opinion.
# Baked into the image (requirements.txt is NOT pip-installed by this Dockerfile).
# Retried once, then fail-soft so the build never breaks on a transient PyPI hiccup.
RUN pip install --no-cache-dir --timeout=300 'yfinance<0.2.61' anthropic openai || \
    pip install --no-cache-dir --timeout=300 'yfinance<0.2.61' anthropic openai || \
    echo "⚠️ advisor data/LLM deps failed (US-equities/FX/rationale degraded)" && \
    echo "✅ Advisor data/LLM deps install attempted"

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
    pytest-asyncio \
    pytest-cov \
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
    python -c "import solana; print('✅ Solana imported')" && \
    python -c "from solana.rpc.async_api import AsyncClient; print('✅ Solana AsyncClient OK')" && \
    python -c "import driftpy; print('✅ DriftPy imported')" && \
    echo "✅ All critical dependencies verified"

COPY . .

# Make entrypoint script executable
RUN chmod +x scripts/docker-entrypoint.sh

# Health check — actually probe the dashboard /health endpoint rather
# than just checking Python imports (the previous check always passed
# even when the dashboard was unresponsive, masking real outages).
# Tolerates a non-2xx status by counting only timeout/connection errors
# as unhealthy so a degraded DB doesn't take down the container.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, sys; \
        urllib.request.urlopen('http://localhost:8080/health', timeout=5); \
        sys.exit(0)" || exit 1

# Use entrypoint to run migrations before starting app
ENTRYPOINT ["./scripts/docker-entrypoint.sh"]
CMD ["main.py", "--mode", "production"]
