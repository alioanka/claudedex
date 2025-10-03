FROM python:3.11-slim

WORKDIR /app

# System deps + TA-Lib C (needed by Python TA-Lib)
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential gcc g++ make wget curl ca-certificates postgresql-client; \
    update-ca-certificates; \
    wget -q https://github.com/TA-Lib/ta-lib/releases/download/0.4.0/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib.tgz; \
    tar -xzf /tmp/ta-lib.tgz -C /tmp; \
    cd /tmp/ta-lib-0.4.0; \
    ./configure --prefix=/usr; \
    make -j"$(nproc)"; \
    make install; \
    rm -rf /tmp/ta-lib*; \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# App dirs & perms
RUN mkdir -p /app/logs /app/data /app/config && \
    chmod -R 777 /app/logs /app/data /app/config

# Healthcheck (adjust port if different)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

CMD ["python", "main.py", "--mode", "production"]
