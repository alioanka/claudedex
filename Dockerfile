FROM python:3.11-slim

WORKDIR /app

# System deps + TA-Lib C (source build with mirror fallback; stable path)
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates curl wget gcc g++ make build-essential postgresql-client; \
    update-ca-certificates; \
    TA_GH="https://github.com/TA-Lib/ta-lib/releases/download/0.4.0/ta-lib-0.4.0-src.tar.gz"; \
    TA_SF="https://downloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"; \
    # download with fallback
    (curl -fL "$TA_GH" -o /tmp/ta-lib.tgz || curl -fL "$TA_SF" -o /tmp/ta-lib.tgz); \
    test -s /tmp/ta-lib.tgz; \
    # extract into a known folder regardless of archive’s top-level dir name
    mkdir -p /tmp/ta-src; \
    tar -xzf /tmp/ta-lib.tgz -C /tmp/ta-src --strip-components=1; \
    cd /tmp/ta-src; \
    ./configure --prefix=/usr; \
    make -j"$(nproc)"; \
    make install; \
    # cleanup
    rm -rf /tmp/ta-* /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY . .

# App dirs & perms
RUN mkdir -p /app/logs /app/data /app/config && \
    chmod -R 777 /app/logs /app/data /app/config

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

CMD ["python", "main.py", "--mode", "production"]
