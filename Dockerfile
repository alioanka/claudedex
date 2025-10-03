FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*


# ---- TA-Lib C library (required for Python talib) ----
    RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ wget ca-certificates \
&& update-ca-certificates \
&& wget -q https://github.com/TA-Lib/ta-lib/releases/download/0.4.0/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib.tgz \
&& tar -xzf /tmp/ta-lib.tgz -C /tmp \
&& cd /tmp/ta-lib && ./configure --prefix=/usr && make && make install \
&& cd / && rm -rf /tmp/ta-lib* \
&& rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/config && \
    chmod -R 777 /app/logs /app/data /app/config

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Run as root for now (fix permissions later)
CMD ["python", "main.py", "--mode", "production"]