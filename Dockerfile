FROM python:3.11-slim
WORKDIR /app

# --- System deps + TA-Lib (use APT if available, else source) ---
RUN set -eux; \
    apt-get update; \
    # add tools often needed for TA-Lib builds
    apt-get install -y --no-install-recommends \
      ca-certificates curl wget \
      build-essential gcc g++ make \
      autoconf automake libtool patch file \
      postgresql-client; \
    update-ca-certificates; \
    # try OS packages first (Debian/Ubuntu repos)
    if apt-get install -y --no-install-recommends libta-lib0 libta-lib-dev 2>/dev/null; then \
      echo "TA-Lib C installed via APT"; \
    else \
      echo "TA-Lib not in APT, building from source..."; \
      TA_GH="https://github.com/TA-Lib/ta-lib/releases/download/0.4.0/ta-lib-0.4.0-src.tar.gz"; \
      TA_SF="https://downloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"; \
      (curl -fL "$TA_GH" -o /tmp/ta-lib.tgz || curl -fL "$TA_SF" -o /tmp/ta-lib.tgz); \
      test -s /tmp/ta-lib.tgz; \
      mkdir -p /tmp/ta-src; \
      tar -xzf /tmp/ta-lib.tgz -C /tmp/ta-src --strip-components=1; \
      cd /tmp/ta-src; \
      ./configure --prefix=/usr; \
      make -j"$(nproc)"; \
      make install; \
      rm -rf /tmp/ta-*; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# --- Python deps: preinstall numpy + TA-Lib, then the rest ---
COPY requirements.txt .
RUN pip install --no-cache-dir "numpy<2" "TA-Lib==0.4.32"
RUN pip install --no-cache-dir -r requirements.txt

# Fail build immediately if talib isn't importable
RUN python - <<'PY'
import talib, numpy
print("BUILD CHECK talib", talib.__version__, "numpy", numpy.__version__)
PY

# --- App ---
COPY . .
RUN mkdir -p /app/logs /app/data /app/config && chmod -R 777 /app/logs /app/data /app/config

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1

CMD ["python", "main.py", "--mode", "production"]
