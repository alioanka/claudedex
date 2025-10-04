# ---------- base stage: compile TA-Lib once ----------
  FROM python:3.11-slim AS talib-base
  ARG TA_VER=0.4.0
  RUN set -eux; \
      apt-get update; \
      apt-get install -y --no-install-recommends \
        ca-certificates curl wget build-essential gcc g++ make \
        autoconf automake libtool patch file; \
      update-ca-certificates; \
      curl -fsSL "https://downloads.sourceforge.net/ta-lib/ta-lib-${TA_VER}-src.tar.gz" -o /tmp/ta-lib.tgz; \
      mkdir -p /tmp/ta-src; \
      tar -xzf /tmp/ta-lib.tgz -C /tmp/ta-src --strip-components=1; \
      cd /tmp/ta-src; \
      ./configure --prefix=/usr CFLAGS='-O2 -fPIC'; \
      make -j"$(nproc)"; \
      make install; \
      ldconfig || true; \
      rm -rf /tmp/ta-* /var/lib/apt/lists/*
  
  # ---------- runtime image ----------
  FROM python:3.11-slim
  WORKDIR /app
  ENV PYTHONPATH=/app
  
  # copy only the compiled TA-Lib artifacts from base
  COPY --from=talib-base /usr/lib/ /usr/lib/
  COPY --from=talib-base /usr/include/ /usr/include/
  COPY --from=talib-base /usr/lib/pkgconfig/ /usr/lib/pkgconfig/
  
  # python deps (pre-pin numpy then TA-Lib to be safe)
  COPY requirements.txt .
  RUN pip install --no-cache-dir "numpy<2" "TA-Lib==0.4.32" && \
      pip install --no-cache-dir -r requirements.txt
  
  # optional build-time import smoke test
  RUN python - <<'PY'
  import talib, numpy; print("talib", talib.__version__, "numpy", numpy.__version__)
  PY
  
  # app
  COPY . .
  CMD ["python", "main.py", "--mode", "production"]
  