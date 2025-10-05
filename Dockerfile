# ---------- base: build TA-Lib C ----------
  FROM python:3.11-slim AS talib-c
  ARG TA_VER=0.4.0
  ENV DEBIAN_FRONTEND=noninteractive
  RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates curl wget build-essential gcc g++ make \
      autoconf automake libtool patch file; \
    update-ca-certificates; \
    curl -fsSL "https://downloads.sourceforge.net/ta-lib/ta-lib-${TA_VER}-src.tar.gz" -o /tmp/ta-lib.tgz; \
    mkdir -p /tmp/ta-src; tar -xzf /tmp/ta-lib.tgz -C /tmp/ta-src --strip-components=1; \
    cd /tmp/ta-src; ./configure --prefix=/usr CFLAGS='-O2 -fPIC'; \
    make -j1; make install; ldconfig || true; \
    cd /; rm -rf /tmp/ta-src /tmp/ta-lib.tgz /var/lib/apt/lists/*
  
  # ---------- runtime ----------
  FROM python:3.11-slim
  WORKDIR /app
  ENV PYTHONPATH=/app \
      PIP_PREFER_BINARY=1 \
      PIP_ONLY_BINARY=numpy,TA-Lib
  
  # bring in TA-Lib C
  COPY --from=talib-c /usr/lib/ /usr/lib/
  COPY --from=talib-c /usr/include/ /usr/include/
  
  # python deps (use wheels for numpy & TA-Lib)
  COPY requirements.txt .
  RUN pip install --no-cache-dir "numpy<2" "TA-Lib==0.4.32" \
   && pip install --no-cache-dir -r requirements.txt
  
  # quick import smoke test
  RUN python -c "import talib, numpy; print('talib', talib.__version__, 'numpy', numpy.__version__)"
  
  # app
  COPY . .
  CMD ["python", "main.py", "--mode", "production"]
  