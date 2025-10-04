# ---------- base stage: build TA-Lib C once ----------
  FROM python:3.11-slim AS talib-base
  ARG TA_VER=0.4.0
  ENV DEBIAN_FRONTEND=noninteractive
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
      export MAKEFLAGS=-j1; \
      make -j1; \
      make install; \
      ldconfig || true; \
      rm -rf /tmp/ta-* /var/lib/apt/lists/*; \
      \
      # build TA-Lib Python wheel here (compiler available)
      python -m pip install -U pip wheel setuptools && \
      pip install --no-cache-dir "numpy<2" && \
      pip wheel --no-binary :all: TA-Lib==0.4.32 -w /wheels
  
  # ---------- runtime image ----------
  FROM python:3.11-slim
  WORKDIR /app
  ENV PYTHONPATH=/app
  
  # TA-Lib C artifacts
  COPY --from=talib-base /usr/lib/ /usr/lib/
  COPY --from=talib-base /usr/include/ /usr/include/
  
  # Prebuilt TA-Lib wheel
  COPY --from=talib-base /wheels /wheels
  
  # python deps
  COPY requirements.txt .
  RUN pip install --no-cache-dir "numpy<2" \
   && pip install --no-cache-dir /wheels/TA_Lib-0.4.32-*.whl \
   && pip install --no-cache-dir -r requirements.txt
  
  # sanity check
  RUN python -c "import talib, numpy; print('talib', talib.__version__, 'numpy', numpy.__version__)"
  
  # app
  COPY . .
  
  CMD ["python", "main.py", "--mode", "production"]
  