# ---------- stage 1: build TA-Lib C (race-free) ----------
  FROM python:3.11-slim AS talib-c
  ARG TA_VER=0.4.0
  ENV DEBIAN_FRONTEND=noninteractive \
      MAKEFLAGS=-j1
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
      make; \
      make install; \
      ldconfig || true; \
      rm -rf /tmp/ta-* /var/lib/apt/lists/*
  
  # ---------- stage 2: build Python TA-Lib wheel ----------
  FROM python:3.11-slim AS talib-wheel
  ENV DEBIAN_FRONTEND=noninteractive
  # bring in the C headers/libs so pip can link against them
  COPY --from=talib-c /usr/include/ /usr/include/
  COPY --from=talib-c /usr/lib/ /usr/lib/
  RUN set -eux; \
      apt-get update; \
      apt-get install -y --no-install-recommends build-essential gcc g++; \
      python -m pip install -U pip wheel setuptools; \
      pip install --no-cache-dir "numpy<2"; \
      pip wheel --no-binary :all: TA-Lib==0.4.32 -w /wheels; \
      ls -l /wheels; \
      apt-get purge -y --auto-remove build-essential gcc g++; \
      rm -rf /var/lib/apt/lists/*
  
  # ---------- stage 3: runtime ----------
  FROM python:3.11-slim
  WORKDIR /app
  ENV PYTHONPATH=/app
  
  # TA-Lib C artifacts
  COPY --from=talib-c /usr/lib/ /usr/lib/
  COPY --from=talib-c /usr/include/ /usr/include/
  
  # Prebuilt Python TA-Lib wheel
  COPY --from=talib-wheel /wheels /wheels
  
  # Python deps
  COPY requirements.txt .
  RUN pip install --no-cache-dir "numpy<2" \
   && pip install --no-cache-dir /wheels/TA_Lib-0.4.32-*.whl \
   && pip install --no-cache-dir -r requirements.txt
  
  # sanity check
  RUN python -c "import talib, numpy; print('talib', talib.__version__, 'numpy', numpy.__version__)"
  
  # app code
  COPY . .
  
  CMD ["python", "main.py", "--mode", "production"]
  