# ---------- stage 1: build TA-Lib C and Python wheel ----------
  FROM python:3.11-slim AS talib-build
  ARG TA_VER=0.4.0
  ENV DEBIAN_FRONTEND=noninteractive \
      MAKEFLAGS=-j1 \
      TA_INCLUDE_PATH=/usr/include \
      TA_LIBRARY_PATH=/usr/lib \
      PIP_DISABLE_PIP_VERSION_CHECK=1 \
      PIP_ROOT_USER_ACTION=ignore
  
  RUN set -eux; \
      apt-get update; \
      apt-get install -y --no-install-recommends \
        ca-certificates curl wget build-essential gcc g++ make \
        autoconf automake libtool patch file; \
      update-ca-certificates; \
      # build TA-Lib C
      curl -fsSL "https://downloads.sourceforge.net/ta-lib/ta-lib-${TA_VER}-src.tar.gz" -o /tmp/ta-lib.tgz; \
      mkdir -p /tmp/ta-src; \
      tar -xzf /tmp/ta-lib.tgz -C /tmp/ta-src --strip-components=1; \
      cd /tmp/ta-src; \
      ./configure --prefix=/usr CFLAGS='-O2 -fPIC'; \
      make; \
      make install; \
      ldconfig || true; \
      rm -rf /tmp/ta-* /var/lib/apt/lists/*; \
      \
      # build Python TA-Lib wheel WITHOUT build isolation (fast) and capture logs
      python -m pip install -U pip wheel setuptools; \
      pip install --no-cache-dir "numpy<2"; \
      pip wheel -vv --no-binary :all: --no-build-isolation TA-Lib==0.4.32 -w /wheels \
        2>&1 | tee /tmp/ta_wheel.log
  
  # ---------- stage 2: runtime ----------
  FROM python:3.11-slim
  WORKDIR /app
  ENV PYTHONPATH=/app \
      PIP_DISABLE_PIP_VERSION_CHECK=1 \
      PIP_ROOT_USER_ACTION=ignore
  
  # TA-Lib C artifacts
  COPY --from=talib-build /usr/lib/ /usr/lib/
  COPY --from=talib-build /usr/include/ /usr/include/
  # Prebuilt Python TA-Lib wheel
  COPY --from=talib-build /wheels /wheels
  
  # Python deps
  COPY requirements.txt .
  RUN pip install --no-cache-dir "numpy<2" \
   && pip install --no-cache-dir /wheels/TA_Lib-0.4.32-*.whl \
   && pip install --no-cache-dir -r requirements.txt
  
  # Quick sanity check
  RUN python -c "import talib, numpy; print('talib', talib.__version__, 'numpy', numpy.__version__)"
  
  # App code
  COPY . .
  
  CMD ["python", "main.py", "--mode", "production"]
  