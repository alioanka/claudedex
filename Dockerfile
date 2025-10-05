# ---------- build libTA-Lib + wheel ----------
  FROM python:3.11-slim AS talib-build
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
    cd /tmp/ta-src; ./configure --prefix=/usr CFLAGS='-O2 -fPIC'; make; make install; ldconfig || true; \
    cd /; rm -rf /tmp/ta-* /var/lib/apt/lists/*; \
    python -m pip install -U pip wheel setuptools; \
    pip install --no-cache-dir "numpy<2"; \
    pip wheel --no-binary :all: --no-build-isolation TA-Lib==0.4.32 -w /wheels
  
  # ---------- runtime ----------
  FROM python:3.11-slim
  WORKDIR /app
  ENV PYTHONPATH=/app
  
  # bring in the TA-Lib shared libs first
  COPY --from=talib-build /usr/lib/ /usr/lib/
  COPY --from=talib-build /usr/include/ /usr/include/
  
  # deps
  COPY requirements.txt .
  RUN python -m pip install -U pip; \
      pip install --no-cache-dir "numpy<2" /wheels/TA_Lib-0.4.32-*.whl; \
      pip install --no-cache-dir -r requirements.txt
  
  # app
  COPY . .
  CMD ["python", "main.py", "--mode", "production"]
  