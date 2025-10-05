FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY requirements.txt .

# Install all requirements including newer TA-Lib
RUN pip install --no-cache-dir -r requirements.txt

# Smoke test
RUN python -c "import talib; print('TA-Lib version:', talib.__version__)"

COPY . .

CMD ["python", "main.py", "--mode", "production"]