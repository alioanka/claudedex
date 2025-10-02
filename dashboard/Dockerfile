# Dashboard Dockerfile - For separate dashboard container if needed
FROM node:18-alpine AS frontend-builder

WORKDIR /app/dashboard

# Copy dashboard frontend files
COPY dashboard/package*.json ./
RUN npm install

COPY dashboard/static ./static
COPY dashboard/templates ./templates

# Build/minify frontend assets if needed
RUN npm run build || true

# Python backend stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/dashboard/static ./dashboard/static

# Create necessary directories
RUN mkdir -p logs data config models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DASHBOARD_HOST=0.0.0.0
ENV DASHBOARD_PORT=8080

# Expose dashboard port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Run the application
CMD ["python", "main.py"]