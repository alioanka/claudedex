# scripts/deploy.sh
#!/bin/bash

# Deployment script for production

set -e

echo "🚀 Starting deployment..."

# Configuration
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Pre-deployment checks
pre_deploy_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found"
        exit 1
    fi
    
    # Check environment file
    if [ ! -f ".env.$ENVIRONMENT" ]; then
        log_error "Environment file .env.$ENVIRONMENT not found"
        exit 1
    fi
    
    # Run tests
    log_info "Running tests..."
    pytest tests/smoke/ -v
    
    if [ $? -ne 0 ]; then
        log_error "Tests failed"
        exit 1
    fi
}

# Build application
build() {
    log_info "Building application..."
    
    docker build -t trading-bot:$VERSION .
    
    if [ $? -ne 0 ]; then
        log_error "Build failed"
        exit 1
    fi
    
    log_info "Build successful"
}

# Deploy application
deploy() {
    log_info "Deploying to $ENVIRONMENT..."
    
    # Stop existing containers
    docker-compose -f docker-compose.$ENVIRONMENT.yml down
    
    # Start new containers
    docker-compose -f docker-compose.$ENVIRONMENT.yml up -d
    
    # Wait for health check
    sleep 10
    
    # Verify deployment
    if curl -f http://localhost:8080/health; then
        log_info "Deployment successful"
    else
        log_error "Health check failed"
        exit 1
    fi
}

# Post-deployment tasks
post_deploy() {
    log_info "Running post-deployment tasks..."
    
    # Run migrations
    docker exec trading-bot python scripts/migrate.py
    
    # Warm up cache
    docker exec trading-bot python scripts/warm_cache.py
    
    # Send notification
    curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
        -H 'Content-Type: application/json' \
        -d "{\"text\":\"🚀 Trading Bot $VERSION deployed to $ENVIRONMENT\"}"
}

# Main execution
main() {
    pre_deploy_checks
    build
    deploy
    post_deploy
    
    log_info "✅ Deployment complete!"
}

main