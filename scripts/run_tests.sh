# scripts/run_tests.sh
#!/bin/bash
"""
Shell script to run tests with various options
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 Trading Bot Test Runner"
echo "=========================="

# Parse command line arguments
TEST_TYPE=${1:-all}
COVERAGE=${2:-yes}
PARALLEL=${3:-yes}

# Function to run tests
run_tests() {
    local test_args=""
    
    # Add base arguments
    test_args="-v --tb=short"
    
    # Add coverage if requested
    if [ "$COVERAGE" = "yes" ]; then
        test_args="$test_args --cov=. --cov-report=html --cov-report=term-missing"
    fi
    
    # Add parallel execution if requested
    if [ "$PARALLEL" = "yes" ]; then
        test_args="$test_args -n auto"
    fi
    
    # Run specific test type
    case $TEST_TYPE in
        unit)
            echo "Running unit tests..."
            pytest $test_args tests/unit/
            ;;
        integration)
            echo "Running integration tests..."
            pytest $test_args tests/integration/
            ;;
        performance)
            echo "Running performance tests..."
            pytest $test_args tests/performance/ --benchmark-only
            ;;
        security)
            echo "Running security tests..."
            pytest $test_args tests/security/
            ;;
        smoke)
            echo "Running smoke tests..."
            pytest $test_args -m smoke tests/
            ;;
        all)
            echo "Running all tests..."
            pytest $test_args tests/
            ;;
        *)
            echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
            echo "Usage: ./run_tests.sh [unit|integration|performance|security|smoke|all] [yes|no] [yes|no]"
            exit 1
            ;;
    esac
    
    return $?
}

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated${NC}"
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install test dependencies if needed
if ! pip show pytest > /dev/null 2>&1; then
    echo "Installing test dependencies..."
    pip install -r test-requirements.txt
fi

# Start required services
echo "Starting test services..."
docker-compose -f docker-compose.test.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 5

# Run tests
run_tests
TEST_RESULT=$?

# Stop test services
echo "Stopping test services..."
docker-compose -f docker-compose.test.yml down

# Generate reports
if [ "$COVERAGE" = "yes" ] && [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ Tests passed!${NC}"
    echo "📊 Coverage report available at: htmlcov/index.html"
    
    # Open coverage report in browser (optional)
    if command -v open > /dev/null; then
        open htmlcov/index.html
    elif command -v xdg-open > /dev/null; then
        xdg-open htmlcov/index.html
    fi
else
    echo -e "${RED}❌ Tests failed!${NC}"
fi

exit $TEST_RESULT
