#!/bin/bash

# Test Runner Script
# Runs all tests, generates coverage reports, and runs linters

set -e  # Exit on error

echo "=================================="
echo "  AI Research Assistant Test Suite"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "Virtual environment not activated. Activating..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    else
        print_error "Virtual environment not found. Please create one first."
        exit 1
    fi
fi

# Install/update test dependencies
print_status "Installing test dependencies..."
pip install -q pytest pytest-cov pytest-asyncio pytest-xdist pytest-timeout black flake8 mypy isort

# Clean previous coverage data
print_status "Cleaning previous coverage data..."
rm -rf htmlcov .coverage coverage.xml

echo ""
echo "=================================="
echo "  1. Code Formatting (Black)"
echo "=================================="
echo ""

if black --check backend/ tests/ --exclude venv; then
    print_status "Code formatting check passed"
else
    print_warning "Code formatting issues found. Run 'black backend/ tests/' to fix"
fi

echo ""
echo "=================================="
echo "  2. Import Sorting (isort)"
echo "=================================="
echo ""

if isort --check-only backend/ tests/ --skip venv; then
    print_status "Import sorting check passed"
else
    print_warning "Import sorting issues found. Run 'isort backend/ tests/' to fix"
fi

echo ""
echo "=================================="
echo "  3. Linting (Flake8)"
echo "=================================="
echo ""

if flake8 backend/ tests/ --max-line-length=120 --exclude=venv --extend-ignore=E203,W503; then
    print_status "Linting check passed"
else
    print_error "Linting issues found"
fi

echo ""
echo "=================================="
echo "  4. Type Checking (MyPy)"
echo "=================================="
echo ""

if mypy backend/ --ignore-missing-imports --no-strict-optional; then
    print_status "Type checking passed"
else
    print_warning "Type checking issues found"
fi

echo ""
echo "=================================="
echo "  5. Unit Tests"
echo "=================================="
echo ""

pytest tests/unit/ -v -m "unit" --cov=backend --cov-report=term-missing

if [ $? -eq 0 ]; then
    print_status "Unit tests passed"
else
    print_error "Unit tests failed"
    exit 1
fi

echo ""
echo "=================================="
echo "  6. Integration Tests"
echo "=================================="
echo ""

# Check if services are running
if ! curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    print_warning "Backend service not running. Skipping integration tests."
    print_warning "Start services with: docker-compose up -d"
else
    pytest tests/integration/ -v -m "integration" --cov=backend --cov-append --cov-report=term-missing
    
    if [ $? -eq 0 ]; then
        print_status "Integration tests passed"
    else
        print_error "Integration tests failed"
        exit 1
    fi
fi

echo ""
echo "=================================="
echo "  7. Performance Tests (Optional)"
echo "=================================="
echo ""

read -p "Run performance tests? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pytest tests/performance/ -v -m "performance" --cov=backend --cov-append --cov-report=term-missing
    
    if [ $? -eq 0 ]; then
        print_status "Performance tests passed"
    else
        print_warning "Some performance tests failed (this is informational)"
    fi
fi

echo ""
echo "=================================="
echo "  8. Coverage Report"
echo "=================================="
echo ""

# Generate coverage reports
pytest --cov=backend --cov-report=html --cov-report=term --cov-report=xml --cov-fail-under=80 tests/unit/ tests/integration/

if [ $? -eq 0 ]; then
    print_status "Coverage threshold (80%) met"
else
    print_error "Coverage below 80% threshold"
fi

echo ""
print_status "HTML coverage report: htmlcov/index.html"
print_status "XML coverage report: coverage.xml"

echo ""
echo "=================================="
echo "  9. Test Summary"
echo "=================================="
echo ""

# Count test files
unit_tests=$(find tests/unit -name "test_*.py" | wc -l)
integration_tests=$(find tests/integration -name "test_*.py" | wc -l)
performance_tests=$(find tests/performance -name "test_*.py" | wc -l)

echo "Test files:"
echo "  Unit tests: $unit_tests"
echo "  Integration tests: $integration_tests"
echo "  Performance tests: $performance_tests"
echo ""

# Open coverage report in browser (optional)
if command -v xdg-open > /dev/null; then
    read -p "Open coverage report in browser? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        xdg-open htmlcov/index.html
    fi
elif command -v open > /dev/null; then
    read -p "Open coverage report in browser? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open htmlcov/index.html
    fi
fi

echo ""
print_status "All tests completed!"
echo ""
