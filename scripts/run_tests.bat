@echo off
REM Test Runner Script for Windows
REM Runs all tests, generates coverage reports, and runs linters

echo ==================================
echo   AI Research Assistant Test Suite
echo ==================================
echo.

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo [!] Virtual environment not activated. Activating...
    if exist venv\Scripts\activate.bat (
        call venv\Scripts\activate.bat
    ) else (
        echo [X] Virtual environment not found. Please create one first.
        exit /b 1
    )
)

REM Install/update test dependencies
echo [+] Installing test dependencies...
pip install -q pytest pytest-cov pytest-asyncio pytest-xdist pytest-timeout black flake8 mypy isort

REM Clean previous coverage data
echo [+] Cleaning previous coverage data...
if exist htmlcov rmdir /s /q htmlcov
if exist .coverage del .coverage
if exist coverage.xml del coverage.xml

echo.
echo ==================================
echo   1. Code Formatting (Black)
echo ==================================
echo.

black --check backend tests --exclude venv
if %errorlevel% equ 0 (
    echo [✓] Code formatting check passed
) else (
    echo [!] Code formatting issues found. Run 'black backend tests' to fix
)

echo.
echo ==================================
echo   2. Import Sorting (isort)
echo ==================================
echo.

isort --check-only backend tests --skip venv
if %errorlevel% equ 0 (
    echo [✓] Import sorting check passed
) else (
    echo [!] Import sorting issues found. Run 'isort backend tests' to fix
)

echo.
echo ==================================
echo   3. Linting (Flake8)
echo ==================================
echo.

flake8 backend tests --max-line-length=120 --exclude=venv --extend-ignore=E203,W503
if %errorlevel% equ 0 (
    echo [✓] Linting check passed
) else (
    echo [X] Linting issues found
)

echo.
echo ==================================
echo   4. Type Checking (MyPy)
echo ==================================
echo.

mypy backend --ignore-missing-imports --no-strict-optional
if %errorlevel% equ 0 (
    echo [✓] Type checking passed
) else (
    echo [!] Type checking issues found
)

echo.
echo ==================================
echo   5. Unit Tests
echo ==================================
echo.

pytest tests\unit -v -m "unit" --cov=backend --cov-report=term-missing
if %errorlevel% neq 0 (
    echo [X] Unit tests failed
    exit /b 1
)
echo [✓] Unit tests passed

echo.
echo ==================================
echo   6. Integration Tests
echo ==================================
echo.

REM Check if backend is running
curl -s http://localhost:8000/api/v1/health >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Backend service not running. Skipping integration tests.
    echo [!] Start services with: docker-compose up -d
) else (
    pytest tests\integration -v -m "integration" --cov=backend --cov-append --cov-report=term-missing
    if %errorlevel% neq 0 (
        echo [X] Integration tests failed
        exit /b 1
    )
    echo [✓] Integration tests passed
)

echo.
echo ==================================
echo   7. Performance Tests (Optional)
echo ==================================
echo.

set /p run_perf="Run performance tests? (y/N): "
if /i "%run_perf%"=="y" (
    pytest tests\performance -v -m "performance" --cov=backend --cov-append --cov-report=term-missing
    if %errorlevel% equ 0 (
        echo [✓] Performance tests passed
    ) else (
        echo [!] Some performance tests failed (this is informational)
    )
)

echo.
echo ==================================
echo   8. Coverage Report
echo ==================================
echo.

pytest --cov=backend --cov-report=html --cov-report=term --cov-report=xml --cov-fail-under=80 tests\unit tests\integration
if %errorlevel% equ 0 (
    echo [✓] Coverage threshold (80%%) met
) else (
    echo [X] Coverage below 80%% threshold
)

echo.
echo [+] HTML coverage report: htmlcov\index.html
echo [+] XML coverage report: coverage.xml

echo.
echo ==================================
echo   9. Test Summary
echo ==================================
echo.

REM Count test files
for /f %%i in ('dir /b /s tests\unit\test_*.py ^| find /c /v ""') do set unit_tests=%%i
for /f %%i in ('dir /b /s tests\integration\test_*.py ^| find /c /v ""') do set integration_tests=%%i
for /f %%i in ('dir /b /s tests\performance\test_*.py ^| find /c /v ""') do set performance_tests=%%i

echo Test files:
echo   Unit tests: %unit_tests%
echo   Integration tests: %integration_tests%
echo   Performance tests: %performance_tests%
echo.

REM Open coverage report in browser (optional)
set /p open_report="Open coverage report in browser? (y/N): "
if /i "%open_report%"=="y" (
    start htmlcov\index.html
)

echo.
echo [✓] All tests completed!
echo.
