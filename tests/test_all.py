# tests/test_all.py
"""
Run all tests with coverage report
"""
import pytest
import sys
from pathlib import Path

def run_all_tests():
    """Run all test suites"""
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Test categories to run
    test_suites = [
        "tests/unit",
        "tests/integration", 
        "tests/performance",
        "tests/security",
        "tests/smoke"
    ]
    
    # Pytest arguments
    args = [
        "-v",  # Verbose
        "--tb=short",  # Short traceback
        "--cov=.",  # Coverage for entire project
        "--cov-report=html",  # HTML coverage report
        "--cov-report=term-missing",  # Terminal report with missing lines
        "--cov-report=json",  # JSON coverage for CI/CD
        "--junit-xml=test-results.xml",  # JUnit XML for CI/CD
        "--benchmark-autosave",  # Save benchmark results
        "--benchmark-compare",  # Compare with previous benchmarks
        "--maxfail=5",  # Stop after 5 failures
        "-n", "auto",  # Run tests in parallel (requires pytest-xdist)
    ] + test_suites
    
    # Run tests
    exit_code = pytest.main(args)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST EXECUTION COMPLETE")
    print("="*50)
    
    if exit_code == 0:
        print("✅ All tests passed!")
        print("📊 Coverage report: htmlcov/index.html")
        print("📈 Benchmark results: .benchmarks/")
    else:
        print("❌ Some tests failed. Check the output above.")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(run_all_tests())