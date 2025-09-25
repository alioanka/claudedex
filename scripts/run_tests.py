# scripts/run_tests.py
"""
Python test runner with advanced features
"""
import sys
import os
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Optional
import json

class TestRunner:
    """Advanced test runner for the trading bot"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        
    def setup_environment(self):
        """Setup test environment"""
        print("🔧 Setting up test environment...")
        
        # Add project to path
        sys.path.insert(0, str(self.project_root))
        
        # Check for test database
        if not self.check_test_database():
            self.create_test_database()
        
        # Check for Redis
        if not self.check_redis():
            print("⚠️  Redis not running. Starting Redis container...")
            self.start_redis()
    
    def check_test_database(self) -> bool:
        """Check if test database exists"""
        try:
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                database="tradingbot_test",
                user="test_user",
                password="test_password"
            )
            conn.close()
            return True
        except:
            return False
    
    def create_test_database(self):
        """Create test database"""
        print("📦 Creating test database...")
        subprocess.run([
            "psql", "-U", "postgres", "-c",
            "CREATE DATABASE tradingbot_test;"
        ])
    
    def check_redis(self) -> bool:
        """Check if Redis is running"""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379)
            r.ping()
            return True
        except:
            return False
    
    def start_redis(self):
        """Start Redis container"""
        subprocess.run([
            "docker", "run", "-d", "--name", "redis-test",
            "-p", "6379:6379", "redis:alpine"
        ])
        time.sleep(2)  # Wait for Redis to start
    
    def run_tests(
        self,
        test_type: str = "all",
        coverage: bool = True,
        parallel: bool = True,
        verbose: bool = True,
        markers: Optional[List[str]] = None
    ):
        """Run tests with specified options"""
        
        # Build pytest command
        cmd = ["pytest"]
        
        # Add verbosity
        if verbose:
            cmd.append("-v")
        
        # Add coverage
        if coverage:
            cmd.extend([
                "--cov=.",
                "--cov-report=html",
                "--cov-report=json",
                "--cov-report=term-missing"
            ])
        
        # Add parallel execution
        if parallel:
            cmd.extend(["-n", "auto"])
        
        # Add markers
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        # Add test path based on type
        if test_type == "unit":
            cmd.append("tests/unit/")
        elif test_type == "integration":
            cmd.append("tests/integration/")
        elif test_type == "performance":
            cmd.extend(["tests/performance/", "--benchmark-only"])
        elif test_type == "security":
            cmd.append("tests/security/")
        elif test_type == "smoke":
            cmd.extend(["-m", "smoke", "tests/"])
        else:
            cmd.append("tests/")
        
        # Add output formats
        cmd.extend([
            "--junit-xml=test-results.xml",
            "--html=test-report.html",
            "--self-contained-html"
        ])
        
        # Run tests
        print(f"🧪 Running {test_type} tests...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse results
        self.parse_results(result)
        
        return result.returncode
    
    def parse_results(self, result):
        """Parse test results"""
        # Parse coverage if available
        coverage_file = self.project_root / "coverage.json"
        if coverage_file.exists():
            with open(coverage_file) as f:
                coverage_data = json.load(f)
                self.test_results["coverage"] = coverage_data["totals"]["percent_covered"]
        
        # Parse test counts from output
        output_lines = result.stdout.split("\n")
        for line in output_lines:
            if "passed" in line and "failed" in line:
                # Extract test counts
                parts = line.split()
                for i, part in enumerate(parts):
                    if "passed" in part:
                        self.test_results["passed"] = int(parts[i-1])
                    if "failed" in part:
                        self.test_results["failed"] = int(parts[i-1])
    
    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*60)
        print("📊 TEST REPORT")
        print("="*60)
        
        if "passed" in self.test_results:
            print(f"✅ Passed: {self.test_results.get('passed', 0)}")
        
        if "failed" in self.test_results:
            print(f"❌ Failed: {self.test_results.get('failed', 0)}")
        
        if "coverage" in self.test_results:
            coverage = self.test_results["coverage"]
            if coverage >= 80:
                print(f"📈 Coverage: {coverage:.1f}% 🎉")
            elif coverage >= 60:
                print(f"📈 Coverage: {coverage:.1f}% ⚠️")
            else:
                print(f"📈 Coverage: {coverage:.1f}% ❌")
        
        print("\n📁 Reports generated:")
        print("  - HTML Coverage: htmlcov/index.html")
        print("  - Test Report: test-report.html")
        print("  - JUnit XML: test-results.xml")
        print("="*60)
    
    def cleanup(self):
        """Cleanup test environment"""
        print("\n🧹 Cleaning up...")
        
        # Stop test containers
        subprocess.run(["docker", "stop", "redis-test"], capture_output=True)
        subprocess.run(["docker", "rm", "redis-test"], capture_output=True)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run trading bot tests")
    parser.add_argument(
        "type",
        choices=["all", "unit", "integration", "performance", "security", "smoke"],
        default="all",
        nargs="?",
        help="Type of tests to run"
    )
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel execution")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("-m", "--markers", nargs="+", help="Run tests with specific markers")
    
    args = parser.parse_args()
    
    # Create runner
    runner = TestRunner()
    
    try:
        # Setup environment
        runner.setup_environment()
        
        # Run tests
        exit_code = runner.run_tests(
            test_type=args.type,
            coverage=not args.no_coverage,
            parallel=not args.no_parallel,
            verbose=not args.quiet,
            markers=args.markers
        )
        
        # Generate report
        runner.generate_report()
        
        return exit_code
        
    finally:
        # Always cleanup
        runner.cleanup()

if __name__ == "__main__":
    sys.exit(main())