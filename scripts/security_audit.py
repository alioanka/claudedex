"""Security audit script"""
import asyncio
import os
import json
from pathlib import Path
import hashlib
from datetime import datetime

async def security_audit():
    """Perform comprehensive security audit"""
    
    print("SECURITY AUDIT")
    print("="*60)
    
    issues = []
    warnings = []
    
    # Check environment variables
    print("\nChecking environment configuration...")
    critical_vars = [
        'DB_PASSWORD', 'WALLET_PRIVATE_KEY', 'API_SECRET_KEY',
        'ENCRYPTION_KEY', 'JWT_SECRET'
    ]
    
    for var in critical_vars:
        value = os.getenv(var)
        if not value:
            issues.append(f"Missing critical env var: {var}")
        elif len(value) < 32:
            warnings.append(f"Weak {var} (less than 32 chars)")
    
    if issues:
        print("   Critical issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("   Environment vars OK")
    
    # Check file permissions
    print("\nChecking file permissions...")
    sensitive_files = [
        '.env',
        'config/security.yaml',
        'config/wallets.yaml'
    ]
    
    for file in sensitive_files:
        if os.path.exists(file):
            stat = os.stat(file)
            mode = oct(stat.st_mode)[-3:]
            if mode != '600':
                warnings.append(f"{file} has insecure permissions: {mode}")
    
    if not any('permissions' in w for w in warnings):
        print("   File permissions OK")
    
    # Check for exposed secrets in code
    print("\nScanning for exposed secrets...")
    patterns = [
        'password', 'secret', 'private_key', 'api_key', 'token'
    ]
    
    exposed_count = 0
    for py_file in Path('.').rglob('*.py'):
        if 'venv' in str(py_file) or 'tests' in str(py_file):
            continue
        
        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().lower()
            for pattern in patterns:
                if f'{pattern}=' in content or f'{pattern} =' in content:
                    if 'getenv' not in content and 'config' not in str(py_file):
                        exposed_count += 1
                        warnings.append(f"Possible hardcoded secret in {py_file}")
                        break
    
    if exposed_count == 0:
        print("   No exposed secrets found")
    else:
        print(f"   Found {exposed_count} potential issues")
    
    # Check database security
    print("\nChecking database security...")
    db_user = os.getenv('DB_USER', '')
    if db_user in ['root', 'postgres', 'admin']:
        warnings.append("Using privileged database user")
    
    if not os.getenv('DB_SSL_MODE'):
        warnings.append("Database SSL not configured")
    
    # Check API rate limiting
    print("\nChecking API security...")
    if not os.getenv('RATE_LIMIT_ENABLED'):
        warnings.append("Rate limiting not enabled")
    
    # Check wallet security
    print("\nChecking wallet security...")
    wallet_file = Path('config/wallets.yaml')
    if wallet_file.exists():
        with open(wallet_file, 'r') as f:
            content = f.read()
            if 'private_key' in content.lower():
                issues.append("Private keys may be stored in plaintext")
    
    # Check log security
    print("\nChecking log security...")
    for log_file in Path('logs').rglob('*.log'):
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().lower()
            if 'private' in content or 'secret' in content:
                warnings.append(f"Sensitive data in logs: {log_file}")
                break
    
    # Generate report
    print("\n" + "="*60)
    print("AUDIT SUMMARY")
    print("="*60)
    
    print(f"\nCritical Issues: {len(issues)}")
    for issue in issues:
        print(f"  - {issue}")
    
    print(f"\nWarnings: {len(warnings)}")
    for warning in warnings[:10]:  # Show first 10
        print(f"  - {warning}")
    
    if len(warnings) > 10:
        print(f"  ... and {len(warnings) - 10} more")
    
    # Save report
    report_dir = Path('reports')
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f'security_audit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'critical_issues': issues,
        'warnings': warnings,
        'status': 'FAILED' if issues else 'PASSED'
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {report_file}")
    
    if issues:
        print("\nSTATUS: FAILED - Critical issues must be resolved")
        return False
    elif warnings:
        print("\nSTATUS: PASSED WITH WARNINGS")
        return True
    else:
        print("\nSTATUS: PASSED")
        return True

if __name__ == "__main__":
    asyncio.run(security_audit())