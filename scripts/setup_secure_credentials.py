#!/usr/bin/env python3
"""
Secure Credentials Setup Script for ClaudeDex

This script helps you set up a secure credential management system
where the encryption key is stored SEPARATELY from the encrypted data.

Usage:
    python scripts/setup_secure_credentials.py --init       # Initialize new setup
    python scripts/setup_secure_credentials.py --add        # Add a credential
    python scripts/setup_secure_credentials.py --list       # List credential keys (not values)
    python scripts/setup_secure_credentials.py --export-env # Export to env format

SECURITY NOTES:
    - The encryption key is stored in /secure/encryption.key (NOT in .env)
    - Encrypted credentials are stored in credentials.encrypted
    - NEVER commit either file to git
    - NEVER store the key and encrypted data on the same backup
"""

import os
import sys
import json
import argparse
import getpass
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from cryptography.fernet import Fernet
except ImportError:
    print("Error: cryptography package not installed")
    print("Run: pip install cryptography")
    sys.exit(1)


class SecureCredentialManager:
    """Manage encrypted credentials with separate key storage"""

    def __init__(self, key_file: str = None, creds_file: str = None):
        self.key_file = Path(key_file or os.getenv('ENCRYPTION_KEY_FILE', '/secure/encryption.key'))
        self.creds_file = Path(creds_file or os.getenv('CREDENTIALS_FILE', 'credentials.encrypted'))
        self.fernet = None
        self._data = {}

    def initialize(self) -> bool:
        """Initialize new secure credential store"""
        print("\n" + "="*60)
        print("SECURE CREDENTIALS INITIALIZATION")
        print("="*60)

        # Check if already initialized
        if self.key_file.exists():
            response = input(f"\nKey file already exists at {self.key_file}. Overwrite? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return False

        # Create key directory with secure permissions
        key_dir = self.key_file.parent
        if not key_dir.exists():
            print(f"\nCreating secure directory: {key_dir}")
            key_dir.mkdir(parents=True, mode=0o700)

        # Generate new encryption key
        print("\nGenerating new encryption key...")
        key = Fernet.generate_key()

        # Save key with secure permissions
        print(f"Saving key to: {self.key_file}")
        with open(self.key_file, 'wb') as f:
            f.write(key)
        os.chmod(self.key_file, 0o600)

        # Initialize empty credentials file
        self.fernet = Fernet(key)
        self._data = {
            '_metadata': {
                'created_at': datetime.utcnow().isoformat(),
                'version': 1
            }
        }
        self._save()

        print("\n" + "-"*60)
        print("INITIALIZATION COMPLETE")
        print("-"*60)
        print(f"Key file:         {self.key_file}")
        print(f"Credentials file: {self.creds_file}")
        print("\nIMPORTANT:")
        print("  1. NEVER commit these files to git")
        print("  2. NEVER store the key and encrypted file together in backups")
        print("  3. Back up the key file to a SEPARATE secure location")
        print("-"*60)

        return True

    def load(self) -> bool:
        """Load existing credentials"""
        if not self.key_file.exists():
            print(f"Error: Key file not found: {self.key_file}")
            print("Run with --init to initialize first")
            return False

        with open(self.key_file, 'rb') as f:
            key = f.read().strip()

        self.fernet = Fernet(key)

        if self.creds_file.exists():
            with open(self.creds_file, 'rb') as f:
                encrypted = f.read()
            try:
                decrypted = self.fernet.decrypt(encrypted)
                self._data = json.loads(decrypted)
            except Exception as e:
                print(f"Error decrypting credentials: {e}")
                return False
        else:
            self._data = {'_metadata': {'version': 1}}

        return True

    def _save(self):
        """Save encrypted credentials"""
        data = json.dumps(self._data).encode()
        encrypted = self.fernet.encrypt(data)
        with open(self.creds_file, 'wb') as f:
            f.write(encrypted)
        os.chmod(self.creds_file, 0o600)

    def add_credential(self, key: str = None, value: str = None, category: str = 'general'):
        """Add or update a credential"""
        if not self.fernet:
            if not self.load():
                return False

        if not key:
            key = input("Credential key (e.g., BINANCE_API_KEY): ").strip()

        if not key:
            print("Error: Key cannot be empty")
            return False

        if not value:
            # Use getpass for sensitive input
            value = getpass.getpass(f"Value for {key} (hidden): ").strip()

        if not value:
            print("Error: Value cannot be empty")
            return False

        # Store credential
        self._data[key] = {
            'value': value,
            'category': category,
            'added_at': datetime.utcnow().isoformat(),
            'rotated_at': None
        }

        self._save()
        print(f"Credential '{key}' saved successfully")
        return True

    def get_credential(self, key: str) -> str:
        """Get a credential value"""
        if not self.fernet:
            if not self.load():
                return None

        cred = self._data.get(key)
        if cred and isinstance(cred, dict):
            return cred.get('value')
        return cred

    def list_credentials(self):
        """List all credential keys (not values)"""
        if not self.fernet:
            if not self.load():
                return

        print("\n" + "="*60)
        print("STORED CREDENTIALS")
        print("="*60)

        categories = {}
        for key, data in self._data.items():
            if key.startswith('_'):
                continue
            cat = data.get('category', 'general') if isinstance(data, dict) else 'general'
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(key)

        for cat, keys in sorted(categories.items()):
            print(f"\n[{cat.upper()}]")
            for key in sorted(keys):
                print(f"  - {key}")

        print("\n" + "="*60)
        print(f"Total: {len(self._data) - 1} credentials")  # -1 for metadata
        print("="*60)

    def export_env_format(self, output_file: str = None):
        """Export credentials in .env format (for migration)"""
        if not self.fernet:
            if not self.load():
                return

        lines = ["# Exported credentials - DO NOT COMMIT TO GIT\n"]
        lines.append(f"# Exported at: {datetime.utcnow().isoformat()}\n\n")

        for key, data in sorted(self._data.items()):
            if key.startswith('_'):
                continue
            value = data.get('value') if isinstance(data, dict) else data
            lines.append(f"{key}={value}\n")

        if output_file:
            with open(output_file, 'w') as f:
                f.writelines(lines)
            os.chmod(output_file, 0o600)
            print(f"Exported to: {output_file}")
        else:
            print("".join(lines))

    def import_from_env(self, env_file: str):
        """Import credentials from .env file"""
        if not self.fernet:
            if not self.load():
                return False

        if not os.path.exists(env_file):
            print(f"Error: File not found: {env_file}")
            return False

        # Categories for common credential types
        categories = {
            'PRIVATE_KEY': 'wallet',
            'API_KEY': 'api',
            'API_SECRET': 'api',
            'SECRET': 'security',
            'PASSWORD': 'database',
            'TOKEN': 'api',
        }

        count = 0
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")

                    # Skip empty values and placeholders
                    if not value or value.startswith('your_') or value == 'null':
                        continue

                    # Determine category
                    cat = 'general'
                    for pattern, category in categories.items():
                        if pattern in key.upper():
                            cat = category
                            break

                    self._data[key] = {
                        'value': value,
                        'category': cat,
                        'added_at': datetime.utcnow().isoformat(),
                        'imported_from': env_file
                    }
                    count += 1

        self._save()
        print(f"Imported {count} credentials from {env_file}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Secure Credentials Manager for ClaudeDex',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Initialize new secure store:
    python scripts/setup_secure_credentials.py --init

  Add a credential:
    python scripts/setup_secure_credentials.py --add

  Add credential non-interactively:
    python scripts/setup_secure_credentials.py --add --key BINANCE_API_KEY --value "your_key"

  List stored credentials:
    python scripts/setup_secure_credentials.py --list

  Import from .env file:
    python scripts/setup_secure_credentials.py --import-env .env

  Get a credential value:
    python scripts/setup_secure_credentials.py --get BINANCE_API_KEY
        """
    )

    parser.add_argument('--init', action='store_true', help='Initialize new secure credential store')
    parser.add_argument('--add', action='store_true', help='Add a credential')
    parser.add_argument('--key', help='Credential key (for --add)')
    parser.add_argument('--value', help='Credential value (for --add)')
    parser.add_argument('--category', default='general', help='Credential category (for --add)')
    parser.add_argument('--list', action='store_true', help='List all credential keys')
    parser.add_argument('--get', metavar='KEY', help='Get a credential value')
    parser.add_argument('--export-env', metavar='FILE', help='Export to .env format')
    parser.add_argument('--import-env', metavar='FILE', help='Import from .env file')
    parser.add_argument('--key-file', help='Path to encryption key file')
    parser.add_argument('--creds-file', help='Path to credentials file')

    args = parser.parse_args()

    # Create manager
    manager = SecureCredentialManager(
        key_file=args.key_file,
        creds_file=args.creds_file
    )

    # Handle commands
    if args.init:
        manager.initialize()
    elif args.add:
        manager.add_credential(key=args.key, value=args.value, category=args.category)
    elif args.list:
        manager.list_credentials()
    elif args.get:
        value = manager.get_credential(args.get)
        if value:
            print(value)
        else:
            print(f"Credential not found: {args.get}", file=sys.stderr)
            sys.exit(1)
    elif args.export_env:
        manager.export_env_format(args.export_env)
    elif args.import_env:
        manager.import_from_env(args.import_env)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
