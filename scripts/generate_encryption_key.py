#!/usr/bin/env python3
"""
Generate a new Fernet encryption key for ENCRYPTION_KEY in .env

Usage:
    python scripts/generate_encryption_key.py

This will print a new encryption key that you can add to your .env file as:
    ENCRYPTION_KEY=<generated_key>

WARNING: If you change your ENCRYPTION_KEY, all previously encrypted data
         in the database will become unreadable! You must re-run
         import_env_secrets.py after changing the key.
"""

from cryptography.fernet import Fernet

def main():
    # Generate a new Fernet key
    key = Fernet.generate_key().decode()

    print("=" * 80)
    print("NEW ENCRYPTION KEY GENERATED")
    print("=" * 80)
    print()
    print("Add this line to your .env file:")
    print()
    print(f"ENCRYPTION_KEY={key}")
    print()
    print("⚠️  IMPORTANT:")
    print("1. Save this key securely - you cannot recover encrypted data without it")
    print("2. After updating .env, run: python scripts/import_env_secrets.py")
    print("3. This will re-encrypt all your secrets with the new key")
    print("4. Rebuild your bot: docker-compose up -d --build")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
