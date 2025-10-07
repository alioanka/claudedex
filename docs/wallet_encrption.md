Good progress! Now the issue is that your ENCRYPTION_KEY is not a valid Fernet key.
From your .env:
bashENCRYPTION_KEY=OOODS8kYrcjwsLK67Zg13GwH9CrzYAiE
This is only 32 characters, but Fernet requires 44 characters of base64-encoded data (which represents 32 bytes).
Solution 1: Generate a Proper Fernet Key
Run this Python command to generate a valid key:
bashpython3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
This will output something like:
8vB2XqN-jKdE5F7G9H0I1J2K3L4M5N6O7P8Q9R0S1T4=
Copy that and update your .env:
bashENCRYPTION_KEY=8vB2XqN-jKdE5F7G9H0I1J2K3L4M5N6O7P8Q9R0S1T4=
Solution 2: Use Your Existing Key (Convert It)
Or use the EncryptionManager.encrypt_data() method which handles key derivation. Update the decryption code in main.py:
python# Replace the decryption block with:
decrypted_key = encrypted_key
if encrypted_key and encrypted_key.startswith('gAAAAAB') and encryption_key:
    try:
        # Use EncryptionManager's method which handles key derivation
        decrypted_key = security_manager.decrypt_data(encrypted_key, encryption_key)
        self.logger.info("✅ Successfully decrypted private key")
    except Exception as e:
        self.logger.error(f"Failed to decrypt private key: {e}")
        raise ValueError("Cannot decrypt PRIVATE_KEY - check ENCRYPTION_KEY")
But the easiest fix is to just generate a new proper Fernet key and re-encrypt your private key with it:
bash# Generate new Fernet key
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Then encrypt your private key with it:
python3 << EOF
from cryptography.fernet import Fernet
key = b'YOUR_NEW_FERNET_KEY_HERE'  # From above
f = Fernet(key)
private_key = 'PRIVATE_KEY'
encrypted = f.encrypt(private_key.encode())
print(encrypted.decode())
EOF
Then update your .env with both the new key and encrypted private key.