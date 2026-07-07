-- Migration 146: seed Secure-Credentials cards for the Wave-F5 multi-key slots
--
-- The /credentials page renders one card per secure_credentials row; the
-- Wave-F5 multi-key rotation (numbered HELIUS_API_KEY_2..4, ETHERSCAN_API_KEY_2)
-- and the new BIRDEYE_API provider (mig 144) never got card rows, so the
-- operator had no UI field to enter them. pool_engine probes these EXACT key
-- names at startup and registers each as its own rotating endpoint.
-- Mirrors the mig 082 pattern (PLACEHOLDER rows, idempotent).

INSERT INTO secure_credentials
    (key_name, display_name, description, category, subcategory, module, is_required, is_sensitive, encrypted_value)
VALUES
('HELIUS_API_KEY_2', 'Helius API Key #2',
 'Second Helius key for multi-key rotation. Used round-robin with the other '
 'Helius keys by solana/sniper/copy-trading; a 429 on one key rotates to the '
 'next. Free-tier accounts: dashboard.helius.dev.',
 'api', 'helius', NULL, FALSE, TRUE, 'PLACEHOLDER'),
('HELIUS_API_KEY_3', 'Helius API Key #3',
 'Third Helius key for multi-key rotation (see HELIUS_API_KEY_2).',
 'api', 'helius', NULL, FALSE, TRUE, 'PLACEHOLDER'),
('HELIUS_API_KEY_4', 'Helius API Key #4',
 'Fourth Helius key for multi-key rotation (see HELIUS_API_KEY_2).',
 'api', 'helius', NULL, FALSE, TRUE, 'PLACEHOLDER'),
('ETHERSCAN_API_KEY_2', 'Etherscan API Key #2',
 'Second Etherscan key for multi-key rotation (V2 API, one key covers 50+ '
 'chains). Used by copy-trading EVM wallet monitoring/discovery.',
 'api', 'etherscan', NULL, FALSE, TRUE, 'PLACEHOLDER'),
('BIRDEYE_API_KEY', 'Birdeye API Key',
 'Birdeye (birdeye.so) data key — unlocks the Birdeye candidate source in '
 'copy-trading wallet discovery. Free tier available. Optional; discovery '
 'degrades gracefully without it.',
 'api', 'birdeye', NULL, FALSE, TRUE, 'PLACEHOLDER'),
('BIRDEYE_API_KEY_2', 'Birdeye API Key #2',
 'Second Birdeye key for multi-key rotation. Optional.',
 'api', 'birdeye', NULL, FALSE, TRUE, 'PLACEHOLDER')
ON CONFLICT (key_name) DO NOTHING;

-- DOWN (reversible):
-- DELETE FROM secure_credentials WHERE key_name IN
--   ('HELIUS_API_KEY_2','HELIUS_API_KEY_3','HELIUS_API_KEY_4',
--    'ETHERSCAN_API_KEY_2','BIRDEYE_API_KEY','BIRDEYE_API_KEY_2')
--   AND encrypted_value='PLACEHOLDER';
