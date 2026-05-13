# secrets_manager.get() cache-encrypted-skip audit

**Date**: 2026-05-13. **Result**: NOT a bug. Behavior is intentional
defensive safety.

## Finding
`security/secrets_manager.py:214-301` (sync `get()`) and `:393+`
(async `get_async()`) both skip the cache when a cached value starts
with the Fernet ciphertext prefix `gAAAAAB`.

## Why this is correct
1. L290 (sync) / L466 (async) GUARD cache writes — only non-encrypted
   plaintext is ever written to `_cache`.
2. The read-side skip is a defense-in-depth check: if encrypted state
   ever leaks into `_cache` (test monkeypatching, future bug, manual
   `_cache[key] = ciphertext`), the skip causes a fresh fetch which
   re-decrypts properly via the Docker/DB/env code paths.
3. The final safety check at L277-287 decrypts any in-flight encrypted
   value before return so callers never see ciphertext.

## Latent observability gap (addressed in this commit)
If the cache-plaintext invariant is ever violated, the silent re-fetch
swallows the anomaly. Symptoms would surface as a performance
regression (cache hit ratio drops), not a correctness bug. Added a
`logger.warning(...)` on the unexpected-cache-encrypted path so future
regressions are noisy. Test added to assert the safety net behavior.

## Verified
- Cache writes (L290) reject encrypted values.
- Cache reads (L231-235, L406-409) reject encrypted values.
- Final return path (L277-287, L460+) decrypts any encrypted leftovers.
- New regression test in tests/integration/test_secrets_migration.py
  ::TestSecretsManagerContract::test_cache_encrypted_value_triggers_refetch_with_warning
