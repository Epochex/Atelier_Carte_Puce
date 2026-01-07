# src/security/__init__.py
"""
Security package.

This package centralizes:
- Password hashing / verification (PBKDF2 + optional pepper)
- HMAC-based challenge-response primitives (nonce/counter/context/tag)
- Replay protection (nonce cache, in-memory demo-grade)
- Template integrity checks (anti-tamper)
- Audit context encoding (compact, log-friendly)

Backward compatibility:
Existing imports like:
    from src.security import pbkdf2_hash_pin, verify_pin
should continue to work.
"""

from .password_hashing import pbkdf2_hash_pin, verify_pin
from .hmac_challenge_response import (
    generate_nonce,
    compute_hmac_tag,
    verify_hmac_tag,
    new_card_key,
    key_id_from_key,
    build_hmac_message,
)
from .replay_protection import NonceReplayProtector
from .template_integrity import sha256_file_hex, verify_file_sha256
from .audit_logging import (
    build_audit_context,
    encode_audit_context,
    compact_reason,
)
