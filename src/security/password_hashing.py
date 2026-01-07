# src/security/password_hashing.py
from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import Optional, Tuple


_DEFAULT_ITERATIONS = 200_000


def _load_pepper_bytes() -> bytes:
    """
    Optional server-side secret (pepper) to harden against offline guessing after DB compromise.
    - If unset: return b"" (no pepper).
    - If set: supports either raw string or base64 prefixed with "base64:".

    Environment variable:
      CARTEPUCE_PASSWORD_PEPPER
        examples:
          export CARTEPUCE_PASSWORD_PEPPER="my-long-random-pepper"
          export CARTEPUCE_PASSWORD_PEPPER="base64:8cS7...=="
    """
    v = (os.environ.get("CARTEPUCE_PASSWORD_PEPPER") or "").strip()
    if not v:
        return b""
    if v.startswith("base64:"):
        return base64.b64decode(v[len("base64:") :].encode("utf-8"))
    return v.encode("utf-8")


def pbkdf2_hash_pin(
    pin: str,
    salt: Optional[bytes] = None,
    iterations: int = _DEFAULT_ITERATIONS,
) -> Tuple[bytes, bytes]:
    """
    Return (salt, derived_key).

    NOTE:
    - The pepper is NOT stored in DB.
    - If you set pepper later, you should re-enroll users for strict correctness,
      but verify_pin() is backward-compatible by trying both modes.
    """
    if salt is None:
        salt = os.urandom(16)

    pepper = _load_pepper_bytes()
    material = pin.encode("utf-8") + pepper

    dk = hashlib.pbkdf2_hmac("sha256", material, salt, iterations, dklen=32)
    return salt, dk


def verify_pin(
    pin: str,
    salt: bytes,
    expected_hash: bytes,
    iterations: int = _DEFAULT_ITERATIONS,
) -> bool:
    """
    Constant-time verification.

    Backward compatibility strategy:
    - If pepper is configured, try with pepper first.
    - If fails and pepper is configured, try without pepper (older DB entries).
    """
    pepper = _load_pepper_bytes()

    # Try with current pepper mode
    material = pin.encode("utf-8") + pepper
    dk = hashlib.pbkdf2_hmac("sha256", material, salt, iterations, dklen=32)
    if hmac.compare_digest(dk, expected_hash):
        return True

    # Fallback: no-pepper compatibility
    if pepper:
        dk2 = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, iterations, dklen=32)
        return hmac.compare_digest(dk2, expected_hash)

    return False
