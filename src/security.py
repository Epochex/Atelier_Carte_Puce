from __future__ import annotations
import os
import hashlib
import hmac


def pbkdf2_hash_pin(pin: str, salt: bytes | None = None, iterations: int = 200_000) -> tuple[bytes, bytes]:
    if salt is None:
        salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, iterations, dklen=32)
    return salt, dk


def verify_pin(pin: str, salt: bytes, expected_hash: bytes, iterations: int = 200_000) -> bool:
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, iterations, dklen=32)
    return hmac.compare_digest(dk, expected_hash)
