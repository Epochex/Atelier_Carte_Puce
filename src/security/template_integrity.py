# src/security/template_integrity.py
from __future__ import annotations

import hashlib
import os


def sha256_file_hex(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file_sha256(path: str, expected_hex: str) -> bool:
    """
    Returns True if the file exists and matches expected sha256 hex.
    """
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        return False
    expected = (expected_hex or "").strip().lower()
    if len(expected) != 64:
        return False
    actual = sha256_file_hex(path)
    return actual == expected
