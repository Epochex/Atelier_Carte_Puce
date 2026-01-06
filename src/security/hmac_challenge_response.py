# src/security/hmac_challenge_response.py
from __future__ import annotations

import hashlib
import hmac
import os
from typing import Tuple


_HMAC_VERSION_PREFIX = b"CP-AUTH-HMAC-V1\x00"


def generate_nonce(length: int = 16) -> bytes:
    """
    Random nonce for each authentication attempt.
    16 bytes is sufficient for demo-grade replay protection.
    """
    if length < 12:
        raise ValueError("nonce_length_too_small")
    return os.urandom(length)


def new_card_key(length: int = 32) -> bytes:
    """
    Generate per-card secret key K_card.
    """
    if length < 16:
        raise ValueError("card_key_length_too_small")
    return os.urandom(length)


def key_id_from_key(k_card: bytes, length: int = 8) -> str:
    """
    Derive a public key identifier (not secret) from K_card.
    Can be stored on card as key_id to locate DB record.

    Default: 8 bytes (16 hex chars).
    """
    if length < 4:
        raise ValueError("key_id_length_too_small")
    digest = hashlib.sha256(k_card).digest()
    return digest[:length].hex()


def _card_uid_hex_to_bytes(card_uid_hex: str) -> bytes:
    s = (card_uid_hex or "").strip().lower()
    if len(s) != 32:
        # Your current card_uid is 16 bytes hex => 32 chars
        raise ValueError("card_uid_expected_16_bytes_hex")
    return bytes.fromhex(s)


def build_hmac_message(
    card_uid_hex: str,
    nonce: bytes,
    counter: int,
    context: str,
) -> bytes:
    """
    Canonical message: prefix || card_uid(16B) || nonce || counter(4B BE) || context(utf-8)
    """
    if not isinstance(nonce, (bytes, bytearray)) or len(nonce) < 12:
        raise ValueError("invalid_nonce")
    if counter < 0 or counter > 0xFFFFFFFF:
        raise ValueError("counter_out_of_range")

    card_uid = _card_uid_hex_to_bytes(card_uid_hex)
    counter_be = int(counter).to_bytes(4, "big", signed=False)
    ctx = (context or "").encode("utf-8")

    return _HMAC_VERSION_PREFIX + card_uid + bytes(nonce) + counter_be + ctx


def compute_hmac_tag(k_card: bytes, message: bytes, tag_len: int = 32) -> bytes:
    """
    Compute HMAC-SHA256 tag (optionally truncated).
    """
    if not isinstance(k_card, (bytes, bytearray)) or len(k_card) < 16:
        raise ValueError("invalid_k_card")
    full = hmac.new(bytes(k_card), message, hashlib.sha256).digest()
    if tag_len <= 0 or tag_len > len(full):
        raise ValueError("invalid_tag_len")
    return full[:tag_len]


def verify_hmac_tag(k_card: bytes, message: bytes, tag: bytes) -> bool:
    """
    Constant-time verification.
    """
    if not isinstance(tag, (bytes, bytearray)) or len(tag) == 0:
        return False
    expected = compute_hmac_tag(k_card, message, tag_len=len(tag))
    return hmac.compare_digest(expected, bytes(tag))
