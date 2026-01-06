# src/security/replay_protection.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ReplayDecision:
    ok: bool
    reason: str


class NonceReplayProtector:
    """
    In-memory nonce replay protector.
    Scope: current process only.

    Strategy:
      - (card_id, nonce_hex) -> expiry_ts
      - If seen and not expired => replay

    Parameters:
      ttl_seconds: how long a nonce is considered valid for anti-replay purposes.
      max_entries: crude memory bound; oldest entries are evicted when over limit.
    """

    def __init__(self, ttl_seconds: int = 120, max_entries: int = 4096):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds_must_be_positive")
        if max_entries < 64:
            raise ValueError("max_entries_too_small")
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: Dict[Tuple[str, str], float] = {}

    def _purge_expired(self, now: float) -> None:
        expired = [k for k, exp in self._store.items() if exp <= now]
        for k in expired:
            self._store.pop(k, None)

    def _evict_if_needed(self) -> None:
        if len(self._store) <= self.max_entries:
            return
        # Evict earliest-expiring items first
        items = sorted(self._store.items(), key=lambda kv: kv[1])
        for (k, _exp) in items[: max(1, len(self._store) - self.max_entries)]:
            self._store.pop(k, None)

    def check_and_remember(self, card_id: str, nonce: bytes) -> ReplayDecision:
        now = time.time()
        self._purge_expired(now)

        nonce_hex = bytes(nonce).hex()
        key = (card_id, nonce_hex)

        if key in self._store:
            return ReplayDecision(ok=False, reason="replay_nonce_seen")

        self._store[key] = now + self.ttl_seconds
        self._evict_if_needed()
        return ReplayDecision(ok=True, reason="ok")
