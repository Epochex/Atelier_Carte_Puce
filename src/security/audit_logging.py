# src/security/audit_logging.py
from __future__ import annotations

import json
import os
import platform
from typing import Any, Dict, Optional


def _device_identity() -> str:
    """
    Best-effort device identifier for audit logs.
    Not a security guarantee; used to support threat analysis (stolen reader / tampered terminal).
    """
    host = platform.node() or "unknown-host"
    sysname = platform.system() or "unknown-os"
    return f"{sysname}:{host}"


def build_audit_context(
    *,
    nonce: Optional[bytes] = None,
    counter: Optional[int] = None,
    key_id: Optional[str] = None,
    tag: Optional[bytes] = None,
    context: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable dict for audit logs.
    """
    d: Dict[str, Any] = {
        "device": _device_identity(),
    }

    if nonce is not None:
        d["nonce"] = bytes(nonce).hex()
    if counter is not None:
        d["counter"] = int(counter)
    if key_id is not None:
        d["key_id"] = str(key_id)
    if tag is not None:
        d["tag"] = bytes(tag).hex()
    if context is not None:
        d["context"] = str(context)

    if extra:
        # Ensure extra is JSON-serializable
        for k, v in extra.items():
            d[str(k)] = v

    return d


def encode_audit_context(ctx: Dict[str, Any], max_len: int = 512) -> str:
    """
    Encode context as compact JSON string.
    If too long, truncate deterministically.
    """
    s = json.dumps(ctx, separators=(",", ":"), ensure_ascii=False)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def compact_reason(reason: str, audit_context_json: Optional[str] = None) -> str:
    """
    Pack reason + audit context into a single string suitable for existing auth_logs.reason.

    Example:
      "bad_pin|ctx={...}"
    """
    r = (reason or "").strip() or "unknown"
    if not audit_context_json:
        return r
    return f"{r}|ctx={audit_context_json}"
