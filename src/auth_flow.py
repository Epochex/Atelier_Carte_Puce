from __future__ import annotations

import os
import time
import hashlib
import cv2
from dataclasses import dataclass
from typing import Optional

from .config import AppConfig
from .db import get_user_by_card, log_auth, is_locked, record_pin_failure, clear_auth_state
from .security import verify_pin, verify_file_sha256, build_audit_context, encode_audit_context, compact_reason
from .camera import CameraParams, capture_frame
from .bio import compare_biometric


@dataclass
class AuthResult:
    decision: str
    reason: str
    user_id: Optional[str] = None
    bio_score: Optional[float] = None


def _sha256_8_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).digest()[:8].hex()


def _tpl_hash8_from_sha256_hex(sha256_hex: str) -> Optional[str]:
    h = (sha256_hex or "").strip().lower()
    if len(h) != 64:
        return None
    try:
        return bytes.fromhex(h)[:8].hex()
    except ValueError:
        return None


def _ctx(**kwargs):
    return encode_audit_context(build_audit_context(extra=kwargs))


def run_auth_flow(
    cfg: AppConfig,
    conn,
    card_id: str,
    pin: str,
    card_atr: Optional[str] = None,
    card_user_hash8: Optional[str] = None,
    card_tpl_hash8: Optional[str] = None,
) -> AuthResult:
    user = get_user_by_card(conn, card_id)
    if not user:
        log_auth(conn, card_id, card_atr, None, False, None, "DENY", compact_reason("unknown_card", _ctx(card_id=card_id)))
        return AuthResult(decision="DENY", reason="unknown_card")

    user_id = user["user_id"]

    now_epoch = int(time.time())
    locked, locked_until = is_locked(conn, user_id, now_epoch=now_epoch)
    if locked:
        log_auth(conn, card_id, card_atr, user_id, False, None, "DENY", compact_reason("locked_out", _ctx(card_id=card_id, user_id=user_id, locked_until_epoch=locked_until)))
        return AuthResult(decision="DENY", reason="locked_out", user_id=user_id)

    template_path = user.get("template_path")
    template_sha256 = user.get("template_sha256")

    if cfg.auth.enforce_template_integrity:
        if not template_path or not template_sha256:
            log_auth(conn, card_id, card_atr, user_id, False, None, "DENY", compact_reason("no_biometric_template", _ctx(card_id=card_id, user_id=user_id)))
            return AuthResult(decision="DENY", reason="no_biometric_template", user_id=user_id)

        if not verify_file_sha256(template_path, template_sha256):
            log_auth(conn, card_id, card_atr, user_id, False, None, "DENY", compact_reason("template_tampered", _ctx(card_id=card_id, user_id=user_id, template_path=template_path)))
            return AuthResult(decision="DENY", reason="template_tampered", user_id=user_id)

    if cfg.auth.enforce_card_binding:
        if not card_user_hash8 or not card_tpl_hash8:
            log_auth(conn, card_id, card_atr, user_id, False, None, "DENY", compact_reason("card_binding_missing", _ctx(card_id=card_id, user_id=user_id)))
            return AuthResult(decision="DENY", reason="card_binding_missing", user_id=user_id)

        expected_user_hash8 = _sha256_8_hex(user_id)
        expected_tpl_hash8 = _tpl_hash8_from_sha256_hex(template_sha256) if template_sha256 else None

        cu = (card_user_hash8 or "").strip().lower()
        ct = (card_tpl_hash8 or "").strip().lower()

        if cu != expected_user_hash8:
            log_auth(conn, card_id, card_atr, user_id, False, None, "DENY", compact_reason("card_user_binding_mismatch", _ctx(card_id=card_id, user_id=user_id, expected=expected_user_hash8, got=cu)))
            return AuthResult(decision="DENY", reason="card_user_binding_mismatch", user_id=user_id)

        if expected_tpl_hash8 is None or ct != expected_tpl_hash8:
            log_auth(conn, card_id, card_atr, user_id, False, None, "DENY", compact_reason("card_template_binding_mismatch", _ctx(card_id=card_id, user_id=user_id, expected=expected_tpl_hash8, got=ct)))
            return AuthResult(decision="DENY", reason="card_template_binding_mismatch", user_id=user_id)

    pwd_ok = verify_pin(pin, user["pwd_salt"], user["pwd_hash"])
    if not pwd_ok:
        st = record_pin_failure(conn, user_id, now_epoch=now_epoch, max_attempts=cfg.auth.max_pin_attempts, lockout_seconds=cfg.auth.lockout_seconds)
        if st.get("locked_until_epoch") and int(st["locked_until_epoch"]) > now_epoch:
            log_auth(conn, card_id, card_atr, user_id, False, None, "DENY", compact_reason("locked_out", _ctx(card_id=card_id, user_id=user_id, locked_until_epoch=st["locked_until_epoch"], fail_count=st["fail_count"])))
            return AuthResult(decision="DENY", reason="locked_out", user_id=user_id)
        log_auth(conn, card_id, card_atr, user_id, False, None, "DENY", compact_reason("bad_pin", _ctx(card_id=card_id, user_id=user_id, fail_count=st["fail_count"])))
        return AuthResult(decision="DENY", reason="bad_pin", user_id=user_id)

    clear_auth_state(conn, user_id)

    if int(cfg.auth.required_factors) <= 2:
        log_auth(conn, card_id, card_atr, user_id, True, None, "ALLOW", compact_reason("ok_2fa", _ctx(card_id=card_id, user_id=user_id)))
        return AuthResult(decision="ALLOW", reason="ok_2fa", user_id=user_id, bio_score=None)

    frame = capture_frame(CameraParams(
        index=cfg.camera.index,
        warmup_frames=cfg.camera.warmup_frames,
        width=cfg.camera.width,
        height=cfg.camera.height,
    ))

    if template_path and os.path.exists(template_path):
        template = cv2.imread(template_path)
    else:
        template = None

    if template is None:
        log_auth(conn, card_id, card_atr, user_id, True, None, "DENY", compact_reason("template_read_error", _ctx(card_id=card_id, user_id=user_id, template_path=template_path)))
        return AuthResult(decision="DENY", reason="template_read_error", user_id=user_id)

    score = compare_biometric(
        captured_bgr=frame,
        template_bgr=template,
        use_face_crop=cfg.biometric.use_face_crop,
        nfeatures=cfg.biometric.orb_nfeatures,
    )

    if score >= cfg.biometric.score_threshold:
        log_auth(conn, card_id, card_atr, user_id, True, score, "ALLOW", compact_reason("ok", _ctx(card_id=card_id, user_id=user_id, bio_score=score)))
        return AuthResult(decision="ALLOW", reason="ok", user_id=user_id, bio_score=score)

    log_auth(conn, card_id, card_atr, user_id, True, score, "DENY", compact_reason("biometric_mismatch", _ctx(card_id=card_id, user_id=user_id, bio_score=score)))
    return AuthResult(decision="DENY", reason="biometric_mismatch", user_id=user_id, bio_score=score)
