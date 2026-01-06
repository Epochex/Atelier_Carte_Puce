# FILE: /home/vboxuser/Desktop/CartePuce/src/auth_flow.py
from __future__ import annotations
import os
import cv2
from dataclasses import dataclass
from typing import Optional

from .config import AppConfig
from .db import get_user_by_card, log_auth
from .security import verify_pin
from .camera import CameraParams, capture_frame
from .bio import compare_biometric


@dataclass
class AuthResult:
    decision: str  # "ALLOW" | "DENY"
    reason: str
    user_id: Optional[str] = None
    bio_score: Optional[float] = None


def run_auth_flow(cfg: AppConfig, conn, card_id: str, card_atr: Optional[str], pin: str) -> AuthResult:
    """
    card_id: actually the card_uid (read or initialized from the card via APDU)
    """
    user = get_user_by_card(conn, card_id)
    if not user:
        log_auth(conn, card_id, card_atr, None, False, None, "DENY", "unknown_card")
        return AuthResult(decision="DENY", reason="unknown_card")

    user_id = user["user_id"]

    # 1) PIN verification
    pwd_ok = verify_pin(pin, user["pwd_salt"], user["pwd_hash"])
    if not pwd_ok:
        log_auth(conn, card_id, card_atr, user_id, False, None, "DENY", "bad_pin")
        return AuthResult(decision="DENY", reason="bad_pin", user_id=user_id)

    # 2) camera catch
    frame = capture_frame(CameraParams(
        index=cfg.camera.index,
        warmup_frames=cfg.camera.warmup_frames,
        width=cfg.camera.width,
        height=cfg.camera.height,
    ))

    # 3) biologi model invoke
    template_path = user.get("template_path")
    if not template_path or not os.path.exists(template_path):
        log_auth(conn, card_id, card_atr, user_id, True, None, "DENY", "no_biometric_template")
        return AuthResult(decision="DENY", reason="no_biometric_template", user_id=user_id)

    template = cv2.imread(template_path)
    if template is None:
        log_auth(conn, card_id, card_atr, user_id, True, None, "DENY", "template_read_error")
        return AuthResult(decision="DENY", reason="template_read_error", user_id=user_id)

    # compartion
    score = compare_biometric(
        captured_bgr=frame,
        template_bgr=template,
        use_face_crop=cfg.biometric.use_face_crop,
        nfeatures=cfg.biometric.orb_nfeatures,
    )

    if score >= cfg.biometric.score_threshold:
        log_auth(conn, card_id, card_atr, user_id, True, score, "ALLOW", "ok")
        return AuthResult(decision="ALLOW", reason="ok", user_id=user_id, bio_score=score)

    log_auth(conn, card_id, card_atr, user_id, True, score, "DENY", "biometric_mismatch")
    return AuthResult(decision="DENY", reason="biometric_mismatch", user_id=user_id, bio_score=score)
