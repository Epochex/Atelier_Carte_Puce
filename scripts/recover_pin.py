import hashlib
import getpass
import cv2

from src.config import load_config
from src.db import (
    connect,
    init_db,
    get_user_by_card,
    update_user_pin,
    clear_auth_state,
    log_auth,
)
from src.security import pbkdf2_hash_pin, verify_file_sha256
from src.camera import CameraParams, capture_frame
from src.bio import compare_biometric
from src.card import open_card


def _sha256_8_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).digest()[:8].hex()


def _tpl_hash8_from_sha256_hex(sha256_hex: str):
    if not sha256_hex or len(sha256_hex) != 64:
        return None
    return bytes.fromhex(sha256_hex)[:8].hex()


def main():
    cfg = load_config("config.yaml")
    conn = connect(cfg.db_path)
    init_db(conn)

    s = open_card(10)
    card_id = s.get_uid()
    if not card_id:
        raise SystemExit("no_card")

    rec = s.read_app_record()
    if not rec:
        log_auth(conn, card_id, s.atr_hex, None, False, None, "DENY", "card_binding_missing")
        raise SystemExit("card_binding_missing")

    user = get_user_by_card(conn, card_id)
    if not user:
        log_auth(conn, card_id, s.atr_hex, None, False, None, "DENY", "unknown_card")
        raise SystemExit("unknown_card")

    user_id = user["user_id"]
    template_path = user.get("template_path")
    template_sha256 = user.get("template_sha256")

    if cfg.auth.enforce_template_integrity:
        if not verify_file_sha256(template_path, template_sha256):
            log_auth(conn, card_id, s.atr_hex, user_id, False, None, "DENY", "template_tampered")
            raise SystemExit("template_tampered")

    if cfg.auth.enforce_card_binding:
        if rec.user_hash8.lower() != _sha256_8_hex(user_id):
            raise SystemExit("card_user_binding_mismatch")
        if rec.tpl_hash8.lower() != _tpl_hash8_from_sha256_hex(template_sha256):
            raise SystemExit("card_template_binding_mismatch")

    frame = capture_frame(CameraParams(
        index=cfg.camera.index,
        warmup_frames=cfg.camera.warmup_frames,
        width=cfg.camera.width,
        height=cfg.camera.height,
    ))

    template = cv2.imread(template_path)
    score = compare_biometric(
        captured_bgr=frame,
        template_bgr=template,
        use_face_crop=cfg.biometric.use_face_crop,
        nfeatures=cfg.biometric.orb_nfeatures,
    )

    if score < cfg.biometric.score_threshold:
        raise SystemExit("biometric_mismatch")

    new_pin1 = getpass.getpass("Enter new PIN/mot de passe: ").strip()
    new_pin2 = getpass.getpass("Re-enter new PIN/mot de passe: ").strip()
    if not new_pin1 or new_pin1 != new_pin2:
        raise SystemExit("pin_mismatch")

    salt, ph = pbkdf2_hash_pin(new_pin1)
    update_user_pin(conn, user_id, salt, ph)
    clear_auth_state(conn, user_id)

    log_auth(conn, card_id, s.atr_hex, user_id, True, score, "ALLOW", "pin_recovered")
    print(f"PIN recovered: user_id={user_id} card_id={card_id} bio_score={score:.3f}")


if __name__ == "__main__":
    main()
