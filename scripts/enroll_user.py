import os
import argparse
import cv2

from src.config import load_config
from src.db import connect, init_db, upsert_user, upsert_biometric
from src.security import pbkdf2_hash_pin
from src.camera import CameraParams, capture_frame
from src.bio import sha256_file
from src.card import open_card


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    conn = connect(cfg.db_path)
    init_db(conn)

    pin = input("Enter PIN/mot de passe: ").strip()
    if not pin:
        raise SystemExit("PIN empty")

    salt, ph = pbkdf2_hash_pin(pin)

    os.makedirs("data/templates", exist_ok=True)

    frame = capture_frame(CameraParams(
        index=cfg.camera.index,
        warmup_frames=cfg.camera.warmup_frames,
        width=cfg.camera.width,
        height=cfg.camera.height,
    ))

    template_path = f"data/templates/{args.user_id}.png"
    cv2.imwrite(template_path, frame)
    tpl_sha = sha256_file(template_path)

    s = open_card(10)
    card_uid, wrote = s.provision_or_load_uid(user_id=args.user_id, tpl_sha256_hex=tpl_sha)

    print(f"[card] card_id={card_uid} atr={s.atr_hex}")
    print(f"[card] write_app_record={wrote}")

    upsert_user(conn, args.user_id, card_uid, salt, ph)
    upsert_biometric(conn, args.user_id, template_path, tpl_sha, algo="ORB+facecrop")

    print(f"ENROLL OK: user_id={args.user_id}, template={template_path}")


if __name__ == "__main__":
    main()
