import os
import argparse
import cv2

from src.config import load_config
from src.db import connect, init_db, upsert_user, upsert_biometric
from src.security import pbkdf2_hash_pin
from src.card import get_card_id
from src.camera import CameraParams, capture_frame
from src.bio import sha256_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True, help="e.g. alice")
    parser.add_argument("--simulate-card", default=None, help="use this as card_id (skip real card)")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    conn = connect(cfg.db_path)
    init_db(conn)

    # 1) 卡
    card = get_card_id(simulate_card=args.simulate_card, timeout_seconds=30)
    print(f"[card] card_id = {card.card_id}")

    # 2) PIN
    pin = input("Enter PIN/mot de passe: ").strip()
    if not pin:
        raise SystemExit("PIN empty")

    salt, ph = pbkdf2_hash_pin(pin)

    # 3) 摄像头采集模板
    os.makedirs("data/templates", exist_ok=True)
    frame = capture_frame(CameraParams(
        index=cfg.camera.index,
        warmup_frames=cfg.camera.warmup_frames,
        width=cfg.camera.width,
        height=cfg.camera.height,
    ))

    template_path = f"data/templates/{args.user_id}.png"
    cv2.imwrite(template_path, frame)
    t_sha = sha256_file(template_path)

    # 4) 写 DB
    upsert_user(conn, args.user_id, card.card_id, salt, ph)
    upsert_biometric(conn, args.user_id, template_path, t_sha, algo="ORB+facecrop")
    print(f"ENROLL OK: user_id={args.user_id}, template={template_path}")

if __name__ == "__main__":
    main()
