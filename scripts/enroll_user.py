import os
import time
import argparse
import cv2
import getpass

from src.config import load_config
from src.db import connect, init_db, upsert_user, upsert_biometric
from src.security import pbkdf2_hash_pin
from src.camera import CameraParams, capture_frame
from src.bio import sha256_file, compare_biometric
from src.card import open_card


def _cap(cfg):
    return capture_frame(
        CameraParams(
            index=cfg.camera.index,
            warmup_frames=cfg.camera.warmup_frames,
            width=cfg.camera.width,
            height=cfg.camera.height,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--max-attempts", type=int, default=12, help="Max capture attempts for a usable template")
    parser.add_argument("--sleep-ms", type=int, default=250, help="Sleep between attempts (ms)")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Override biometric threshold for enrollment sanity check (default: cfg.biometric.score_threshold)",
    )
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    conn = connect(cfg.db_path)
    init_db(conn)

    pin = getpass.getpass("Enter PIN/mot de passe: ").strip()
    if not pin:
        raise SystemExit("PIN empty")

    salt, ph = pbkdf2_hash_pin(pin)
    os.makedirs("data/templates", exist_ok=True)

    threshold = args.min_score if args.min_score is not None else cfg.biometric.score_threshold

    # Capture a template that can PASS a live-vs-template sanity check immediately.
    best = None  # (score, template_frame_bgr)
    for i in range(1, args.max_attempts + 1):
        tpl_frame = _cap(cfg)

        # Basic extractability check: frame vs itself should be high.
        self_score = compare_biometric(
            captured_bgr=tpl_frame,
            template_bgr=tpl_frame,
            use_face_crop=cfg.biometric.use_face_crop,
            nfeatures=cfg.biometric.orb_nfeatures,
        )

        if self_score < 0.5:
            print(f"[ENROLL] attempt={i}/{args.max_attempts} reject: self-extract failed self_score={self_score:.3f}")
            time.sleep(args.sleep_ms / 1000.0)
            continue

        # Immediate sanity check: capture another live frame and compare to candidate template frame.
        live_frame = _cap(cfg)
        score = compare_biometric(
            captured_bgr=live_frame,
            template_bgr=tpl_frame,
            use_face_crop=cfg.biometric.use_face_crop,
            nfeatures=cfg.biometric.orb_nfeatures,
        )

        print(f"[ENROLL] attempt={i}/{args.max_attempts} sanity_score={score:.3f} (threshold={threshold:.3f})")

        if best is None or score > best[0]:
            best = (score, tpl_frame)

        if score >= threshold:
            # Good enough: finalize
            template_path = f"data/templates/{args.user_id}.png"
            cv2.imwrite(template_path, tpl_frame)
            tpl_sha = sha256_file(template_path)

            s = open_card(10)
            card_uid, wrote = s.provision_or_load_uid(
                user_id=args.user_id,
                tpl_sha256_hex=tpl_sha,
            )

            print(f"[card] card_id={card_uid} atr={s.atr_hex}")
            print(f"[card] write_app_record={wrote}")

            upsert_user(conn, args.user_id, card_uid, salt, ph, card_atr=s.atr_hex)
            upsert_biometric(conn, args.user_id, template_path, tpl_sha, algo="ENROLL+SANITYCHECK")

            print(f"ENROLL OK: user_id={args.user_id}, template={template_path}, sanity_score={score:.3f}")
            return

        time.sleep(args.sleep_ms / 1000.0)

    # If we get here, enrollment failed to reach threshold.
    if best is not None:
        best_score, best_frame = best
        template_path = f"data/templates/{args.user_id}.png"
        cv2.imwrite(template_path, best_frame)
        print(f"[ENROLL] FAILED: could not reach threshold={threshold:.3f}. Best_score={best_score:.3f}")
        print(f"[ENROLL] Saved best-effort template to {template_path} for debugging.")
    raise SystemExit("enroll_failed_no_good_template")


if __name__ == "__main__":
    main()
