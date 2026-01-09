import argparse
from pathlib import Path

import cv2

from src.config import load_config
from src.camera import CameraParams, capture_frame
from src.bio import compare_biometric


def _safe_imwrite(path: str, img) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(p), img)
    if not ok:
        raise SystemExit(f"[ERR] cv2.imwrite failed: {p}")

    if (not p.exists()) or p.stat().st_size < 1024:
        size = p.stat().st_size if p.exists() else 0
        raise SystemExit(f"[ERR] suspicious write: {p} size={size} bytes")

    return p.resolve()


def main():
    parser = argparse.ArgumentParser(description="Camera + biometric quick test")
    parser.add_argument(
        "--template",
        default=None,
        help="Optional template image path to compare against (e.g., data/templates/lin.png)",
    )
    parser.add_argument(
        "--out",
        default="data/vision_test.png",
        help="Where to save the captured frame (default: data/vision_test.png)",
    )
    args = parser.parse_args()

    cfg = load_config("config.yaml")

    frame = capture_frame(
        CameraParams(
            index=cfg.camera.index,
            warmup_frames=cfg.camera.warmup_frames,
            width=cfg.camera.width,
            height=cfg.camera.height,
        )
    )

    out_path = _safe_imwrite(args.out, frame)
    print(f"saved {out_path} ({out_path.stat().st_size} bytes)")

    if args.template:
        tpl = cv2.imread(args.template)
        if tpl is None:
            raise SystemExit(f"cannot_read_template:{args.template}")

        score = compare_biometric(
            captured_bgr=frame,
            template_bgr=tpl,
            use_face_crop=cfg.biometric.use_face_crop,
            nfeatures=cfg.biometric.orb_nfeatures,
        )
        print(f"live_vs_template score={score:.3f} template={args.template}")
    else:
        score = compare_biometric(frame, frame)
        print(f"self-compare score={score:.3f}")


if __name__ == "__main__":
    main()
