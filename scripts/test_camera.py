import os
from pathlib import Path

import cv2

from src.config import load_config
from src.camera import CameraParams, capture_frame


def _safe_imwrite(path: str, img) -> Path:
    """
    Write image to disk robustly:
    - ensure parent dir exists
    - validate cv2.imwrite return
    - validate file exists and size looks sane
    - return absolute Path
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(p), img)
    if not ok:
        raise SystemExit(f"[ERR] cv2.imwrite failed: {p}")

    if (not p.exists()) or p.stat().st_size < 1024:
        # 1KB is an arbitrary "sanity floor" to catch empty/failed writes
        size = p.stat().st_size if p.exists() else 0
        raise SystemExit(f"[ERR] suspicious write: {p} size={size} bytes")

    return p.resolve()


def main():
    cfg = load_config("config.yaml")

    frame = capture_frame(
        CameraParams(
            index=cfg.camera.index,
            warmup_frames=cfg.camera.warmup_frames,
            width=cfg.camera.width,
            height=cfg.camera.height,
        )
    )

    # Keep the old output for backward compatibility
    out_camera = _safe_imwrite("data/camera_test.png", frame)

    # Also write the canonical image used by ght_face_eyes demos
    out_vision = _safe_imwrite("data/vision_test.png", frame)

    print(f"OK: saved {out_camera} ({out_camera.stat().st_size} bytes)")
    print(f"OK: saved {out_vision} ({out_vision.stat().st_size} bytes)  [for ght_face_eyes]")


if __name__ == "__main__":
    main()
