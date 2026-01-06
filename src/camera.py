from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraParams:
    index: int = 0
    warmup_frames: int = 10
    width: int = 640
    height: int = 480


def capture_frame(params: CameraParams) -> np.ndarray:
    cap = cv2.VideoCapture(params.index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index={params.index}. "
            f"Check VirtualBox settings (Devices -> Webcams) and permissions."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, params.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, params.height)

    for _ in range(max(1, params.warmup_frames)):
        cap.read()

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Camera capture failed (no frame returned).")

    return frame
