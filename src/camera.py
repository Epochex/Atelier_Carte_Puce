from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class CameraParams:
    index: int = 0
    warmup_frames: int = 10
    width: int = 640
    height: int = 480


def capture_frame(params: CameraParams) -> np.ndarray:
    cap = cv2.VideoCapture(params.index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 index={params.index}（检查 VirtualBox Webcams 是否已勾选）")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, params.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, params.height)

    # warmup
    for _ in range(max(1, params.warmup_frames)):
        cap.read()

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("摄像头抓帧失败")

    return frame  # BGR
