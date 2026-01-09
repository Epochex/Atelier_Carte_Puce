# FILE: scripts/test_vision.py
import cv2

from src.config import load_config
from src.camera import CameraParams, capture_frame
from src.bio import compare_biometric

def main():
    cfg = load_config("config.yaml")

    frame = capture_frame(CameraParams(
        index=cfg.camera.index,
        warmup_frames=cfg.camera.warmup_frames,
        width=cfg.camera.width,
        height=cfg.camera.height,
    ))

    # 用同一帧与自身比较，理想情况下分数应接近 1
    score = compare_biometric(frame, frame)
    print(f"self-compare score={score:.3f}")

    cv2.imwrite("data/vision_test.png", frame)
    print("saved data/vision_test.png")

if __name__ == "__main__":
    main()
