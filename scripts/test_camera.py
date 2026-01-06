import cv2
from src.config import load_config
from src.camera import CameraParams, capture_frame

def main():
    cfg = load_config("config.yaml")
    frame = capture_frame(CameraParams(
        index=cfg.camera.index,
        warmup_frames=cfg.camera.warmup_frames,
        width=cfg.camera.width,
        height=cfg.camera.height,
    ))
    cv2.imwrite("data/camera_test.png", frame)
    print("OK: saved data/camera_test.png")

if __name__ == "__main__":
    main()
