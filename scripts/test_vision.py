import argparse
import cv2

from src.config import load_config
from src.camera import CameraParams, capture_frame
from src.bio import compare_biometric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default=None, help="Template image path to compare against live capture")
    args = parser.parse_args()

    cfg = load_config("config.yaml")

    frame = capture_frame(CameraParams(
        index=cfg.camera.index,
        warmup_frames=cfg.camera.warmup_frames,
        width=cfg.camera.width,
        height=cfg.camera.height,
    ))

    cv2.imwrite("data/vision_test.png", frame)
    print("saved data/vision_test.png")

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
