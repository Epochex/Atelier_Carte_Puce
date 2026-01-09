# FILE: src/bio.py
from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Optional, Tuple

import cv2
import numpy as np

from .vision_backend import detect_face_eyes_by_ght


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _clamp_roi(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(x0, w))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h))
    y1 = max(0, min(y1, h))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _iris_roi_from_eye(bgr: np.ndarray, eye: Tuple[int, int], r: int) -> Optional[np.ndarray]:
    """
    Derive an iris ROI from eye center and radius.
    We use a conservative crop around the circle center.
    """
    if bgr is None or eye is None or r is None:
        return None
    h, w = bgr.shape[:2]
    ex, ey = int(eye[0]), int(eye[1])

    # Iris is inside the detected eye circle; crop slightly smaller to reduce eyelids/skin.
    rr = max(3, int(round(r * 0.85)))
    x0, y0, x1, y1 = _clamp_roi(ex - rr, ey - rr, ex + rr, ey + rr, w, h)
    if (x1 - x0) < 8 or (y1 - y0) < 8:
        return None
    return bgr[y0:y1, x0:x1]


def _iris_color_signature(iris_bgr: np.ndarray, bins_per_channel: int = 8) -> np.ndarray:
    """
    Compact biometric signature: normalized HSV histogram (H,S,V) concatenated.
    Output: float32 vector, L1-normalized.
    """
    hsv = cv2.cvtColor(iris_bgr, cv2.COLOR_BGR2HSV)

    # H: [0,180), S: [0,256), V: [0,256)
    h_hist = cv2.calcHist([hsv], [0], None, [bins_per_channel], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [bins_per_channel], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [bins_per_channel], [0, 256])

    sig = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()]).astype(np.float32)
    s = float(np.sum(sig))
    if s > 0:
        sig /= s
    return sig


def _hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Histogram similarity in [0,1] using Bhattacharyya distance mapped to score.
    score = 1 - dist, with dist in [0,1] (OpenCV convention for normalized hist).
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    dist = float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
    dist = max(0.0, min(1.0, dist))
    return 1.0 - dist


def _extract_signature_via_ght(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """
    1) Write image to temp file (because teammate binary accepts --image).
    2) Call GHT detector -> eye centers + radius.
    3) Crop iris ROI for both eyes, build signatures, and average them.
    """
    if image_bgr is None:
        return None, "empty_image"

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp_path = tf.name

    try:
        cv2.imwrite(tmp_path, image_bgr)
        det = detect_face_eyes_by_ght(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not det.eyes_ok or det.eye1 is None or det.eye2 is None or det.eye_r is None:
        return None, f"eyes_not_found:{det.raw}"

    iris1 = _iris_roi_from_eye(image_bgr, det.eye1, det.eye_r)
    iris2 = _iris_roi_from_eye(image_bgr, det.eye2, det.eye_r)
    if iris1 is None or iris2 is None:
        return None, "iris_crop_failed"

    s1 = _iris_color_signature(iris1, bins_per_channel=8)
    s2 = _iris_color_signature(iris2, bins_per_channel=8)
    sig = 0.5 * (s1 + s2)
    # re-normalize
    z = float(np.sum(sig))
    if z > 0:
        sig /= z
    return sig, "ok"


def compare_biometric(
    captured_bgr: np.ndarray,
    template_bgr: np.ndarray,
    use_face_crop: bool = True,   # keep parameter for backward compatibility (ignored)
    nfeatures: int = 800,         # keep parameter for backward compatibility (ignored)
) -> float:
    """
    New biometric comparison:
      - Use teammate GHT detector (ellipse face + circle eyes) to locate eyes.
      - Build iris color histogram signature (doc 3.3 option #1).
      - Compare signatures to produce score in [0,1].
    """
    sig_a, ra = _extract_signature_via_ght(captured_bgr)
    if sig_a is None:
        return 0.0

    sig_b, rb = _extract_signature_via_ght(template_bgr)
    if sig_b is None:
        return 0.0

    return _hist_similarity(sig_a, sig_b)
