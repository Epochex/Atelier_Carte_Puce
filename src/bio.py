from __future__ import annotations
import hashlib
import os
from typing import Tuple, Optional

import cv2
import numpy as np


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _largest_face_bbox(faces) -> Optional[Tuple[int, int, int, int]]:
    if faces is None or len(faces) == 0:
        return None
    # faces: (x,y,w,h)
    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    return tuple(int(v) for v in faces[0])


def preprocess_image(img_bgr: np.ndarray, use_face_crop: bool = True, size: int = 320) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if use_face_crop:
        try:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            bbox = _largest_face_bbox(faces)
            if bbox:
                x, y, w, h = bbox
                crop = gray[y:y+h, x:x+w]
                crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
                return crop
        except Exception:
            pass

    # fallback: center crop
    h, w = gray.shape[:2]
    m = min(h, w)
    x0 = (w - m) // 2
    y0 = (h - m) // 2
    crop = gray[y0:y0+m, x0:x0+m]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def orb_score(img1_gray: np.ndarray, img2_gray: np.ndarray, nfeatures: int = 800) -> float:
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if not matches:
        return 0.0

    matches = sorted(matches, key=lambda m: m.distance)
    good = [m for m in matches if m.distance < 50]  
    score = len(good) / max(len(matches), 1)
    return float(score)


def compare_biometric(captured_bgr: np.ndarray, template_bgr: np.ndarray,
                      use_face_crop: bool, nfeatures: int) -> float:
    a = preprocess_image(captured_bgr, use_face_crop=use_face_crop)
    b = preprocess_image(template_bgr, use_face_crop=use_face_crop)
    return orb_score(a, b, nfeatures=nfeatures)
