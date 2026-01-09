# FILE: src/bio.py
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

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


# -----------------------------
# Eye-based signature (H+S hist)
# -----------------------------
def _eye_roi_from_eye(bgr: np.ndarray, eye: Tuple[int, int], r: int) -> Optional[np.ndarray]:
    if bgr is None or eye is None or r is None:
        return None
    h, w = bgr.shape[:2]
    ex, ey = int(eye[0]), int(eye[1])

    rr = max(10, int(round(r * 1.20)))  # slightly larger to reduce bbox jitter sensitivity
    x0, y0, x1, y1 = _clamp_roi(ex - rr, ey - rr, ex + rr, ey + rr, w, h)
    if (x1 - x0) < 24 or (y1 - y0) < 24:
        return None
    return bgr[y0:y1, x0:x1]


def _hs_signature(eye_bgr: np.ndarray, h_bins: int = 24, s_bins: int = 24) -> np.ndarray:
    hsv = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2HSV)

    # Use only H and S to reduce sensitivity to illumination (V).
    hs = hsv[:, :, :2]

    hist = cv2.calcHist([hs], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
    hist = hist.astype(np.float32).flatten()

    s = float(np.sum(hist))
    if s > 0:
        hist /= s
    return hist


def _corr_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Correlation in [-1,1] mapped to [0,1].
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    corr = float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
    # numeric safety
    if corr < -1.0:
        corr = -1.0
    if corr > 1.0:
        corr = 1.0
    return 0.5 * (corr + 1.0)


# -----------------------------
# Face-based fallback (LBP hist)
# -----------------------------
def _lbp8u(gray: np.ndarray) -> np.ndarray:
    """
    Basic LBP (8 neighbors, radius=1) returning uint8 codes.
    """
    g = gray
    h, w = g.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    c = g[1:-1, 1:-1]

    lbp |= ((g[0:-2, 0:-2] >= c) << 7).astype(np.uint8)
    lbp |= ((g[0:-2, 1:-1] >= c) << 6).astype(np.uint8)
    lbp |= ((g[0:-2, 2:  ] >= c) << 5).astype(np.uint8)
    lbp |= ((g[1:-1, 2:  ] >= c) << 4).astype(np.uint8)
    lbp |= ((g[2:  , 2:  ] >= c) << 3).astype(np.uint8)
    lbp |= ((g[2:  , 1:-1] >= c) << 2).astype(np.uint8)
    lbp |= ((g[2:  , 0:-2] >= c) << 1).astype(np.uint8)
    lbp |= ((g[1:-1, 0:-2] >= c) << 0).astype(np.uint8)

    return lbp


def _face_signature_lbp(face_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)

    lbp = _lbp8u(gray)
    hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).astype(np.float32).flatten()
    s = float(np.sum(hist))
    if s > 0:
        hist /= s
    return hist


def _detect_face_haar(image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    face_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cas = cv2.CascadeClassifier(face_path)
    if face_cas.empty():
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cas.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(80, 80),
    )
    if faces is None or len(faces) == 0:
        return None

    x, y, w, h = sorted(faces, key=lambda t: t[2] * t[3], reverse=True)[0]
    return int(x), int(y), int(w), int(h)


@dataclass
class EyeGeom:
    eye1: Tuple[int, int]
    eye2: Tuple[int, int]
    r: int
    method: str
    debug: str = ""


def _detect_eyes_fallback_haar(image_bgr: np.ndarray) -> Optional[EyeGeom]:
    face_box = _detect_face_haar(image_bgr)
    if face_box is None:
        return None
    x, y, w, h = face_box

    eye_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
    eye_cas = cv2.CascadeClassifier(eye_path)
    if eye_cas.empty():
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    face_gray = gray[y:y+h, x:x+w]

    eyes = eye_cas.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=8,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(18, 18),
        maxSize=(max(20, w // 2), max(20, h // 2)),
    )
    if eyes is None or len(eyes) < 2:
        return None

    cand = []
    for (ex, ey, ew, eh) in eyes:
        cy = ey + eh / 2.0
        if cy > 0.65 * h:
            continue
        cand.append((ex, ey, ew, eh))
    if len(cand) < 2:
        cand = list(eyes)

    cand = sorted(cand, key=lambda t: t[2] * t[3], reverse=True)[:2]
    if len(cand) < 2:
        return None

    (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = cand
    c1 = (int(x + ex1 + ew1 / 2), int(y + ey1 + eh1 / 2))
    c2 = (int(x + ex2 + ew2 / 2), int(y + ey2 + eh2 / 2))
    if c2[0] < c1[0]:
        c1, c2 = c2, c1
        ew1, eh1, ew2, eh2 = ew2, eh2, ew1, eh1

    approx_r = int(round(0.30 * min(ew1, eh1, ew2, eh2)))
    approx_r = max(10, min(approx_r, 28))

    return EyeGeom(eye1=c1, eye2=c2, r=approx_r, method="haar", debug=f"face={face_box} eyes={cand} r={approx_r}")


def _detect_eyes_primary_ght(image_bgr: np.ndarray) -> Tuple[Optional[EyeGeom], str]:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tmp_path = tf.name
    try:
        cv2.imwrite(tmp_path, image_bgr)
        det = detect_face_eyes_by_ght(tmp_path, headless=True)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not det.eyes_ok or det.eye1 is None or det.eye2 is None or det.eye_r is None:
        return None, f"ght_eyes_not_found:{det.raw}"

    return EyeGeom(eye1=det.eye1, eye2=det.eye2, r=int(det.eye_r), method="ght", debug="ok"), "ok"


def _extract_eye_signature(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    eg, reason = _detect_eyes_primary_ght(image_bgr)
    if eg is None:
        eg = _detect_eyes_fallback_haar(image_bgr)
        if eg is None:
            return None, f"eyes_not_found:{reason}"
    # build signature from both eyes
    roi1 = _eye_roi_from_eye(image_bgr, eg.eye1, eg.r)
    roi2 = _eye_roi_from_eye(image_bgr, eg.eye2, eg.r)
    if roi1 is None or roi2 is None:
        return None, f"eye_roi_failed:{eg.method}:{eg.debug}"

    s1 = _hs_signature(roi1)
    s2 = _hs_signature(roi2)
    sig = 0.5 * (s1 + s2)
    z = float(np.sum(sig))
    if z > 0:
        sig /= z
    return sig, f"ok:eye:{eg.method}"


def _extract_face_signature(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    face_box = _detect_face_haar(image_bgr)
    if face_box is None:
        return None, "face_not_found"
    x, y, w, h = face_box

    # crop with small margin
    mx = int(round(0.08 * w))
    my = int(round(0.10 * h))
    x0, y0, x1, y1 = _clamp_roi(x - mx, y - my, x + w + mx, y + h + my, image_bgr.shape[1], image_bgr.shape[0])
    face = image_bgr[y0:y1, x0:x1]
    if face.size == 0:
        return None, "face_crop_failed"

    sig = _face_signature_lbp(face)
    return sig, f"ok:face:haar:{face_box}"


def compare_biometric(captured_bgr: np.ndarray, template_bgr: np.ndarray, use_face_crop: bool = True, nfeatures: int = 800) -> float:
    """
    Robust biometric score in [0,1]:
      - primary: eye HS-hist correlation
      - fallback/augment: face LBP-hist correlation

    If both available -> weighted blend (eye dominates), else use what exists.
    If none available -> 0.0
    """
    if captured_bgr is None or template_bgr is None:
        return 0.0

    eye_a, ra = _extract_eye_signature(captured_bgr)
    eye_b, rb = _extract_eye_signature(template_bgr)

    face_a, rfa = _extract_face_signature(captured_bgr)
    face_b, rfb = _extract_face_signature(template_bgr)

    have_eye = (eye_a is not None and eye_b is not None)
    have_face = (face_a is not None and face_b is not None)

    if not have_eye and not have_face:
        return 0.0

    scores = []
    weights = []

    if have_eye:
        scores.append(_corr_similarity(eye_a, eye_b))
        weights.append(0.65)

    if have_face:
        scores.append(_corr_similarity(face_a, face_b))
        weights.append(0.35 if have_eye else 1.0)

    wsum = float(sum(weights))
    if wsum <= 0:
        return 0.0
    score = float(sum(s * w for s, w in zip(scores, weights)) / wsum)

    # clamp
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return score


def compare_biometric_debug(captured_bgr: np.ndarray, template_bgr: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (score, debug_dict) to diagnose demo_run mismatches.
    """
    dbg: Dict[str, Any] = {}

    eye_a, ra = _extract_eye_signature(captured_bgr) if captured_bgr is not None else (None, "captured_empty")
    eye_b, rb = _extract_eye_signature(template_bgr) if template_bgr is not None else (None, "template_empty")
    face_a, rfa = _extract_face_signature(captured_bgr) if captured_bgr is not None else (None, "captured_empty")
    face_b, rfb = _extract_face_signature(template_bgr) if template_bgr is not None else (None, "template_empty")

    dbg["captured_eye"] = ra
    dbg["template_eye"] = rb
    dbg["captured_face"] = rfa
    dbg["template_face"] = rfb

    have_eye = (eye_a is not None and eye_b is not None)
    have_face = (face_a is not None and face_b is not None)

    dbg["have_eye"] = have_eye
    dbg["have_face"] = have_face

    eye_score = None
    face_score = None

    if have_eye:
        eye_score = _corr_similarity(eye_a, eye_b)
    if have_face:
        face_score = _corr_similarity(face_a, face_b)

    dbg["eye_score"] = eye_score
    dbg["face_score"] = face_score

    score = compare_biometric(captured_bgr, template_bgr)
    dbg["final_score"] = score
    return score, dbg
