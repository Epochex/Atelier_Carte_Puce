# FILE: src/vision_backend.py
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FaceEyesDet:
    face_ok: bool
    eyes_ok: bool
    face_center: Optional[Tuple[int, int]] = None
    eye1: Optional[Tuple[int, int]] = None
    eye2: Optional[Tuple[int, int]] = None
    eye_r: Optional[int] = None
    raw: str = ""


_FACE_RE = re.compile(r"Face=\((\-?\d+),(\-?\d+)\)")
_EYES_RE = re.compile(r"Eyes=\((\-?\d+),(\-?\d+)\)\s+\((\-?\d+),(\-?\d+)\).*?\sr=(\d+)")


def _default_bin_path() -> str:
    # CartePuce/vision/bin/ght_face_eyes
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    return os.path.join(root, "vision", "bin", "ght_face_eyes")


def detect_face_eyes_by_ght(image_path: str, bin_path: Optional[str] = None, timeout_sec: int = 5) -> FaceEyesDet:
    """
    Call  C++ GHT detector:
      ght_face_eyes --image <path>
    Parse stdout for Face / Eyes.

    Returns FaceEyesDet with coordinates in pixel space of the input image.
    """
    if not image_path or not os.path.exists(image_path):
        return FaceEyesDet(face_ok=False, eyes_ok=False, raw="image_not_found")

    exe = bin_path or _default_bin_path()
    if not os.path.exists(exe):
        return FaceEyesDet(face_ok=False, eyes_ok=False, raw=f"vision_binary_not_found:{exe}")

    try:
        cp = subprocess.run(
            [exe, "--image", image_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return FaceEyesDet(face_ok=False, eyes_ok=False, raw="vision_timeout")
    except Exception as e:
        return FaceEyesDet(face_ok=False, eyes_ok=False, raw=f"vision_exec_error:{e}")

    out = (cp.stdout or "").strip()
    err = (cp.stderr or "").strip()
    raw = (out + ("\n" + err if err else "")).strip()

    face_ok = "Face=NOTFOUND" not in out and bool(_FACE_RE.search(out))
    eyes_m = _EYES_RE.search(out)
    eyes_ok = "Eyes=NOTFOUND" not in out and (eyes_m is not None)

    det = FaceEyesDet(face_ok=face_ok, eyes_ok=eyes_ok, raw=raw)

    fm = _FACE_RE.search(out)
    if fm:
        det.face_center = (int(fm.group(1)), int(fm.group(2)))

    if eyes_m:
        det.eye1 = (int(eyes_m.group(1)), int(eyes_m.group(2)))
        det.eye2 = (int(eyes_m.group(3)), int(eyes_m.group(4)))
        det.eye_r = int(eyes_m.group(5))

    return det
