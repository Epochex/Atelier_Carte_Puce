# FILE: src/vision_backend.py
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class FaceEyesDet:
    face_ok: bool
    eyes_ok: bool
    face_center: Optional[Tuple[int, int]] = None
    eye1: Optional[Tuple[int, int]] = None
    eye2: Optional[Tuple[int, int]] = None
    eye_r: Optional[int] = None
    raw: str = ""


_FACE_RE = re.compile(r"Face\s*=\s*\(\s*(\-?\d+)\s*,\s*(\-?\d+)\s*\)")
_EYES_RE = re.compile(
    r"Eyes\s*=\s*\(\s*(\-?\d+)\s*,\s*(\-?\d+)\s*\)\s+"
    r"\(\s*(\-?\d+)\s*,\s*(\-?\d+)\s*\).*?\br\s*=\s*(\d+)"
)


def _default_bin_path() -> str:
    # CartePuce/vision/bin/ght_face_eyes
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    return os.path.join(root, "vision", "bin", "ght_face_eyes")


def detect_face_eyes_by_ght(
    image_path: str,
    bin_path: Optional[str] = None,
    timeout_sec: int = 5,
    headless: bool = True,
) -> FaceEyesDet:
    """
    Call C++ GHT detector and parse stdout for Face/Eyes.

      ght_face_eyes --image <path> [--no-gui]

    Headless mode avoids GUI windows + waitKey(0) blocking in automation.
    """
    if not image_path or not os.path.exists(image_path):
        return FaceEyesDet(face_ok=False, eyes_ok=False, raw="image_not_found")

    exe = bin_path or _default_bin_path()
    if not os.path.exists(exe):
        return FaceEyesDet(face_ok=False, eyes_ok=False, raw=f"vision_binary_not_found:{exe}")

    cmd: List[str] = [exe, "--image", image_path]
    if headless:
        # supported by your patched C++ (also accepts --headless)
        cmd.append("--no-gui")

    try:
        cp = subprocess.run(
            cmd,
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

    raw = (
        f"cmd={cmd}\n"
        f"returncode={cp.returncode}\n"
        f"--- stdout ---\n{out}\n"
        f"--- stderr ---\n{err}\n"
    ).strip()

    parse_text = out if out else err
    if not parse_text:
        return FaceEyesDet(face_ok=False, eyes_ok=False, raw="no_output_from_vision_binary\n" + raw)

    face_notfound = re.search(r"Face\s*=\s*NOTFOUND", parse_text) is not None
    eyes_notfound = re.search(r"Eyes\s*=\s*NOTFOUND", parse_text) is not None

    fm = _FACE_RE.search(parse_text)
    em = _EYES_RE.search(parse_text)

    face_ok = (not face_notfound) and (fm is not None)
    eyes_ok = (not eyes_notfound) and (em is not None)

    det = FaceEyesDet(face_ok=face_ok, eyes_ok=eyes_ok, raw=raw)

    if fm:
        det.face_center = (int(fm.group(1)), int(fm.group(2)))

    if em:
        det.eye1 = (int(em.group(1)), int(em.group(2)))
        det.eye2 = (int(em.group(3)), int(em.group(4)))
        det.eye_r = int(em.group(5))

    return det
