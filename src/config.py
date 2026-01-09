from __future__ import annotations
import yaml
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CameraConfig:
    index: int = 0
    warmup_frames: int = 10
    width: int = 640
    height: int = 480


@dataclass
class BiometricConfig:
    score_threshold: float = 0.5
    use_face_crop: bool = True
    orb_nfeatures: int = 800


@dataclass
class AuthConfig:
    required_factors: int = 3
    max_pin_attempts: int = 5
    lockout_seconds: int = 300
    enforce_card_binding: bool = True
    enforce_template_integrity: bool = True


@dataclass
class AppConfig:
    db_path: str = "data/app.db"
    camera: CameraConfig = CameraConfig()
    biometric: BiometricConfig = BiometricConfig()
    auth: AuthConfig = AuthConfig()


def load_config(path: str = "config.yaml") -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    cam = raw.get("camera", {}) or {}
    bio = raw.get("biometric", {}) or {}
    auth = raw.get("auth", {}) or {}

    return AppConfig(
        db_path=str(raw.get("db_path", "data/app.db")),
        camera=CameraConfig(
            index=int(cam.get("index", 0)),
            warmup_frames=int(cam.get("warmup_frames", 10)),
            width=int(cam.get("width", 640)),
            height=int(cam.get("height", 480)),
        ),
        biometric=BiometricConfig(
            score_threshold=float(bio.get("score_threshold", 0.18)),
            use_face_crop=bool(bio.get("use_face_crop", True)),
            orb_nfeatures=int(bio.get("orb_nfeatures", 800)),
        ),
        auth=AuthConfig(
            required_factors=int(auth.get("required_factors", 3)),
            max_pin_attempts=int(auth.get("max_pin_attempts", 5)),
            lockout_seconds=int(auth.get("lockout_seconds", 300)),
            enforce_card_binding=bool(auth.get("enforce_card_binding", True)),
            enforce_template_integrity=bool(auth.get("enforce_template_integrity", True)),
        ),
    )
