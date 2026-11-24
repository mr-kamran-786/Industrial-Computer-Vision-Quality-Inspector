"""Central configuration for the Industrial CV Quality Inspector.

All runtime-tunable values are exposed through the ``settings`` object, which is
populated from environment variables (prefix ``ICVQI_``) or an optional ``.env``
file. Domain constants that describe the inspection problem itself (defect
classes, severity mapping, per-class decision thresholds, model registry) live as
module-level constants so they can be imported directly and reasoned about in one
place.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

STORAGE_DIR = PROJECT_ROOT / "backend" / "app" / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
RESULTS_DIR = STORAGE_DIR / "results"
DATABASE_DIR = STORAGE_DIR / "database"


# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #

CLASS_NAMES: Dict[int, str] = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper",
}


class Severity:
    """Defect severity levels, ordered by impact on product function."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


SEVERITY_RANK: Dict[str, int] = {
    Severity.MINOR: 1,
    Severity.MAJOR: 2,
    Severity.CRITICAL: 3,
}

# Severity is what turns raw detections into a quality decision. Functional
# killers (broken/shorted nets, missing holes) are critical; the rest are major
# cosmetic/reliability risks. Tune per your IPC-A-610 acceptance class.
CLASS_SEVERITY: Dict[str, str] = {
    "missing_hole": Severity.CRITICAL,
    "open_circuit": Severity.CRITICAL,
    "short": Severity.CRITICAL,
    "mouse_bite": Severity.MAJOR,
    "spur": Severity.MAJOR,
    "spurious_copper": Severity.MAJOR,
}

# Per-class confidence thresholds. Critical classes use a lower threshold so we
# bias toward catching real defects (recall); cosmetic classes use a higher
# threshold to suppress false calls. A class not listed here falls back to
# ``settings.global_confidence_threshold``.
#
# These are the hand-picked defaults. If the ML pipeline has produced tuned
# thresholds (``ml/registry/recommended_thresholds.json`` via
# ``python -m ml.evaluation.tune_thresholds``), they override the defaults below,
# closing the loop from evaluation to production decisions.
DEFAULT_PER_CLASS_THRESHOLDS: Dict[str, float] = {
    "missing_hole": 0.25,
    "open_circuit": 0.25,
    "short": 0.25,
    "mouse_bite": 0.30,
    "spur": 0.30,
    "spurious_copper": 0.35,
}

RECOMMENDED_THRESHOLDS_FILE = (
    PROJECT_ROOT / "ml" / "registry" / "recommended_thresholds.json"
)


def _load_tuned_thresholds() -> Dict[str, float]:
    """Overlay tuned thresholds from the ML registry if the file exists/valid."""
    import json
    import logging

    if not RECOMMENDED_THRESHOLDS_FILE.exists():
        return dict(DEFAULT_PER_CLASS_THRESHOLDS)

    merged = dict(DEFAULT_PER_CLASS_THRESHOLDS)
    try:
        with open(RECOMMENDED_THRESHOLDS_FILE, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        tuned = payload.get("thresholds", {})
        for class_name, value in tuned.items():
            if class_name in CLASS_NAMES.values() and isinstance(value, (int, float)):
                merged[class_name] = float(value)
        logging.getLogger("icvqi.config").info(
            "Loaded tuned thresholds", extra={"source": str(RECOMMENDED_THRESHOLDS_FILE)}
        )
    except (OSError, ValueError, KeyError):
        logging.getLogger("icvqi.config").warning(
            "Failed to read tuned thresholds; using defaults",
            extra={"source": str(RECOMMENDED_THRESHOLDS_FILE)},
        )
    return merged


PER_CLASS_THRESHOLDS: Dict[str, float] = _load_tuned_thresholds()

# Model registry. ``version`` is recorded on every inspection so a decision can
# always be reproduced and defended in an audit.
MODEL_CONFIGS: Dict[str, Dict] = {
    "nano": {
        "key": "nano",
        "name": "YOLO Nano Edge v1",
        "mode": "Edge / Fast",
        "version": "pcb_defect_yolo_v1",
        "description": "Lightweight model for low-resource GPU or edge deployment.",
        "path": PROJECT_ROOT / "ml" / "models" / "pcb_defect_yolo_v1.pt",
    },
    "medium": {
        "key": "medium",
        "name": "YOLO Medium Accuracy v1",
        "mode": "Accuracy / Production",
        "version": "pcb_defect_yolo_medium_v1",
        "description": "Higher-accuracy model for production-quality inspection.",
        "path": PROJECT_ROOT / "ml" / "models" / "pcb_defect_yolo_medium_v1.pt",
    },
}


# --------------------------------------------------------------------------- #
# Environment-driven settings
# --------------------------------------------------------------------------- #

class Settings(BaseSettings):
    """Runtime settings, overridable via ``ICVQI_*`` env vars or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ICVQI_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Industrial Computer Vision Quality Inspector"
    app_version: str = "2.0.0"
    environment: str = "development"

    # Persistence. SQLite by default; point at Postgres for production, e.g.
    # ICVQI_DATABASE_URL=postgresql+psycopg://user:pass@host:5432/icvqi
    database_url: str = f"sqlite:///{(DATABASE_DIR / 'inspections.db').as_posix()}"

    # Web / security
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    # Comma-separated API keys. When empty, auth is DISABLED (dev only) and a
    # warning is logged on startup. Set at least one key in production.
    api_keys: List[str] = []

    # Upload guardrails
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB
    allowed_extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    # Inference / decision policy
    image_size: int = 640
    default_model_key: str = "nano"
    global_confidence_threshold: float = 0.25
    # Severities that force a FAIL verdict when detected above threshold.
    fail_severities: List[str] = [Severity.CRITICAL, Severity.MAJOR]
    # Number of below-fail-severity (e.g. minor) defects tolerated before FAIL.
    max_acceptable_minor_defects: int = 0

    @field_validator(
        "cors_origins",
        "api_keys",
        "allowed_extensions",
        "fail_severities",
        mode="before",
    )
    @classmethod
    def _split_csv(cls, value):
        """Allow list settings to be provided as comma-separated strings."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def auth_enabled(self) -> bool:
        return len(self.api_keys) > 0

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Backwards-compatible aliases used across the codebase.
DEFAULT_MODEL_KEY = settings.default_model_key
CONFIDENCE_THRESHOLD = settings.global_confidence_threshold
IMAGE_SIZE = settings.image_size
