"""Central configuration for the ML pipeline.

Kept intentionally standalone (no backend imports) so the training / evaluation /
export scripts can run in a bare ML environment (e.g. a Colab or a training box)
without pulling in the API dependencies. The one hand-off back to the service is
``recommended_thresholds.json`` in the registry, which the backend policy loads
if present.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Paths ---------------------------------------------------------------- #
ML_DIR = PROJECT_ROOT / "ml"
DATASET_DIR = ML_DIR / "datasets" / "industrial_defects"
DATA_YAML = DATASET_DIR / "data.yaml"
MODELS_DIR = ML_DIR / "models"
REGISTRY_DIR = ML_DIR / "registry"
REGISTRY_MANIFEST = REGISTRY_DIR / "models.json"
RECOMMENDED_THRESHOLDS = REGISTRY_DIR / "recommended_thresholds.json"
RUNS_DIR = PROJECT_ROOT / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- Domain --------------------------------------------------------------- #
CLASS_NAMES: List[str] = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

# Duplicated from the backend on purpose (keeps ML standalone). Keep in sync.
CLASS_SEVERITY: Dict[str, str] = {
    "missing_hole": "critical",
    "open_circuit": "critical",
    "short": "critical",
    "mouse_bite": "major",
    "spur": "major",
    "spurious_copper": "major",
}

# Fbeta used during threshold tuning, by severity. beta>1 favours recall, which
# is what we want for defects that must not escape the gate.
SEVERITY_FBETA: Dict[str, float] = {
    "critical": 2.0,
    "major": 1.5,
    "minor": 1.0,
}


@dataclass
class TrainConfig:
    """Training hyper-parameters, with defect-detection-aware defaults.

    Notable choices for small PCB defects:
      * higher ``epochs`` and ``patience`` than the original 30/10 baseline;
      * conservative geometric aug (small rotate/shear) — defects are tiny and
        orientation-sensitive; aggressive mosaic is disabled near the end;
      * ``copy_paste`` and ``mixup`` help rare-class recall;
      * cosine LR + warmup for stable convergence.
    """

    base_model: str = "yolo11n.pt"
    epochs: int = 150
    imgsz: int = 640
    batch: int = 16
    patience: int = 30
    seed: int = 0
    optimizer: str = "auto"
    lr0: float = 0.01
    cos_lr: bool = True
    warmup_epochs: float = 3.0
    cache: bool = False
    workers: int = 8

    # Augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.5
    hsv_v: float = 0.4
    degrees: float = 5.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 2.0
    fliplr: float = 0.5
    flipud: float = 0.2
    mosaic: float = 1.0
    close_mosaic: int = 15  # disable mosaic for the last N epochs
    mixup: float = 0.1
    copy_paste: float = 0.1

    run_name: str = "pcb_defect_v2"

    def to_dict(self) -> dict:
        return asdict(self)

    def ultralytics_kwargs(self) -> dict:
        """Kwargs passed straight to ``model.train(...)``."""
        return {
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "patience": self.patience,
            "seed": self.seed,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "cos_lr": self.cos_lr,
            "warmup_epochs": self.warmup_epochs,
            "cache": self.cache,
            "workers": self.workers,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "fliplr": self.fliplr,
            "flipud": self.flipud,
            "mosaic": self.mosaic,
            "close_mosaic": self.close_mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
        }


def severity_for(class_name: str) -> str:
    return CLASS_SEVERITY.get(class_name, "major")


def fbeta_for(class_name: str) -> float:
    return SEVERITY_FBETA.get(severity_for(class_name), 1.0)
