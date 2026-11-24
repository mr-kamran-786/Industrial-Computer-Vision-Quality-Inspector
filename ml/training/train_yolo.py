"""Train the PCB defect detector.

Improvements over the original 30-epoch baseline:
  * defect-aware augmentation and longer schedule (see ``ml.config.TrainConfig``);
  * reproducible (seeded), CLI-configurable;
  * on completion, copies the best weights into ``ml/models/<version>.pt`` and
    registers the model (config + checksum) in the model registry.

Run:  python -m ml.training.train_yolo --version pcb_defect_v2 --epochs 150
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ml.config import (
    DATA_YAML,
    MODELS_DIR,
    RUNS_DIR,
    TrainConfig,
)
from ml.registry.registry import register_model


def _resolve_device():
    import torch

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return 0
    print("GPU not found. Training will use CPU (slow).")
    return "cpu"


def train(config: TrainConfig, data_yaml: Path, version: str) -> Path:
    from ultralytics import YOLO

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    device = _resolve_device()
    model = YOLO(config.base_model)

    results = model.train(
        data=str(data_yaml),
        device=device,
        project=str(RUNS_DIR / "detect"),
        name=version,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
        **config.ultralytics_kwargs(),
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Best weights not found: {best}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    deployed = MODELS_DIR / f"{version}.pt"
    shutil.copy2(best, deployed)

    # Pull final validation metrics if available.
    metrics = {}
    try:
        box = results.box  # type: ignore[attr-defined]
        metrics = {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        }
    except Exception:  # noqa: BLE001 - metrics are best-effort here
        pass

    register_model(
        version=version,
        weights_path=deployed,
        metrics=metrics,
        train_config=config.to_dict(),
        notes=f"Trained from {config.base_model} on {data_yaml.name}.",
    )

    print(f"\nTraining complete. Deployed weights: {deployed}")
    print(f"Registered model version: {version}")
    return deployed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PCB defect YOLO model.")
    parser.add_argument("--version", default="pcb_defect_v2", help="Model version / run name")
    parser.add_argument("--data", default=str(DATA_YAML))
    parser.add_argument("--base-model", default=None, help="Override base checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = TrainConfig(run_name=args.version)
    if args.base_model:
        config.base_model = args.base_model
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch is not None:
        config.batch = args.batch
    if args.imgsz is not None:
        config.imgsz = args.imgsz
    if args.seed is not None:
        config.seed = args.seed

    train(config=config, data_yaml=Path(args.data), version=args.version)


if __name__ == "__main__":
    main()
