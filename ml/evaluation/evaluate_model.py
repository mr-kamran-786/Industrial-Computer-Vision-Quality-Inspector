"""Evaluate a trained model on a dataset split.

Produces a human-readable markdown report AND a machine-readable JSON metrics
file, and records the metrics against the model in the registry so quality can
be tracked across model versions.

Run:  python -m ml.evaluation.evaluate_model --model ml/models/pcb_defect_yolo_v1.pt
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from ml.config import DATA_YAML, REPORTS_DIR, RUNS_DIR
from ml.registry.registry import register_model


def _safe_float(value):
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def evaluate(
    model_path: Path,
    data_yaml: Path = DATA_YAML,
    split: str = "test",
    imgsz: int = 640,
    register: bool = True,
) -> Dict:
    from ultralytics import YOLO

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    version = model_path.stem
    run_name = f"{version}_{split}_eval"

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        project=str(RUNS_DIR / "evaluation"),
        name=run_name,
        exist_ok=True,
        plots=True,
        save_json=True,
        verbose=False,
    )
    box = metrics.box

    per_class = {}
    for row, class_id in enumerate(box.ap_class_index):
        per_class[model.names[int(class_id)]] = round(float(box.maps[int(class_id)]), 4)

    summary = {
        "model": version,
        "split": split,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "precision": _safe_float(box.mp),
        "recall": _safe_float(box.mr),
        "map50": _safe_float(box.map50),
        "map50_95": _safe_float(box.map),
        "per_class_map50_95": per_class,
    }

    _write_json(summary, version, split)
    _write_markdown(summary, version, split)

    if register:
        register_model(
            version=version,
            weights_path=model_path,
            metrics={
                k: summary[k]
                for k in ("precision", "recall", "map50", "map50_95")
            },
            notes=f"Evaluated on {split} split.",
        )

    return summary


def _write_json(summary: Dict, version: str, split: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{version}_{split}_metrics.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def _write_markdown(summary: Dict, version: str, split: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{version}_{split}_evaluation.md"

    def fmt(value):
        return f"{value:.4f}" if isinstance(value, float) else "N/A"

    lines = [
        f"# Evaluation Report — {version} ({split})",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Precision | {fmt(summary['precision'])} |",
        f"| Recall | {fmt(summary['recall'])} |",
        f"| mAP@50 | {fmt(summary['map50'])} |",
        f"| mAP@50-95 | {fmt(summary['map50_95'])} |",
        "",
        "## Per-Class mAP@50-95",
        "",
        "| Class | mAP@50-95 |",
        "|---|---:|",
    ]
    for class_name, value in summary["per_class_map50_95"].items():
        lines.append(f"| {class_name} | {value} |")
    lines += [
        "",
        "> Recall is the metric to watch for a quality gate: it is the share of "
        "real defects the model catches. Use `tune_thresholds` to trade precision "
        "for recall per class.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PCB defect model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default=str(DATA_YAML))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    summary = evaluate(
        model_path=Path(args.model),
        data_yaml=Path(args.data),
        split=args.split,
        imgsz=args.imgsz,
        register=not args.no_register,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
