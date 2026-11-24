"""Per-class confidence-threshold tuning.

Picks, for each defect class, the confidence threshold that maximises an
F-beta score, where beta is chosen by severity (critical classes use beta=2 to
favour recall — we would rather over-flag a critical defect than let it escape).
An optional precision floor prevents choosing a threshold that produces an
unusable false-call rate.

The tuner reuses Ultralytics' own precision/recall curves from ``model.val()``
(``box.px``, ``box.p_curve``, ``box.r_curve``) so it is consistent with the
reported metrics. The output ``recommended_thresholds.json`` is loaded by the
backend policy engine, closing the loop from evaluation to production decisions.

Run:  python -m ml.evaluation.tune_thresholds --model ml/models/pcb_defect_yolo_v1.pt
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ml.config import (
    DATA_YAML,
    RECOMMENDED_THRESHOLDS,
    REGISTRY_DIR,
    REPORTS_DIR,
    fbeta_for,
    severity_for,
)


def fbeta_curve(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    """F-beta at every point of a precision/recall curve (0 where undefined)."""
    b2 = beta * beta
    denom = b2 * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (1 + b2) * (precision * recall) / denom
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def select_threshold_from_curves(
    px: Sequence[float],
    p_curve: Sequence[float],
    r_curve: Sequence[float],
    beta: float = 1.0,
    precision_floor: float = 0.0,
    default: float = 0.25,
    clip: tuple = (0.05, 0.95),
) -> Dict[str, float]:
    """Choose the confidence threshold maximising F-beta.

    Pure function (no model dependency) so it is unit-testable. ``px`` is the
    confidence axis; ``p_curve``/``r_curve`` are precision/recall at each px.
    """
    px = np.asarray(px, dtype=float)
    precision = np.asarray(p_curve, dtype=float)
    recall = np.asarray(r_curve, dtype=float)

    f = fbeta_curve(precision, recall, beta)

    allowed = precision >= precision_floor
    if precision_floor > 0 and allowed.any():
        candidate_idx = np.where(allowed)[0]
        best = int(candidate_idx[int(np.argmax(f[candidate_idx]))])
    elif f.max() > 0:
        best = int(np.argmax(f))
    else:
        return {
            "threshold": round(float(default), 3),
            "precision": None,
            "recall": None,
            "fbeta": None,
            "note": "no positive F-beta; using default",
        }

    threshold = float(np.clip(px[best], clip[0], clip[1]))
    return {
        "threshold": round(threshold, 3),
        "precision": round(float(precision[best]), 4),
        "recall": round(float(recall[best]), 4),
        "fbeta": round(float(f[best]), 4),
    }


def run_tuning(
    model_path: Path,
    data_yaml: Path = DATA_YAML,
    split: str = "val",
    imgsz: int = 640,
    precision_floor: float = 0.3,
) -> Dict:
    from ultralytics import YOLO  # heavy import kept local

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml), split=split, imgsz=imgsz, verbose=False, plots=False
    )
    box = metrics.box
    names = model.names

    per_class: List[Dict] = []
    thresholds: Dict[str, float] = {}
    for row, class_id in enumerate(box.ap_class_index):
        class_name = names[int(class_id)]
        beta = fbeta_for(class_name)
        selection = select_threshold_from_curves(
            box.px,
            box.p_curve[row],
            box.r_curve[row],
            beta=beta,
            precision_floor=precision_floor,
        )
        thresholds[class_name] = selection["threshold"]
        per_class.append(
            {
                "class": class_name,
                "severity": severity_for(class_name),
                "beta": beta,
                **selection,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": Path(model_path).stem,
        "split": split,
        "precision_floor": precision_floor,
        "thresholds": thresholds,
        "per_class": per_class,
    }

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with open(RECOMMENDED_THRESHOLDS, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    _write_report(payload)
    return payload


def _write_report(payload: Dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "threshold_tuning.md"
    lines = [
        "# Per-Class Threshold Tuning Report",
        "",
        f"Generated: {payload['generated_at']}",
        f"Model: `{payload['model']}` | Split: {payload['split']} | "
        f"Precision floor: {payload['precision_floor']}",
        "",
        "| Class | Severity | beta | Threshold | Precision | Recall | F-beta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["per_class"]:
        lines.append(
            f"| {row['class']} | {row['severity']} | {row['beta']} | "
            f"{row['threshold']} | {row.get('precision')} | "
            f"{row.get('recall')} | {row.get('fbeta')} |"
        )
    lines += [
        "",
        "These thresholds are written to `ml/registry/recommended_thresholds.json`",
        "and loaded automatically by the backend policy engine on startup.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune per-class confidence thresholds.")
    parser.add_argument("--model", required=True, help="Path to .pt weights")
    parser.add_argument("--data", default=str(DATA_YAML))
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--precision-floor", type=float, default=0.3)
    args = parser.parse_args()

    payload = run_tuning(
        model_path=Path(args.model),
        data_yaml=Path(args.data),
        split=args.split,
        imgsz=args.imgsz,
        precision_floor=args.precision_floor,
    )
    print("Recommended thresholds:")
    for name, value in payload["thresholds"].items():
        print(f"  {name:<16} {value}")
    print(f"\nWritten to: {RECOMMENDED_THRESHOLDS}")


if __name__ == "__main__":
    main()
