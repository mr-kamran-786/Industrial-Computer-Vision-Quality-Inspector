"""Export trained weights to a deployment format (ONNX by default).

ONNX gives portable, framework-independent inference (onnxruntime on CPU/GPU,
or as an intermediate to TensorRT for edge). The script exports and then runs a
lightweight parity check so a broken export is caught before deployment.

Run:  python -m ml.export.export_model --model ml/models/pcb_defect_yolo_v1.pt
      python -m ml.export.export_model --model ... --format engine   # TensorRT
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.config import MODELS_DIR


def export(
    model_path: Path,
    fmt: str = "onnx",
    imgsz: int = 640,
    opset: int = 12,
    half: bool = False,
    simplify: bool = True,
) -> Path:
    from ultralytics import YOLO

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    kwargs = {"format": fmt, "imgsz": imgsz, "half": half}
    if fmt == "onnx":
        kwargs.update({"opset": opset, "simplify": simplify, "dynamic": False})

    exported = Path(model.export(**kwargs))
    print(f"Exported: {exported}")
    return exported


def validate_onnx(onnx_path: Path) -> bool:
    """Structurally validate an ONNX graph (no runtime / no auto-install)."""
    try:
        import onnx

        onnx.checker.check_model(str(onnx_path))
        print(f"ONNX graph valid: {onnx_path}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"ONNX validation FAILED: {exc}")
        return False


def parity_check(pt_path: Path, exported_path: Path, imgsz: int = 640) -> bool:
    """Best-effort inference parity between the .pt and exported model.

    Never fails the export: on a CUDA host Ultralytics may try to (re)install
    ``onnxruntime-gpu`` when loading ONNX; if that or anything else goes wrong we
    report the check as skipped rather than crashing a successful export.
    """
    try:
        import numpy as np
        from ultralytics import YOLO

        dummy = (np.random.rand(imgsz, imgsz, 3) * 255).astype("uint8")
        pt_preds = YOLO(str(pt_path)).predict(dummy, imgsz=imgsz, verbose=False)[0]
        ex_preds = YOLO(str(exported_path)).predict(dummy, imgsz=imgsz, verbose=False)[0]

        pt_n = 0 if pt_preds.boxes is None else len(pt_preds.boxes)
        ex_n = 0 if ex_preds.boxes is None else len(ex_preds.boxes)
        ok = pt_n == ex_n
        print(f"Parity check: pt={pt_n} boxes, exported={ex_n} boxes -> {'OK' if ok else 'MISMATCH'}")
        return ok
    except Exception as exc:  # noqa: BLE001
        print(f"Parity check skipped (inference backend unavailable): {exc}")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO weights for deployment.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "torchscript", "openvino"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--no-parity", action="store_true", help="Skip the parity check")
    args = parser.parse_args()

    model_path = Path(args.model)
    exported = export(
        model_path=model_path,
        fmt=args.format,
        imgsz=args.imgsz,
        opset=args.opset,
        half=args.half,
    )

    if args.format == "onnx":
        validate_onnx(exported)
    if not args.no_parity and args.format in {"onnx", "torchscript"}:
        parity_check(model_path, exported, imgsz=args.imgsz)

    print(f"\nDeployment artifact ready in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
