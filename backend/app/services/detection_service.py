"""YOLO-based defect detection service.

Runs inference, then delegates the PASS/FAIL decision to the policy engine so
per-class thresholds, severity weighting and explainable reasons are applied
consistently. Records model provenance and inference latency on every result.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.core import policy
from backend.app.core.config import (
    CLASS_NAMES,
    DEFAULT_MODEL_KEY,
    IMAGE_SIZE,
    MODEL_CONFIGS,
)

logger = logging.getLogger("icvqi.detection")


class DetectionService:
    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._device: Optional[Any] = None
        self._torch = None
        self._yolo_cls = None

    # -- lazy heavy imports/state ------------------------------------------- #

    @property
    def device(self):
        """Resolve the compute device lazily (avoids importing torch at import)."""
        if self._device is None:
            import torch

            self._torch = torch
            if torch.cuda.is_available():
                self._device = 0
                logger.info(
                    "CUDA available", extra={"gpu": torch.cuda.get_device_name(0)}
                )
            else:
                self._device = "cpu"
                logger.info("CUDA not available; using CPU")
        return self._device

    def _yolo(self):
        if self._yolo_cls is None:
            from ultralytics import YOLO

            self._yolo_cls = YOLO
        return self._yolo_cls

    # -- model registry ----------------------------------------------------- #

    def get_available_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": key,
                "name": config["name"],
                "mode": config["mode"],
                "version": config["version"],
                "description": config["description"],
                "available": config["path"].exists(),
                "path": str(config["path"]),
            }
            for key, config in MODEL_CONFIGS.items()
        ]

    def is_ready(self) -> bool:
        """True if the default model file exists and can be served."""
        default = MODEL_CONFIGS.get(DEFAULT_MODEL_KEY)
        return bool(default and default["path"].exists())

    def _load_model(self, model_key: str):
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model_key: {model_key}")

        config = MODEL_CONFIGS[model_key]
        model_path: Path = config["path"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found for '{model_key}' at: {model_path}"
            )

        if model_key not in self._models:
            logger.info("Loading model", extra={"model_key": model_key, "path": str(model_path)})
            self._models[model_key] = self._yolo()(str(model_path))
            logger.info("Model loaded", extra={"model_key": model_key})

        return self._models[model_key]

    # -- inference ---------------------------------------------------------- #

    def predict_image(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
        model_key: str = DEFAULT_MODEL_KEY,
    ) -> Dict[str, Any]:
        from PIL import Image

        model = self._load_model(model_key)
        config = MODEL_CONFIGS[model_key]

        image = Image.open(image_path).convert("RGB")

        floor = policy.confidence_floor()
        start = time.perf_counter()
        results = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=floor,
            device=self.device,
            verbose=False,
        )
        inference_ms = round((time.perf_counter() - start) * 1000, 2)

        result = results[0]
        raw_detections: List[Dict[str, Any]] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw_detections.append(
                    {
                        "class_id": class_id,
                        "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                        "confidence": round(confidence, 4),
                        "bbox": {
                            "x1": round(float(x1), 2),
                            "y1": round(float(y1), 2),
                            "x2": round(float(x2), 2),
                            "y2": round(float(y2), 2),
                        },
                    }
                )

        # Apply the quality-decision policy (per-class thresholds + severity).
        decision = policy.evaluate(raw_detections)

        if output_path is not None:
            # Annotate from the counted detections so the evidence image always
            # matches the recorded verdict (not the raw model output).
            self._save_annotated(image_path, decision.counted_detections, output_path)

        return {
            "filename": image_path.name,
            "model_key": model_key,
            "model_name": config["name"],
            "model_version": config["version"],
            "confidence_floor": floor,
            "inference_ms": inference_ms,
            "total_detections": len(decision.counted_detections),
            "is_defective": decision.is_defective,
            "quality_status": decision.quality_status,
            "highest_severity": decision.highest_severity,
            "decision_reason": decision.decision_reason,
            "detections": decision.counted_detections,
        }

    # Severity -> BGR color (OpenCV) for annotation.
    _SEVERITY_COLOR = {
        "critical": (0, 0, 220),   # red
        "major": (0, 140, 255),    # orange
        "minor": (0, 200, 220),    # yellow
    }

    def _save_annotated(
        self,
        image_path: Path,
        detections: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read source image: {image_path}")

        for det in detections:
            bbox = det["bbox"]
            color = self._SEVERITY_COLOR.get(det.get("severity"), (0, 200, 0))
            p1 = (int(bbox["x1"]), int(bbox["y1"]))
            p2 = (int(bbox["x2"]), int(bbox["y2"]))
            cv2.rectangle(image, p1, p2, color, 2)

            label = f"{det['class_name']} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly = max(p1[1] - 6, th + 4)
            cv2.rectangle(image, (p1[0], ly - th - 4), (p1[0] + tw + 4, ly + 2), color, -1)
            cv2.putText(
                image, label, (p1[0] + 2, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Failed to save annotated image to: {output_path}")


detection_service = DetectionService()
