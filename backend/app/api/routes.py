import logging
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from backend.app.core.config import (
    CLASS_NAMES,
    CLASS_SEVERITY,
    DEFAULT_MODEL_KEY,
    PER_CLASS_THRESHOLDS,
    RESULTS_DIR,
    UPLOAD_DIR,
    settings,
)
from backend.app.core.security import require_api_key
from backend.app.db.analytics import get_analytics_summary, get_analytics_trend
from backend.app.db.crud import (
    apply_review,
    create_inspection_record,
    get_inspection_by_id,
    get_recent_inspections,
)
from backend.app.db.database import get_db
from backend.app.schemas.analytics import AnalyticsSummary, AnalyticsTrend
from backend.app.schemas.detection import (
    InspectionDetail,
    InspectionSummary,
    PredictionResponse,
)
from backend.app.schemas.review import ReviewRequest, ReviewResponse
from backend.app.services.detection_service import detection_service

logger = logging.getLogger("icvqi.api")

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


@router.get("/ready")
def readiness_check():
    """Readiness probe: reports whether the default model can actually serve."""
    ready = detection_service.is_ready()
    if not ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Default model file is not available.",
        )
    return {"status": "ready"}


@router.get("/model/info")
def model_info():
    return {
        "model_type": "YOLO object detection",
        "task": "PCB defect detection",
        "default_model_key": DEFAULT_MODEL_KEY,
        "classes": list(CLASS_NAMES.values()),
        "class_severity": CLASS_SEVERITY,
        "per_class_thresholds": PER_CLASS_THRESHOLDS,
        "fail_severities": settings.fail_severities,
        "models": detection_service.get_available_models(),
    }


def _validate_upload(file_bytes: bytes, extension: str) -> None:
    if extension not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{extension}'. Allowed: {settings.allowed_extensions}",
        )
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(file_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.max_upload_bytes} bytes.",
        )
    # Verify the bytes are actually a decodable image (defends against a file
    # that merely carries an image extension).
    try:
        Image.open(BytesIO(file_bytes)).verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a valid image.")


@router.post(
    "/predict/image",
    response_model=PredictionResponse,
    dependencies=[Depends(require_api_key)],
)
async def predict_image(
    file: UploadFile = File(...),
    model_key: str = Form(DEFAULT_MODEL_KEY),
    serial_number: Optional[str] = Form(None),
    batch_lot: Optional[str] = Form(None),
    production_line: Optional[str] = Form(None),
    station_id: Optional[str] = Form(None),
    operator_id: Optional[str] = Form(None),
    shift: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    original_filename = file.filename or "uploaded_image.jpg"
    file_extension = Path(original_filename).suffix.lower()

    file_bytes = await file.read()
    _validate_upload(file_bytes, file_extension)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    inspection_id = uuid4().hex
    saved_filename = f"{inspection_id}{file_extension}"
    saved_path = UPLOAD_DIR / saved_filename
    result_filename = f"{inspection_id}_annotated.jpg"
    result_path = RESULTS_DIR / result_filename

    with open(saved_path, "wb") as handle:
        handle.write(file_bytes)

    try:
        prediction = detection_service.predict_image(
            image_path=saved_path,
            output_path=result_path,
            model_key=model_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        logger.error("Model unavailable", extra={"model_key": model_key})
        raise HTTPException(status_code=503, detail=str(exc))

    result_image_url = f"/static/results/{result_filename}"
    traceability = {
        "serial_number": serial_number,
        "batch_lot": batch_lot,
        "production_line": production_line,
        "station_id": station_id,
        "operator_id": operator_id,
        "shift": shift,
    }

    create_inspection_record(
        db=db,
        inspection_id=inspection_id,
        original_filename=original_filename,
        saved_filename=saved_filename,
        result_filename=result_filename,
        result_image_url=result_image_url,
        prediction=prediction,
        traceability=traceability,
    )

    logger.info(
        "Inspection complete",
        extra={
            "inspection_id": inspection_id,
            "verdict": prediction["quality_status"],
            "defects": prediction["total_detections"],
            "model_key": model_key,
            "inference_ms": prediction["inference_ms"],
            "serial_number": serial_number,
        },
    )

    prediction["inspection_id"] = inspection_id
    prediction["result_image_url"] = result_image_url
    prediction["traceability"] = traceability
    return prediction


@router.get("/inspections", response_model=List[InspectionSummary])
def list_inspections(
    limit: int = 20,
    production_line: Optional[str] = None,
    quality_status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    limit = max(1, min(limit, 200))
    inspections = get_recent_inspections(
        db=db,
        limit=limit,
        production_line=production_line,
        quality_status=quality_status,
    )
    return [
        {
            "inspection_id": inspection.id,
            "original_filename": inspection.original_filename,
            "quality_status": inspection.quality_status,
            "is_defective": inspection.is_defective,
            "total_detections": inspection.total_detections,
            "highest_severity": inspection.highest_severity,
            "serial_number": inspection.serial_number,
            "production_line": inspection.production_line,
            "reviewed": inspection.reviewed,
            "review_is_correct": inspection.review_is_correct,
            "result_image_url": inspection.result_image_url,
            "created_at": inspection.created_at,
        }
        for inspection in inspections
    ]


@router.get("/inspections/{inspection_id}", response_model=InspectionDetail)
def get_inspection_detail(inspection_id: str, db: Session = Depends(get_db)):
    inspection = get_inspection_by_id(db=db, inspection_id=inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail="Inspection not found.")

    return {
        "inspection_id": inspection.id,
        "original_filename": inspection.original_filename,
        "saved_filename": inspection.saved_filename,
        "result_filename": inspection.result_filename,
        "result_image_url": inspection.result_image_url,
        "quality_status": inspection.quality_status,
        "is_defective": inspection.is_defective,
        "total_detections": inspection.total_detections,
        "highest_severity": inspection.highest_severity,
        "decision_reason": inspection.decision_reason,
        "model_key": inspection.model_key,
        "model_name": inspection.model_name,
        "model_version": inspection.model_version,
        "inference_ms": inspection.inference_ms,
        "traceability": {
            "serial_number": inspection.serial_number,
            "batch_lot": inspection.batch_lot,
            "production_line": inspection.production_line,
            "station_id": inspection.station_id,
            "operator_id": inspection.operator_id,
            "shift": inspection.shift,
        },
        "reviewed": inspection.reviewed,
        "review_verdict": inspection.review_verdict,
        "review_is_correct": inspection.review_is_correct,
        "reviewed_by": inspection.reviewed_by,
        "reviewed_at": inspection.reviewed_at,
        "review_notes": inspection.review_notes,
        "created_at": inspection.created_at,
        "detections": [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "severity": d.severity,
                "threshold": d.threshold_used,
                "bbox": {"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2},
            }
            for d in inspection.detections
        ],
    }


@router.post(
    "/inspections/{inspection_id}/review",
    response_model=ReviewResponse,
    dependencies=[Depends(require_api_key)],
)
def review_inspection(
    inspection_id: str,
    review: ReviewRequest,
    db: Session = Depends(get_db),
):
    """Record an operator's ground-truth judgement (feeds escape/false-call rates)."""
    inspection = get_inspection_by_id(db=db, inspection_id=inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail="Inspection not found.")

    updated = apply_review(
        db=db,
        inspection=inspection,
        verdict=review.verdict,
        reviewed_by=review.reviewed_by,
        notes=review.notes,
    )
    logger.info(
        "Inspection reviewed",
        extra={
            "inspection_id": inspection_id,
            "ai_verdict": updated.quality_status,
            "human_verdict": updated.review_verdict,
            "correct": updated.review_is_correct,
            "reviewed_by": updated.reviewed_by,
        },
    )
    return {
        "inspection_id": updated.id,
        "quality_status": updated.quality_status,
        "review_verdict": updated.review_verdict,
        "review_is_correct": updated.review_is_correct,
        "reviewed_by": updated.reviewed_by,
        "reviewed_at": updated.reviewed_at,
        "review_notes": updated.review_notes,
    }


@router.get("/analytics/summary", response_model=AnalyticsSummary)
def analytics_summary(db: Session = Depends(get_db)):
    return get_analytics_summary(db=db)


@router.get("/analytics/trend", response_model=AnalyticsTrend)
def analytics_trend(days: int = 14, db: Session = Depends(get_db)):
    days = max(1, min(days, 90))
    return get_analytics_trend(db=db, days=days)
