from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.app.db.models import Inspection, InspectionDetection


def create_inspection_record(
    db: Session,
    inspection_id: str,
    original_filename: str,
    saved_filename: str,
    result_filename: str,
    result_image_url: str,
    prediction: Dict[str, Any],
    traceability: Optional[Dict[str, Any]] = None,
) -> Inspection:
    traceability = traceability or {}

    inspection = Inspection(
        id=inspection_id,
        original_filename=original_filename,
        saved_filename=saved_filename,
        result_filename=result_filename,
        result_image_url=result_image_url,
        quality_status=prediction["quality_status"],
        is_defective=prediction["is_defective"],
        total_detections=prediction["total_detections"],
        highest_severity=prediction.get("highest_severity"),
        decision_reason=prediction.get("decision_reason"),
        model_key=prediction["model_key"],
        model_name=prediction["model_name"],
        model_version=prediction["model_version"],
        confidence_floor=prediction.get("confidence_floor"),
        inference_ms=prediction.get("inference_ms"),
        serial_number=traceability.get("serial_number"),
        batch_lot=traceability.get("batch_lot"),
        production_line=traceability.get("production_line"),
        station_id=traceability.get("station_id"),
        operator_id=traceability.get("operator_id"),
        shift=traceability.get("shift"),
    )

    for detection in prediction["detections"]:
        bbox = detection["bbox"]
        inspection.detections.append(
            InspectionDetection(
                class_id=detection["class_id"],
                class_name=detection["class_name"],
                confidence=detection["confidence"],
                severity=detection.get("severity"),
                threshold_used=detection.get("threshold"),
                x1=bbox["x1"],
                y1=bbox["y1"],
                x2=bbox["x2"],
                y2=bbox["y2"],
            )
        )

    db.add(inspection)
    db.commit()
    db.refresh(inspection)
    return inspection


def get_recent_inspections(
    db: Session,
    limit: int = 20,
    production_line: Optional[str] = None,
    quality_status: Optional[str] = None,
) -> List[Inspection]:
    query = db.query(Inspection)
    if production_line:
        query = query.filter(Inspection.production_line == production_line)
    if quality_status:
        query = query.filter(Inspection.quality_status == quality_status)
    return query.order_by(Inspection.created_at.desc()).limit(limit).all()


def get_inspection_by_id(
    db: Session,
    inspection_id: str,
) -> Optional[Inspection]:
    return (
        db.query(Inspection)
        .filter(Inspection.id == inspection_id)
        .first()
    )


def apply_review(
    db: Session,
    inspection: Inspection,
    verdict: str,
    reviewed_by: str,
    notes: Optional[str] = None,
) -> Inspection:
    """Record an operator's ground-truth judgement on an inspection."""
    inspection.reviewed = True
    inspection.review_verdict = verdict
    inspection.review_is_correct = verdict == inspection.quality_status
    inspection.reviewed_by = reviewed_by
    inspection.reviewed_at = datetime.utcnow()
    inspection.review_notes = notes

    db.add(inspection)
    db.commit()
    db.refresh(inspection)
    return inspection
