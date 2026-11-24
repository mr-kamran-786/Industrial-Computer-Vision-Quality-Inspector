from datetime import datetime, timedelta
from typing import Any, Dict

from sqlalchemy import case, desc, func
from sqlalchemy.orm import Session

from backend.app.db.models import Inspection, InspectionDetection


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100, 2)


def get_analytics_trend(db: Session, days: int = 14) -> Dict[str, Any]:
    """Per-day yield / defect-rate / DPPM trend for SPC-style charts."""
    since = datetime.utcnow() - timedelta(days=days)
    day = func.date(Inspection.created_at)

    rows = (
        db.query(
            day.label("day"),
            func.count(Inspection.id).label("total"),
            func.sum(
                case((Inspection.quality_status == "FAIL", 1), else_=0)
            ).label("fail"),
        )
        .filter(Inspection.created_at >= since)
        .group_by(day)
        .order_by(day)
        .all()
    )

    points = []
    for row in rows:
        total = int(row.total)
        fail = int(row.fail or 0)
        points.append(
            {
                "date": str(row.day),
                "total": total,
                "pass_count": total - fail,
                "fail_count": fail,
                "defect_rate_percent": _rate(fail, total),
                "dppm": round((fail / total) * 1_000_000, 1) if total else 0.0,
            }
        )

    return {"days": days, "points": points}


def get_analytics_summary(db: Session) -> Dict[str, Any]:
    total_inspections = db.query(func.count(Inspection.id)).scalar() or 0

    fail_count = (
        db.query(func.count(Inspection.id))
        .filter(Inspection.quality_status == "FAIL")
        .scalar()
        or 0
    )
    pass_count = total_inspections - fail_count

    total_detected_defects = (
        db.query(func.count(InspectionDetection.id)).scalar() or 0
    )

    average_confidence = db.query(
        func.avg(InspectionDetection.confidence)
    ).scalar()
    if average_confidence is not None:
        average_confidence = round(float(average_confidence), 4)

    defect_rate_percent = _rate(fail_count, total_inspections)
    yield_percent = _rate(pass_count, total_inspections)
    dppm = round((fail_count / total_inspections) * 1_000_000, 1) if total_inspections else 0.0

    # Defect distribution by class
    defect_rows = (
        db.query(
            InspectionDetection.class_name,
            func.count(InspectionDetection.id).label("count"),
        )
        .group_by(InspectionDetection.class_name)
        .order_by(desc("count"))
        .all()
    )
    defect_distribution = [
        {"class_name": row.class_name, "count": int(row.count)}
        for row in defect_rows
    ]
    most_common_defect = defect_distribution[0]["class_name"] if defect_distribution else None

    # Distribution by severity
    severity_rows = (
        db.query(
            InspectionDetection.severity,
            func.count(InspectionDetection.id).label("count"),
        )
        .filter(InspectionDetection.severity.isnot(None))
        .group_by(InspectionDetection.severity)
        .order_by(desc("count"))
        .all()
    )
    severity_distribution = [
        {"severity": row.severity, "count": int(row.count)}
        for row in severity_rows
    ]

    # Ground-truth feedback metrics
    reviewed_count = (
        db.query(func.count(Inspection.id))
        .filter(Inspection.reviewed.is_(True))
        .scalar()
        or 0
    )

    escape_rate_percent = None
    false_call_rate_percent = None
    agreement_rate_percent = None
    if reviewed_count > 0:
        # Escape: AI passed it, human failed it (defect escaped the gate).
        escapes = (
            db.query(func.count(Inspection.id))
            .filter(
                Inspection.reviewed.is_(True),
                Inspection.quality_status == "PASS",
                Inspection.review_verdict == "FAIL",
            )
            .scalar()
            or 0
        )
        # False call: AI failed it, human passed it (good part rejected).
        false_calls = (
            db.query(func.count(Inspection.id))
            .filter(
                Inspection.reviewed.is_(True),
                Inspection.quality_status == "FAIL",
                Inspection.review_verdict == "PASS",
            )
            .scalar()
            or 0
        )
        agreements = (
            db.query(func.count(Inspection.id))
            .filter(
                Inspection.reviewed.is_(True),
                Inspection.review_is_correct.is_(True),
            )
            .scalar()
            or 0
        )
        escape_rate_percent = _rate(escapes, reviewed_count)
        false_call_rate_percent = _rate(false_calls, reviewed_count)
        agreement_rate_percent = _rate(agreements, reviewed_count)

    recent_failures_rows = (
        db.query(Inspection)
        .filter(Inspection.quality_status == "FAIL")
        .order_by(Inspection.created_at.desc())
        .limit(5)
        .all()
    )
    recent_failures = [
        {
            "inspection_id": inspection.id,
            "original_filename": inspection.original_filename,
            "total_detections": inspection.total_detections,
            "highest_severity": inspection.highest_severity,
            "result_image_url": inspection.result_image_url,
            "created_at": inspection.created_at,
        }
        for inspection in recent_failures_rows
    ]

    return {
        "total_inspections": total_inspections,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "defect_rate_percent": defect_rate_percent,
        "yield_percent": yield_percent,
        "dppm": dppm,
        "total_detected_defects": total_detected_defects,
        "average_confidence": average_confidence,
        "most_common_defect": most_common_defect,
        "defect_distribution": defect_distribution,
        "severity_distribution": severity_distribution,
        "reviewed_count": reviewed_count,
        "escape_rate_percent": escape_rate_percent,
        "false_call_rate_percent": false_call_rate_percent,
        "agreement_rate_percent": agreement_rate_percent,
        "recent_failures": recent_failures,
    }
