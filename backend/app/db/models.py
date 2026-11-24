from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from backend.app.db.database import Base


class Inspection(Base):
    """A single quality inspection of one physical unit.

    Carries full traceability (which unit, which line/station/shift/operator),
    model provenance (which model version + threshold produced the verdict), and
    an optional operator review used to measure real escape / false-call rates.
    """

    __tablename__ = "inspections"

    id = Column(String, primary_key=True, index=True)

    # Image artifacts
    original_filename = Column(String, nullable=False)
    saved_filename = Column(String, nullable=False)
    result_filename = Column(String, nullable=False)
    result_image_url = Column(String, nullable=False)

    # AI verdict
    quality_status = Column(String, nullable=False, index=True)
    is_defective = Column(Boolean, nullable=False)
    total_detections = Column(Integer, nullable=False)
    highest_severity = Column(String, nullable=True)
    decision_reason = Column(Text, nullable=True)

    # Model provenance (reproducibility / audit)
    model_key = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    confidence_floor = Column(Float, nullable=True)
    inference_ms = Column(Float, nullable=True)

    # Traceability to the physical part and production context
    serial_number = Column(String, nullable=True, index=True)
    batch_lot = Column(String, nullable=True, index=True)
    production_line = Column(String, nullable=True, index=True)
    station_id = Column(String, nullable=True)
    operator_id = Column(String, nullable=True)
    shift = Column(String, nullable=True)

    # Operator review / ground-truth feedback loop
    reviewed = Column(Boolean, nullable=False, default=False, index=True)
    review_verdict = Column(String, nullable=True)  # PASS / FAIL as judged by human
    review_is_correct = Column(Boolean, nullable=True)  # did AI match the human?
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    review_notes = Column(Text, nullable=True)

    created_at = Column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )

    detections = relationship(
        "InspectionDetection",
        back_populates="inspection",
        cascade="all, delete-orphan",
    )


class InspectionDetection(Base):
    __tablename__ = "inspection_detections"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    inspection_id = Column(
        String,
        ForeignKey("inspections.id"),
        nullable=False,
        index=True,
    )

    class_id = Column(Integer, nullable=False)
    class_name = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    severity = Column(String, nullable=True)
    threshold_used = Column(Float, nullable=True)

    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)

    inspection = relationship(
        "Inspection",
        back_populates="detections",
    )
