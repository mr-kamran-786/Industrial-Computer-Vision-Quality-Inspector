"""Quality-decision policy engine.

Turns raw model detections into an auditable PASS/FAIL verdict. The engine is
deliberately explicit and side-effect free so its decisions can be unit-tested
and explained to an auditor: every verdict comes with the exact reasons that
produced it, the per-class threshold applied, and the severity of each defect.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.core.config import (
    CLASS_SEVERITY,
    PER_CLASS_THRESHOLDS,
    SEVERITY_RANK,
    Severity,
    settings,
)

PASS = "PASS"
FAIL = "FAIL"


def threshold_for(class_name: str) -> float:
    """Confidence threshold for a class, falling back to the global default."""
    return PER_CLASS_THRESHOLDS.get(
        class_name, settings.global_confidence_threshold
    )


def severity_for(class_name: str) -> str:
    """Severity for a class, defaulting to MAJOR for unknown classes."""
    return CLASS_SEVERITY.get(class_name, Severity.MAJOR)


def confidence_floor() -> float:
    """Lowest threshold across all classes.

    Used to configure the model's own ``conf`` so it returns every box that any
    per-class threshold might accept; the policy then applies the stricter,
    class-specific thresholds itself.
    """
    values = list(PER_CLASS_THRESHOLDS.values()) + [
        settings.global_confidence_threshold
    ]
    return max(0.01, min(values))


@dataclass
class PolicyResult:
    quality_status: str
    is_defective: bool
    highest_severity: Optional[str]
    counted_detections: List[Dict[str, Any]]
    filtered_detections: List[Dict[str, Any]]
    reasons: List[str] = field(default_factory=list)

    @property
    def decision_reason(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "No defects above threshold."


def enrich_detection(detection: Dict[str, Any]) -> Dict[str, Any]:
    """Attach severity and the applied threshold to a raw detection."""
    class_name = detection["class_name"]
    return {
        **detection,
        "severity": severity_for(class_name),
        "threshold": threshold_for(class_name),
    }


def evaluate(detections: List[Dict[str, Any]]) -> PolicyResult:
    """Evaluate a list of raw detections into a quality verdict.

    Each detection must contain at least ``class_name`` and ``confidence``.
    """
    counted: List[Dict[str, Any]] = []
    filtered: List[Dict[str, Any]] = []

    for raw in detections:
        enriched = enrich_detection(raw)
        if enriched["confidence"] >= enriched["threshold"]:
            counted.append(enriched)
        else:
            filtered.append(enriched)

    reasons: List[str] = []
    highest_severity: Optional[str] = None
    if counted:
        highest_severity = max(
            (d["severity"] for d in counted),
            key=lambda s: SEVERITY_RANK.get(s, 0),
        )

    fail_severities = set(settings.fail_severities)

    # Rule 1: any defect at or above a fail-triggering severity fails the part.
    failing = [d for d in counted if d["severity"] in fail_severities]
    if failing:
        by_class: Dict[str, int] = {}
        for d in failing:
            by_class[d["class_name"]] = by_class.get(d["class_name"], 0) + 1
        summary = ", ".join(f"{count}x {name}" for name, count in by_class.items())
        reasons.append(f"Fail-severity defect(s) detected: {summary}.")

    # Rule 2: too many below-fail-severity (e.g. minor) defects also fails.
    minor = [d for d in counted if d["severity"] not in fail_severities]
    if len(minor) > settings.max_acceptable_minor_defects:
        reasons.append(
            f"{len(minor)} minor defect(s) exceed tolerance of "
            f"{settings.max_acceptable_minor_defects}."
        )

    is_fail = bool(reasons)

    return PolicyResult(
        quality_status=FAIL if is_fail else PASS,
        is_defective=is_fail,
        highest_severity=highest_severity,
        counted_detections=counted,
        filtered_detections=filtered,
        reasons=reasons,
    )
