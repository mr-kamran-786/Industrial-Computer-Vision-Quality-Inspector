from backend.app.core import policy
from backend.app.core.config import settings


def _det(class_name, confidence):
    return {
        "class_id": 0,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
    }


def test_empty_detections_pass():
    result = policy.evaluate([])
    assert result.quality_status == "PASS"
    assert result.is_defective is False
    assert result.highest_severity is None


def test_critical_defect_fails():
    result = policy.evaluate([_det("short", 0.9)])
    assert result.quality_status == "FAIL"
    assert result.highest_severity == "critical"
    assert len(result.counted_detections) == 1


def test_low_confidence_detection_is_filtered_out():
    # A detection below the class threshold must not count toward the verdict.
    below = policy.threshold_for("spur") - 0.05
    result = policy.evaluate([_det("spur", below)])
    assert result.quality_status == "PASS"
    assert len(result.counted_detections) == 0
    assert len(result.filtered_detections) == 1


def test_major_defect_above_threshold_fails():
    result = policy.evaluate([_det("mouse_bite", 0.95)])
    assert result.quality_status == "FAIL"
    assert result.highest_severity == "major"


def test_counted_detection_is_enriched():
    result = policy.evaluate([_det("short", 0.9)])
    det = result.counted_detections[0]
    assert det["severity"] == "critical"
    assert det["threshold"] == policy.threshold_for("short")


def test_confidence_floor_is_min_threshold():
    assert policy.confidence_floor() <= settings.global_confidence_threshold
    assert policy.confidence_floor() > 0
