import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.db import models  # noqa: F401 (register mappers)
from backend.app.db.database import Base, get_db
from backend.app.main import app
from backend.app.services.detection_service import detection_service


def make_png_bytes(size=(64, 64), color=(120, 120, 120)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def session_factory(tmp_path):
    db_file = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_file.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    yield factory
    engine.dispose()


@pytest.fixture
def client(session_factory, monkeypatch):
    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    def fake_predict(image_path, output_path=None, model_key="nano"):
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"annotated")
        return {
            "filename": image_path.name,
            "model_key": model_key,
            "model_name": "Test Model",
            "model_version": "test_v1",
            "confidence_floor": 0.25,
            "inference_ms": 1.0,
            "total_detections": 1,
            "is_defective": True,
            "quality_status": "FAIL",
            "highest_severity": "critical",
            "decision_reason": "Fail-severity defect(s) detected: 1x short.",
            "detections": [
                {
                    "class_id": 3,
                    "class_name": "short",
                    "confidence": 0.9,
                    "severity": "critical",
                    "threshold": 0.25,
                    "bbox": {"x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0},
                }
            ],
        }

    monkeypatch.setattr(detection_service, "predict_image", fake_predict)
    monkeypatch.setattr(detection_service, "is_ready", lambda: True)
    # Auth disabled by default in tests.
    monkeypatch.setattr(settings, "api_keys", [])

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
