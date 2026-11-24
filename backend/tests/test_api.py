from backend.app.core.config import settings
from backend.app.core.security import API_KEY_HEADER
from backend.tests.conftest import make_png_bytes


def _upload(client, headers=None, **form):
    files = {"file": ("board.png", make_png_bytes(), "image/png")}
    return client.post("/api/predict/image", files=files, data=form, headers=headers)


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready(client):
    resp = client.get("/api/ready")
    assert resp.status_code == 200


def test_model_info(client):
    resp = client.get("/api/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "short" in body["classes"]
    assert body["class_severity"]["short"] == "critical"


def test_predict_records_traceability(client):
    resp = _upload(
        client,
        model_key="nano",
        serial_number="SN-001",
        production_line="LINE-A",
        operator_id="op-7",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["quality_status"] == "FAIL"
    assert body["inspection_id"]
    assert body["traceability"]["serial_number"] == "SN-001"
    assert body["model_version"] == "test_v1"


def test_predict_rejects_non_image(client):
    files = {"file": ("notes.txt", b"hello world", "text/plain")}
    resp = client.post("/api/predict/image", files=files)
    assert resp.status_code == 400


def test_inspection_lifecycle_and_review(client):
    created = _upload(client, serial_number="SN-XYZ").json()
    inspection_id = created["inspection_id"]

    listing = client.get("/api/inspections").json()
    assert any(i["inspection_id"] == inspection_id for i in listing)

    detail = client.get(f"/api/inspections/{inspection_id}").json()
    assert detail["traceability"]["serial_number"] == "SN-XYZ"
    assert detail["detections"][0]["severity"] == "critical"

    # Human agrees it's a FAIL -> AI was correct.
    review = client.post(
        f"/api/inspections/{inspection_id}/review",
        json={"verdict": "FAIL", "reviewed_by": "qa.lead", "notes": "confirmed short"},
    )
    assert review.status_code == 200
    assert review.json()["review_is_correct"] is True


def test_analytics_reflects_review(client):
    created = _upload(client).json()
    client.post(
        f"/api/inspections/{created['inspection_id']}/review",
        json={"verdict": "FAIL", "reviewed_by": "qa.lead"},
    )
    summary = client.get("/api/analytics/summary").json()
    assert summary["total_inspections"] == 1
    assert summary["fail_count"] == 1
    assert summary["dppm"] == 1_000_000.0
    assert summary["reviewed_count"] == 1
    assert summary["agreement_rate_percent"] == 100.0
    assert summary["escape_rate_percent"] == 0.0


def test_analytics_trend(client):
    _upload(client)
    trend = client.get("/api/analytics/trend?days=7").json()
    assert trend["days"] == 7
    assert len(trend["points"]) >= 1
    point = trend["points"][0]
    assert point["total"] >= 1
    assert "dppm" in point and "defect_rate_percent" in point


def test_review_missing_inspection_404(client):
    resp = client.post(
        "/api/inspections/does-not-exist/review",
        json={"verdict": "PASS", "reviewed_by": "qa"},
    )
    assert resp.status_code == 404


def test_auth_enforced_when_keys_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "api_keys", ["secret-key"])

    assert _upload(client).status_code == 401
    assert _upload(client, headers={API_KEY_HEADER: "wrong"}).status_code == 403
    assert _upload(client, headers={API_KEY_HEADER: "secret-key"}).status_code == 200
