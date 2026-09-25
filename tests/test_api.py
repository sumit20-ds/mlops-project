import pytest
from fastapi.testclient import TestClient

from src import config

SAMPLE = {
    "Study_Hours": 4.5,
    "Age": 21,
    "Avg_Daily_Usage_Hours": 4.0,
    "Daily_Unlocks": 134,
    "Physical_Activity_Hours": 2.2,
    "Sleep_Hours_Per_Night": 6.7,
    "Stress_Level": "Medium",
    "Gender": "Male",
    "Academic_Level": "Undergraduate",
    "Most_Used_Platform": "Facebook",
    "Purpose_Of_Use": "Networking",
    "Country": "India",
}

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="model artifact not present")


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_predict(client):
    resp = client.post("/predict", json=SAMPLE)
    assert resp.status_code == 200
    body = resp.json()
    assert "mental_health_score" in body
    assert 0 <= body["mental_health_score"] <= 10
    assert "request_id" in body


def test_predict_invalid_payload(client):
    bad = {**SAMPLE, "Stress_Level": "Extreme"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post("/predict/batch", json={"instances": [SAMPLE, SAMPLE]})
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"predict_requests_total" in resp.content


def test_drift_endpoint_after_predictions(client):
    client.post("/predict", json=SAMPLE)
    resp = client.get("/monitoring/drift")
    assert resp.status_code == 200
    assert resp.json()["status"] in {"ok", "warn", "alert", "no_data"}
