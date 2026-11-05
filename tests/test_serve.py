import importlib

import pytest
from fastapi.testclient import TestClient

NORMAL = {"machine_id": 1, "product_type": "M", "air_temp_k": 300.0, "process_temp_k": 310.0,
          "rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 20}
OVERLOAD = {**NORMAL, "rpm": 1200, "torque_nm": 70.0, "tool_wear_min": 230}


@pytest.fixture()
def client(trained_bundle, monkeypatch):
    bundle_path, _ = trained_bundle
    monkeypatch.setenv("MODEL_BUNDLE", str(bundle_path))
    import serve.app as serve_app
    serve_app = importlib.reload(serve_app)  # 환경변수를 다시 읽게 한다
    return TestClient(serve_app.app)


def test_health(client):
    assert client.get("/health").json()["model_loaded"] is True


def test_predict_schema_and_overload_is_riskier(client):
    normal = client.post("/predict", json=NORMAL).json()
    overload = client.post("/predict", json=OVERLOAD).json()
    assert 0 <= normal["failure_probability"] <= 1
    assert overload["failure_probability"] > normal["failure_probability"]


def test_anomaly_score_higher_for_overload(client):
    normal = client.post("/anomaly", json=NORMAL).json()
    overload = client.post("/anomaly", json=OVERLOAD).json()
    assert 0 <= normal["anomaly_score"] <= 1
    assert overload["raw_score"] > normal["raw_score"]


def test_invalid_input_rejected(client):
    assert client.post("/predict", json={**NORMAL, "rpm": -5}).status_code == 422


def test_missing_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_BUNDLE", str(tmp_path / "none.joblib"))
    import serve.app as serve_app
    serve_app = importlib.reload(serve_app)
    response = TestClient(serve_app.app).post("/predict", json=NORMAL)
    assert response.status_code == 503
