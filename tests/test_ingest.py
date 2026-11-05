import importlib

from fastapi.testclient import TestClient

READING = {"machine_id": 3, "product_type": "L", "air_temp_k": 299.0, "process_temp_k": 309.0,
           "rpm": 1450, "torque_nm": 45.0, "tool_wear_min": 50, "actual_failure": 0}


def test_ingest_store_and_latest(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("SCORE_ON_INGEST", "0")  # 서빙 API 없이 저장만 검증
    import ingest.app as ingest_app
    ingest_app = importlib.reload(ingest_app)
    client = TestClient(ingest_app.app)
    assert client.post("/readings", json=READING).status_code == 201
    client.post("/readings", json={**READING, "torque_nm": 50.0})
    latest = client.get("/machines/3/latest").json()
    assert latest["torque_nm"] == 50.0
    assert client.get("/machines/9/latest").status_code == 404
    assert client.get("/health").json()["rows"] == 2
