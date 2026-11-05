"""센서 수집 API → SQLite 저장.

실행: uvicorn ingest.app:app --port 8001
  POST /readings                      센서 1건 수집 (SERVE_URL이 살아 있으면 예측·이상점수도 함께 저장)
  GET  /readings?machine_id=&limit=   최근 기록
  GET  /machines/{machine_id}/latest  설비별 최신 1건 (에이전트가 사용)
"""
from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import closing
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("ingest")

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at TEXT NOT NULL,
    machine_id INTEGER NOT NULL,
    source_udi INTEGER,
    product_type TEXT NOT NULL,
    air_temp_k REAL NOT NULL,
    process_temp_k REAL NOT NULL,
    rpm REAL NOT NULL,
    torque_nm REAL NOT NULL,
    tool_wear_min REAL NOT NULL,
    actual_failure INTEGER,
    failure_probability REAL,
    predicted_failure INTEGER,
    failure_type TEXT,
    anomaly_score REAL,
    is_anomaly INTEGER
);
CREATE INDEX IF NOT EXISTS idx_readings_machine ON readings(machine_id, id);
"""

SENSOR_FIELDS = ["product_type", "air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]


class IngestReading(BaseModel):
    machine_id: int = Field(..., ge=1)
    source_udi: int | None = None
    product_type: Literal["L", "M", "H"]
    air_temp_k: float
    process_temp_k: float
    rpm: float
    torque_nm: float
    tool_wear_min: float
    actual_failure: int | None = Field(default=None, description="시뮬레이터가 아는 실제 라벨(평가용)")


def db_path() -> Path:
    path = Path(os.getenv("DB_PATH", PROJECT_ROOT / "data" / "sensor.db"))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(db_path())
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


@lru_cache(maxsize=1)
def serve_client() -> httpx.Client:
    """요청마다 새 연결을 만들지 않도록 클라이언트를 재사용한다 (keep-alive)."""
    return httpx.Client(base_url=os.getenv("SERVE_URL", "http://127.0.0.1:8000"), timeout=2.0)


def score_with_serve(payload: dict) -> dict:
    """서빙 API가 꺼져 있으면 점수 없이 저장만 한다 (수집은 멈추지 않는 게 우선)."""
    try:
        client = serve_client()
        prediction = client.post("/predict", json=payload).raise_for_status().json()
        anomaly = client.post("/anomaly", json=payload).raise_for_status().json()
    except httpx.HTTPError as error:
        logger.warning("serve 호출 실패, 점수 없이 저장: %s", error)
        return {}
    return {
        "failure_probability": prediction["failure_probability"],
        "predicted_failure": int(prediction["predicted_failure"]),
        "failure_type": prediction["failure_type"],
        "anomaly_score": anomaly["anomaly_score"],
        "is_anomaly": int(anomaly["is_anomaly"]),
    }


app = FastAPI(title="factory-pdm ingest", version="0.1.0")


@app.post("/readings", status_code=201)
def ingest(reading: IngestReading) -> dict:
    row = reading.model_dump()
    row["received_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if os.getenv("SCORE_ON_INGEST", "1") == "1":
        row.update(score_with_serve({key: row[key] for key in SENSOR_FIELDS + ["machine_id"]}))
    columns = ", ".join(row)
    placeholders = ", ".join(f":{key}" for key in row)
    with closing(connect()) as connection, connection:
        cursor = connection.execute(f"INSERT INTO readings ({columns}) VALUES ({placeholders})", row)
        row["id"] = cursor.lastrowid
    return row


@app.get("/readings")
def list_readings(machine_id: int | None = None, limit: int = 200) -> list[dict]:
    limit = max(1, min(limit, 5000))
    query = "SELECT * FROM readings"
    params: list = []
    if machine_id is not None:
        query += " WHERE machine_id = ?"
        params.append(machine_id)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with closing(connect()) as connection:
        return [dict(record) for record in connection.execute(query, params)]


@app.get("/machines/{machine_id}/latest")
def latest(machine_id: int) -> dict:
    records = list_readings(machine_id=machine_id, limit=1)
    if not records:
        raise HTTPException(status_code=404, detail=f"{machine_id}번 설비 기록 없음")
    return records[0]


@app.get("/health")
def health() -> dict:
    with closing(connect()) as connection:
        count = connection.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    return {"status": "ok", "rows": count}
