"""예측 서빙 API.

실행: uvicorn serve.app:app --port 8000
  GET  /health   모델 로드 여부
  POST /predict  고장 확률 + 고장 유형
  POST /anomaly  이상 점수 (0~1 정규화 + 원점수)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model.scoring import predict_failure, score_anomaly

BUNDLE_PATH = Path(os.getenv(
    "MODEL_BUNDLE", Path(__file__).resolve().parents[1] / "model" / "artifacts" / "bundle.joblib"
))

app = FastAPI(title="factory-pdm serve", version="0.1.0")


class SensorReading(BaseModel):
    machine_id: int | None = Field(default=None, description="설비 번호 (선택)")
    product_type: Literal["L", "M", "H"] = "M"
    air_temp_k: float = Field(..., ge=250, le=350)
    process_temp_k: float = Field(..., ge=250, le=360)
    rpm: float = Field(..., ge=0, le=5000)
    torque_nm: float = Field(..., ge=0, le=150)
    tool_wear_min: float = Field(..., ge=0, le=400)


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"모델 파일 없음: {BUNDLE_PATH} (python -m model.train 먼저 실행)")
    return joblib.load(BUNDLE_PATH)


def get_bundle_or_503() -> dict:
    try:
        return load_bundle()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/health")
def health() -> dict:
    try:
        bundle = load_bundle()
        return {"status": "ok", "model_loaded": True, "serving_model": bundle["serving_model_name"]}
    except FileNotFoundError:
        return {"status": "degraded", "model_loaded": False}


@app.post("/predict")
def predict(reading: SensorReading) -> dict:
    bundle = get_bundle_or_503()
    return {"machine_id": reading.machine_id, **predict_failure(bundle, reading.model_dump())}


@app.post("/anomaly")
def anomaly(reading: SensorReading) -> dict:
    bundle = get_bundle_or_503()
    return {"machine_id": reading.machine_id, **score_anomaly(bundle, reading.model_dump())}
