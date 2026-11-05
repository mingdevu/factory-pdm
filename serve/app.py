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
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model.features import reading_to_frame

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


def reading_features(reading: SensorReading):
    return reading_to_frame(reading.model_dump(exclude={"machine_id"}))


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
    features = reading_features(reading)
    probability = float(bundle["failure_model"].predict_proba(features)[0, 1])
    type_probabilities = bundle["type_model"].predict_proba(features)[0]
    type_classes = list(bundle["type_model"].classes_)
    ranked_types = sorted(zip(type_classes, type_probabilities), key=lambda item: -item[1])
    is_failure = probability >= bundle["failure_threshold"]
    return {
        "machine_id": reading.machine_id,
        "failure_probability": round(probability, 4),
        "threshold": round(bundle["failure_threshold"], 4),
        "predicted_failure": bool(is_failure),
        # 고장 유형은 '고장이라면 어떤 유형일지'의 조건부 추정이다
        "failure_type": ranked_types[0][0] if is_failure else "NONE",
        "failure_type_probabilities": {name: round(float(p), 4) for name, p in ranked_types},
    }


@app.post("/anomaly")
def anomaly(reading: SensorReading) -> dict:
    bundle = get_bundle_or_503()
    features = reading_features(reading)
    raw_score = float(-bundle["anomaly_model"].score_samples(features)[0])
    scale = bundle["anomaly_scale"]
    # 정상 데이터 분위수 기준 구간 선형 변환 (p50→0, p99→0.7, p99.9→0.9, 범위 밖은 0/1로 고정)
    normalized = float(np.interp(raw_score, scale["raw_anchors"], scale["score_anchors"]))
    return {
        "machine_id": reading.machine_id,
        "anomaly_score": round(normalized, 4),
        "raw_score": round(raw_score, 4),
        "threshold_raw": round(bundle["anomaly_threshold"], 4),
        "is_anomaly": raw_score >= bundle["anomaly_threshold"],
    }
