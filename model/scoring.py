"""센서 1건 → 고장 확률·고장 유형·이상 점수 계산.

서빙 API(serve/app.py)와 단독 실행 데모(deploy/streamlit)가 같은 계산을 쓰도록 한 곳에 모았다.
"""
from __future__ import annotations

import numpy as np

from model.features import reading_to_frame

SENSOR_FIELDS = ["product_type", "air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]


def predict_failure(bundle: dict, reading: dict) -> dict:
    features = reading_to_frame({key: reading[key] for key in SENSOR_FIELDS})
    probability = float(bundle["failure_model"].predict_proba(features)[0, 1])
    type_probabilities = bundle["type_model"].predict_proba(features)[0]
    ranked_types = sorted(zip(bundle["type_model"].classes_, type_probabilities), key=lambda item: -item[1])
    is_failure = probability >= bundle["failure_threshold"]
    return {
        "failure_probability": round(probability, 4),
        "threshold": round(bundle["failure_threshold"], 4),
        "predicted_failure": bool(is_failure),
        # 고장 유형은 '고장이라면 어떤 유형일지'의 조건부 추정이다
        "failure_type": str(ranked_types[0][0]) if is_failure else "NONE",
        "failure_type_probabilities": {str(name): round(float(p), 4) for name, p in ranked_types},
    }


def score_anomaly(bundle: dict, reading: dict) -> dict:
    features = reading_to_frame({key: reading[key] for key in SENSOR_FIELDS})
    raw_score = float(-bundle["anomaly_model"].score_samples(features)[0])
    scale = bundle["anomaly_scale"]
    # 정상 데이터 분위수 기준 구간 선형 변환 (p50→0, p99→0.7, p99.9→0.9, 범위 밖은 0/1로 고정)
    normalized = float(np.interp(raw_score, scale["raw_anchors"], scale["score_anchors"]))
    return {
        "anomaly_score": round(normalized, 4),
        "raw_score": round(raw_score, 4),
        "threshold_raw": round(bundle["anomaly_threshold"], 4),
        "is_anomaly": raw_score >= bundle["anomaly_threshold"],
    }
