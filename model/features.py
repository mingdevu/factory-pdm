"""센서 원시값 → 모델 입력 피처 변환.

학습(train.py)과 서빙(serve/app.py)이 같은 함수를 쓰도록 한 곳에 모아 둔다.
(학습/서빙 피처가 어긋나는 training-serving skew를 막기 위함)
"""
from __future__ import annotations

import math

import pandas as pd

# 원본 CSV 컬럼명 → 코드에서 쓰는 짧은 이름
RAW_COLUMN_MAP = {
    "Type": "product_type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}

SENSOR_COLUMNS = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
FAILURE_TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
PRODUCT_TYPES = ["L", "M", "H"]

FEATURE_COLUMNS = SENSOR_COLUMNS + [
    "temp_diff_k",       # 공정온도 - 주변온도 (작으면 방열 불량 → HDF)
    "power_w",           # 토크 × 각속도 (너무 작거나 크면 PWF)
    "strain_min_nm",     # 공구마모 × 토크 (과부하 → OSF)
    "type_L",
    "type_M",
    "type_H",
]


def load_raw(csv_path) -> pd.DataFrame:
    """원본 CSV를 읽어 컬럼명을 정리한다."""
    raw_frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    return raw_frame.rename(columns=RAW_COLUMN_MAP)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """물리적 의미가 있는 파생 피처를 추가한다. 입력 frame은 수정하지 않는다."""
    featured = frame.copy()
    featured["temp_diff_k"] = featured["process_temp_k"] - featured["air_temp_k"]
    angular_velocity = featured["rpm"] * 2 * math.pi / 60
    featured["power_w"] = featured["torque_nm"] * angular_velocity
    featured["strain_min_nm"] = featured["tool_wear_min"] * featured["torque_nm"]
    for product_type in PRODUCT_TYPES:
        featured[f"type_{product_type}"] = (featured["product_type"] == product_type).astype(int)
    return featured


def failure_type_label(frame: pd.DataFrame) -> pd.Series:
    """고장 유형 라벨을 하나로 합친다.

    원본은 유형별 0/1 플래그(멀티라벨)라서, 고장 행에서는 FAILURE_TYPES 순서상 첫 번째로
    켜진 유형을 대표 라벨로 쓴다. 고장이지만 플래그가 하나도 없는 행은 'UNKNOWN'.
    """
    def pick(row) -> str:
        if row["machine_failure"] == 0:
            return "NONE"
        for failure_type in FAILURE_TYPES:
            if row[failure_type] == 1:
                return failure_type
        return "UNKNOWN"

    return frame.apply(pick, axis=1)


def reading_to_frame(reading: dict) -> pd.DataFrame:
    """API로 들어온 센서 1건(dict)을 피처 DataFrame(1행)으로 만든다."""
    frame = pd.DataFrame([reading])
    return add_features(frame)[FEATURE_COLUMNS]
