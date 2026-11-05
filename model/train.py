"""고장 분류 · 고장 유형 분류 · 이상탐지 모델 학습/비교 + MLflow 기록.

실행: python -m model.train
결과:
  - model/artifacts/bundle.joblib  (서빙용 모델 묶음, git 제외)
  - model/reports/metrics.json     (실측 지표, README 표의 근거)
  - MLflow 실행 기록 (mlflow.db + mlruns/)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model.features import FEATURE_COLUMNS, add_features, failure_type_label, load_raw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
ARTIFACT_DIR = PROJECT_ROOT / "model" / "artifacts"
REPORT_DIR = PROJECT_ROOT / "model" / "reports"
RANDOM_SEED = 42
# 고장 유형 분류기가 맞히려는 유형. RNF(무작위 고장)·UNKNOWN은 센서로 설명이 안 되므로 OTHER로 묶는다.
TYPE_CLASSES = ["TWF", "HDF", "PWF", "OSF", "OTHER"]


def best_f1_threshold(y_true, scores) -> float:
    """검증셋에서 F1이 최대가 되는 임계값을 찾는다 (테스트셋은 보지 않는다)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_values = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # thresholds 길이는 precision보다 1 짧다
    best_index = int(np.nanargmax(f1_values[:-1]))
    return float(thresholds[best_index])


def recall_target_threshold(y_true, scores, target_recall: float = 0.9) -> float:
    """검증셋 재현율이 target 이상인 임계값 중 가장 큰 값 (= 정밀도를 가장 덜 희생)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    candidates = [t for r, t in zip(recall[:-1], thresholds) if r >= target_recall]
    return float(max(candidates)) if candidates else float(thresholds.min())


def binary_metrics(y_true, scores, threshold) -> dict:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def split_data(frame: pd.DataFrame):
    """train 60 / val 20 / test 20, 고장 비율을 유지하도록 층화 분할."""
    train_val, test = train_test_split(
        frame, test_size=0.2, stratify=frame["machine_failure"], random_state=RANDOM_SEED
    )
    train, val = train_test_split(
        train_val, test_size=0.25, stratify=train_val["machine_failure"], random_state=RANDOM_SEED
    )
    return train, val, test


def build_failure_models(positive_ratio: float) -> dict:
    """비교할 고장 분류기들. 불균형 처리 유무를 비교하려고 LightGBM을 두 번 넣는다."""
    lgb_params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                      random_state=RANDOM_SEED, verbose=-1)
    return {
        "logreg_balanced": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)
        ),
        "lgbm_no_weight": lgb.LGBMClassifier(**lgb_params),
        # scale_pos_weight = 음성/양성 비율 → 소수 클래스(고장) 오분류에 더 큰 손실
        "lgbm_weighted": lgb.LGBMClassifier(
            **lgb_params, scale_pos_weight=(1 - positive_ratio) / positive_ratio
        ),
    }


def to_type_class(label: str) -> str:
    return label if label in TYPE_CLASSES[:-1] else "OTHER"


def train_all(csv_path=DEFAULT_CSV, artifact_dir=ARTIFACT_DIR, report_dir=REPORT_DIR,
              use_mlflow: bool = True, sample_rows: int | None = None) -> dict:
    frame = add_features(load_raw(csv_path))
    frame["failure_type"] = failure_type_label(frame)
    if sample_rows:
        frame = frame.sample(n=sample_rows, random_state=RANDOM_SEED)
    train, val, test = split_data(frame)
    x_train, x_val, x_test = (part[FEATURE_COLUMNS] for part in (train, val, test))
    y_train, y_val, y_test = (part["machine_failure"].to_numpy() for part in (train, val, test))
    positive_ratio = float(y_train.mean())

    if use_mlflow:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("factory-pdm")

    results: dict = {
        "dataset": {
            "rows": int(len(frame)), "train": int(len(train)), "val": int(len(val)),
            "test": int(len(test)), "failure_rate": float(frame["machine_failure"].mean()),
        },
        "failure_classifiers": {},
    }
    fitted_models = {}

    # 1) 고장 여부 분류기 비교
    for model_name, estimator in build_failure_models(positive_ratio).items():
        started = time.perf_counter()
        estimator.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        threshold = best_f1_threshold(y_val, estimator.predict_proba(x_val)[:, 1])
        test_metrics = binary_metrics(y_test, estimator.predict_proba(x_test)[:, 1], threshold)
        test_metrics["fit_seconds"] = round(fit_seconds, 3)
        # 운영점 2: 고장을 놓치는 비용이 더 크다고 보고 '검증 재현율 ≥ 0.9'로 맞춘 임계값
        recall_threshold = recall_target_threshold(y_val, estimator.predict_proba(x_val)[:, 1])
        recall_metrics = binary_metrics(y_test, estimator.predict_proba(x_test)[:, 1], recall_threshold)
        test_metrics["recall_priority"] = {key: recall_metrics[key] for key in
                                           ("precision", "recall", "f1", "threshold", "confusion_matrix")}
        results["failure_classifiers"][model_name] = test_metrics
        fitted_models[model_name] = (estimator, threshold)
        if use_mlflow:
            with mlflow.start_run(run_name=model_name):
                mlflow.log_params({"model": model_name, "threshold_from": "val_best_f1"})
                mlflow.log_metrics({k: v for k, v in test_metrics.items()
                                    if isinstance(v, float)})
                mlflow.log_metrics({f"recall_priority_{k}": v for k, v in
                                    test_metrics["recall_priority"].items() if isinstance(v, float)})
                mlflow.log_dict(test_metrics, "test_metrics.json")

    # 2) 이상탐지: 정상 데이터로만 학습하는 비지도 모델 (라벨 없는 현장 가정)
    normal_train = x_train[y_train == 0]
    started = time.perf_counter()
    isolation_forest = IsolationForest(n_estimators=300, contamination="auto",
                                       random_state=RANDOM_SEED)
    isolation_forest.fit(normal_train)
    fit_seconds = time.perf_counter() - started
    # score_samples는 클수록 정상 → 부호를 뒤집어 '클수록 이상'으로 만든다
    val_anomaly = -isolation_forest.score_samples(x_val)
    test_anomaly = -isolation_forest.score_samples(x_test)
    anomaly_threshold = best_f1_threshold(y_val, val_anomaly)
    anomaly_metrics = binary_metrics(y_test, test_anomaly, anomaly_threshold)
    anomaly_metrics["fit_seconds"] = round(fit_seconds, 3)
    results["anomaly_detector"] = {"isolation_forest": anomaly_metrics}
    # 0~1 점수 보정 기준: 정상 학습 데이터 원점수의 분위수
    # 정상 중앙값→0, 정상 99분위→0.7(경고), 99.9분위→0.9(위험)로 구간 선형 변환한다.
    # → "0.9 이상"은 '정상 데이터 1000건 중 1건보다 더 이상한 값'이라는 뜻이 된다.
    normal_scores = -isolation_forest.score_samples(normal_train)
    p50, p99, p999 = (float(np.percentile(normal_scores, q)) for q in (50, 99, 99.9))
    anomaly_scale = {"raw_anchors": [p50, p99, p999, p999 + (p999 - p99)],
                     "score_anchors": [0.0, 0.7, 0.9, 1.0]}
    if use_mlflow:
        with mlflow.start_run(run_name="isolation_forest"):
            mlflow.log_params({"model": "isolation_forest", "trained_on": "normal_only"})
            mlflow.log_metrics({k: v for k, v in anomaly_metrics.items() if isinstance(v, float)})

    # 3) 고장 유형 분류기: 고장 행만으로 학습 (고장이라고 판단된 뒤 '무슨 고장인지')
    failed_train = pd.concat([train, val])
    failed_train = failed_train[failed_train["machine_failure"] == 1]
    failed_test = test[test["machine_failure"] == 1]
    type_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                    min_child_samples=5, class_weight="balanced",
                                    random_state=RANDOM_SEED, verbose=-1)
    type_model.fit(failed_train[FEATURE_COLUMNS], failed_train["failure_type"].map(to_type_class))
    type_predictions = type_model.predict(failed_test[FEATURE_COLUMNS])
    type_truth = failed_test["failure_type"].map(to_type_class)
    present_labels = [label for label in TYPE_CLASSES if label in set(type_truth) | set(type_predictions)]
    results["failure_type_classifier"] = {
        "test_failures": int(len(failed_test)),
        "accuracy": float((type_predictions == type_truth.to_numpy()).mean()),
        "macro_f1": float(f1_score(type_truth, type_predictions, average="macro", zero_division=0)),
        "labels": present_labels,
        "confusion_matrix": confusion_matrix(type_truth, type_predictions,
                                             labels=present_labels).tolist(),
    }
    if use_mlflow:
        with mlflow.start_run(run_name="failure_type_lgbm"):
            mlflow.log_metrics({"accuracy": results["failure_type_classifier"]["accuracy"],
                                "macro_f1": results["failure_type_classifier"]["macro_f1"]})

    # 서빙 모델 선택: 테스트셋이 아닌 '검증셋 기준'으로 고르는 게 원칙이지만,
    # 여기서는 사전에 lgbm_weighted를 주 모델로 정해 두고 비교군만 보고한다.
    serving_name = "lgbm_weighted"
    serving_model, serving_threshold = fitted_models[serving_name]

    # 서빙 지연 실측 (단건 predict_proba, 1000회 평균)
    one_row = x_test.iloc[[0]]
    started = time.perf_counter()
    for _ in range(1000):
        serving_model.predict_proba(one_row)
    results["single_row_inference_ms"] = round((time.perf_counter() - started), 3)  # 1000회 총 s = 1회 ms
    results["serving_model"] = serving_name

    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "failure_model": serving_model,
        "failure_threshold": serving_threshold,
        "type_model": type_model,
        "anomaly_model": isolation_forest,
        "anomaly_threshold": anomaly_threshold,
        "anomaly_scale": anomaly_scale,
        "feature_columns": FEATURE_COLUMNS,
        "serving_model_name": serving_name,
    }, artifact_dir / "bundle.joblib")
    (report_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    results = train_all(args.csv, use_mlflow=not args.no_mlflow)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
