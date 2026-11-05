"""factory-pdm 단독 실행 데모 (Streamlit Community Cloud용).

무료 호스팅은 프로세스 하나만 띄울 수 있어서, 수집 API·예측 API·시뮬레이터가 하던 일을
이 앱 안에서 한 번에 처리한다. 화면(dashboard/views.py)과 계산(model/scoring.py)은 원래 코드를 그대로 쓴다.

실행: streamlit run deploy/streamlit/streamlit_app.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.views import render_dashboard  # noqa: E402
from model.scoring import predict_failure, score_anomaly  # noqa: E402
from model.train import ARTIFACT_DIR, DEFAULT_CSV, train_all  # noqa: E402
from simulator.stream import build_readings, inject_overload  # noqa: E402

MACHINES = 5
FAULT_MACHINE = 3          # 데모용 과부하를 주입할 설비
ROWS_PER_TICK = MACHINES   # 한 번 갱신할 때 설비마다 1건씩
MAX_ROWS = 600             # 화면에 유지할 최근 기록 수
WARMUP_ROWS = 100          # 처음 열었을 때 미리 채워 둘 기록 수

st.set_page_config(page_title="factory-pdm 데모", layout="wide")


@st.cache_resource(show_spinner="데이터 다운로드와 모델 학습 중 (첫 실행만, 1분 내외)...")
def load_bundle() -> dict:
    """모델 파일이 없으면 데이터를 받아 학습한다. 서버가 살아 있는 동안 한 번만 실행된다."""
    bundle_path = ARTIFACT_DIR / "bundle.joblib"
    if not bundle_path.exists():
        if not DEFAULT_CSV.exists():
            from scripts.download_data import main as download
            if download() != 0:
                raise RuntimeError("UCI 데이터 다운로드 실패")
        train_all(DEFAULT_CSV, artifact_dir=ARTIFACT_DIR, report_dir=ARTIFACT_DIR, use_mlflow=False)
    return joblib.load(bundle_path)


@st.cache_resource
def stream_rows() -> list[dict]:
    """학습에 쓰지 않은 테스트셋 행만 스트림으로 쓴다 (적중률이 부풀려지지 않게)."""
    return list(build_readings(MACHINES, start=0, limit=None, test_only=True))


def next_rows(bundle: dict, count: int) -> list[dict]:
    """세션별 위치에서 다음 행들을 꺼내 점수를 매긴다. 끝까지 가면 처음부터 반복한다."""
    rows, source = [], stream_rows()
    for _ in range(count):
        position = st.session_state.position % len(source)
        st.session_state.position += 1
        reading = dict(source[position])
        if reading["machine_id"] == FAULT_MACHINE:
            reading = inject_overload(reading, st.session_state.rng)
        prediction = predict_failure(bundle, reading)
        anomaly = score_anomaly(bundle, reading)
        rows.append({
            **reading, "id": st.session_state.position,
            "received_at": pd.Timestamp.now(tz="UTC"),
            "failure_probability": prediction["failure_probability"],
            "predicted_failure": int(prediction["predicted_failure"]),
            "failure_type": prediction["failure_type"],
            "anomaly_score": anomaly["anomaly_score"],
            "is_anomaly": int(anomaly["is_anomaly"]),
        })
    return rows


def append_rows(rows: list[dict]) -> None:
    new = pd.DataFrame(rows)
    st.session_state.frame = pd.concat([st.session_state.frame, new], ignore_index=True).tail(MAX_ROWS)


st.title("설비 예지보전 모니터링 (데모)")
st.caption("데이터: UCI AI4I 2020 (원저자가 만든 **합성 데이터셋**)의 테스트셋 행 · "
           f"{FAULT_MACHINE}번 설비에는 데모용 과부하 패턴을 주입 · "
           "소스: [github.com/mingdevu/factory-pdm](https://github.com/mingdevu/factory-pdm)")

try:
    bundle = load_bundle()
except Exception as error:  # 데이터 다운로드·학습 실패를 화면에 그대로 알린다
    st.error(f"모델 준비 실패: {error}")
    st.stop()

if "frame" not in st.session_state:
    st.session_state.position = 0
    st.session_state.rng = random.Random(0)
    st.session_state.frame = pd.DataFrame()
    append_rows(next_rows(bundle, WARMUP_ROWS))

refresh_seconds = st.sidebar.slider("새로고침 주기(초)", 1, 10, 2)
anomaly_alert_level = st.sidebar.slider("이상 점수 알림 기준", 0.0, 1.0, 0.5, step=0.05)
paused = st.sidebar.toggle("스트림 일시정지", value=False)
st.sidebar.caption("이 데모는 접속한 사람마다 따로 스트림을 돌립니다. 새로고침하면 처음부터 다시 시작합니다.")


@st.fragment(run_every=refresh_seconds)
def live_view() -> None:
    if not paused:
        append_rows(next_rows(bundle, ROWS_PER_TICK))
    frame = st.session_state.frame.copy()
    frame["received_at"] = pd.to_datetime(frame["received_at"])
    render_dashboard(frame, anomaly_alert_level)


tab_live, tab_try = st.tabs(["실시간 모니터링", "직접 예측해 보기"])
with tab_live:
    live_view()

with tab_try:
    st.write("센서값을 바꿔 가며 `/predict`·`/anomaly`와 같은 계산 결과를 확인합니다.")
    left, right = st.columns(2)
    with left:
        product_type = st.selectbox("제품 등급", ["L", "M", "H"])
        air_temp = st.slider("주변온도 [K]", 295.0, 305.0, 300.0, 0.1)
        process_temp = st.slider("공정온도 [K]", 305.0, 315.0, 310.0, 0.1)
        rpm = st.slider("회전수 [rpm]", 1100, 2900, 1500, 10)
        torque = st.slider("토크 [Nm]", 3.0, 80.0, 40.0, 0.5)
        tool_wear = st.slider("공구 사용시간 [min]", 0, 250, 50, 1)
    reading = {"product_type": product_type, "air_temp_k": air_temp, "process_temp_k": process_temp,
               "rpm": rpm, "torque_nm": torque, "tool_wear_min": tool_wear}
    prediction = predict_failure(bundle, reading)
    anomaly = score_anomaly(bundle, reading)
    with right:
        st.metric("고장 확률", f"{prediction['failure_probability']:.1%}",
                  delta="고장 예측" if prediction["predicted_failure"] else "정상", delta_color="off")
        st.metric("예측 고장 유형", prediction["failure_type"])
        st.metric("이상 점수 (0~1)", f"{anomaly['anomaly_score']:.2f}")
        st.caption(f"온도차 {process_temp - air_temp:.1f} K · 전력 {torque * rpm * 2 * 3.14159 / 60:,.0f} W · "
                   f"마모×토크 {tool_wear * torque:,.0f} minNm")
        st.json(prediction["failure_type_probabilities"], expanded=False)
