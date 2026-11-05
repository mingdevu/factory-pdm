"""실시간 설비 모니터링 대시보드.

실행: streamlit run dashboard/app.py
수집 API(/readings)를 주기적으로 읽어 센서 추이 · 이상 알림 · 예측 결과를 보여준다.
"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

INGEST_URL = os.getenv("INGEST_URL", "http://127.0.0.1:8001")
SENSOR_LABELS = {
    "air_temp_k": "주변온도 [K]",
    "process_temp_k": "공정온도 [K]",
    "rpm": "회전수 [rpm]",
    "torque_nm": "토크 [Nm]",
    "tool_wear_min": "공구마모 [min]",
}

st.set_page_config(page_title="factory-pdm 모니터링", layout="wide")
st.title("설비 예지보전 모니터링")
st.caption("데이터: UCI AI4I 2020 (원저자가 실제 설비를 본떠 만든 합성 데이터셋) · 시뮬레이터 스트리밍")

refresh_seconds = st.sidebar.slider("새로고침 주기(초)", 1, 60, int(os.getenv("DASHBOARD_REFRESH_SECONDS", "3")))
history_limit = st.sidebar.slider("조회 건수", 100, 2000, 500, step=100)
anomaly_alert_level = st.sidebar.slider("이상 점수 알림 기준", 0.0, 1.0, 0.5, step=0.05)


def fetch_readings(limit: int) -> pd.DataFrame | None:
    try:
        response = httpx.get(f"{INGEST_URL}/readings", params={"limit": limit}, timeout=3.0)
        response.raise_for_status()
    except httpx.HTTPError as error:
        st.error(f"수집 API({INGEST_URL}) 연결 실패: {error}")
        return None
    frame = pd.DataFrame(response.json())
    if frame.empty:
        return frame
    frame["received_at"] = pd.to_datetime(frame["received_at"])
    return frame.sort_values("id")


@st.fragment(run_every=refresh_seconds)
def render() -> None:
    frame = fetch_readings(history_limit)
    if frame is None:
        return
    if frame.empty:
        st.info("아직 수집된 데이터가 없습니다. 시뮬레이터를 실행하세요: python -m simulator.stream")
        return

    machine_ids = sorted(frame["machine_id"].unique())
    latest_per_machine = frame.groupby("machine_id").tail(1).set_index("machine_id")

    st.subheader("설비별 최신 상태")
    columns = st.columns(len(machine_ids))
    for column, machine_id in zip(columns, machine_ids):
        latest = latest_per_machine.loc[machine_id]
        anomaly_score = latest.get("anomaly_score")
        failure_probability = latest.get("failure_probability")
        is_alert = (pd.notna(anomaly_score) and anomaly_score >= anomaly_alert_level) or \
            bool(latest.get("predicted_failure") == 1)
        column.metric(
            label=f"{'🔴' if is_alert else '🟢'} {machine_id}번 설비",
            value="점수 없음" if pd.isna(anomaly_score) else f"이상 {anomaly_score:.2f}",
            delta=None if pd.isna(failure_probability) else f"고장확률 {failure_probability:.1%}",
            delta_color="off",
        )

    left, right = st.columns([2, 1])
    with left:
        selected_machine = st.selectbox("추이를 볼 설비", machine_ids, key="machine")
        sensor = st.radio("센서", list(SENSOR_LABELS), format_func=SENSOR_LABELS.get,
                          horizontal=True, key="sensor")
        machine_frame = frame[frame["machine_id"] == selected_machine]
        sensor_chart = px.line(machine_frame, x="received_at", y=sensor,
                               labels={sensor: SENSOR_LABELS[sensor], "received_at": "수신 시각"})
        sensor_chart.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(sensor_chart, use_container_width=True)
        if machine_frame["anomaly_score"].notna().any():
            score_chart = px.line(machine_frame, x="received_at",
                                  y=["anomaly_score", "failure_probability"],
                                  labels={"value": "점수", "received_at": "수신 시각", "variable": ""})
            score_chart.add_hline(y=anomaly_alert_level, line_dash="dot")
            score_chart.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(score_chart, use_container_width=True)

    with right:
        st.subheader("이상 알림")
        alerts = frame[(frame["anomaly_score"] >= anomaly_alert_level) |
                       (frame["predicted_failure"] == 1)]
        alerts = alerts.sort_values("id", ascending=False).head(20)
        if alerts.empty:
            st.success("알림 없음")
        else:
            st.dataframe(
                alerts[["received_at", "machine_id", "anomaly_score", "failure_probability",
                        "failure_type", "actual_failure"]],
                hide_index=True, use_container_width=True,
            )
        scored = frame.dropna(subset=["predicted_failure", "actual_failure"])
        if not scored.empty:
            st.subheader("스트림 예측 결과 (실제 라벨 대비)")
            st.caption("시뮬레이터 기본값은 학습에 쓰지 않은 테스트셋 행만 전송한다. 주입된 이상 데이터는 라벨이 없어 제외.")
            hit = int(((scored["predicted_failure"] == 1) & (scored["actual_failure"] == 1)).sum())
            st.write(f"수신 {len(scored)}건 · 실제 고장 {int(scored['actual_failure'].sum())}건 · "
                     f"예측 적중 {hit}건 · 고장 경보 {int(scored['predicted_failure'].sum())}건")


render()
