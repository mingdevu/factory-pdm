"""실시간 설비 모니터링 대시보드 (수집 API 연결 모드).

실행: streamlit run dashboard/app.py
수집 API(/readings)를 주기적으로 읽어 센서 추이 · 이상 알림 · 예측 결과를 보여준다.
API 없이 혼자 도는 데모는 deploy/streamlit/streamlit_app.py를 쓴다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

# `streamlit run dashboard/app.py`는 dashboard/만 import 경로에 넣으므로 프로젝트 루트를 추가한다
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.views import render_dashboard  # noqa: E402

INGEST_URL = os.getenv("INGEST_URL", "http://127.0.0.1:8001")

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
    render_dashboard(frame, anomaly_alert_level)


render()
