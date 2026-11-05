---
title: factory-pdm
emoji: 🏭
colorFrom: pink
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: 설비 센서 이상탐지·고장 예측 실시간 대시보드 데모
---

# factory-pdm 데모

설비 센서 스트림(시뮬레이터)을 받아 **고장 확률·고장 유형·이상 점수**를 실시간으로 보여 주는 예지보전 대시보드입니다.

- 데이터: UCI AI4I 2020 (원저자가 만든 **합성 데이터셋**), 학습에 쓰지 않은 테스트셋 행을 1초 간격으로 반복 전송
- 3번 설비에는 데모용 과부하 패턴을 주입합니다 (라벨 없음)
- 공개 API (읽기 전용): `/api/serve/health`, `POST /api/serve/predict`, `POST /api/serve/anomaly`, `GET /api/ingest/machines/{id}/latest`
- 무료 CPU Space라 한동안 접속이 없으면 잠들고, 다시 깨어날 때 1~2분 걸릴 수 있습니다. 재시작하면 수집 데이터는 초기화됩니다.

소스 코드·실험 결과: https://github.com/mingdevu/factory-pdm
