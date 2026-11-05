#!/usr/bin/env bash
# 컨테이너 안에서 서비스 4개 + nginx를 띄운다.
set -euo pipefail
cd /home/user/app
export DB_PATH=/tmp/sensor.db SERVE_URL=http://127.0.0.1:8000 INGEST_URL=http://127.0.0.1:8001
export DASHBOARD_REFRESH_SECONDS=5

uvicorn serve.app:app --host 127.0.0.1 --port 8000 &
uvicorn ingest.app:app --host 127.0.0.1 --port 8001 &
# 수집 API가 뜰 때까지 기다린 뒤 시뮬레이터를 시작한다
until python -c "import httpx; httpx.get('http://127.0.0.1:8001/health').raise_for_status()" 2>/dev/null; do sleep 1; done
python -m simulator.stream --interval 1 --inject-fault 3 --loop > /tmp/simulator.log 2>&1 &
streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501 \
  --server.headless true --browser.gatherUsageStats false &
# nginx를 포그라운드로 실행 (컨테이너의 메인 프로세스)
exec nginx -e stderr -c /home/user/app/deploy/huggingface/nginx.conf -g 'daemon off;'
