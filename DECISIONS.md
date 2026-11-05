# DECISIONS — factory-pdm

무인 작업 중 내린 판단과 이유를 한 줄씩 기록합니다.

- 데이터: UCI 공식 URL에서 AI4I 2020 다운로드에 성공해서 실제 공개 데이터셋을 썼다. 단, 원저자가 만든 합성 데이터셋이라 README에 '합성 데이터'라고 표시했다.
- 데이터 파일(`data/raw/`)은 git에서 제외하고 `scripts/download_data.py`로 다시 받게 했다. 500KB로 작지만 라이선스·재현 경로를 명확히 하려고.
- 분할은 train/val/test = 60/20/20 층화 분할로 했다. 임계값을 테스트셋으로 정하면 성능이 부풀려지므로 검증셋을 따로 뒀다.
- 불균형 처리는 SMOTE 대신 `scale_pos_weight`(클래스 가중치) + 검증셋 임계값 조정을 택했다. 추가 패키지(imbalanced-learn) 없이 충분하고, 합성 샘플이 물리적으로 불가능한 센서 조합을 만들 위험도 피할 수 있어서.
- 운영점을 두 개(F1 최대, 재현율 0.9 이상) 보고했다. 예지보전은 FN 비용이 커서 F1 하나로는 판단이 부족하기 때문이다. 서빙 기본값은 F1 최대 운영점으로 뒀다.
- 서빙 모델은 `lgbm_weighted`로 미리 정했다. 테스트셋 결과를 보고 고르면 선택 편향이 생기므로, 비교 결과는 보고만 한다(README 한계에 명시).
- 고장 유형 분류는 고장 행만으로 학습하는 별도 모델로 만들었다. RNF와 플래그 없는 고장은 센서로 설명할 수 없어서 OTHER로 묶었다.
- 이상탐지(Isolation Forest)는 정상 데이터로만 학습했다. 현장에 라벨이 없는 상황을 가정했다.
- 이상 점수 정규화: 처음에는 '정상 p99 → 0.5' 선형 변환을 썼는데, 실제로 돌려 보니 과부하를 주입해도 0.5~0.6에 머물러 SOP의 경고(0.7)·위험(0.9) 등급에 닿지 않았다. 그래서 정상 분위수 기준 구간 선형 변환(p50→0, p99→0.7, p99.9→0.9)으로 바꿨다.
- MLflow 추적 저장소는 `sqlite:///mlflow.db`를 썼다. MLflow 3.x에서 파일 저장소(`./mlruns`)는 권장 경로가 아니어서. `mlflow.db`, `mlruns/`는 git에서 제외했다.
- 시뮬레이터 기본값을 '테스트셋 행만 전송'으로 바꿨다. 처음에는 전체 행을 보냈는데, 학습에 쓴 행이 섞여 대시보드의 '예측 적중'이 부풀려지는 것을 발견해서. 전체를 보내려면 `--all-rows`를 쓴다.
- Windows에서 `localhost`가 IPv6(::1)를 먼저 시도해서 요청마다 약 2초 지연이 생겼다(수집 1건 약 4초). 그래서 모든 기본 주소를 `127.0.0.1`로 바꿨고, ingest가 serve를 부를 때 HTTP 연결을 재사용하게 했다. 그 결과 `/readings` p50이 115.7ms가 됐다.
- ingest는 serve가 꺼져 있어도 저장은 계속하게 했다(점수 컬럼만 비움). 수집 누락이 예측 누락보다 더 치명적이라서.
- 대시보드는 SQLite를 직접 읽지 않고 ingest API(`/readings`)를 거치게 했다. 컨테이너로 나눠 띄울 때 DB 파일 공유가 필요 없어서.
- Playwright는 스크린샷용 개발 도구라 `requirements.txt`에 넣지 않았다(README 스크립트 주석에 설치 방법 명시).
- 작업 중 서비스를 재시작하면서 `taskkill /IM python.exe`로 python 프로세스를 한꺼번에 종료한 적이 한 번 있다. 이후에는 포트별 PID만 종료하도록 바꿨다.
- Docker: 처음 확인 때 Docker Desktop은 설치돼 있었지만 데몬이 꺼져 있었다. '미설치' 조건이 아니므로 Docker Desktop을 직접 실행했고, `docker compose build` → `docker compose up -d`로 serve·ingest·simulator·dashboard 4개 컨테이너가 모두 running이고 /health·대시보드(200)가 응답하는 것까지 확인했다. README 스크린샷은 이 Docker 스택에서 찍었다.
- docker-compose의 4개 서비스가 같은 이미지(`factory-pdm:local`)를 공유하게 해서 빌드를 한 번만 하도록 했다.
- 배포(2026-11-06): Hugging Face Docker Space를 먼저 준비하고 로컬 Docker로 검증까지 했지만, 무료 계정에서 Docker Space 생성이 402(PRO 필요)로 막혔다. 사용자와 상의해 Streamlit Community Cloud(무료)로 배포하기로 했다. HF용 파일(`deploy/huggingface/`)은 PRO 전환 시 바로 쓸 수 있게 남겨 두었다.
- Streamlit Cloud는 프로세스를 하나만 띄울 수 있어서, 수집·예측·시뮬레이터를 앱 안에서 처리하는 단독 데모(`deploy/streamlit/streamlit_app.py`)를 따로 만들었다. 같은 계산을 두 번 쓰지 않도록 점수 계산은 `model/scoring.py`, 화면은 `dashboard/views.py`로 분리해 serve·대시보드·데모가 함께 쓴다.
- 클라우드용 의존성은 `deploy/streamlit/requirements.txt`에 최소한으로 두었다(MLflow·FastAPI·pytest 제외). 같은 조건의 리눅스 컨테이너(python:3.11-slim)에서 학습·스트림·직접 예측 탭이 동작하는 것을 확인했다.
