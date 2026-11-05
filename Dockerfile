# 하나의 이미지로 serve / ingest / dashboard를 모두 띄운다 (command만 다르게)
FROM python:3.11-slim
WORKDIR /app
# lightgbm 실행에 필요한 OpenMP 런타임
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# 이미지 빌드 시 데이터 다운로드 + 학습까지 해서 모델 파일을 포함한다
RUN python scripts/download_data.py && python -m model.train --no-mlflow
EXPOSE 8000 8001 8501
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
