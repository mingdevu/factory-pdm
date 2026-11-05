# Hugging Face Spaces(Docker)용 이미지: 컨테이너 하나에서 serve·ingest·simulator·dashboard를 모두 띄우고
# nginx가 7860 포트 하나로 대시보드와 '읽기 전용' API를 내보낸다. (Spaces는 포트를 하나만 공개한다)
# 배포 방법: git archive로 레포를 풀고, 이 파일을 루트의 Dockerfile로, README.space.md를 README.md로 바꿔 업로드
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 nginx \
 && rm -rf /var/lib/apt/lists/*
# Spaces는 UID 1000 사용자로 실행하는 것을 권장한다
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
RUN python scripts/download_data.py && python -m model.train --no-mlflow
EXPOSE 7860
CMD ["bash", "deploy/huggingface/start.sh"]
