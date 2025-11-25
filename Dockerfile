FROM python:3.10-slim

WORKDIR /app

# 1. 의존성 설치
# (headless 버전을 쓰므로 시스템 라이브러리 설치 단계 삭제됨)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. 소스 코드 복사
COPY . .

# 3. 호환성 환경변수
ENV TF_USE_LEGACY_KERAS=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]