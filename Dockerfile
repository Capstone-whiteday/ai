FROM python:3.9-slim

WORKDIR /app

# 시스템 의존성 최소 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 파일 복사
COPY app.py .
COPY requirements.txt .
COPY solar_predict_model.keras .
COPY ppo_solar_model.zip .

# pip 업그레이드 및 패키지 설치
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
