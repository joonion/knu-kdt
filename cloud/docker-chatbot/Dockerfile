# Dockerfile

# Python 3.12 경량 이미지를 기본 이미지로 사용
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# 필요한 Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

COPY src/app.py src/

# 컨테이너가 사용할 포트 명시
EXPOSE 5000

 # 컨테이너 실행 시 실행할 명령
CMD ["python", "src/app.py"]