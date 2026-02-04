FROM python:3.10-slim

# 컨테이너 작업 디렉토리
WORKDIR /app

# 필요한 파일 디렉토리 안으로
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "시세_예측_대시보드.py", "--server.port=8501", "--server.address=0.0.0.0"]