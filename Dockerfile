# 파이썬 이미지
FROM python:3.10-slim

# 컨테이너 작업 디렉토리
WORKDIR /app

# 필요한 파일 디렉토리 안으로
COPY . /app

# 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

# 스트림릿 포트
EXPOSE 8501

# 실행
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]