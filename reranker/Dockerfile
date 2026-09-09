FROM python:3.12-slim

WORKDIR /app

# CPU-only torch to keep the image small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir sentence-transformers==3.3.1 fastapi==0.115.6 "uvicorn[standard]==0.34.0"

COPY app.py .

ENV HF_HOME=/data
EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
