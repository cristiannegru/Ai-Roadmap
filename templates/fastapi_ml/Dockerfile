# Build context = this folder's parent-tree needs src/ + sample.csv.
#   docker build -f templates/fastapi_ml/Dockerfile -t ml-api .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY templates/fastapi_ml/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/samples/titanic_sample.csv ./sample.csv
COPY templates/fastapi_ml/settings.py templates/fastapi_ml/train.py templates/fastapi_ml/app.py ./

ENV CSV_PATH=/app/sample.csv \
    TARGET=survived \
    MODEL_PATH=/app/outputs/model.joblib \
    PYTHONPATH=/app:/app/src

# Bake the artefact into the image so the container serves on boot.
RUN python train.py

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
