# Build context = repo root (needs src/ + data/ + this project).
#   docker build -f projects/07_fastapi_titanic_api/Dockerfile -t titanic-api .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependencies first (better layer caching).
COPY projects/07_fastapi_titanic_api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App code + shared library + sample data for startup training.
COPY src/ ./src/
COPY data/samples/titanic_sample.csv ./data/samples/titanic_sample.csv
COPY projects/07_fastapi_titanic_api/app.py ./app.py

# Non-root runtime user.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
