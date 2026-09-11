# 07 — Titanic Survival API (MLOps bridge)

FastAPI + sklearn with Pydantic validation, health probe, and Dockerfile.

```bash
cd projects/07_fastapi_titanic_api
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
# http://localhost:8000/docs  → interactive Swagger UI
```

```bash
# Docker (context = repo root):
docker build -f projects/07_fastapi_titanic_api/Dockerfile -t titanic-api .
docker run -p 8000:8000 titanic-api
```

## Endpoints

- `GET /health` → `{"status":"ok",...}` (Docker/Render probe)
- `POST /predict` → `{"survived":0/1,"probability":0-1}` (400 on bad input)

Chunk 6 turns this into a reusable template (`templates/fastapi_ml/`) with
`docker-compose.yml`, `render.yaml`, and MLflow tracking.
