# FastAPI ML template (production starter, Chunk 6)

Train → artefact → serving + Docker, all wired. Copy this folder to start
any tabular classifier service.

```bash
cp -r templates/fastapi_ml my-service && cd my-service
pip install -r requirements.txt
python train.py                 # → outputs/model.joblib + runs/train/run.json
uvicorn app:app --port 8000     # → /docs Swagger UI
```

Adapt: `CSV_PATH`/`TARGET` env vars for data, `Features` schema in `app.py`
for your columns, `EXCLUDE` in `train.py` for row-id columns.

## Files

- `settings.py` — env-driven paths/ports (12-factor).
- `train.py` — leak-free pipeline + `RunLogger` run + joblib bundle.
- `app.py` — lifespan artefact load (fail-fast), `/health`, `/info`, `/predict`.
- `Dockerfile` — bakes `train.py` into the image; non-root + HEALTHCHECK.
- `test_app.py` — template-local smoke test (see docstring).
