# MLflow bridge (optional, Chunk 6)

Track offline by default, export to MLflow when you need the UI.

```bash
# 1. Offline (zero deps) — runs/train/run.json appears:
cd templates/fastapi_ml && python train.py

# 2. Export to a live server:
pip install mlflow && mlflow server --host 127.0.0.1 --port 5000 &
cd ../mlflow_tracking
python bridge.py --runs ../fastapi_ml/runs --experiment titanic-baseline
# → http://127.0.0.1:5000
```

`RunLogger` (`src/ai_roadmap/tracking.py`) mirrors the MLflow calls
(`log_param/log_metric/set_tag`) so the bridge is a mechanical replay.
