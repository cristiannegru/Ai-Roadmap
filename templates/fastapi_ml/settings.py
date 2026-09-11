"""Environment-driven settings for the FastAPI ML template.

Adapt to your model by setting env vars (or a `.env` file) — no code edits
needed for paths/ports. Schema + TARGET changes live in train.py / app.py
(marked with TODO).
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(HERE / "outputs" / "model.joblib")))
CSV_PATH = Path(os.getenv("CSV_PATH", str(HERE / "sample.csv")))
TARGET = os.getenv("TARGET", "survived")  # matches bundled sample.csv
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
PORT = int(os.getenv("PORT", "8000"))
