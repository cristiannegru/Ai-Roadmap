"""Project 07 — Titanic survival API (FastAPI + sklearn, Docker-ready).

Category 5 (Startup/MLOps bridge) in docs/projects-roadmap.md.

Run locally:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000
    curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \\
      -d '{"pclass":1,"sex":"female","age":29,"sibsp":0,"fare":100,"embarked":"S"}'

Docker (build context = repo root):
    docker build -f projects/07_fastapi_titanic_api/Dockerfile -t titanic-api .
    docker run -p 8000:8000 titanic-api
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for candidate in [REPO, REPO / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
try:
    from ai_roadmap.data_wrangling import impute_missing
    from ai_roadmap.features import infer_feature_types, make_classification_pipeline
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO / "src"))
    from ai_roadmap.data_wrangling import impute_missing
    from ai_roadmap.features import infer_feature_types, make_classification_pipeline

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

CSV = REPO / "data" / "samples" / "titanic_sample.csv"
TARGET = "survived"


def _resolve_csv() -> Path:
    """Find the training CSV in container (/app/…) or checkout (repo/…) layouts."""
    candidates = [
        HERE / "data" / "samples" / "titanic_sample.csv",  # container: /app/…
        CSV,  # repo checkout: <root>/data/…
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"titanic_sample.csv not found (tried {candidates[0]} and {candidates[1]})"
    )


CSV = _resolve_csv()

app = FastAPI(title="Titanic Survival API", version="1.0.0")


class Passenger(BaseModel):
    """Validated passenger features (400s on bad input, not 500s)."""

    pclass: int = Field(ge=1, le=3, examples=[1])
    sex: Literal["male", "female"]
    age: float = Field(ge=0, le=100, examples=[29])
    sibsp: int = Field(ge=0, le=10, examples=[0])
    fare: float = Field(ge=0, le=1000, examples=[100])
    embarked: Literal["S", "C", "Q"] = "S"


@lru_cache(maxsize=1)
def get_model():
    """Train once (deterministic, ~1s) and reuse for every request."""
    df = impute_missing(pd.read_csv(CSV)).drop_duplicates().reset_index(drop=True)
    num, cat = infer_feature_types(df, target=TARGET, exclude=["passenger_id"])
    pipe = make_classification_pipeline(num, cat)
    pipe.fit(df.drop(columns=[TARGET]), df[TARGET])
    return pipe


def predict_survival(features: dict[str, object]) -> dict[str, object]:
    """Pure predict helper (used by the route and the test-suite)."""
    passenger = Passenger(**features)  # type: ignore[arg-type]
    row = pd.DataFrame([passenger.model_dump()])
    pipe = get_model()
    proba = float(pipe.predict_proba(row)[0][1])
    return {"survived": int(proba >= 0.5), "probability": round(proba, 4)}


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for Docker/Render health checks."""
    return {"status": "ok", "model": "titanic-logreg", "version": app.version}


@app.post("/predict")
def predict(payload: Passenger) -> dict[str, object]:
    """Survival prediction for one passenger."""
    return predict_survival(payload.model_dump())
