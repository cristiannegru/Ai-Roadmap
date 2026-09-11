"""Serving layer for the FastAPI ML template (artefact built by train.py).

Run:
    python train.py
    uvicorn app:app --port 8000
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for candidate in [REPO, REPO / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import joblib
import pandas as pd
import settings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_bundle: dict[str, Any] = {}


def load_bundle() -> dict[str, Any]:
    """Load (and cache) the train.py artefact. Fails fast with a fix hint."""
    if not _bundle:
        if not settings.MODEL_PATH.exists():
            raise RuntimeError(
                f"model artefact missing: {settings.MODEL_PATH}. Run `python train.py` first."
            )
        _bundle.update(joblib.load(settings.MODEL_PATH))
    return _bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fail fast at startup when the artefact is missing (not on 1st request)."""
    load_bundle()
    yield


app = FastAPI(title="ML API template", version=settings.APP_VERSION, lifespan=lifespan)


# TODO(you): replace with your feature schema. Keep bounds tight: bad input
# should be a 422, never a 500.
class Features(BaseModel):
    pclass: int = Field(ge=1, le=3)
    sex: Literal["male", "female"]
    age: float = Field(ge=0, le=100)
    sibsp: int = Field(ge=0, le=10)
    fare: float = Field(ge=0, le=1000)
    embarked: Literal["S", "C", "Q"] = "S"


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for Docker/Render."""
    return {"status": "ok", "version": app.version}


@app.get("/info")
def info() -> dict[str, Any]:
    """Model metadata: target, features, artefact path."""
    bundle = load_bundle()
    return {
        "target": bundle["target"],
        "numeric": bundle["numeric"],
        "categorical": bundle["categorical"],
        "artefact": str(settings.MODEL_PATH),
    }


@app.post("/predict")
def predict(payload: Features) -> dict[str, Any]:
    """Class prediction + probability for one row."""
    bundle = load_bundle()
    try:
        row = pd.DataFrame([payload.model_dump()])
        pipe = bundle["pipeline"]
        proba = float(pipe.predict_proba(row)[0][1])
    except Exception as e:  # noqa: BLE001 — surface feature mismatch as 400
        raise HTTPException(status_code=400, detail=f"prediction failed: {e}") from e
    return {"prediction": int(proba >= 0.5), "probability": round(proba, 4)}
