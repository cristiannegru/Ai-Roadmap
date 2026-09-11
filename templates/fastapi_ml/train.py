"""Train a tabular classifier and save the artefact (offline).

Default demo: titanic samples from the repo (override CSV_PATH/TARGET via env).

Usage:
    python train.py
    CSV_PATH=./my.csv TARGET=churned python train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for candidate in [REPO, REPO / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
try:
    from ai_roadmap.data_wrangling import impute_missing
    from ai_roadmap.features import infer_feature_types, make_classification_pipeline
    from ai_roadmap.metrics import classification_report
    from ai_roadmap.tracking import RunLogger
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO / "src"))
    from ai_roadmap.data_wrangling import impute_missing
    from ai_roadmap.features import infer_feature_types, make_classification_pipeline
    from ai_roadmap.metrics import classification_report
    from ai_roadmap.tracking import RunLogger

import joblib
import pandas as pd
import settings
from sklearn.model_selection import train_test_split

# TODO(you): row identifiers must never become features.
EXCLUDE = ["passenger_id"] if settings.TARGET == "survived" else []


def main() -> int:
    df = impute_missing(pd.read_csv(settings.CSV_PATH)).drop_duplicates().reset_index(drop=True)
    num, cat = infer_feature_types(df, target=settings.TARGET, exclude=EXCLUDE)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[settings.TARGET]),
        df[settings.TARGET],
        test_size=0.2,
        random_state=42,
        stratify=df[settings.TARGET],
    )
    pipe = make_classification_pipeline(num, cat)
    with RunLogger(HERE / "runs", run_id="train") as run:
        run.log_param("model", "logreg")
        run.log_param("target", settings.TARGET)
        run.log_param("numeric", num)
        run.log_param("categorical", cat)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        report = classification_report(y_test.to_numpy(), pred)
        for key in ("accuracy", "precision", "recall", "f1"):
            run.log_metric(key, report[key])
    print({k: round(report[k], 4) for k in ("accuracy", "precision", "recall", "f1")})
    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipe, "numeric": num, "categorical": cat, "target": settings.TARGET},
        settings.MODEL_PATH,
    )
    print(f"saved: {settings.MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
