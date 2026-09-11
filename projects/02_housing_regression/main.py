"""Project 02 — Housing price regression (Ridge pipeline, sample CSV, offline).

Category 1 (Beginner) in docs/projects-roadmap.md.

Usage:
    python main.py                                         # train + evaluate + save
    python main.py --predict --area 1500 --bedrooms 3 --bathrooms 2 \\
        --age 10 --location suburb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for candidate in [REPO, REPO / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
try:
    from ai_roadmap.data_wrangling import impute_missing
    from ai_roadmap.features import (
        infer_feature_types,
        make_regression_pipeline,
        train_test_split_df,
    )
    from ai_roadmap.metrics import r2_score, rmse
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO / "src"))
    from ai_roadmap.data_wrangling import impute_missing
    from ai_roadmap.features import (
        infer_feature_types,
        make_regression_pipeline,
        train_test_split_df,
    )
    from ai_roadmap.metrics import r2_score, rmse

import joblib
import pandas as pd

CSV = REPO / "data" / "samples" / "housing_sample.csv"
ARTEFACT = HERE / "outputs" / "housing_ridge.joblib"


def train(csv_path: Path = CSV) -> tuple[object, dict[str, float]]:
    """Train Ridge on the housing sample. Returns (pipeline, {rmse, r2})."""
    df = impute_missing(pd.read_csv(csv_path)).drop_duplicates().reset_index(drop=True)
    num, cat = infer_feature_types(df, target="price")
    X_train, X_test, y_train, y_test = train_test_split_df(
        df, "price", test_size=0.25, random_state=42, stratify=False
    )
    pipe = make_regression_pipeline(num, cat)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return pipe, {
        "rmse": rmse(y_test.to_numpy(), pred),
        "r2": r2_score(y_test.to_numpy(), pred),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Housing price regression")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--area", type=float, default=1500)
    parser.add_argument("--bedrooms", type=int, default=3)
    parser.add_argument("--bathrooms", type=int, default=2)
    parser.add_argument("--age", type=int, default=10)
    parser.add_argument("--location", default="suburb")
    parser.add_argument("--out", type=Path, default=ARTEFACT)
    args = parser.parse_args(argv)

    if args.predict:
        if not args.out.exists():
            print(f"artefact missing: {args.out}. Run `python main.py` first.")
            return 2
        pipe = joblib.load(args.out)
        row = pd.DataFrame(
            [
                {
                    "area_sqft": args.area,
                    "bedrooms": args.bedrooms,
                    "bathrooms": args.bathrooms,
                    "age_years": args.age,
                    "location": args.location,
                }
            ]
        )
        print(f"predicted price: ${float(pipe.predict(row)[0]):,.0f}")
        return 0

    pipe, report = train()
    print(f"rmse=${report['rmse']:,.0f}  r2={report['r2']:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
