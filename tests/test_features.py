"""Tests for features.py (Phase 3 pipelines)."""

from pathlib import Path

import pandas as pd
import pytest

from ai_roadmap.data_wrangling import impute_missing
from ai_roadmap.features import (
    build_preprocessor,
    infer_feature_types,
    make_classification_pipeline,
    make_regression_pipeline,
    train_test_split_df,
)

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def _titanic_clean() -> pd.DataFrame:
    df = pd.read_csv(SAMPLES / "titanic_sample.csv")
    return impute_missing(df)


def test_infer_feature_types():
    df = _titanic_clean()
    num, cat = infer_feature_types(df, target="survived")
    assert "survived" not in num + cat
    assert {"age", "fare"} <= set(num)
    assert {"sex", "embarked"} <= set(cat)
    with pytest.raises(ValueError):
        infer_feature_types(df, target="nope")


def test_preprocessor_handles_unknown_category():
    df = _titanic_clean()
    num, cat = infer_feature_types(df, target="survived")
    pre = build_preprocessor(num, cat)
    X = df.drop(columns=["survived"])
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(X)
    # New port of embarkation at predict time must not crash.
    X_new = X.iloc[:5].copy()
    X_new["embarked"] = "Z"
    assert pre.transform(X_new).shape[0] == 5


def test_classification_pipeline_fits_and_scores():
    df = _titanic_clean()
    num, cat = infer_feature_types(df, target="survived")
    X_train, X_test, y_train, y_test = train_test_split_df(df, "survived")
    pipe = make_classification_pipeline(num, cat)
    pipe.fit(X_train, y_train)
    assert 0.0 <= pipe.score(X_test, y_test) <= 1.0


def test_regression_pipeline_on_housing():
    df = pd.read_csv(SAMPLES / "housing_sample.csv")
    df = impute_missing(df)
    num, cat = infer_feature_types(df, target="price")
    pipe = make_regression_pipeline(num, cat)
    X_train, X_test, y_train, y_test = train_test_split_df(df, "price", stratify=False)
    pipe.fit(X_train, y_train)
    assert 0.0 <= pipe.score(X_test, y_test) <= 1.0


def test_split_validation_and_stratify_fallback():
    df = _titanic_clean()
    with pytest.raises(ValueError):
        train_test_split_df(df, "missing_col")
    with pytest.raises(ValueError):
        train_test_split_df(df, "survived", test_size=1.5)
    # Tiny frame where stratify is impossible → falls back cleanly.
    tiny = pd.DataFrame({"a": [1.0, 2.0, 3.0], "y": [0, 0, 1]})
    Xtr, Xte, ytr, yte = train_test_split_df(tiny, "y")
    assert len(Xtr) + len(Xte) == 3
