"""Phase 3 — Feature engineering pipelines (scikit-learn).

Maps to ``docs/machine-learning-roadmap.md > Phase 3`` (imputation,
encoding, scaling) and powers Milestone 3:

    "Deployed a classification model using Scikit-Learn."

All builders return *unfitted* sklearn objects — fit them in notebooks,
``projects/``, or tests. Every function is deterministic given
``random_state``.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__all__ = [
    "infer_feature_types",
    "build_preprocessor",
    "make_classification_pipeline",
    "make_regression_pipeline",
    "train_test_split_df",
]

ScaleMethod = Literal["standard", "passthrough"]


def infer_feature_types(
    df: pd.DataFrame, *, target: str, exclude: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """Split columns into ``(numeric, categorical)`` excluding target/excluded.

    Numeric = pandas number dtype. Everything else (object/category/bool)
    is treated as categorical.
    """
    if target not in df.columns:
        raise ValueError(f"target column not found: {target!r}")
    drop = {target} | set(exclude or [])
    numeric = [
        c for c in df.select_dtypes(include="number").columns if c not in drop and c in df.columns
    ]
    categorical = [c for c in df.columns if c not in drop and c not in numeric]
    if not numeric and not categorical:
        raise ValueError("no feature columns left after excluding target")
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scale_method: ScaleMethod = "standard",
) -> ColumnTransformer:
    """ColumnTransformer: median-impute + scale numerics, mode-impute + one-hot cats.

    Unknown categories at predict time are ignored (``handle_unknown="ignore"``),
    so production data with new labels never crashes.
    """
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_method == "standard":
        numeric_steps.append(("scaler", StandardScaler()))
    elif scale_method != "passthrough":
        raise ValueError(f"unknown scale_method: {scale_method}")

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_features:
        transformers.append(("num", Pipeline(steps=numeric_steps), numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipe, categorical_features))
    if not transformers:
        raise ValueError("need at least one numeric or categorical feature")
    return ColumnTransformer(transformers=transformers)


def make_classification_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scale_method: ScaleMethod = "standard",
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """Preprocess + LogisticRegression pipeline (Milestone-3 default).

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"age": [20., 30., 40., 50.],
        ...                    "sex": ["m", "f", "m", "f"],
        ...                    "y": [0, 1, 0, 1]})
        >>> pipe = make_classification_pipeline(["age"], ["sex"])
        >>> _ = pipe.fit(df[["age", "sex"]], df["y"])
        >>> float(pipe.score(df[["age", "sex"]], df["y"])) >= 0.0
        True
    """
    preprocessor = build_preprocessor(
        numeric_features, categorical_features, scale_method=scale_method
    )
    clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def make_regression_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scale_method: ScaleMethod = "standard",
    alpha: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Preprocess + Ridge regression pipeline (Chunk-3 regression starter)."""
    preprocessor = build_preprocessor(
        numeric_features, categorical_features, scale_method=scale_method
    )
    reg = Ridge(alpha=alpha, random_state=random_state)
    return Pipeline(steps=[("preprocess", preprocessor), ("reg", reg)])


def train_test_split_df(
    df: pd.DataFrame,
    target: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into ``(X_train, X_test, y_train, y_test)``.

    Stratifies on the target for classification when every class has
    >= 2 members; falls back to unstratified otherwise (tiny teaching CSVs).
    """
    from sklearn.model_selection import train_test_split

    if target not in df.columns:
        raise ValueError(f"target column not found: {target!r}")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    X = df.drop(columns=[target])
    y = df[target]
    strat = None
    if stratify and y.nunique() <= len(y) // 2 and int(y.value_counts().min()) >= 2:
        strat = y
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)
