"""Phase 2 — Data manipulation & visualization (Pandas + Matplotlib/Seaborn).

Maps to ``docs/machine-learning-roadmap.md > Phase 2``.
The one-call pipeline :func:`clean_dataframe` powers Milestone 2:

    "Build a data wrangling script cleaning a custom CSV with Pandas."

Non-plotting functions are pure pandas and fully tested. Plotting helpers
return the ``matplotlib.figure.Figure`` so tests can run headless
(``matplotlib.use("Agg")``) without opening windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")  # headless-safe: never pop up windows in scripts/CI

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

__all__ = [
    "load_csv",
    "basic_info",
    "impute_missing",
    "encode_categoricals",
    "scale_numeric",
    "remove_duplicates",
    "detect_outliers_iqr",
    "plot_numeric_hist",
    "plot_correlation_heatmap",
    "clean_dataframe",
]

ImputeStrategy = Literal["mean", "median", "mode", "constant"]
ScaleMethod = Literal["standard", "minmax"]


def load_csv(path: str | Path, **read_kwargs: object) -> pd.DataFrame:
    """Load a CSV, raising a clear error if missing or empty.

    Args:
        path: CSV file path.
        read_kwargs: forwarded to :func:`pandas.read_csv` (e.g. ``nrows=100``).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    df = pd.read_csv(p, **read_kwargs)  # type: ignore[arg-type]
    if df.empty:
        raise ValueError(f"CSV is empty: {p}")
    return df


def basic_info(df: pd.DataFrame) -> dict[str, object]:
    """Return shape, dtypes, and per-column missing counts (JSON-friendly)."""
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(df[c].isna().sum()) for c in df.columns},
        "duplicated_rows": int(df.duplicated().sum()),
    }


def impute_missing(
    df: pd.DataFrame,
    *,
    numeric_strategy: ImputeStrategy = "mean",
    categorical_strategy: ImputeStrategy = "mode",
    fill_value: object = 0,
) -> pd.DataFrame:
    """Fill NaNs: numerics via mean/median/mode/constant, categoricals via mode/constant.

    Returns a new DataFrame (input is never mutated).
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in out.columns if c not in numeric_cols]

    def _fill(series: pd.Series, strategy: ImputeStrategy) -> pd.Series:
        if series.isna().sum() == 0:
            return series
        if strategy == "mean":
            return series.fillna(series.mean(numeric_only=True))
        if strategy == "median":
            return series.fillna(series.median(numeric_only=True))
        if strategy == "mode":
            modes = series.mode(dropna=True)
            return series.fillna(modes.iloc[0] if len(modes) else fill_value)
        return series.fillna(fill_value)  # constant

    for col in numeric_cols:
        out[col] = _fill(out[col], numeric_strategy)
    for col in cat_cols:
        out[col] = _fill(out[col], categorical_strategy)
    return out


def encode_categoricals(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """One-hot encode object/category columns via ``pandas.get_dummies``."""
    cols = columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df.copy()
    return pd.get_dummies(df, columns=cols, drop_first=drop_first, dtype=float)


def scale_numeric(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    method: ScaleMethod = "standard",
) -> pd.DataFrame:
    """Scale numeric columns: ``standard`` (z-score) or ``minmax`` ([0, 1]).

    Zero-variance columns map to 0.0 instead of NaN/inf.
    """
    out = df.copy()
    cols = columns or out.select_dtypes(include="number").columns.tolist()
    for col in cols:
        s = pd.to_numeric(out[col], errors="coerce")
        if method == "standard":
            std = s.std()
            out[col] = 0.0 if std == 0 or pd.isna(std) else (s - s.mean()) / std
        elif method == "minmax":
            rng = s.max() - s.min()
            out[col] = 0.0 if rng == 0 or pd.isna(rng) else (s - s.min()) / rng
        else:
            raise ValueError(f"unknown scale method: {method}")
    return out


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicated rows. Returns ``(deduped_df, n_removed)``."""
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    return out, before - len(out)


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """Boolean mask of outliers via the 1.5×IQR rule (numeric column only)."""
    if column not in df.columns:
        raise ValueError(f"unknown column: {column}")
    s = pd.to_numeric(df[column], errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (s < lower) | (s > upper)


def plot_numeric_hist(df: pd.DataFrame, column: str) -> plt.Figure:
    """Histogram + KDE for one numeric column. Returns the Figure (headless-safe)."""
    if column not in df.columns:
        raise ValueError(f"unknown column: {column}")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(pd.to_numeric(df[column], errors="coerce").dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution — {column}")
    ax.set_xlabel(column)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Correlation heatmap over numeric columns. Returns the Figure."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("need at least 2 numeric columns for a correlation heatmap")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(numeric.corr(numeric_only=True), annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    return fig


def clean_dataframe(
    df: pd.DataFrame,
    *,
    numeric_strategy: ImputeStrategy = "mean",
    scale: ScaleMethod | None = None,
    one_hot: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """One-call Milestone-2 pipeline: dedupe → impute → (optional) scale/encode.

    Returns ``(cleaned_df, report)`` where report records rows removed,
    missing values before/after, and applied options — ideal for printing
    in notebooks and asserting in tests.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", "x"]})
        >>> clean, report = clean_dataframe(df)
        >>> (report["missing_before"], report["missing_after"])
        ({'a': 1, 'b': 0}, {'a': 0, 'b': 0})
    """
    missing_before = {c: int(df[c].isna().sum()) for c in df.columns}
    deduped, n_removed = remove_duplicates(df)
    imputed = impute_missing(
        deduped, numeric_strategy=numeric_strategy, categorical_strategy="mode"
    )
    final = imputed
    if one_hot:
        final = encode_categoricals(final)
    if scale is not None:
        final = scale_numeric(final, method=scale)
    report: dict[str, object] = {
        "rows_before": int(len(df)),
        "rows_after": int(len(final)),
        "duplicates_removed": int(n_removed),
        "missing_before": missing_before,
        "missing_after": {c: int(final[c].isna().sum()) for c in final.columns},
        "options": {"numeric_strategy": numeric_strategy, "scale": scale, "one_hot": one_hot},
    }
    return final.reset_index(drop=True), report
