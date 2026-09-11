"""Tests for data_wrangling (Phase 2) — uses the synthetic sample CSVs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ai_roadmap.data_wrangling import (
    basic_info,
    clean_dataframe,
    detect_outliers_iqr,
    encode_categoricals,
    impute_missing,
    load_csv,
    plot_correlation_heatmap,
    plot_numeric_hist,
    remove_duplicates,
    scale_numeric,
)

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
TITANIC = SAMPLES / "titanic_sample.csv"


def test_sample_csv_contract():
    """The teaching dataset must keep its deliberate mess (missing + 1 duplicate)."""
    df = load_csv(TITANIC)
    assert list(df.columns) == [
        "passenger_id",
        "pclass",
        "sex",
        "age",
        "sibsp",
        "fare",
        "embarked",
        "survived",
    ]
    assert len(df) == 61
    assert int(df["age"].isna().sum()) == 6
    assert int(df.duplicated().sum()) == 1


def test_load_csv_missing_and_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "nope.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_csv(empty)


def test_basic_info_reports_missing_and_dupes():
    df = load_csv(TITANIC)
    info = basic_info(df)
    assert info["shape"] == [61, 8]
    assert info["missing"]["age"] == 6
    assert info["duplicated_rows"] == 1


def test_impute_missing_never_mutates_and_fills_all():
    df = load_csv(TITANIC)
    out = impute_missing(df)
    assert int(out.isna().sum().sum()) == 0
    assert int(df.isna().sum().sum()) > 0  # input untouched


def test_remove_duplicates_reports_count():
    df = pd.DataFrame({"a": [1, 1, 2]})
    out, removed = remove_duplicates(df)
    assert (len(out), removed) == (2, 1)


def test_encode_and_scale():
    df = load_csv(TITANIC)
    clean = impute_missing(df)
    enc = encode_categoricals(clean, columns=["sex", "embarked"])
    assert "sex" not in enc.columns and any(c.startswith("sex_") for c in enc.columns)
    scaled = scale_numeric(clean, columns=["age", "fare"], method="standard")
    assert abs(scaled["age"].mean()) < 1e-9
    mm = scale_numeric(clean, columns=["fare"], method="minmax")
    assert mm["fare"].min() >= 0.0 and mm["fare"].max() <= 1.0


def test_detect_outliers_iqr_flags_extreme():
    df = pd.DataFrame({"fare": [10.0, 11.0, 12.0, 11.5, 500.0]})
    mask = detect_outliers_iqr(df, "fare")
    assert bool(mask.iloc[-1]) is True
    assert int(mask.sum()) == 1


def test_plots_return_figures_headless():
    df = load_csv(TITANIC)
    clean = impute_missing(df)
    fig1 = plot_numeric_hist(clean, "age")
    fig2 = plot_correlation_heatmap(clean[["pclass", "age", "sibsp", "fare", "survived"]])
    assert isinstance(fig1, plt.Figure) and isinstance(fig2, plt.Figure)
    plt.close(fig1)
    plt.close(fig2)


def test_clean_dataframe_milestone2_contract():
    """Milestone 2: one call removes the duplicate and all NaNs, with a report."""
    df = load_csv(TITANIC)
    clean, report = clean_dataframe(df)
    assert report["duplicates_removed"] == 1
    assert report["rows_before"] == 61 and report["rows_after"] == 60
    assert all(v == 0 for v in report["missing_after"].values())
    assert len(clean) == 60
