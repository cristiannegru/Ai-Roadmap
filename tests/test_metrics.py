"""Tests for metrics.py — from-scratch values + sklearn agreement."""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import r2_score as sk_r2

from ai_roadmap.metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    f1,
    mae,
    mse,
    precision,
    r2_score,
    recall,
    rmse,
    roc_auc,
)


def test_regression_known_values():
    yt = np.array([3.0, -0.5, 2.0, 7.0])
    yp = np.array([2.5, 0.0, 2.0, 8.0])
    assert mae(yt, yp) == pytest.approx(0.5)
    assert mse(yt, yp) == pytest.approx(0.375)
    assert rmse(yt, yp) == pytest.approx(np.sqrt(0.375))
    assert mae(yt, yp) == pytest.approx(mean_absolute_error(yt, yp))
    assert mse(yt, yp) == pytest.approx(mean_squared_error(yt, yp))
    assert r2_score(yt, yp) == pytest.approx(sk_r2(yt, yp))


def test_r2_constant_target_returns_zero():
    assert r2_score(np.array([5.0, 5.0, 5.0]), np.array([1.0, 2.0, 3.0])) == 0.0


def test_classification_matches_sklearn():
    yt = np.array([0, 1, 1, 0, 1, 1, 0, 0])
    yp = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    assert accuracy(yt, yp) == pytest.approx(accuracy_score(yt, yp))
    assert precision(yt, yp) == pytest.approx(precision_score(yt, yp))
    assert recall(yt, yp) == pytest.approx(recall_score(yt, yp))
    assert f1(yt, yp) == pytest.approx(f1_score(yt, yp))
    cm = confusion_matrix(yt, yp)
    assert cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"] == len(yt)


def test_zero_division_safe():
    yt = np.array([0, 0, 0])
    yp = np.array([0, 0, 0])
    assert precision(yt, np.array([0, 0, 1])) == pytest.approx(0.0)
    assert recall(yt, yp) == 0.0  # no actual positives
    assert f1(yt, yp) == 0.0


def test_roc_auc_agrees_with_sklearn():
    yt = np.array([0, 0, 1, 1])
    ys = np.array([0.1, 0.4, 0.35, 0.8])
    assert roc_auc(yt, ys) == pytest.approx(roc_auc_score(yt, ys))


def test_classification_report_keys():
    rep = classification_report(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
    assert {"accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"} <= set(rep)
    assert rep["tp"] + rep["tn"] + rep["fp"] + rep["fn"] == 4.0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        mae(np.array([1.0]), np.array([1.0, 2.0]))
