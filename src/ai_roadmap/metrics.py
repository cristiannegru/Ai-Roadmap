"""Phase 3 — Evaluation metrics from scratch (NumPy) + sklearn agreement.

Maps to ``docs/machine-learning-roadmap.md > Evaluation Metrics`` and
``docs/interview-preparation.md`` (precision/recall/F1, ROC-AUC).

Rule: regression + hard classification metrics are implemented with pure
NumPy so learners see the formulas. Tests assert agreement with sklearn
within 1e-9. ``roc_auc`` wraps sklearn (ranking integral) to avoid a
subtle reimplementation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "mae",
    "mse",
    "rmse",
    "r2_score",
    "accuracy",
    "confusion_matrix",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "classification_report",
]


def _as_float_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape or yt.size == 0:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    return yt, yp


# ── Regression ──────────────────────────────────────────


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    yt, yp = _as_float_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    yt, yp = _as_float_arrays(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 - SS_res/SS_tot. Returns 0.0 for a constant target (no variance)."""
    yt, yp = _as_float_arrays(y_true, y_pred)
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    ss_res = float(np.sum((yt - yp) ** 2))
    return 1.0 - ss_res / ss_tot


# ── Classification ──────────────────────────────────────


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, *, pos_label: int = 1
) -> dict[str, int]:
    """Return ``{"tn", "fp", "fn", "tp"}`` for binary labels."""
    yt, yp = _as_float_arrays(y_true, y_pred)
    if not set(np.unique(yt)) <= {0.0, 1.0} or not set(np.unique(yp)) <= {0.0, 1.0}:
        raise ValueError("confusion_matrix expects binary 0/1 labels")
    tp = int(np.sum((yt == pos_label) & (yp == pos_label)))
    tn = int(np.sum((yt != pos_label) & (yp != pos_label)))
    fp = int(np.sum((yt != pos_label) & (yp == pos_label)))
    fn = int(np.sum((yt == pos_label) & (yp != pos_label)))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction correct."""
    yt, yp = _as_float_arrays(y_true, y_pred)
    return float(np.mean(yt == yp))


def precision(
    y_true: np.ndarray, y_pred: np.ndarray, *, pos_label: int = 1, zero_division: float = 0.0
) -> float:
    """TP / (TP + FP); ``zero_division`` when no predicted positives."""
    cm = confusion_matrix(y_true, y_pred, pos_label=pos_label)
    denom = cm["tp"] + cm["fp"]
    return cm["tp"] / denom if denom else float(zero_division)


def recall(
    y_true: np.ndarray, y_pred: np.ndarray, *, pos_label: int = 1, zero_division: float = 0.0
) -> float:
    """TP / (TP + FN); ``zero_division`` when no actual positives."""
    cm = confusion_matrix(y_true, y_pred, pos_label=pos_label)
    denom = cm["tp"] + cm["fn"]
    return cm["tp"] / denom if denom else float(zero_division)


def f1(
    y_true: np.ndarray, y_pred: np.ndarray, *, pos_label: int = 1, zero_division: float = 0.0
) -> float:
    """Harmonic mean of precision and recall (0.0 when both are 0)."""
    p = precision(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    r = recall(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    return 2 * p * r / (p + r) if (p + r) else float(zero_division)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC from continuous scores (wraps sklearn for a correct ranking integral)."""
    from sklearn.metrics import roc_auc_score

    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    if yt.shape != ys.shape or yt.size == 0:
        raise ValueError("y_true and y_score must have the same non-zero length")
    return float(roc_auc_score(yt, ys))


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, *, pos_label: int = 1
) -> dict[str, float]:
    """Single dict with accuracy/precision/recall/f1 + raw tn/fp/fn/tp counts."""
    cm = confusion_matrix(y_true, y_pred, pos_label=pos_label)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred, pos_label=pos_label),
        "recall": recall(y_true, y_pred, pos_label=pos_label),
        "f1": f1(y_true, y_pred, pos_label=pos_label),
        "tn": float(cm["tn"]),
        "fp": float(cm["fp"]),
        "fn": float(cm["fn"]),
        "tp": float(cm["tp"]),
    }
