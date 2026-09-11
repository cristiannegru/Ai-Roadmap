"""Phase 1 — Math utilities (NumPy): linear algebra, stats, probability.

Maps to ``docs/machine-learning-roadmap.md > Phase 1``:
vectors/matrices, eigenvalues, descriptive stats, Bayes' theorem,
softmax/sigmoid/ReLU and closed-form linear regression.

All functions are pure (no I/O), fully typed, and tested in
``tests/test_math_utils.py``. Run:

    pytest tests/test_math_utils.py -q
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "dot_product",
    "cosine_similarity",
    "euclidean_distance",
    "matrix_multiply",
    "eigen_decomposition",
    "solve_linear_system",
    "describe",
    "bayes_theorem",
    "sigmoid",
    "relu",
    "softmax",
    "linear_regression_closed_form",
]


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two 1-D arrays.

    Examples:
        >>> import numpy as np
        >>> dot_product(np.array([1., 2.]), np.array([3., 4.]))
        11.0
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError(f"expected two same-shaped 1-D arrays, got {a.shape} vs {b.shape}")
    return float(np.dot(a, b))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; 0.0 if either vector is all zeros.

    Same formula tested in ``docs/interview-preparation.md`` Q&A.

    Examples:
        >>> import numpy as np
        >>> round(cosine_similarity(np.array([1., 2., 3.]), np.array([2., 3., 4.])), 4)
        0.9926
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean (L2) distance between two same-shaped vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.linalg.norm(a - b))


def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix product ``a @ b`` with a readable shape-mismatch error."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"cannot multiply shapes {a.shape} and {b.shape}")
    return a @ b


def eigen_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(eigenvalues, eigenvectors)`` of a square matrix, sorted descending.

    Eigenvectors are columns of the returned matrix (NumPy convention).
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"expected square matrix, got shape {m.shape}")
    values, vectors = np.linalg.eig(m)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``Ax = b`` for ``x`` (used in closed-form regression derivation)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.linalg.solve(a, b)


def describe(values: np.ndarray) -> dict[str, float]:
    """Descriptive stats: count, mean, median, std (population), min, max.

    Examples:
        >>> import numpy as np
        >>> d = describe(np.array([1., 2., 3., 4.]))
        >>> (d["mean"], d["min"], d["max"], d["count"])
        (2.5, 1.0, 4.0, 4.0)
    """
    x = np.asarray(values, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("describe() received an empty array")
    return {
        "count": float(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def bayes_theorem(prior: float, likelihood: float, evidence: float) -> float:
    """Posterior ``P(H|E) = P(E|H) * P(H) / P(E)``.

    Examples:
        >>> round(bayes_theorem(0.01, 0.9, 0.108), 4)
        0.0833
    """
    if not (0 <= prior <= 1 and 0 <= likelihood <= 1):
        raise ValueError("prior and likelihood must be probabilities in [0, 1]")
    if evidence <= 0:
        raise ValueError("evidence P(E) must be > 0")
    return (likelihood * prior) / evidence


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Sigmoid ``1 / (1 + exp(-x))`` — numerically stable via clipping."""
    arr = np.asarray(x, dtype=float)
    clipped = np.clip(arr, -500, 500)
    out = 1.0 / (1.0 + np.exp(-clipped))
    return float(out) if np.ndim(x) == 0 else out


def relu(x: np.ndarray | float) -> np.ndarray | float:
    """ReLU ``max(0, x)`` elementwise."""
    arr = np.asarray(x, dtype=float)
    out = np.maximum(arr, 0.0)
    return float(out) if np.ndim(x) == 0 else out


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the last axis (subtracts max for numerical stability)."""
    z = np.asarray(logits, dtype=float)
    if z.size == 0:
        raise ValueError("softmax() received an empty array")
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def linear_regression_closed_form(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope + intercept via normal equation ``w = (X^T X)^{-1} X^T y``.

    Fits ``y ≈ slope * x + intercept`` for 1-D input. Teaching companion to
    the CampusX linear-regression video linked in Phase 3 docs.

    Examples:
        >>> import numpy as np
        >>> slope, intercept = linear_regression_closed_form(
        ...     np.array([1., 2., 3.]), np.array([2., 4., 6.]))
        >>> (round(slope, 6), round(intercept, 6))
        (2.0, 0.0)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape or x.size < 2:
        raise ValueError("x and y must have the same length >= 2")
    design = np.column_stack([x, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])
