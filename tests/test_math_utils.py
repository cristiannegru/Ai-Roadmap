"""Tests for math_utils (Phase 1 math)."""

import numpy as np
import pytest

from ai_roadmap.math_utils import (
    bayes_theorem,
    cosine_similarity,
    describe,
    dot_product,
    eigen_decomposition,
    euclidean_distance,
    linear_regression_closed_form,
    matrix_multiply,
    relu,
    sigmoid,
    softmax,
    solve_linear_system,
)


def test_dot_product():
    assert dot_product(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 11.0


def test_dot_product_shape_mismatch():
    with pytest.raises(ValueError):
        dot_product(np.array([1.0]), np.array([1.0, 2.0]))


def test_cosine_similarity_known_value():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    assert round(cosine_similarity(a, b), 4) == 0.9926


def test_cosine_similarity_zero_vector_safe():
    assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


def test_euclidean_distance():
    assert euclidean_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == 5.0


def test_matrix_multiply_and_shape_error():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_allclose(matrix_multiply(a, b), a @ b)
    with pytest.raises(ValueError):
        matrix_multiply(np.zeros((2, 3)), np.zeros((2, 2)))


def test_eigen_sorted_descending():
    vals, _ = eigen_decomposition(np.array([[2.0, 0.0], [0.0, 1.0]]))
    assert list(vals) == sorted(vals, reverse=True)
    assert np.allclose(sorted(vals), [1.0, 2.0])


def test_solve_linear_system():
    x = solve_linear_system(np.array([[2.0, 0.0], [0.0, 2.0]]), np.array([4.0, 6.0]))
    np.testing.assert_allclose(x, [2.0, 3.0])


def test_describe_stats():
    d = describe(np.array([1.0, 2.0, 3.0, 4.0]))
    assert (d["mean"], d["min"], d["max"], d["count"]) == (2.5, 1.0, 4.0, 4.0)
    with pytest.raises(ValueError):
        describe(np.array([]))


def test_bayes_and_validation():
    assert round(bayes_theorem(0.01, 0.9, 0.108), 4) == 0.0833
    with pytest.raises(ValueError):
        bayes_theorem(1.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        bayes_theorem(0.5, 0.5, 0.0)


def test_activations():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert relu(-3.0) == 0.0 and relu(2.5) == 2.5
    s = softmax(np.array([2.0, 1.0, 0.1]))
    assert s.sum() == pytest.approx(1.0)
    assert int(np.argmax(s)) == 0


def test_linear_regression_recovers_slope():
    slope, intercept = linear_regression_closed_form(
        np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])
    )
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(0.0, abs=1e-6)
