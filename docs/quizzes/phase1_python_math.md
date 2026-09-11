# Quiz — Phase 1: Python & Math Foundations

Self-test for `docs/machine-learning-roadmap.md > Phase 1` and notebooks `01`, `02` (§§1–2).
Hands-on answers live in `src/ai_roadmap/python_basics.py` and `src/ai_roadmap/math_utils.py`.

## Q1 — Why does `safe_divide` exist? What does `safe_divide(8, 0)` return?
<details><summary>Answer</summary>

Real data contains zeros (e.g. no predicted positives). Dividing would crash precision/recall.
`safe_divide(8, 0)` returns the default `0.0` instead of raising `ZeroDivisionError`.
</details>

## Q2 — `chunk_list([1, 2, 3, 4, 5], 2)` returns what? Where does this idea reappear?
<details><summary>Answer</summary>

`[[1, 2], [3, 4], [5]]`. The same batching idea powers RAG document chunking in Chunk 4
(`text_chunking.chunk_text`), just with overlap and provenance.
</details>

## Q3 — Why does `flatten_nested` keep strings atomic?
<details><summary>Answer</summary>

Strings are iterable, so naive flattening would explode `"ab"` into `["a", "b"]`.
The function checks `isinstance(item, (str, bytes))` first and appends them whole.
</details>

## Q4 — Cosine similarity of `[1, 2, 3]` and `[2, 3, 4]` is ~0.9926. What does a value near 1 mean, and what happens with a zero vector?
<details><summary>Answer</summary>

Near 1 = vectors point almost the same direction (high semantic overlap — the basis of
vector search). A zero vector has no direction, so our implementation safely returns `0.0`.
</details>

## Q5 — A disease has 1% prior, a test is 90% sensitive, 10.8% test positive. Posterior?
<details><summary>Answer</summary>

Bayes: `P(H|E) = 0.9 × 0.01 / 0.108 ≈ 0.083`. Even with a good test, a rare condition means
most positives are false — the base-rate lesson behind precision in imbalanced data.
</details>

## Q6 — What do `sigmoid`, `relu`, and `softmax` output, and where is each used?
<details><summary>Answer</summary>

- `sigmoid`: (0, 1) squashing — binary probabilities.
- `relu`: `max(0, x)` — hidden-layer activation that keeps gradients alive.
- `softmax`: positive vector summing to 1 — multi-class probabilities.
</details>

## Q7 — `eigen_decomposition` returns eigenvalues sorted how? Why sort?
<details><summary>Answer</summary>

Descending. Sorted eigenvalues rank principal directions by explained variance — exactly what
PCA uses to decide how many components to keep.
</details>

## Q8 — What does `linear_regression_closed_form` compute, and when does it beat gradient descent?
<details><summary>Answer</summary>

Slope + intercept via the normal equation (`lstsq`). For small, well-conditioned problems it is
exact and instant; gradient descent wins when data is too large for matrix inversion.
</details>

## Q9 — `describe()` returns which six stats, and why population (not sample) std?
<details><summary>Answer</summary>

`count, mean, median, std, min, max`. Population std (`ddof=0`) describes the data you have;
sample std corrects for estimating a larger population — EDA uses the former.
</details>

## Q10 — What does the `Experiment` class preview, and what are its two dicts?
<details><summary>Answer</summary>

MLflow-style tracking with stdlib only. `params` (inputs/hyperparameters) vs `metrics`
(outputs/scores) — the same split `RunLogger` and MLflow use in Chunk 6.
</details>
