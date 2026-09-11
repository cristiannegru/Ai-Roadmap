# Quiz — Phase 2: Data Manipulation & Wrangling

Self-test for `docs/machine-learning-roadmap.md > Phase 2` and notebook `02` (§§3–6).
Code: `src/ai_roadmap/data_wrangling.py`. Dataset: `data/samples/titanic_sample.csv`.

## Q1 — The titanic sample has 61 rows but 60 passengers. Explain.
<details><summary>Answer</summary>

One deliberate duplicate row. `clean_dataframe` reports `duplicates_removed: 1` and returns
60 rows — duplicates silently inflate metrics if never removed.
</details>

## Q2 — Which columns have missing values, and what does the pipeline fill them with?
<details><summary>Answer</summary>

`age` ×6 (mean by default; median/mode/constant available) and `embarked` ×3 (mode).
`impute_missing` never mutates its input — it returns a new DataFrame.
</details>

## Q3 — When would you pick median over mean imputation?
<details><summary>Answer</summary>

Skewed distributions with outliers (e.g. `fare`): the median resists extremes while the mean
gets dragged. Our default is mean; pass `numeric_strategy="median"` to switch.
</details>

## Q4 — What does `encode_categoricals` do to `sex`/`embarked`, and what is `drop_first` for?
<details><summary>Answer</summary>

One-hot encodes them into float columns (`sex_male`, …). `drop_first=True` drops one level per
variable to avoid the dummy-variable trap (perfect multicollinearity in linear models).
</details>

## Q5 — Standard vs min-max scaling: formulas and when each?
<details><summary>Answer</summary>

Standard: `(x − mean)/std` (zero mean, unit variance — for regularized linear models, PCA).
Min-max: `(x − min)/(max − min)` → [0, 1] (for neural nets, images). Zero-variance columns map to 0.0.
</details>

## Q6 — State the 1.5×IQR outlier rule used by `detect_outliers_iqr`.
<details><summary>Answer</summary>

Outlier if `x < Q1 − 1.5×IQR` or `x > Q3 + 1.5×IQR` with `IQR = Q3 − Q1`. Non-parametric —
no normality assumption, unlike z-score thresholds.
</details>

## Q7 — What does `basic_info` return, and why is it the first call on new data?
<details><summary>Answer</summary>

Shape, column list, dtypes, per-column missing counts, duplicated-row count. It sizes the
cleaning job (how much is missing? any dupes?) before any transformation.
</details>

## Q8 — What is in the `clean_dataframe` report dict?
<details><summary>Answer</summary>

`rows_before/rows_after`, `duplicates_removed`, `missing_before/missing_after`, and the
applied `options`. Assert on it in tests (`missing_after` all zero) and print it in notebooks.
</details>

## Q9 — Why do plotting helpers return the `Figure` instead of calling `plt.show()`?
<details><summary>Answer</summary>

Headless safety: with the `Agg` backend, tests/CI save figures (`fig.savefig`) without ever
opening a window. Callers decide to show, save, or close.
</details>

## Q10 — You run `clean_dataframe(df, scale="standard", one_hot=True)`. How does the output shape change?
<details><summary>Answer</summary>

Rows: −1 (duplicate removed). Columns: categoricals replaced by several one-hot floats
(`sex`→1 col, `embarked`→2 cols with `drop_first`), numerics rescaled in place. (Exercise 1 in notebook 02.)
</details>
