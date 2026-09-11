# Quiz — Phase 3: Classical Machine Learning

Self-test for `docs/machine-learning-roadmap.md > Phase 3` and notebook `03`.
Code: `src/ai_roadmap/features.py`, `src/ai_roadmap/metrics.py`.

## Q1 — Why must imputation/scaling/one-hot live *inside* the sklearn Pipeline?
<details><summary>Answer</summary>

Otherwise statistics (mean, scaler, categories) leak from the test set into training when fit
on the full frame. Inside the pipeline they fit on train folds only — notebook 03's core lesson.
</details>

## Q2 — What does `handle_unknown="ignore"` protect against?
<details><summary>Answer</summary>

New categorical levels at predict time (e.g. a new embarkation port). Without it, `OneHotEncoder`
raises; with it, unseen levels encode as all-zeros and prediction proceeds.
</details>

## Q3 — `train_test_split_df` stratifies "when possible". When does it fall back, and why?
<details><summary>Answer</summary>

When a class has fewer than 2 members (stratified splitting is impossible). Tiny teaching frames
hit this; the fallback keeps the split working instead of crashing.
</details>

## Q4 — Define precision, recall, F1 — and which one matters when missing a case is costly?
<details><summary>Answer</summary>

Precision = TP/(TP+FP) (of alarms, how many real). Recall = TP/(TP+FN) (of real cases, how many
caught). F1 = harmonic mean. Missing cases (disease, fraud) → optimize **recall**.
</details>

## Q5 — Your model predicts all negatives on 95%-negative data. Accuracy? What is wrong?
<details><summary>Answer</summary>

Accuracy = 95% — and useless. Accuracy lies under class imbalance; report precision/recall/F1
and the confusion matrix instead. Our `zero_division` guard returns 0.0 rather than crashing.
</details>

## Q6 — What does ROC-AUC measure, and what inputs does it need?
<details><summary>Answer</summary>

Probability that a random positive scores above a random negative (ranking quality across all
thresholds). Needs continuous *scores*, not hard labels — hence `roc_auc(y_true, y_score)`.
</details>

## Q7 — R² of 0.996 (housing project) means what? What does R² = 0 mean?
<details><summary>Answer</summary>

99.6% of price variance explained by the model. R² = 0 means "as good as predicting the mean";
negative means worse. Constant targets return 0.0 (no variance to explain).
</details>

## Q8 — LogReg vs RandomForest on titanic: how was the comparison kept fair?
<details><summary>Answer</summary>

Same features, same split, same preprocessor — only the estimator changed. Comparing models on
different preprocessing or splits measures plumbing, not models.
</details>

## Q9 — Why save the whole pipeline with joblib, not just the classifier?
<details><summary>Answer</summary>

The pipeline *is* the model: imputer medians, scaler stats, one-hot categories plus weights.
Saving only the classifier orphans it — deployment would need to rebuild preprocessing exactly.
</details>

## Q10 — Ridge (`alpha=1.0`) vs plain linear regression: what does the penalty buy?
<details><summary>Answer</summary>

L2 shrinkage stabilizes coefficients under multicollinearity (e.g. correlated `area`/`bedrooms`)
and curbs overfitting. Larger `alpha` = stronger shrinkage toward zero.
</details>
