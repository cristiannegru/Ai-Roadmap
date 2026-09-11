# 01 — Spam Classifier (Beginner)

TF-IDF + Naive Bayes on synthetic SMS data. Fully offline.

```bash
cd projects/01_spam_classifier
pip install -r requirements.txt
python main.py                                   # train → F1 ~1.0 → saves outputs/
python main.py --predict "FREE prize, claim now"  # → SPAM (prob ~1.0)
```

## How it works

`build_dataset()` → 80/20 stratified split → `TfidfVectorizer(stop_words)` +
`MultinomialNB` → evaluated with our from-scratch `classification_report`
(`src/ai_roadmap/metrics.py`).

## Go production

Replace `build_dataset()` with the UCI SMS Spam CSV — the pipeline,
evaluation, and `--predict` CLI stay identical. Serve via the pattern in
`projects/07_fastapi_titanic_api/`.
