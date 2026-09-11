"""Project 01 — Spam classifier (TF-IDF + Naive Bayes, synthetic data, offline).

Category 1 (Beginner) in docs/projects-roadmap.md.

Usage:
    python main.py                        # train + evaluate + save artefact
    python main.py --predict "FREE prize, claim cash now"

Swap to production: replace build_dataset() with an SMS-spam CSV
(e.g. UCI SMS Spam Collection) — pipeline code is unchanged.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in [HERE.parent.parent, HERE.parent.parent / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
try:
    from ai_roadmap.metrics import classification_report
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(HERE.parent.parent / "src"))
    from ai_roadmap.metrics import classification_report

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ARTEFACT = HERE / "outputs" / "spam_classifier.joblib"

SPAM_TEMPLATES = [
    "FREE prize! Claim your ${amt} cash reward now, {name}!",
    "Congratulations {name}, you WON ${amt}! Click to collect your prize.",
    "URGENT: your account will be suspended. Verify now to win ${amt}.",
    "Cheap meds online, no prescription. Order now and save {pct}%!",
    "You inherited ${amt}! Send your bank details to claim, {name}.",
    "Limited offer: double your crypto in 24h. Act now, {name}!",
]

HAM_TEMPLATES = [
    "Hi {name}, are we still on for lunch tomorrow?",
    "The quarterly report is attached. Let me know your thoughts.",
    "Reminder: team standup moved to {time} in room {room}.",
    "Can you review my pull request when you get a chance?",
    "Happy birthday {name}! Hope you have a great day.",
    "The deploy finished in {n} minutes with no errors.",
]


def build_dataset(n_per_class: int = 120, seed: int = 42) -> tuple[list[str], list[int]]:
    """Generate a balanced synthetic SMS dataset. Returns (texts, labels)."""
    rng = random.Random(seed)
    names = ["alex", "priya", "sam", "mia", "leo", "ana"]
    texts, labels = [], []
    for _ in range(n_per_class):
        texts.append(
            rng.choice(SPAM_TEMPLATES).format(
                amt=rng.choice([100, 500, 1000, 5000]),
                name=rng.choice(names),
                pct=rng.choice([50, 70, 90]),
            )
        )
        labels.append(1)
        texts.append(
            rng.choice(HAM_TEMPLATES).format(
                name=rng.choice(names),
                time=rng.choice(["9am", "10:30", "2pm"]),
                room=rng.choice(["A1", "B2", "C3"]),
                n=rng.choice([3, 5, 12]),
            )
        )
        labels.append(0)
    order = list(range(len(texts)))
    rng.shuffle(order)
    return [texts[i] for i in order], [labels[i] for i in order]


def train(texts: list[str], labels: list[int]) -> Pipeline:
    """Fit a TF-IDF + MultinomialNB pipeline."""
    pipe = Pipeline(
        steps=[("tfidf", TfidfVectorizer(stop_words="english")), ("nb", MultinomialNB())]
    )
    pipe.fit(texts, labels)
    return pipe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Spam classifier (synthetic SMS data)")
    parser.add_argument("--predict", default=None, help="classify one message")
    parser.add_argument("--n-per-class", type=int, default=120)
    parser.add_argument("--out", type=Path, default=ARTEFACT)
    args = parser.parse_args(argv)

    if args.predict is not None:
        if not args.out.exists():
            print(f"artefact missing: {args.out}. Run `python main.py` first.")
            return 2
        pipe = joblib.load(args.out)
        pred = int(pipe.predict([args.predict])[0])
        proba = pipe.predict_proba([args.predict])[0]
        print(f"label: {'SPAM' if pred else 'HAM'} (spam prob {proba[1]:.3f})")
        return 0

    texts, labels = build_dataset(args.n_per_class)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipe = train(X_train, y_train)
    pred = pipe.predict(X_test)
    report = classification_report(np.array(y_test), np.array(pred))
    print(f"train: {len(X_train)}  test: {len(X_test)}")
    print(
        f"accuracy={report['accuracy']:.3f} precision={report['precision']:.3f} "
        f"recall={report['recall']:.3f} f1={report['f1']:.3f}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
