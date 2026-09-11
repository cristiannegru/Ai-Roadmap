"""Project 03 — Resume parser & skill matcher (stdlib only: regex + keywords).

Category 2 (Intermediate) in docs/projects-roadmap.md.

Usage:
    python main.py --resume sample_resume.txt
    python main.py --resume my_resume.txt --require python sql --min-years 2

Production swap: replace keyword matching with spaCy NER + PDFMiner text
extraction (see the tutorial linked in docs/projects-roadmap.md) — the
parse → match → verdict shape stays the same.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

SKILLS = [
    "python",
    "sql",
    "pandas",
    "numpy",
    "scikit-learn",
    "pytorch",
    "tensorflow",
    "docker",
    "kubernetes",
    "aws",
    "gcp",
    "git",
    "fastapi",
    "langchain",
    "machine learning",
    "deep learning",
    "nlp",
    "computer vision",
    "mlops",
    "spark",
    "airflow",
    "linux",
    "javascript",
    "react",
]

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")
YEARS_RE = re.compile(r"(\d+)\s*\+?\s*years?", re.IGNORECASE)


def _is_phone(candidate: str) -> bool:
    """Reject year-ranges like 2020-2022: need ≥10 digits + real separators."""
    digits = sum(ch.isdigit() for ch in candidate)
    return digits >= 10 and any(sep in candidate for sep in ("+", " ", "-", "(", ")"))


def parse_resume(text: str) -> dict[str, object]:
    """Extract contact + skills + experience signals from resume text."""
    lowered = text.lower()
    emails = EMAIL_RE.findall(text)
    phones = [p.strip() for p in PHONE_RE.findall(text) if _is_phone(p)]
    skills = sorted({s for s in SKILLS if s in lowered})
    years = [int(m.group(1)) for m in YEARS_RE.finditer(text)]
    return {
        "emails": emails,
        "phones": phones,
        "skills": skills,
        "num_skills": len(skills),
        "years_experience_max": max(years) if years else 0,
        "num_lines": len(text.splitlines()),
    }


def match_job(
    parsed: dict[str, object], required: list[str], min_years: int = 0
) -> dict[str, object]:
    """Score a parsed resume against required skills + min experience."""
    have = set(str(s).lower() for s in parsed["skills"])  # type: ignore[union-attr]
    need = [r.lower() for r in required]
    matched = [r for r in need if r in have]
    missing = [r for r in need if r not in have]
    years_ok = int(parsed["years_experience_max"]) >= min_years  # type: ignore[arg-type]
    score = (len(matched) / len(need) * 100) if need else 100.0
    return {
        "score": round(score, 1),
        "matched": matched,
        "missing": missing,
        "years_ok": years_ok,
        "verdict": "SHORTLIST" if score >= 60 and years_ok else "REVIEW",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resume parser & skill matcher")
    parser.add_argument("--resume", type=Path, default=HERE / "sample_resume.txt")
    parser.add_argument("--require", nargs="*", default=["python", "sql", "docker"])
    parser.add_argument("--min-years", type=int, default=2)
    args = parser.parse_args(argv)

    if not args.resume.exists():
        print(f"resume not found: {args.resume}")
        return 2
    parsed = parse_resume(args.resume.read_text(encoding="utf-8"))
    result = match_job(parsed, args.require, args.min_years)
    print(json.dumps({"parsed": parsed, "match": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
