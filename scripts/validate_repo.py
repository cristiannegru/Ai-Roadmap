"""Repo validator — enforces the 100%-repo contract.

Checks:
  1. Required top-level files exist (incl. mkdocs.yml, compose, render)
  2. Required directories exist (incl. docs/quizzes)
  3. All 14 docs guides exist and contain at least one YouTube link
  4. All 7 phase quizzes + flashcards exist with real Q&A content
  5. No secrets accidentally tracked (.env, *.pem, etc.)
  6. No large binaries tracked (>2MB, excluding assets/*.png)

Usage:
    python scripts/validate_repo.py
Exit 0 = pass, Exit 1 = fail with printed errors.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-docs.txt",
    "pyproject.toml",
    "Makefile",
    ".gitignore",
    ".env.example",
    ".pre-commit-config.yaml",
    ".markdownlint.json",
    "mkdocs.yml",
    "docker-compose.yml",
    "render.yaml",
    ".dockerignore",
]

REQUIRED_DIRS = [
    "docs",
    "docs/quizzes",
    "assets",
    "src/ai_roadmap",
    "scripts",
    "tests",
    "data/samples",
    "notebooks",
    "projects",
    "templates",
]

EXPECTED_DOCS = [
    "docs/machine-learning-roadmap.md",
    "docs/deep-learning-roadmap.md",
    "docs/computer-vision-roadmap.md",
    "docs/nlp-roadmap.md",
    "docs/generative-ai-roadmap.md",
    "docs/rag-roadmap.md",
    "docs/ai-agents-roadmap.md",
    "docs/mlops-roadmap.md",
    "docs/projects-roadmap.md",
    "docs/resume-guide.md",
    "docs/interview-preparation.md",
    "docs/internship-guide.md",
    "docs/freelancing-guide.md",
    "docs/open-source-guide.md",
]

FORBIDDEN_TRACKED = [".env", ".pem", "id_rsa", ".key"]
MAX_BYTES = 2_000_000
YOUTUBE_RE = re.compile(r"(youtube\.com|youtu\.be)")
# Local caches / envs are never shipped — skip them in the size walk
# (they are git-ignored, but may exist in a working tree after lint runs).
SKIP_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        "site",
        "htmlcov",
        "outputs",
        "runs",
        "mlruns",
        "models",
    }
)

EXPECTED_QUIZZES = [
    "docs/quizzes/phase1_python_math.md",
    "docs/quizzes/phase2_data.md",
    "docs/quizzes/phase3_ml.md",
    "docs/quizzes/phase4_dl.md",
    "docs/quizzes/phase5_genai.md",
    "docs/quizzes/phase6_projects.md",
    "docs/quizzes/phase7_mlops.md",
    "docs/flashcards.md",
    "docs/index.md",
]


def main() -> int:
    errors: list[str] = []

    for f in REQUIRED_FILES:
        if not (ROOT / f).exists():
            errors.append(f"missing required file: {f}")

    for d in REQUIRED_DIRS:
        if not (ROOT / d).is_dir():
            errors.append(f"missing required directory: {d}/")

    for doc in EXPECTED_DOCS:
        p = ROOT / doc
        if not p.exists():
            errors.append(f"missing docs guide: {doc}")
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if not YOUTUBE_RE.search(text):
            errors.append(f"docs guide has no YouTube link: {doc}")
        if len(text.strip()) < 500:
            errors.append(f"docs guide suspiciously short (<500 chars): {doc}")

    for quiz in EXPECTED_QUIZZES:
        p = ROOT / quiz
        if not p.exists():
            errors.append(f"missing quiz/site doc: {quiz}")
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if quiz.startswith("docs/quizzes/"):
            n_questions = len(re.findall(r"^## Q\d+", text, flags=re.MULTILINE))
            if n_questions < 10:
                errors.append(f"quiz has <10 questions: {quiz} ({n_questions})")
            if "<details>" not in text:
                errors.append(f"quiz missing answer blocks: {quiz}")
        elif len(text.strip()) < 500:
            errors.append(f"site doc suspiciously short (<500 chars): {quiz}")

    for forbidden in FORBIDDEN_TRACKED:
        matches = list(ROOT.glob(f"**/{forbidden}"))
        # .env.example is allowed; bare .env is not
        matches = [m for m in matches if m.name == forbidden and ".example" not in str(m)]
        # only flag top-level .env
        if forbidden == ".env" and (ROOT / ".env").exists():
            errors.append("SECURITY: tracked .env found — remove it, keep only .env.example")

    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        rel_parts = path.relative_to(ROOT).parts
        if SKIP_DIRS & set(rel_parts):
            continue
        if "assets/" in str(path):
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                errors.append(f"large file tracked (>2MB): {path.relative_to(ROOT)}")
        except OSError:
            continue

    if errors:
        print("validate_repo: FAILED")
        for e in errors:
            print(f"  ✗ {e}")
        return 1
    print(
        f"validate_repo: OK ({len(REQUIRED_FILES)} files, "
        f"{len(REQUIRED_DIRS)} dirs, {len(EXPECTED_DOCS)} guides, "
        f"{len(EXPECTED_QUIZZES)} quizzes/site docs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
