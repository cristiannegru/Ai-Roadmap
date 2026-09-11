"""Content-contract tests for docs (quizzes, flashcards, site nav).

Mirrors scripts/validate_repo.py quiz checks inside pytest so CI enforces
learning content, not just code.
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

QUIZZES = [
    "docs/quizzes/phase1_python_math.md",
    "docs/quizzes/phase2_data.md",
    "docs/quizzes/phase3_ml.md",
    "docs/quizzes/phase4_dl.md",
    "docs/quizzes/phase5_genai.md",
    "docs/quizzes/phase6_projects.md",
    "docs/quizzes/phase7_mlops.md",
]


def test_all_quizzes_have_ten_answered_questions():
    assert len(QUIZZES) == 7
    for quiz in QUIZZES:
        text = (ROOT / quiz).read_text(encoding="utf-8")
        assert len(re.findall(r"^## Q\d+", text, flags=re.MULTILINE)) >= 10, quiz
        assert text.count("<details>") >= 10, quiz


def test_flashcards_cover_all_phases():
    cards = (ROOT / "docs" / "flashcards.md").read_text(encoding="utf-8")
    assert len(re.findall(r"^\| \d+ \|", cards, flags=re.MULTILINE)) >= 40
    for phase in ("Phase 1", "Phase 5", "Phase 7", "Career"):
        assert phase in cards


def test_mkdocs_nav_matches_files():
    nav_files: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            nav_files.append(node)

    walk(yaml.safe_load((ROOT / "mkdocs.yml").read_text())["nav"])
    assert len(nav_files) >= 20
    for target in nav_files:
        assert (ROOT / "docs" / target).exists(), target
