"""Tests for scripts/validate_repo.py (repo hygiene contract)."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "validate_repo", ROOT / "scripts" / "validate_repo.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_repo"] = module
    spec.loader.exec_module(module)
    return module


def test_cache_dirs_skipped_in_size_walk():
    mod = _load()
    for dirname in (
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "outputs",
        "runs",
    ):
        assert dirname in mod.SKIP_DIRS


def test_validator_passes_on_repo():
    assert _load().main() == 0
