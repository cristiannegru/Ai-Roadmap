"""Phase 1 — Python foundations (stdlib only, zero heavy deps).

Every function here maps to ``docs/machine-learning-roadmap.md > Phase 1``.
Designed to be read top-to-bottom by a beginner, then reused in later chunks.

Run doctests:
    python -m doctest src/ai_roadmap/python_basics.py -v
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

__all__ = [
    "safe_divide",
    "chunk_list",
    "flatten_nested",
    "count_words",
    "read_text_file",
    "write_text_file",
    "Experiment",
]

T = TypeVar("T")


def safe_divide(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    """Divide safely — return ``default`` instead of raising on zero division.

    Examples:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 0, default=-1.0)
        -1.0
    """
    if denominator == 0:
        return default
    return numerator / denominator


def chunk_list(items: list[T], chunk_size: int) -> list[list[T]]:
    """Split a list into consecutive chunks of at most ``chunk_size``.

    Used later for RAG document chunking (Chunk 4) — same idea, smaller scale.

    Examples:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
        >>> chunk_list([], 3)
        []
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def flatten_nested(nested: Iterable[Any]) -> list[Any]:
    """Flatten one level of nesting (lists/tuples/sets) into a single list.

    Strings are treated as atomic values, not iterables to expand.

    Examples:
        >>> flatten_nested([[1, 2], (3, 4), 5])
        [1, 2, 3, 4, 5]
        >>> flatten_nested(["ab", ["cd"]])
        ['ab', 'cd']
    """
    flat: list[Any] = []
    for item in nested:
        if isinstance(item, (list, tuple, set)) and not isinstance(item, (str, bytes)):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def count_words(text: str, *, top_n: int = 10) -> list[tuple[str, int]]:
    """Return the ``top_n`` most common words (lowercased, punctuation-stripped).

    A tiny preview of the NLP tokenization covered in Phase 5 / Chunk 4.

    Examples:
        >>> count_words("Cat cat dog. Dog! cat?", top_n=2)
        [('cat', 3), ('dog', 2)]
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be > 0, got {top_n}")
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    words = [w for w in cleaned.split() if w]
    return Counter(words).most_common(top_n)


def read_text_file(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Read a text file, raising a clear error if it does not exist.

    Examples:
        >>> import tempfile, os
        >>> with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        ...     _ = f.write("hello")
        ...     name = f.name
        >>> read_text_file(name)
        'hello'
        >>> os.unlink(name)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {p}")
    return p.read_text(encoding=encoding)


def write_text_file(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Write ``content`` to ``path``, creating parent dirs as needed. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding=encoding)
    return p


@dataclass
class Experiment:
    """Minimal OOP example — tracks one ML experiment's config + result.

    Mirrors what MLflow does in Chunk 6, but with pure stdlib so Phase 1
    learners meet classes before frameworks.

    Examples:
        >>> exp = Experiment(name="baseline", params={"lr": 0.01})
        >>> exp.log_metric("accuracy", 0.9)
        >>> exp.to_dict()["metrics"]
        {'accuracy': 0.9}
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def log_param(self, key: str, value: Any) -> None:
        """Record a hyperparameter."""
        self.params[key] = value

    def log_metric(self, key: str, value: float) -> None:
        """Record a numeric metric."""
        self.metrics[key] = float(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot."""
        return asdict(self)
