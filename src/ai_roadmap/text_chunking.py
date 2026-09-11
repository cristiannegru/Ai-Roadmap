"""Phase 5 — Text chunking for RAG (no heavy deps, stdlib + typing only).

Maps to ``docs/rag-roadmap.md`` (chunking strategies). The same
:func:`chunk_text` powers :mod:`ai_roadmap.rag_pipeline`,
``templates/streamlit_rag``, and notebook ``05_rag_minimal``.

Two units:
- ``words`` (default): splits on whitespace, best for prose docs.
- ``chars``: splits on raw characters, best for code/logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["Chunk", "chunk_text", "chunk_documents"]

Unit = Literal["words", "chars"]


@dataclass
class Chunk:
    """One retrievable unit with provenance for citations."""

    text: str
    chunk_index: int
    doc_id: str
    start: int  # token/char offset in the original doc (unit-dependent)
    end: int


def _split_units(text: str, unit: Unit) -> list[str]:
    if unit == "words":
        return text.split()
    if unit == "chars":
        return list(text)
    raise ValueError(f"unknown unit: {unit!r} (expected 'words' or 'chars')")


def _join_units(units: list[str], unit: Unit) -> str:
    return " ".join(units) if unit == "words" else "".join(units)


def chunk_text(
    text: str,
    *,
    chunk_size: int = 200,
    overlap: int = 50,
    unit: Unit = "words",
    doc_id: str = "doc",
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks.

    Args:
        text: source document text.
        chunk_size: max units per chunk (> 0).
        overlap: units shared with the previous chunk (0 <= overlap < chunk_size).
        unit: ``"words"`` or ``"chars"``.
        doc_id: provenance id copied onto every chunk.

    Returns:
        List of :class:`Chunk` (empty list for blank input).

    Examples:
        >>> chunks = chunk_text("a b c d e f", chunk_size=4, overlap=2)
        >>> [c.text for c in chunks]
        ['a b c d', 'c d e f']
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < chunk_size, got {overlap}")
    if not text.strip():
        return []
    units = _split_units(text, unit)
    step = chunk_size - overlap
    chunks: list[Chunk] = []
    for idx, start in enumerate(range(0, len(units), step)):
        window = units[start : start + chunk_size]
        if not window:
            break
        chunks.append(
            Chunk(
                text=_join_units(window, unit),
                chunk_index=idx,
                doc_id=doc_id,
                start=start,
                end=start + len(window),
            )
        )
        if start + chunk_size >= len(units):
            break
    return chunks


def chunk_documents(
    docs: list[dict[str, str]],
    *,
    chunk_size: int = 200,
    overlap: int = 50,
    unit: Unit = "words",
) -> list[Chunk]:
    """Chunk ``[{"id": ..., "text": ...}]`` docs, preserving each doc's id.

    Raises:
        ValueError: if a doc lacks ``"id"`` or ``"text"`` keys.
    """
    out: list[Chunk] = []
    for doc in docs:
        if "id" not in doc or "text" not in doc:
            raise ValueError(f"each doc needs 'id' + 'text' keys, got {sorted(doc)}")
        out.extend(
            chunk_text(
                doc["text"],
                chunk_size=chunk_size,
                overlap=overlap,
                unit=unit,
                doc_id=doc["id"],
            )
        )
    return out
