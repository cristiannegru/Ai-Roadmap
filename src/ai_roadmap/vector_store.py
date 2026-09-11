"""Phase 5 — In-memory vector store (NumPy only, Chroma-like minimal API).

``SimpleVectorStore`` is deliberately tiny: brute-force cosine search over
L2-normalized rows is exact and instant at teaching scale (<10k chunks).
Production swap: replace with Chroma/Qdrant/Pinecone — the ``add`` /
``search`` / ``save`` / ``load`` shapes mirror what Chunk 4's Streamlit
template needs, so call sites don't change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["SimpleVectorStore"]


class SimpleVectorStore:
    """Brute-force cosine store over normalized embedding rows."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        self.dim = dim
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.metadatas: list[dict[str, str]] = []
        self._vectors = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        texts: list[str],
        metadatas: list[dict[str, str]] | None = None,
    ) -> None:
        """Append rows. Raises on dim mismatch, length mismatch, or duplicate ids."""
        mat = np.asarray(vectors, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[1] != self.dim:
            raise ValueError(f"vectors must be (n, {self.dim}), got {mat.shape}")
        if not (len(ids) == len(texts) == mat.shape[0]):
            raise ValueError("ids/texts/vectors length mismatch")
        dupes = set(ids) & set(self.ids)
        if dupes:
            raise ValueError(f"duplicate ids: {sorted(dupes)[:5]}")
        self.ids.extend(ids)
        self.texts.extend(texts)
        self.metadatas.extend(list(metadatas) if metadatas is not None else [{} for _ in ids])
        self._vectors = np.vstack([self._vectors, mat]) if len(self) else mat.copy()

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[dict[str, object]]:
        """Top-k cosine hits → ``[{"id", "score", "text", "metadata"}]`` desc.

        Args:
            query_vector: ``(dim,)`` (need not be normalized; handled here).
            k: number of hits (clamped to store size).
        """
        if len(self) == 0:
            return []
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        q = np.asarray(query_vector, dtype=np.float32).ravel()
        if q.shape != (self.dim,):
            raise ValueError(f"query must be ({self.dim},), got {q.shape}")
        norm = float(np.linalg.norm(q))
        qn = q / norm if norm > 0 else q
        scores = self._vectors @ qn
        order = np.argsort(scores)[::-1][: min(k, len(self))]
        return [
            {
                "id": self.ids[i],
                "score": float(scores[i]),
                "text": self.texts[i],
                "metadata": self.metadatas[i],
            }
            for i in order
        ]

    def save(self, path: str | Path) -> Path:
        """Persist to ``<path>.npz`` (vectors) + ``<path>.json`` (ids/texts/meta)."""
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(base.with_suffix(".npz"), vectors=self._vectors, dim=self.dim)
        base.with_suffix(".json").write_text(
            json.dumps(
                {"ids": self.ids, "texts": self.texts, "metadatas": self.metadatas},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return base

    @classmethod
    def load(cls, path: str | Path) -> SimpleVectorStore:
        """Load a store saved with :meth:`save`."""
        base = Path(path)
        try:
            blob = np.load(base.with_suffix(".npz"))
            meta = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"vector store not found at {base}") from e
        store = cls(int(blob["dim"]))
        store.add(meta["ids"], np.asarray(blob["vectors"]), meta["texts"], meta["metadatas"])
        return store
