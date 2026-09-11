"""Phase 5 — Offline fake embeddings (NumPy only, zero API keys).

Maps to ``docs/rag-roadmap.md`` (embeddings) and ``docs/interview-preparation.md``
(cosine similarity). :class:`HashingEmbedder` is a deterministic hashed
bag-of-words: same text → same vector, paraphrases with shared words score
high, unrelated texts score low. Good enough to teach retrieval and to run
the whole RAG stack in CI without network or keys.

Swap path to production: replace ``HashingEmbedder`` with OpenAI/Cohere/
``sentence-transformers`` embeddings — :class:`SimpleVectorStore` and
:mod:`ai_roadmap.rag_pipeline` only require an ``(n, dim)`` float matrix.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

__all__ = ["HashingEmbedder", "tokenize"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (``don't`` → ``["don", "t"]``)."""
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Deterministic hashed bag-of-words embedder with L2-normalized rows.

    Term weighting is TF-IDF over hash bins: call :meth:`fit` on the corpus
    once, then distinctive terms (``hybrid``, ``relu``) outweigh stopwords
    (``with``, ``the``). Without :meth:`fit` every bin has weight 1.0
    (pure hashed counts) — useful for tiny demos.

    Args:
        dim: embedding width (default 1024). Larger = fewer hash collisions.
        seed: salt mixed into every token hash (changes the whole space).

    Examples:
        >>> emb = HashingEmbedder(dim=64)
        >>> _ = emb.fit(["cat sat on mat", "dog ran in park"])
        >>> vecs = emb.embed_texts(["cat sat", "dog ran"])
        >>> vecs.shape
        (2, 64)
    """

    def __init__(self, dim: int = 1024, seed: int = 42) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        self.dim = dim
        self.seed = seed
        self._idf: np.ndarray | None = None

    def _token_index(self, token: str) -> int:
        digest = hashlib.md5(f"{self.seed}:{token}".encode()).hexdigest()
        return int(digest, 16) % self.dim

    def fit(self, corpus: list[str]) -> HashingEmbedder:
        """Compute smoothed IDF weights from a corpus. Returns self.

        ``idf[b] = log((1 + N) / (1 + df[b])) + 1`` where ``df[b]`` counts
        corpus texts containing bin ``b``. Always >= 1.0, so unfitted and
        fitted spaces differ only by weighting, never by sign.
        """
        n = len(corpus)
        if n == 0:
            raise ValueError("fit() received an empty corpus")
        df = np.zeros(self.dim, dtype=np.float64)
        for text in corpus:
            bins = {self._token_index(t) for t in tokenize(text)}
            for b in bins:
                df[b] += 1
        self._idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
        return self

    @property
    def is_fit(self) -> bool:
        """Whether IDF weights have been computed."""
        return self._idf is not None

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch → ``(len(texts), dim)`` float32, rows L2-normalized.

        Term counts are multiplied by the fitted IDF weights (or 1.0 when
        unfitted). Empty/whitespace texts map to the zero vector
        (cosine-safe: similarity 0).
        """
        idf = self._idf if self._idf is not None else np.ones(self.dim, dtype=np.float32)
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokenize(text):
                mat[i, self._token_index(token)] += 1.0
            mat[i] *= idf
            norm = float(np.linalg.norm(mat[i]))
            if norm > 0:
                mat[i] /= norm
        return mat

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one query → ``(dim,)`` float32."""
        return self.embed_texts([query])[0]

    def save(self, path: str | Path) -> Path:
        """Persist dim/seed/idf to ``<path>`` (.npz). Needed because the query
        side must use the exact IDF weights fitted at index time."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            dim=np.array(self.dim),
            seed=np.array(self.seed),
            idf=self._idf if self._idf is not None else np.zeros(0, dtype=np.float32),
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> HashingEmbedder:
        """Load an embedder saved with :meth:`save`."""
        p = Path(path)
        try:
            blob = np.load(p, allow_pickle=False)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"embedder not found at {p}") from e
        emb = cls(int(blob["dim"]), int(blob["seed"]))
        idf = np.asarray(blob["idf"])
        if idf.size:
            emb._idf = idf.astype(np.float32)
        return emb
