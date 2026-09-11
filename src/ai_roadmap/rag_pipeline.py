"""Phase 5 — Minimal RAG pipeline: ingest → retrieve → generate → evaluate.

Demo-first design: :func:`generate_answer` is *extractive* (quotes the top
retrieved chunk with citations), so the whole pipeline runs offline with
zero API keys. To go production, replace ``generate_answer`` with an LLM
call (OpenAI/Anthropic/local Ollama) — retrieval code stays identical.

Pipeline:
    docs (id/text) → chunk → embed → SimpleVectorStore → top-k → answer
"""

from __future__ import annotations

from pathlib import Path

from ai_roadmap.embeddings import HashingEmbedder
from ai_roadmap.text_chunking import Chunk, chunk_documents
from ai_roadmap.vector_store import SimpleVectorStore

__all__ = [
    "build_index",
    "retrieve",
    "generate_answer",
    "answer_query",
    "hit_at_k",
    "load_docs_from_dir",
]


def load_docs_from_dir(docs_dir: str | Path) -> list[dict[str, str]]:
    """Load ``*.md`` + ``*.txt`` files → ``[{"id", "text"}]`` sorted by name."""
    root = Path(docs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"docs dir not found: {root}")
    docs: list[dict[str, str]] = []
    for path in sorted([*root.glob("*.md"), *root.glob("*.txt")]):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"id": path.stem, "text": text})
    if not docs:
        raise ValueError(f"no readable *.md/*.txt docs in {root}")
    return docs


def build_index(
    docs: list[dict[str, str]],
    *,
    chunk_size: int = 120,
    overlap: int = 20,
    dim: int = 1024,
    seed: int = 42,
) -> tuple[SimpleVectorStore, HashingEmbedder, list[Chunk]]:
    """Chunk + embed + index docs. Returns ``(store, embedder, chunks)``.

    The embedder is IDF-fit on the produced chunks, so queries embedded
    with the returned embedder live in the same weighted space. Persist it
    with :meth:`HashingEmbedder.save` whenever the store is saved.
    """
    chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("no chunks produced — docs are empty?")
    embedder = HashingEmbedder(dim=dim, seed=seed).fit([c.text for c in chunks])
    vectors = embedder.embed_texts([c.text for c in chunks])
    store = SimpleVectorStore(dim=dim)
    store.add(
        ids=[f"{c.doc_id}:::{c.chunk_index}" for c in chunks],
        vectors=vectors,
        texts=[c.text for c in chunks],
        metadatas=[{"doc_id": c.doc_id, "chunk": str(c.chunk_index)} for c in chunks],
    )
    return store, embedder, chunks


def retrieve(
    store: SimpleVectorStore,
    embedder: HashingEmbedder,
    query: str,
    k: int = 3,
) -> list[dict[str, object]]:
    """Embed ``query`` and return top-k store hits (empty query → [])."""
    if not query.strip():
        return []
    return store.search(embedder.embed_query(query), k=k)


def generate_answer(query: str, hits: list[dict[str, object]]) -> dict[str, object]:
    """Extractive demo answer: quote the top chunk + cite all hit ids.

    Returns ``{"answer", "citations", "backing_chunk"}``. No LLM call —
    deterministic and testable. Production swap: format ``hits`` into an
    LLM prompt and return its completion with the same citations.
    """
    if not hits:
        return {
            "answer": "I couldn't find anything relevant in the indexed docs.",
            "citations": [],
            "backing_chunk": None,
        }
    top = hits[0]
    snippet = str(top["text"])[:600]
    citations = [str(h["id"]) for h in hits]
    return {
        "answer": (
            f"Based on the indexed docs, the most relevant passage for "
            f"'{query}' is:\n\n> {snippet}\n\n"
            f"(Sources: {', '.join(citations)})"
        ),
        "citations": citations,
        "backing_chunk": top["id"],
    }


def answer_query(
    store: SimpleVectorStore,
    embedder: HashingEmbedder,
    query: str,
    k: int = 3,
) -> dict[str, object]:
    """One-call RAG: retrieve then generate. Merges hits + answer dicts."""
    hits = retrieve(store, embedder, query, k=k)
    result = generate_answer(query, hits)
    result["hits"] = hits
    result["query"] = query
    return result


def hit_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str] | set[str],
    k: int = 3,
) -> float:
    """1.0 if any of the top-k retrieved ids is relevant, else 0.0.

    Compares on ``doc_id`` prefix (chunk ids look like ``doc:::idx``).
    """
    relevant_docs = {r.split(":::")[0] for r in relevant_ids}
    top_docs = {r.split(":::")[0] for r in retrieved_ids[:k]}
    return 1.0 if relevant_docs & top_docs else 0.0
