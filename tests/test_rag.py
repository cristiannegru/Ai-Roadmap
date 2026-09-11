"""Tests for embeddings + vector store + RAG pipeline (offline, no keys)."""

from pathlib import Path

import numpy as np
import pytest

from ai_roadmap.embeddings import HashingEmbedder
from ai_roadmap.rag_pipeline import (
    answer_query,
    build_index,
    generate_answer,
    hit_at_k,
    load_docs_from_dir,
    retrieve,
)
from ai_roadmap.vector_store import SimpleVectorStore

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "data" / "samples" / "rag_docs"


def _docs() -> list[dict[str, str]]:
    return load_docs_from_dir(SAMPLE_DOCS)


def test_load_docs_contract():
    docs = _docs()
    assert {d["id"] for d in docs} == {"ml_basics", "deep_learning", "rag_systems", "ai_agents"}
    with pytest.raises(FileNotFoundError):
        load_docs_from_dir(Path("/tmp/no_such_docs_xyz"))
    empty = Path("/tmp/ai_roadmap_empty_docs")
    empty.mkdir(exist_ok=True)
    for p in empty.glob("*"):
        p.unlink()
    with pytest.raises(ValueError):
        load_docs_from_dir(empty)


def test_embedder_deterministic_normalized():
    emb = HashingEmbedder(dim=64)
    assert not emb.is_fit
    a = emb.embed_texts(["hybrid search reranker", "relu backpropagation"])
    b = emb.embed_texts(["hybrid search reranker", "relu backpropagation"])
    assert a.shape == (2, 64)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), [1.0, 1.0], atol=1e-5)
    assert emb.embed_texts(["   "])[0].sum() == 0.0  # blank → zero vector
    with pytest.raises(ValueError):
        HashingEmbedder(dim=64).fit([])


def test_embedder_fit_idf_and_persistence(tmp_path):
    emb = HashingEmbedder(dim=64).fit(["cat sat on mat", "dog ran in park", "cat dog"])
    assert emb.is_fit
    # 'cat' appears twice → lower idf than unique words; weights stay >= 1.
    assert float(emb._idf.min()) >= 1.0  # type: ignore[union-attr]
    vecs = emb.embed_texts(["cat sat", "xylophone quantum"])
    assert vecs.shape == (2, 64)
    path = emb.save(tmp_path / "emb.npz")
    reloaded = HashingEmbedder.load(path)
    assert (reloaded.dim, reloaded.seed, reloaded.is_fit) == (64, 42, True)
    np.testing.assert_allclose(reloaded.embed_texts(["cat sat"]), vecs[:1])
    with pytest.raises(FileNotFoundError):
        HashingEmbedder.load(tmp_path / "missing.npz")


def test_store_search_ranking_and_validation():
    emb = HashingEmbedder(dim=128)
    store = SimpleVectorStore(dim=128)
    vecs = emb.embed_texts(["hybrid search reranker vectors", "relu conv pooling"])
    store.add(["r", "c"], vecs, ["rag text", "cnn text"], [{"doc_id": "r"}, {"doc_id": "c"}])
    hits = store.search(emb.embed_query("what is hybrid search"), k=1)
    assert hits[0]["id"] == "r"
    assert len(store) == 2
    with pytest.raises(ValueError):
        store.add(["r"], vecs[:1], ["dupe"], [{}])  # duplicate id
    with pytest.raises(ValueError):
        store.search(np.zeros(4), k=1)  # wrong dim
    assert SimpleVectorStore(dim=8).search(np.zeros(8)) == []  # empty store


def test_store_save_load_roundtrip(tmp_path):
    emb = HashingEmbedder(dim=32)
    store = SimpleVectorStore(dim=32)
    store.add(["a"], emb.embed_texts(["hello world"]), ["hello world"], [{"doc_id": "a"}])
    base = store.save(tmp_path / "idx")
    assert base.with_suffix(".npz").exists() and base.with_suffix(".json").exists()
    reloaded = SimpleVectorStore.load(tmp_path / "idx")
    assert len(reloaded) == 1 and reloaded.ids == ["a"]
    with pytest.raises(FileNotFoundError):
        SimpleVectorStore.load(tmp_path / "missing")


def test_rag_end_to_end_topics_and_eval():
    docs = _docs()
    store, embedder, chunks = build_index(docs)
    assert len(chunks) >= 4 and len(store) == len(chunks)
    cases = {
        "How does hybrid search with reranking work?": "rag_systems",
        "What is the ReAct pattern with tools?": "ai_agents",
        "Why does ReLU help backpropagation?": "deep_learning",
        "How to prevent overfitting with cross validation?": "ml_basics",
    }
    scores = []
    for query, expected in cases.items():
        hits = retrieve(store, embedder, query, k=2)
        assert hits and hits[0]["score"] > 0
        assert hits[0]["id"].split(":::")[0] == expected  # top-1 exact topic
        scores.append(hit_at_k([h["id"] for h in hits], [expected], k=2))
    assert sum(scores) / len(scores) >= 0.75
    assert retrieve(store, embedder, "   ") == []  # blank query


def test_generate_answer_cites_and_handles_empty():
    result = answer_query(*build_index(_docs())[:2], "hybrid search reranker", k=2)
    assert result["citations"] and "rag_systems" in str(result["citations"][0])
    assert "Sources:" in str(result["answer"])
    empty_result = generate_answer("anything", [])
    assert empty_result["citations"] == [] and "couldn't find" in str(empty_result["answer"])
