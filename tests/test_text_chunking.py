"""Tests for text_chunking (Phase 5 RAG)."""

import pytest

from ai_roadmap.text_chunking import chunk_documents, chunk_text


def test_words_split_with_overlap():
    chunks = chunk_text("a b c d e f", chunk_size=4, overlap=2)
    assert [c.text for c in chunks] == ["a b c d", "c d e f"]
    assert chunks[0].start == 0 and chunks[1].start == 2


def test_no_overlap_and_short_text():
    chunks = chunk_text("a b c d e f g h", chunk_size=4, overlap=0)
    assert [c.text for c in chunks] == ["a b c d", "e f g h"]
    single = chunk_text("hello world", chunk_size=10, overlap=0)
    assert len(single) == 1 and single[0].text == "hello world"


def test_blank_returns_empty_and_invalid_args_raise():
    assert chunk_text("   ") == []
    with pytest.raises(ValueError):
        chunk_text("a b", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("a b", chunk_size=4, overlap=4)
    with pytest.raises(ValueError):
        chunk_text("a b", chunk_size=4, overlap=-1)


def test_chars_unit():
    chunks = chunk_text("abcdef", chunk_size=4, overlap=2, unit="chars")
    assert [c.text for c in chunks] == ["abcd", "cdef"]


def test_chunk_documents_preserves_ids_and_validates():
    docs = [{"id": "d1", "text": "a b c d e"}, {"id": "d2", "text": "x y"}]
    chunks = chunk_documents(docs, chunk_size=3, overlap=1)
    assert {c.doc_id for c in chunks} == {"d1", "d2"}
    with pytest.raises(ValueError):
        chunk_documents([{"id": "d1"}])
