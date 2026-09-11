"""Tests for python_basics (Phase 1, stdlib only)."""

import pytest

from ai_roadmap.python_basics import (
    Experiment,
    chunk_list,
    count_words,
    flatten_nested,
    read_text_file,
    safe_divide,
    write_text_file,
)


def test_safe_divide_normal_and_zero():
    assert safe_divide(10, 2) == 5.0
    assert safe_divide(10, 0) == 0.0
    assert safe_divide(10, 0, default=-1.0) == -1.0


def test_chunk_list_even_uneven_empty():
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunk_list([1, 2, 3], 3) == [[1, 2, 3]]
    assert chunk_list([], 3) == []


def test_chunk_list_invalid_size():
    with pytest.raises(ValueError):
        chunk_list([1, 2], 0)


def test_flatten_nested_keeps_strings_atomic():
    assert flatten_nested([[1, 2], (3, 4), 5]) == [1, 2, 3, 4, 5]
    assert flatten_nested(["ab", ["cd"]]) == ["ab", "cd"]


def test_count_words_case_punctuation():
    assert count_words("Cat cat dog. Dog! cat?", top_n=2) == [("cat", 3), ("dog", 2)]


def test_count_words_invalid_top_n():
    with pytest.raises(ValueError):
        count_words("hello", top_n=0)


def test_file_roundtrip(tmp_path):
    p = write_text_file(tmp_path / "sub" / "note.txt", "hello")
    assert read_text_file(p) == "hello"


def test_read_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_text_file(tmp_path / "nope.txt")


def test_experiment_logs_and_serializes():
    exp = Experiment(name="demo", params={"lr": 0.01})
    exp.log_param("epochs", 5)
    exp.log_metric("accuracy", 0.9)
    d = exp.to_dict()
    assert d["name"] == "demo"
    assert d["params"] == {"lr": 0.01, "epochs": 5}
    assert d["metrics"] == {"accuracy": 0.9}
