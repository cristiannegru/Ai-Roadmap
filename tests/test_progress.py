"""Tests for the progress tracker (Chunk 1 infrastructure)."""

from ai_roadmap.progress import MILESTONES, Progress, load, save


def test_milestones_count_and_ids():
    assert len(MILESTONES) == 7
    assert [m["id"] for m in MILESTONES] == [1, 2, 3, 4, 5, 6, 7]


def test_percent_math():
    assert Progress(done=[]).percent == 0.0
    assert Progress(done=[1]).percent == round(100 / 7, 1)
    assert Progress(done=[1, 2, 3, 4, 5, 6, 7]).percent == 100.0


def test_save_load_roundtrip(tmp_path):
    state = tmp_path / "progress.json"
    save(Progress(done=[1, 3]), state)
    assert load(state).done == [1, 3]


def test_load_missing_and_corrupt(tmp_path):
    assert load(tmp_path / "nope.json").done == []
    bad = tmp_path / "bad.json"
    bad.write_text("not-json{{{", encoding="utf-8")
    assert load(bad).done == []


def test_load_filters_invalid_ids(tmp_path):
    state = tmp_path / "p.json"
    state.write_text('{"done": [1, 99, -2]}', encoding="utf-8")
    assert load(state).done == [1]


def test_summary_marks_done():
    s = Progress(done=[1]).summary()
    assert "[x] Milestone 1" in s
    assert "[ ] Milestone 2" in s
