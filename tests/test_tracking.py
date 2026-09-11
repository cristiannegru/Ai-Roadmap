"""Tests for tracking.py (offline run logger + MLflow bridge guard)."""

import json

import pytest

from ai_roadmap.tracking import RunLogger, summarize_runs


def test_context_manager_persists_run(tmp_path):
    with RunLogger(tmp_path, run_id="abc123") as run:
        run.log_param("model", "logreg")
        run.log_param("lr", 0.01)
        run.log_metric("f1", 0.81)
        run.set_tag("owner", "test")
    payload = json.loads((tmp_path / "abc123" / "run.json").read_text())
    assert payload["status"] == "finished"
    assert payload["params"] == {"model": "logreg", "lr": 0.01}
    assert payload["metrics"] == {"f1": 0.81}
    assert payload["tags"] == {"owner": "test"}
    assert payload["duration_s"] >= 0


def test_failure_status_on_exception(tmp_path):
    with pytest.raises(RuntimeError), RunLogger(tmp_path, run_id="bad") as run:
        run.log_metric("x", 1.0)
        raise RuntimeError("boom")
    payload = json.loads((tmp_path / "bad" / "run.json").read_text())
    assert payload["status"] == "failed"


def test_summarize_runs_sorted_and_skips_corrupt(tmp_path):
    RunLogger(tmp_path, run_id="b-run").finish()
    RunLogger(tmp_path, run_id="a-run").finish()
    (tmp_path / "half" / "run.json").parent.mkdir(parents=True)
    (tmp_path / "half" / "run.json").write_text("{not json", encoding="utf-8")
    summaries = summarize_runs(tmp_path)
    assert [s["run_id"] for s in summaries] == ["a-run", "b-run"]
    assert summarize_runs(tmp_path / "nope") == []


def test_to_mlflow_without_mlflow_errors_cleanly(tmp_path):
    mlflow_spec = __import__("importlib").util.find_spec("mlflow")
    if mlflow_spec is not None:
        pytest.skip("mlflow installed — skipping missing-dep guard test")
    run = RunLogger(tmp_path, run_id="x")
    with pytest.raises(ImportError, match="pip install mlflow"):
        run.to_mlflow()
