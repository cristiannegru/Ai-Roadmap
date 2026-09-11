"""End-to-end tests for the 7 portfolio projects (all offline, hermetic tmp outs)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT / "projects"
RAG_DOCS = ROOT / "data" / "samples" / "rag_docs"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ── 01 spam ─────────────────────────────────────────────


def test_01_dataset_balanced_and_deterministic():
    m = _load("p01", PROJECTS / "01_spam_classifier" / "main.py")
    texts, labels = m.build_dataset(20)
    assert len(texts) == 40 and sum(labels) == 20
    t2, l2 = m.build_dataset(20)
    assert (texts, labels) == (t2, l2)


def test_01_train_predict_and_cli(tmp_path):
    m = _load("p01b", PROJECTS / "01_spam_classifier" / "main.py")
    texts, labels = m.build_dataset(60)
    pipe = m.train(texts, labels)
    assert int(pipe.predict(["FREE prize claim cash now"])[0]) == 1
    assert int(pipe.predict(["see you at lunch tomorrow?"])[0]) == 0
    out = tmp_path / "spam.joblib"
    assert m.main(["--n-per-class", "60", "--out", str(out)]) == 0
    assert out.exists()
    assert m.main(["--predict", "WINNER! claim your prize", "--out", str(out)]) == 0


# ── 02 housing ──────────────────────────────────────────


def test_02_train_and_predict_cli(tmp_path):
    m = _load("p02", PROJECTS / "02_housing_regression" / "main.py")
    pipe, report = m.train()
    assert report["rmse"] > 0 and -1.0 <= report["r2"] <= 1.0
    out = tmp_path / "housing.joblib"
    assert m.main(["--out", str(out)]) == 0
    assert m.main(["--predict", "--out", str(out), "--location", "suburb"]) == 0


# ── 03 resume ───────────────────────────────────────────


def test_03_parse_sample_and_match():
    m = _load("p03", PROJECTS / "03_resume_parser" / "main.py")
    text = (PROJECTS / "03_resume_parser" / "sample_resume.txt").read_text()
    parsed = m.parse_resume(text)
    assert parsed["emails"] == ["priya.sharma@example.com"]
    assert parsed["phones"] == ["+1-555-010-2030"]  # year-ranges excluded
    assert {"python", "pytorch", "docker"} <= set(parsed["skills"])
    assert parsed["years_experience_max"] >= 4
    verdict = m.match_job(parsed, ["python", "pytorch", "aws"], min_years=3)
    assert verdict["matched"] == ["python", "pytorch"] and verdict["missing"] == ["aws"]
    assert verdict["verdict"] == "SHORTLIST"
    junior = m.match_job(parsed, ["python"], min_years=10)
    assert junior["verdict"] == "REVIEW" and not junior["years_ok"]


# ── 04 vision ───────────────────────────────────────────


def test_04_detects_three_synthetic_objects(tmp_path):
    m = _load("p04", PROJECTS / "04_vision_detector" / "main.py")
    img = m.make_synthetic_image()
    boxes = m.detect_boxes(img)
    assert len(boxes) == 3
    # Rect 1 spans x[15,55) y[20,45); allow ±3px noise tolerance.
    x0, y0, x1, y1 = boxes[0]
    assert abs(x0 - 15) <= 3 and abs(y0 - 20) <= 3
    assert isinstance(m.draw_boxes(img, boxes), m.Image.Image)
    with pytest.raises(ValueError):
        m.detect_boxes(np.zeros((8, 8, 3), dtype=np.uint8))
    out = tmp_path / "det.png"
    assert m.main(["--out", str(out)]) == 0 and out.exists()


# ── 05 rag chatbot ──────────────────────────────────────


def test_05_ingest_and_ask(tmp_path):
    m = _load("p05", PROJECTS / "05_pdf_rag_chatbot" / "main.py")
    index = tmp_path / "rag"
    assert m.main(["ingest", "--docs", str(RAG_DOCS), "--index", str(index)]) == 0
    assert index.with_suffix(".npz").exists()
    assert m.main(["ask", "--query", "What is hybrid search?", "--index", str(index)]) == 0
    assert m.main(["ask", "--query", "x", "--index", str(tmp_path / "missing")]) == 2


# ── 06 crew ─────────────────────────────────────────────


def test_06_brief_contains_citations(tmp_path):
    m = _load("p06", PROJECTS / "06_crewai_research_team" / "main.py")
    out = tmp_path / "brief.md"
    assert m.main(["--query", "What is the ReAct pattern?", "--out", str(out)]) == 0
    brief = out.read_text()
    assert "Sources:" in brief and "ai_agents" in brief


# ── 07 fastapi ──────────────────────────────────────────


def test_07_predict_survival_and_validation():
    m = _load("p07", PROJECTS / "07_fastapi_titanic_api" / "app.py")
    good = {"pclass": 1, "sex": "female", "age": 29.0, "sibsp": 0, "fare": 100.0, "embarked": "S"}
    result = m.predict_survival(good)
    assert set(result) == {"survived", "probability"}
    assert result["survived"] in (0, 1) and 0.0 <= result["probability"] <= 1.0
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        m.predict_survival({**good, "pclass": 9})


def test_07_http_routes():
    m = _load("p07b", PROJECTS / "07_fastapi_titanic_api" / "app.py")
    from fastapi.testclient import TestClient

    client = TestClient(m.app)
    assert client.get("/health").json()["status"] == "ok"
    resp = client.post(
        "/predict",
        json={"pclass": 3, "sex": "male", "age": 30, "sibsp": 0, "fare": 10, "embarked": "S"},
    )
    assert resp.status_code == 200 and "probability" in resp.json()
    assert client.post("/predict", json={"pclass": 9}).status_code == 422


def test_07_csv_resolves_in_container_layout(tmp_path, monkeypatch):
    """Regression: /app container layout must resolve, not just repo checkout."""
    import shutil

    m = _load("p07c", PROJECTS / "07_fastapi_titanic_api" / "app.py")
    assert m.CSV.exists()  # checkout layout
    container = tmp_path / "app"  # container layout: /app/{app.py, data/…}
    target = container / "data" / "samples"
    target.mkdir(parents=True)
    shutil.copy(ROOT / "data" / "samples" / "titanic_sample.csv", target / "titanic_sample.csv")
    monkeypatch.setattr(m, "HERE", container)
    assert m._resolve_csv() == target / "titanic_sample.csv"
