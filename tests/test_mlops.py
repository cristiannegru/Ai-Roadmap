"""Tests for Chunk 6 MLOps artefacts (compose/render YAML + load-test math)."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_compose_defines_api_and_rag():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert {"api", "rag"} <= set(compose["services"])
    assert compose["services"]["api"]["ports"] == ["8000:8000"]
    assert "Dockerfile" in compose["services"]["rag"]["build"]["dockerfile"]


def test_render_blueprint_points_at_api_dockerfile():
    render = yaml.safe_load((ROOT / "render.yaml").read_text())
    svc = render["services"][0]
    assert svc["healthCheckPath"] == "/health"
    assert svc["dockerfilePath"].endswith("07_fastapi_titanic_api/Dockerfile")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_dockerfiles_expose_src_on_pythonpath():
    """Regression: ai_roadmap lives in <workdir>/src — images must add it.

    Without /app/src on PYTHONPATH, `RUN python train.py` fails at build and
    `uvicorn app:app` crashes at boot (caught by image-layout simulation).
    """
    dockerfiles = [
        ROOT / "projects" / "07_fastapi_titanic_api" / "Dockerfile",
        ROOT / "templates" / "fastapi_ml" / "Dockerfile",
        ROOT / "templates" / "streamlit_rag" / "Dockerfile",
    ]
    for dockerfile in dockerfiles:
        text = dockerfile.read_text(encoding="utf-8")
        assert "/app/src" in text, dockerfile


def test_load_test_rejects_bad_args():
    lt = _load("load_test", ROOT / "scripts" / "load_test.py")
    with pytest.raises(ValueError):
        lt.run_load_test("http://localhost:1", n=0)
    assert lt.main(["--url", "http://localhost:1", "--n", "1", "--timeout", "1"]) == 1


def test_mlflow_bridge_without_mlflow_exits_2(tmp_path):
    if importlib.util.find_spec("mlflow") is not None:
        pytest.skip("mlflow installed")
    bridge = _load("mlflow_bridge", ROOT / "templates" / "mlflow_tracking" / "bridge.py")
    assert bridge.main(["--runs", str(tmp_path)]) == 2
