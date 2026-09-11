"""Template-local smoke test (run inside this folder, not repo CI).

cd templates/fastapi_ml
pip install -r requirements.txt
PYTHONPATH=../../src:. pytest test_app.py -q
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))

import settings


def test_train_then_serve():
    import train

    assert train.main() == 0
    assert settings.MODEL_PATH.exists()
    import app
    from fastapi.testclient import TestClient

    with TestClient(app.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        info = client.get("/info").json()
        assert info["target"] == "survived" and "pclass" in str(info["numeric"])
        resp = client.post(
            "/predict",
            json={
                "pclass": 1,
                "sex": "female",
                "age": 29,
                "sibsp": 0,
                "fare": 100,
                "embarked": "S",
            },
        )
        assert resp.status_code == 200 and "probability" in resp.json()
        assert client.post("/predict", json={"pclass": 9}).status_code == 422
