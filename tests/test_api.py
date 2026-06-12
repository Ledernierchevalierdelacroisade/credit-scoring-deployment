import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict():
    payload = {
        "AMT_CREDIT": 500000,
        "AMT_INCOME_TOTAL": 200000,
        "AMT_ANNUITY": 25000,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -1500
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert "probability" in data
    assert "prediction" in data
    assert "threshold" in data