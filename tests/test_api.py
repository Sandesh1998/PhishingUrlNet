"""
PhishNet - API Tests
Sandesh Poudel | B01818884 | UWS MSc Cybersecurity
"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status(self):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "models_loaded" in data

    def test_health_version(self):
        resp = client.get("/health")
        assert resp.json()["version"] == "1.0.0"


class TestPredictEndpoint:
    def test_missing_url_returns_422(self):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_predict_schema(self):
        # This will return 503 if no models loaded — that's expected in CI
        resp = client.post("/predict", json={"url": "http://google.com"})
        assert resp.status_code in (200, 503)

    def test_invalid_url_returns_422(self):
        resp = client.post("/predict", json={"url": ""})
        assert resp.status_code == 422


class TestBatchPredictEndpoint:
    def test_empty_list_returns_422(self):
        resp = client.post("/predict/batch", json={"urls": []})
        assert resp.status_code == 422

    def test_oversized_batch(self):
        urls = [f"http://example{i}.com" for i in range(501)]
        resp = client.post("/predict/batch", json={"urls": urls})
        # Expect 400 (too many) or 503 (no models) — not 200
        assert resp.status_code in (400, 503)
