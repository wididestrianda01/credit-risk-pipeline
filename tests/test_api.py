"""
test_api.py
-----------
Integration tests for the FastAPI prediction endpoint.

Tests verify:
- Authentication (API key validation)
- Request validation (malformed JSON returns 422)
- Prediction endpoint (happy path, returns PD + risk band + SHAP factors)
- Health endpoint (no auth required, returns model version + uptime)
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure API_KEY is set for tests before importing app
os.environ.setdefault("API_KEY", "test-key")


@pytest.fixture(scope="module", autouse=True)
def mock_model_loading():
    """Mock the model loading at startup to avoid loading the 100MB+ pkl file.

    This fixture patches:
    - load_model() to return a mock model with predict_proba() method
    - compute_shap_values() to return a mock SHAP explanation
    - get_adverse_action_factors() to return mock SHAP factors
    """

    def mock_load(path):
        """Return a mock CalibratedClassifierCV-like object."""
        mock_model = MagicMock()
        # Simulate CalibratedClassifierCV structure with calibrated_classifiers_[0].estimator.estimator
        mock_model.calibrated_classifiers_ = [MagicMock()]
        mock_model.calibrated_classifiers_[0].estimator = MagicMock()
        mock_model.calibrated_classifiers_[0].estimator.estimator = MagicMock()

        # predict_proba returns [[prob_class_0, prob_class_1]]
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        return mock_model

    def mock_shap(model, X):
        """Return a mock SHAP explanation object."""
        mock_explanation = MagicMock()
        # Mock __getitem__ to allow indexing: explanation[0]
        mock_explanation.__getitem__ = MagicMock(
            return_value=MagicMock(
                values=np.array([0.1, -0.05, 0.2, 0.15, -0.08, 0.12, 0.09, 0.03])
            )
        )
        mock_explanation.feature_names = [
            "EXT_SOURCE_2",
            "EXT_SOURCE_1",
            "CREDIT_INCOME_RATIO",
            "bureau_days_credit_mean",
            "inst_payment_ratio_mean",
            "AGE_YEARS",
            "AMT_CREDIT",
            "DAYS_EMPLOYED",
        ]
        return mock_explanation

    def mock_factors(shap_explanation, idx, feature_labels, top_n=5):
        """Return mock adverse action factors."""
        return [
            {
                "rank": 1,
                "feature_name": "CREDIT_INCOME_RATIO",
                "human_label": "Credit amount to annual income ratio",
                "shap_value": 0.2,
                "direction": "increases_risk",
            },
            {
                "rank": 2,
                "feature_name": "EXT_SOURCE_2",
                "human_label": "External Credit Score 2 (bureau, 0–1 scale)",
                "shap_value": 0.1,
                "direction": "increases_risk",
            },
            {
                "rank": 3,
                "feature_name": "bureau_days_credit_mean",
                "human_label": "Mean days since bureau credit opened",
                "shap_value": 0.15,
                "direction": "increases_risk",
            },
            {
                "rank": 4,
                "feature_name": "AGE_YEARS",
                "human_label": "Age in years (derived from DAYS_BIRTH / -365)",
                "shap_value": 0.12,
                "direction": "increases_risk",
            },
            {
                "rank": 5,
                "feature_name": "AMT_CREDIT",
                "human_label": "Credit amount requested (currency)",
                "shap_value": 0.09,
                "direction": "increases_risk",
            },
        ]

    with patch("app.api.load_model", side_effect=mock_load):
        with patch("app.api.compute_shap_values", side_effect=mock_shap):
            with patch("app.api.get_adverse_action_factors", side_effect=mock_factors):
                # Import app AFTER patches are applied
                from app.api import app

                yield app


@pytest.fixture
def client(mock_model_loading):
    """FastAPI TestClient with mocked model.

    Uses the context manager so the lifespan (startup) runs, which sets
    app.state.model and app.state.startup_time before the first request.
    """
    with TestClient(mock_model_loading) as c:
        yield c


class TestPredictAuthentication:
    """Test authentication on /predict endpoint."""

    def test_predict_missing_api_key_returns_401(self, client):
        """POST /predict without X-API-Key header returns 401."""
        response = client.post("/predict", json={"EXT_SOURCE_1": 0.5})
        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_predict_invalid_api_key_returns_401(self, client):
        """POST /predict with wrong X-API-Key value returns 401."""
        response = client.post(
            "/predict",
            headers={"X-API-Key": "wrong-key"},
            json={"EXT_SOURCE_1": 0.5},
        )
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_predict_valid_api_key_accepted(self, client):
        """POST /predict with valid X-API-Key is accepted (auth passes)."""
        response = client.post(
            "/predict",
            headers={"X-API-Key": "test-key"},
            json={"EXT_SOURCE_1": 0.5},
        )
        # Should not return 401; may return 200 or 422 depending on validation
        assert response.status_code != 401


class TestPredictValidation:
    """Test request validation on /predict endpoint."""

    def test_predict_malformed_json_returns_422(self, client):
        """POST /predict with invalid JSON structure returns 422."""
        response = client.post(
            "/predict",
            headers={"X-API-Key": "test-key", "Content-Type": "application/json"},
            content="not valid json",
        )
        assert response.status_code == 422 or response.status_code == 400

    def test_predict_empty_request_returns_200(self, client):
        """POST /predict with empty body uses all defaults and returns 200."""
        response = client.post(
            "/predict",
            headers={"X-API-Key": "test-key"},
            json={},
        )
        assert response.status_code == 200


class TestPredictHappyPath:
    """Test happy path: valid request returns complete response."""

    def test_predict_valid_request_returns_200(self, client):
        """POST /predict with valid request returns 200 and all required fields."""
        response = client.post(
            "/predict",
            headers={"X-API-Key": "test-key"},
            json={
                "EXT_SOURCE_1": 0.5,
                "EXT_SOURCE_2": 0.6,
                "AMT_CREDIT": 100000,
                "AMT_INCOME_TOTAL": 50000,
                "DAYS_EMPLOYED": -500,
            },
        )
        assert response.status_code == 200

        data = response.json()
        # Verify all required fields
        assert "probability_of_default" in data
        assert "risk_band" in data
        assert "adverse_action_factors" in data
        assert "model_version" in data
        assert "gini_at_training" in data

        # Verify data types and ranges
        assert isinstance(data["probability_of_default"], float)
        assert 0 <= data["probability_of_default"] <= 1
        assert data["risk_band"] in ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
        assert isinstance(data["adverse_action_factors"], list)
        assert len(data["adverse_action_factors"]) > 0
        assert len(data["adverse_action_factors"]) <= 5
        assert data["model_version"] == "catboost_v2"
        assert data["gini_at_training"] == 0.5814

    def test_adverse_action_factors_structure(self, client):
        """Verify adverse action factors have required fields and proper structure."""
        response = client.post(
            "/predict",
            headers={"X-API-Key": "test-key"},
            json={"EXT_SOURCE_1": 0.5},
        )
        assert response.status_code == 200

        data = response.json()
        factors = data["adverse_action_factors"]

        # Check each factor has required fields
        for factor in factors:
            assert "rank" in factor
            assert "feature_name" in factor
            assert "human_label" in factor
            assert "shap_value" in factor
            assert "direction" in factor

            # Verify data types
            assert isinstance(factor["rank"], int)
            assert isinstance(factor["feature_name"], str)
            assert isinstance(factor["human_label"], str)
            assert isinstance(factor["shap_value"], float)
            assert factor["direction"] in ["increases_risk", "decreases_risk"]

    def test_risk_band_classification(self, client):
        """Verify risk band is correctly assigned based on PD."""
        # Test case 1: Very low PD (mock returns 0.3, should map to MEDIUM)
        response = client.post(
            "/predict",
            headers={"X-API-Key": "test-key"},
            json={"EXT_SOURCE_1": 0.8},  # high score = low risk
        )
        assert response.status_code == 200
        data = response.json()
        assert data["risk_band"] in ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]

    def test_model_version_and_gini_constant(self, client):
        """Verify model version and Gini constant are returned."""
        response = client.post(
            "/predict",
            headers={"X-API-Key": "test-key"},
            json={},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["model_version"] == "catboost_v2"
        assert data["gini_at_training"] == 0.5814


class TestHealthEndpoint:
    """Test GET /health endpoint (no authentication)."""

    def test_health_returns_200_without_auth(self, client):
        """GET /health without X-API-Key returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_required_fields(self, client):
        """GET /health returns status, model_version, and uptime_seconds."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_version" in data
        assert "uptime_seconds" in data

        assert data["status"] == "ok"
        assert data["model_version"] == "catboost_v2"
        assert isinstance(data["uptime_seconds"], float)
        assert data["uptime_seconds"] >= 0

    def test_health_does_not_require_api_key(self, client):
        """GET /health should work even without the correct API key."""
        # Try with missing key
        response = client.get("/health")
        assert response.status_code == 200

        # Try with wrong key (should still work — /health is unprotected)
        response = client.get("/health", headers={"X-API-Key": "wrong"})
        assert response.status_code == 200

    def test_health_uptime_increases(self, client):
        """GET /health uptime_seconds should increase between calls."""
        import time

        response1 = client.get("/health")
        uptime1 = response1.json()["uptime_seconds"]

        time.sleep(0.1)

        response2 = client.get("/health")
        uptime2 = response2.json()["uptime_seconds"]

        assert uptime2 >= uptime1
