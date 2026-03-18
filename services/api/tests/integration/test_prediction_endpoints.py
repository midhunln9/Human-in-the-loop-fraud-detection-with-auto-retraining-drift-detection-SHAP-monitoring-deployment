import json
from unittest.mock import patch, MagicMock
import pytest


class TestPredictEndpoint:
    """Integration tests for the /predict endpoint."""

    def test_predict_returns_prediction_and_probability(self, client, sample_prediction_data):
        """Test that predict endpoint returns prediction and probability."""
        with patch("app.routes.prediction_endpoints.sns_publish") as mock_sns:
            response = client.post("/predict", json=sample_prediction_data)

            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert isinstance(data["prediction"], int)
            assert isinstance(data["probability"], float)

    def test_predict_returns_expected_values_from_mock_model(self, client, sample_prediction_data):
        """Test that predict endpoint returns values from the mock model."""
        with patch("app.routes.prediction_endpoints.sns_publish") as mock_sns:
            response = client.post("/predict", json=sample_prediction_data)

            data = response.json()
            assert data["prediction"] == 1
            assert data["probability"] == 0.85

    def test_predict_publishes_to_sns(self, client, sample_prediction_data):
        """Test that predict endpoint publishes to SNS."""
        with patch("app.routes.prediction_endpoints.sns_publish") as mock_sns:
            client.post("/predict", json=sample_prediction_data)

            mock_sns.publish.assert_called_once()

    def test_predict_publishes_correct_payload_to_sns(self, client, sample_prediction_data):
        """Test that predict endpoint publishes correct payload structure to SNS."""
        with patch("app.routes.prediction_endpoints.sns_publish") as mock_sns:
            client.post("/predict", json=sample_prediction_data)

            call_args = mock_sns.publish.call_args
            payload = call_args.kwargs["message"]

            assert "features" in payload
            assert "prediction" in payload
            assert len(payload["features"]) == 29
            assert payload["prediction"] == 1

    def test_predict_returns_500_when_model_not_loaded(self, client_no_model, sample_prediction_data):
        """Test that predict endpoint returns 500 when model is not loaded."""
        with patch("app.routes.prediction_endpoints.sns_publish") as mock_sns:
            response = client_no_model.post("/predict", json=sample_prediction_data)

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Model not loaded" in data["detail"]

    def test_predict_does_not_publish_when_model_not_loaded(self, client_no_model, sample_prediction_data):
        """Test that predict endpoint does not publish to SNS when model is not loaded."""
        with patch("app.routes.prediction_endpoints.sns_publish") as mock_sns:
            client_no_model.post("/predict", json=sample_prediction_data)

            mock_sns.publish.assert_not_called()

    def test_predict_returns_validation_error_for_missing_field(self, client):
        """Test that predict endpoint returns 422 for missing required field."""
        incomplete_data = {"V1": 1.0}  # Missing most fields

        with patch("app.routes.prediction_endpoints.sns_publish") as mock_sns:
            response = client.post("/predict", json=incomplete_data)

            assert response.status_code == 422

    def test_predict_returns_validation_error_for_invalid_type(self, client):
        """Test that predict endpoint returns 422 for invalid data type."""
        invalid_data = {f"V{i}": "invalid" for i in range(1, 29)}
        invalid_data["Amount"] = "not_a_number"

        with patch("app.routes.prediction_endpoints.sns_publish") as mock_sns:
            response = client.post("/predict", json=invalid_data)

            assert response.status_code == 422

    def test_predict_accepts_valid_float_values(self, client):
        """Test that predict endpoint accepts valid float values."""
        valid_data = {f"V{i}": float(i) for i in range(1, 29)}
        valid_data["Amount"] = 100.50

        with patch("app.routes.prediction_endpoints.sns_publish") as mock_sns:
            response = client.post("/predict", json=valid_data)

            assert response.status_code == 200

    def test_predict_accepts_integer_values(self, client):
        """Test that predict endpoint accepts integer values (coerced to float)."""
        valid_data = {f"V{i}": i for i in range(1, 29)}
        valid_data["Amount"] = 100

        with patch("app.routes.prediction_endpoints.sns_publish") as mock_sns:
            response = client.post("/predict", json=valid_data)

            assert response.status_code == 200

    def test_predict_returns_json_content_type(self, client, sample_prediction_data):
        """Test that predict endpoint returns JSON content type."""
        with patch("app.routes.prediction_endpoints.sns_publish") as mock_sns:
            response = client.post("/predict", json=sample_prediction_data)

            assert response.headers["content-type"] == "application/json"

    def test_predict_accepts_negative_values(self, client):
        """Test that predict endpoint accepts negative float values."""
        valid_data = {f"V{i}": -float(i) for i in range(1, 29)}
        valid_data["Amount"] = -50.25

        with patch("app.routes.prediction_endpoints.sns_publish") as mock_sns:
            response = client.post("/predict", json=valid_data)

            assert response.status_code == 200


class TestBatchPredictEndpoint:
    """Integration tests for the /batch-predict endpoint."""

    def test_batch_predict_returns_predictions_and_probabilities(self, client, sample_batch_data):
        """Test that batch predict endpoint returns predictions and probabilities."""
        response = client.post("/batch-predict", json=sample_batch_data)

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "probability" in data
        assert isinstance(data["predictions"], list)
        assert isinstance(data["probability"], list)

    def test_batch_predict_returns_correct_number_of_results(self, client, sample_batch_data):
        """Test that batch predict returns correct number of predictions."""
        response = client.post("/batch-predict", json=sample_batch_data)

        data = response.json()
        assert len(data["predictions"]) == 2
        assert len(data["probability"]) == 2

    def test_batch_predict_returns_expected_values(self, client, sample_batch_data):
        """Test that batch predict returns expected values from mock model."""
        response = client.post("/batch-predict", json=sample_batch_data)

        data = response.json()
        assert all(p == 1 for p in data["predictions"])
        assert all(p == 0.85 for p in data["probability"])

    def test_batch_predict_returns_500_when_model_not_loaded(self, client_no_model, sample_batch_data):
        """Test that batch predict endpoint returns 500 when model is not loaded."""
        response = client_no_model.post("/batch-predict", json=sample_batch_data)

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Model not loaded" in data["detail"]

    def test_batch_predict_single_item(self, client, sample_prediction_data):
        """Test that batch predict works with a single item."""
        data = {"data": [sample_prediction_data]}

        response = client.post("/batch-predict", json=data)

        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 1
        assert len(data["probability"]) == 1

    def test_batch_predict_empty_batch_raises_error(self, client):
        """Test that batch predict endpoint raises error for empty batch.

        The current implementation does not handle empty batches properly.
        This test documents the current behavior where empty batches cause an IndexError
        in the model.predict_proba() call.
        """
        data = {"data": []}

        # Empty batch causes IndexError in current implementation
        # because model.predict_proba returns empty 1D array which can't be indexed [:,1]
        with pytest.raises(IndexError):
            client.post("/batch-predict", json=data)

    def test_batch_predict_returns_validation_error_for_missing_data_field(self, client):
        """Test that batch predict endpoint returns 422 for missing data field."""
        response = client.post("/batch-predict", json={})

        assert response.status_code == 422

    def test_batch_predict_returns_validation_error_for_invalid_item(self, client, sample_prediction_data):
        """Test that batch predict endpoint returns 422 for invalid item in batch."""
        invalid_item = sample_prediction_data.copy()
        del invalid_item["V1"]

        data = {"data": [sample_prediction_data, invalid_item]}

        response = client.post("/batch-predict", json=data)

        assert response.status_code == 422

    def test_batch_predict_returns_json_content_type(self, client, sample_batch_data):
        """Test that batch predict endpoint returns JSON content type."""
        response = client.post("/batch-predict", json=sample_batch_data)

        assert response.headers["content-type"] == "application/json"

    def test_batch_predict_accepts_large_batch(self, client, sample_prediction_data):
        """Test that batch predict handles larger batch sizes."""
        data = {"data": [sample_prediction_data.copy() for _ in range(10)]}

        response = client.post("/batch-predict", json=data)

        assert response.status_code == 200
        result = response.json()
        assert len(result["predictions"]) == 10
        assert len(result["probability"]) == 10


class TestPredictionEndpointsErrorHandling:
    """Tests for error handling in prediction endpoints."""

    def test_predict_rejects_get_request(self, client):
        """Test that GET request to /predict is not allowed."""
        response = client.get("/predict")

        assert response.status_code == 405

    def test_predict_rejects_put_request(self, client):
        """Test that PUT request to /predict is not allowed."""
        response = client.put("/predict")

        assert response.status_code == 405

    def test_batch_predict_rejects_get_request(self, client):
        """Test that GET request to /batch-predict is not allowed."""
        response = client.get("/batch-predict")

        assert response.status_code == 405

    def test_predict_rejects_empty_body(self, client):
        """Test that empty body to /predict returns validation error."""
        response = client.post("/predict", json={})

        assert response.status_code == 422

    def test_batch_predict_rejects_null_data(self, client):
        """Test that null data field returns validation error."""
        response = client.post("/batch-predict", json={"data": None})

        assert response.status_code == 422
