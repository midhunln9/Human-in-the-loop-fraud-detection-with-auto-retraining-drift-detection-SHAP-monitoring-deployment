import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import numpy as np
from sklearn.base import BaseEstimator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app
from app.schemas.real_time_prediction import RealTimePredictionRequest
from app.schemas.batch_prediction import BatchPredictionRequest


class MockModel(BaseEstimator):
    """Mock ML model for testing - inherits from BaseEstimator for health check."""

    def __init__(self, prediction_value=1, probability_value=0.85):
        self.prediction_value = prediction_value
        self.probability_value = probability_value

    def predict(self, X):
        n_samples = len(X) if hasattr(X, "__len__") else 1
        return np.array([self.prediction_value] * n_samples)

    def predict_proba(self, X):
        n_samples = len(X) if hasattr(X, "__len__") else 1
        prob_class_0 = 1 - self.probability_value
        prob_class_1 = self.probability_value
        return np.array([[prob_class_0, prob_class_1]] * n_samples)


class MockPreprocessor(BaseEstimator):
    """Mock preprocessor for testing."""

    def transform(self, X):
        if hasattr(X, "values"):
            return X.values
        return np.array(X) if not isinstance(X, np.ndarray) else X


@pytest.fixture
def mock_model():
    """Fixture providing a mock ML model."""
    return MockModel(prediction_value=1, probability_value=0.85)


@pytest.fixture
def mock_preprocessor():
    """Fixture providing a mock preprocessor."""
    return MockPreprocessor()


@pytest.fixture
def sample_prediction_data():
    """Fixture providing a sample valid prediction request."""
    return {
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061083,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376641382,
        "V28": -0.0210530534538215,
        "Amount": 149.62,
    }


@pytest.fixture
def sample_batch_data(sample_prediction_data):
    """Fixture providing a sample valid batch prediction request."""
    data1 = sample_prediction_data.copy()
    data2 = sample_prediction_data.copy()
    data2["V1"] = 1.1918571114
    data2["Amount"] = 2.69
    return {"data": [data1, data2]}


@pytest.fixture
def mock_boto3_client():
    """Fixture providing a mocked boto3 client."""
    with patch("boto3.client") as mock_client:
        mock_sns = MagicMock()
        mock_secrets = MagicMock()

        mock_secrets.get_secret_value.return_value = {
            "SecretString": json.dumps(
                {
                    "WANDB_API_KEY": "test-api-key",
                    "wandb_project": "test-project",
                    "wandb_entity": "test-entity",
                    "model_artifact_name": "test-model",
                    "model_file_name": "model.joblib",
                    "preprocessor_artifact_name": "test-preprocessor",
                    "preprocessor_file_name": "preprocessor.joblib",
                }
            )
        }

        def mock_client_creator(service_name, **kwargs):
            if service_name == "sns":
                return mock_sns
            elif service_name == "secretsmanager":
                return mock_secrets
            return MagicMock()

        mock_client.side_effect = mock_client_creator
        yield {"sns": mock_sns, "secretsmanager": mock_secrets}


@pytest.fixture
def client(mock_model, mock_preprocessor):
    """Fixture providing a FastAPI TestClient with mocked app state."""
    app.state.model = mock_model
    app.state.preprocessor = mock_preprocessor
    return TestClient(app)


@pytest.fixture
def client_no_model(mock_preprocessor):
    """Fixture providing a TestClient with no model loaded."""
    app.state.model = None
    app.state.preprocessor = mock_preprocessor
    return TestClient(app)


@pytest.fixture
def mock_wandb_artifacts(tmp_path):
    """Fixture that mocks wandb and creates temporary artifact files."""
    import joblib

    model_path = tmp_path / "model.joblib"
    preprocessor_path = tmp_path / "preprocessor.joblib"

    joblib.dump(MockModel(), model_path)
    joblib.dump(MockPreprocessor(), preprocessor_path)

    mock_artifact = MagicMock()
    mock_artifact.download = MagicMock(return_value=str(tmp_path))

    mock_run = MagicMock()
    mock_run.use_artifact.return_value = mock_artifact

    with patch("wandb.init") as mock_init:
        mock_init.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_init.return_value.__exit__ = MagicMock(return_value=False)
        yield {
            "model_path": model_path,
            "preprocessor_path": preprocessor_path,
            "mock_init": mock_init,
            "mock_run": mock_run,
            "mock_artifact": mock_artifact,
        }
