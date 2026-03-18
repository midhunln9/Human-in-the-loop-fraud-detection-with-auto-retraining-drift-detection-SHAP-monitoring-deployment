import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app, load_app_secrets, load_model, on_start_up


class TestHealthEndpoint:
    """Integration tests for the health check endpoint."""

    def test_health_check_returns_ok_when_model_loaded(self, client):
        """Test that health check returns ok status when model is loaded."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_check_returns_false_when_model_not_loaded(self, client_no_model):
        """Test that health check returns model_loaded=false when model is not loaded."""
        response = client_no_model.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is False

    def test_health_check_returns_json_content_type(self, client):
        """Test that health check returns JSON content type."""
        response = client.get("/health")

        assert response.headers["content-type"] == "application/json"

    def test_health_check_method_not_allowed_post(self, client):
        """Test that POST request to health endpoint is not allowed."""
        response = client.post("/health")

        assert response.status_code == 405

    def test_health_check_method_not_allowed_put(self, client):
        """Test that PUT request to health endpoint is not allowed."""
        response = client.put("/health")

        assert response.status_code == 405


class TestAppMetadata:
    """Tests for FastAPI application metadata."""

    def test_app_title_is_correct(self, client):
        """Test that the FastAPI app has the correct title."""
        assert app.title == "Credit Card Fraud Detection API"

    def test_app_description_is_correct(self, client):
        """Test that the FastAPI app has the correct description."""
        assert app.description == "API for credit card fraud detection"

    def test_app_has_lifespan_handler(self):
        """Test that the app has a lifespan handler configured."""
        assert app.router.lifespan_context is not None


class TestLoadAppSecrets:
    """Tests for load_app_secrets function."""

    @patch("app.main.boto3.client")
    @patch.dict(
        "os.environ",
        {"AWS_ACCESS_KEY_ID": "test-key", "AWS_SECRET_ACCESS_KEY": "test-secret"},
    )
    def test_load_app_secrets_returns_parsed_secret(self, mock_boto_client):
        """Test that load_app_secrets returns parsed secret dictionary."""
        mock_secrets_manager = MagicMock()
        mock_secrets_manager.get_secret_value.return_value = {
            "SecretString": json.dumps(
                {
                    "WANDB_API_KEY": "test-wandb-key",
                    "wandb_project": "test-project",
                    "wandb_entity": "test-entity",
                }
            )
        }
        mock_boto_client.return_value = mock_secrets_manager

        result = load_app_secrets()

        assert result["WANDB_API_KEY"] == "test-wandb-key"
        assert result["wandb_project"] == "test-project"
        assert result["wandb_entity"] == "test-entity"

    @patch("app.main.boto3.client")
    @patch.dict(
        "os.environ",
        {"AWS_ACCESS_KEY_ID": "test-key", "AWS_SECRET_ACCESS_KEY": "test-secret"},
    )
    def test_load_app_secrets_sets_wandb_api_key_env(self, mock_boto_client):
        """Test that load_app_secrets sets WANDB_API_KEY environment variable."""
        mock_secrets_manager = MagicMock()
        mock_secrets_manager.get_secret_value.return_value = {
            "SecretString": json.dumps({"WANDB_API_KEY": "test-wandb-key"})
        }
        mock_boto_client.return_value = mock_secrets_manager

        import os

        load_app_secrets()

        assert os.environ["WANDB_API_KEY"] == "test-wandb-key"

    @patch("app.main.boto3.client")
    @patch.dict(
        "os.environ",
        {"AWS_ACCESS_KEY_ID": "test-key", "AWS_SECRET_ACCESS_KEY": "test-secret"},
    )
    def test_load_app_secrets_uses_correct_secret_name(self, mock_boto_client):
        """Test that load_app_secrets uses the correct secret name."""
        mock_secrets_manager = MagicMock()
        mock_secrets_manager.get_secret_value.return_value = {
            "SecretString": json.dumps({"WANDB_API_KEY": "test-key"})
        }
        mock_boto_client.return_value = mock_secrets_manager

        load_app_secrets()

        mock_secrets_manager.get_secret_value.assert_called_once_with(
            SecretId="fraud_detection/api"
        )

    @patch("app.main.boto3.client")
    @patch.dict(
        "os.environ",
        {"AWS_ACCESS_KEY_ID": "test-key", "AWS_SECRET_ACCESS_KEY": "test-secret"},
    )
    def test_load_app_secrets_uses_correct_region(self, mock_boto_client):
        """Test that load_app_secrets uses the correct region."""
        mock_secrets_manager = MagicMock()
        mock_secrets_manager.get_secret_value.return_value = {
            "SecretString": json.dumps({"WANDB_API_KEY": "test-key"})
        }
        mock_boto_client.return_value = mock_secrets_manager

        load_app_secrets(region_name="us-west-2")

        mock_boto_client.assert_called_once()
        call_kwargs = mock_boto_client.call_args.kwargs
        assert call_kwargs["region_name"] == "us-west-2"

    @patch("app.main.boto3.client")
    @patch.dict(
        "os.environ",
        {"AWS_ACCESS_KEY_ID": "test-key", "AWS_SECRET_ACCESS_KEY": "test-secret"},
    )
    def test_load_app_secrets_raises_on_client_error(self, mock_boto_client):
        """Test that load_app_secrets raises RuntimeError on AWS client error."""
        from botocore.exceptions import ClientError

        mock_secrets_manager = MagicMock()
        mock_secrets_manager.get_secret_value.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "Secret not found"}},
            "GetSecretValue",
        )
        mock_boto_client.return_value = mock_secrets_manager

        with pytest.raises(RuntimeError) as exc_info:
            load_app_secrets()

        assert "Failed to load secret" in str(exc_info.value)


class TestLoadModel:
    """Tests for load_model function."""

    @patch("app.main.wandb.init")
    @patch("app.main.joblib.load")
    def test_load_model_returns_loaded_model(self, mock_joblib_load, mock_wandb_init):
        """Test that load_model returns the loaded model."""
        mock_run = MagicMock()
        mock_artifact = MagicMock()
        mock_run.use_artifact.return_value = mock_artifact
        mock_wandb_init.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_wandb_init.return_value.__exit__ = MagicMock(return_value=False)

        expected_model = {"model": "test"}
        mock_joblib_load.return_value = expected_model

        result = load_model(
            project_name="test-project",
            entity="test-entity",
            artifact_name="test-artifact",
            file_name="model.joblib",
        )

        assert result == expected_model

    @patch("app.main.wandb.init")
    @patch("app.main.joblib.load")
    def test_load_model_uses_production_version(self, mock_joblib_load, mock_wandb_init):
        """Test that load_model uses :production version alias."""
        mock_run = MagicMock()
        mock_artifact = MagicMock()
        mock_run.use_artifact.return_value = mock_artifact
        mock_wandb_init.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_wandb_init.return_value.__exit__ = MagicMock(return_value=False)

        load_model(
            project_name="test-project",
            entity="test-entity",
            artifact_name="test-artifact",
            file_name="model.joblib",
        )

        mock_run.use_artifact.assert_called_once_with("test-artifact:production")


class TestLifespan:
    """Tests for the lifespan context manager."""

    @pytest.mark.asyncio
    @patch("app.main.load_app_secrets")
    @patch("app.main.load_model")
    @patch("app.main.setup_logging")
    async def test_lifespan_loads_secrets_and_model(
        self, mock_setup_logging, mock_load_model, mock_load_secrets
    ):
        """Test that lifespan loads secrets and model on startup."""
        mock_secrets = {
            "wandb_project": "test-project",
            "wandb_entity": "test-entity",
            "model_artifact_name": "test-model",
            "model_file_name": "model.joblib",
            "preprocessor_artifact_name": "test-preprocessor",
            "preprocessor_file_name": "preprocessor.joblib",
        }
        mock_load_secrets.return_value = mock_secrets
        mock_load_model.return_value = {"model": "test"}

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        async with on_start_up(mock_app):
            pass

        mock_load_secrets.assert_called_once()
        assert mock_load_model.call_count == 2

    @pytest.mark.asyncio
    @patch("app.main.load_app_secrets")
    @patch("app.main.load_model")
    @patch("app.main.setup_logging")
    async def test_lifespan_sets_app_state(
        self, mock_setup_logging, mock_load_model, mock_load_secrets
    ):
        """Test that lifespan sets model and preprocessor on app state."""
        mock_secrets = {
            "wandb_project": "test-project",
            "wandb_entity": "test-entity",
            "model_artifact_name": "test-model",
            "model_file_name": "model.joblib",
            "preprocessor_artifact_name": "test-preprocessor",
            "preprocessor_file_name": "preprocessor.joblib",
        }
        mock_load_secrets.return_value = mock_secrets

        mock_model = {"type": "model"}
        mock_preprocessor = {"type": "preprocessor"}
        mock_load_model.side_effect = [mock_model, mock_preprocessor]

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        async with on_start_up(mock_app):
            pass

        assert mock_app.state.model == mock_model
        assert mock_app.state.preprocessor == mock_preprocessor

    @pytest.mark.asyncio
    @patch("app.main.load_app_secrets")
    @patch("app.main.load_model")
    @patch("app.main.setup_logging")
    async def test_lifespan_raises_when_model_none(
        self, mock_setup_logging, mock_load_model, mock_load_secrets
    ):
        """Test that lifespan raises RuntimeError when model loading fails."""
        mock_secrets = {
            "wandb_project": "test-project",
            "wandb_entity": "test-entity",
            "model_artifact_name": "test-model",
            "model_file_name": "model.joblib",
            "preprocessor_artifact_name": "test-preprocessor",
            "preprocessor_file_name": "preprocessor.joblib",
        }
        mock_load_secrets.return_value = mock_secrets
        mock_load_model.side_effect = [None, {"type": "preprocessor"}]

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        with pytest.raises(RuntimeError) as exc_info:
            async with on_start_up(mock_app):
                pass

        assert "Error starting or shutting down the application" in str(exc_info.value)
