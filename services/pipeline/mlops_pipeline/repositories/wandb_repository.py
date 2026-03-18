import wandb
from mlops_pipeline.configs.wandb_config import WandbConfig
from mlops_pipeline.protocols.model_versioning_protocol import ModelVersioningProtocol
from sklearn.base import BaseEstimator
import tempfile
import os
import joblib
import logging
from typing import Tuple
from mlops_pipeline.exceptions import ArtifactError

logger = logging.getLogger(__name__)


class WandbRepository(ModelVersioningProtocol):
    """Weights & Biases implementation of the ModelVersioningProtocol.

    This class provides W&B-based model versioning operations for logging,
    loading, and promoting model and preprocessor artifacts.

    Attributes:
        config: Configuration for W&B including project, entity, and artifact names.
    """

    def __init__(self, config: WandbConfig):
        """Initialize the WandbRepository with configuration.

        Args:
            config: The W&B configuration containing project and artifact settings.
        """
        self.config = config

    def create_and_log_model_artifact_to_run(self, model: BaseEstimator, alias: str, metric: float) -> None:
        """Create and log a model artifact to W&B.

        Args:
            model: The trained model to log.
            alias: The alias to assign to the artifact (e.g., "staging").
            metric: The PR-AUC performance metric to store as metadata.

        Raises:
            ArtifactError: If artifact creation or logging fails.

        Returns:
            None
        """
        try:
            with wandb.init(entity=self.config.entity, project=self.config.project) as run:
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_path = os.path.join(temp_dir, self.config.model_file_name)
                    joblib.dump(model, model_path)
                    artifact = wandb.Artifact(self.config.model_artifact_name, type=self.config.model_artifact_type,
                    metadata={"pr_auc": metric})
                    artifact.add_file(model_path)
                    run.log_artifact(artifact, aliases=[alias])
            logger.info(f"Model artifact created and logged successfully")
        except Exception as e:
            logger.error(f"Error creating and logging model artifact: {e}")
            raise ArtifactError(f"Error creating and logging model artifact") from e
    
    def create_and_log_preprocessor_artifact_to_run(self, preprocessor: BaseEstimator, alias: str) -> None:
        """Create and log a preprocessor artifact to W&B.

        Args:
            preprocessor: The fitted preprocessor pipeline to log.
            alias: The alias to assign to the artifact (e.g., "staging").

        Raises:
            ArtifactError: If artifact creation or logging fails.

        Returns:
            None
        """
        try:
            with wandb.init(entity=self.config.entity, project=self.config.project) as run:
                with tempfile.TemporaryDirectory() as temp_dir:
                    preprocessor_path = os.path.join(temp_dir, self.config.preprocessor_file_name)
                    joblib.dump(preprocessor, preprocessor_path)
                    artifact = wandb.Artifact(self.config.preprocessor_artifact_name, type=self.config.preprocessor_artifact_type)
                    artifact.add_file(preprocessor_path)
                    run.log_artifact(artifact, aliases=[alias])
            logger.info(f"Preprocessor artifact created and logged successfully")
        except Exception as e:
            logger.error(f"Error creating and logging preprocessor artifact: {e}")
            raise ArtifactError(f"Error creating and logging preprocessor artifact") from e
        
    def stream_load_from_alias(self, artifact_name: str, alias: str) -> Tuple[BaseEstimator, float]:
        """Load a model artifact from W&B by its alias.

        Args:
            artifact_name: The name of the artifact to load.
            alias: The alias of the artifact version to load.

        Returns:
            A tuple containing the loaded model and its PR-AUC metric.
            Returns (None, None) if the artifact is not found.

        Raises:
            ArtifactError: If the download operation fails unexpectedly.
        """
        try:
            with wandb.init(entity=self.config.entity, project=self.config.project) as run:
                try:
                    artifact = run.use_artifact(f"{artifact_name}:{alias}")
                    with tempfile.TemporaryDirectory() as temp_dir:
                        artifact.download(root=temp_dir)
                        return joblib.load(os.path.join(temp_dir, self.config.model_file_name)), artifact.metadata.get("pr_auc", None)
                except wandb.errors.CommError:
                    logger.warning(f"Artifact {artifact_name}:{alias} not found. Returning None.")
        except Exception as e:
            logger.error(f"Error downloading artifact: {e}")
            raise ArtifactError(f"Error downloading artifact") from e
        
    def promote_artifact(self, artifact_name: str, from_alias: str, to_alias: str) -> None:
        """Promote an artifact from one alias to another in W&B.

        Args:
            artifact_name: The name of the artifact to promote.
            from_alias: The current alias of the artifact.
            to_alias: The target alias to promote the artifact to.

        Raises:
            ArtifactError: If the promotion operation fails.

        Returns:
            None
        """
        try:
            api = wandb.Api()
            artifact = api.artifact(f"{self.config.entity}/{self.config.project}/{artifact_name}:{from_alias}")
            artifact.aliases.append(to_alias)
            if from_alias in artifact.aliases:
                artifact.aliases.remove(from_alias)
            artifact.save()
            logger.info(f"Artifact promoted from {from_alias} to {to_alias}")
        except Exception as e:
            logger.error(f"Error promoting artifact: {e}")
            raise ArtifactError(f"Error promoting artifact") from e