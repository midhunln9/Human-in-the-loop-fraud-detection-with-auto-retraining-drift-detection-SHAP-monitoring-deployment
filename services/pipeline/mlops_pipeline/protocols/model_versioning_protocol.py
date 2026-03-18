from typing import Protocol, Any
from sklearn.base import BaseEstimator


class ModelVersioningProtocol(Protocol):
    """Protocol defining the interface for model versioning operations.

    This protocol specifies the contract that any model versioning implementation
    must fulfill to track, log, and manage model artifacts within the MLOps pipeline.
    """

    def create_and_log_model_artifact_to_run(self, model: BaseEstimator, alias: str, metric: float) -> None:
        """Create and log a model artifact to the versioning system.

        Args:
            model: The trained model to log as an artifact.
            alias: The alias to assign to the artifact (e.g., "staging", "production").
            metric: The performance metric (PR-AUC) associated with the model.

        Returns:
            None
        """
        ...

    def create_and_log_preprocessor_artifact_to_run(self, preprocessor: BaseEstimator, alias: str) -> None:
        """Create and log a preprocessor artifact to the versioning system.

        Args:
            preprocessor: The fitted preprocessor to log as an artifact.
            alias: The alias to assign to the artifact (e.g., "staging", "production").

        Returns:
            None
        """
        ...

    def stream_load_from_alias(self, artifact_name: str, alias: str) -> Any:
        """Load an artifact from the versioning system by its alias.

        Args:
            artifact_name: The name of the artifact to load.
            alias: The alias of the artifact version to load.

        Returns:
            A tuple containing the loaded artifact and its associated metric.
        """
        ...

    def promote_artifact(self, artifact_name: str, from_alias: str, to_alias: str) -> None:
        """Promote an artifact from one alias to another.

        Args:
            artifact_name: The name of the artifact to promote.
            from_alias: The current alias of the artifact.
            to_alias: The target alias to promote the artifact to.

        Returns:
            None
        """
        ...