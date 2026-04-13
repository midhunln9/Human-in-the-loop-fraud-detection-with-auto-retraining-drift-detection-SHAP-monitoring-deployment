from dataclasses import dataclass


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases model versioning.

    This dataclass holds all configuration parameters for W&B artifact
    management including artifact names, types, and file naming conventions.

    Attributes:
        entity: The W&B entity (team or user) for the project.
        project: The W&B project name.
        preprocessor_file_name: Filename for serialized preprocessor artifacts.
        preprocessor_artifact_name: Name identifier for preprocessor artifacts.
        preprocessor_artifact_type: Type classification for preprocessor artifacts.
        model_artifact_name: Name identifier for model artifacts.
        model_file_name: Filename for serialized model artifacts.
        model_artifact_type: Type classification for model artifacts.
    """

    entity: str = "midhun61025-student"
    project: str = "mlops-fraud-detection"
    preprocessor_file_name: str = "preprocessor.joblib"
    preprocessor_artifact_name: str = "preprocessor_artifact"
    preprocessor_artifact_type: str = "preprocessor"
    model_artifact_name: str = "model_artifact"
    model_file_name: str = "model.joblib"
    model_artifact_type: str = "model"
