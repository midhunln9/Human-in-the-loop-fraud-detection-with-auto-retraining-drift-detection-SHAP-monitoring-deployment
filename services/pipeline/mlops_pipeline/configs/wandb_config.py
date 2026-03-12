from dataclasses import dataclass

@dataclass
class WandbConfig:
    entity: str = "midhunln9-liverpool"
    project: str = "mlops-fraud-detection"
    preprocessor_file_name: str = "preprocessor.joblib"
    preprocessor_artifact_name: str = "preprocessor_artifact"
    preprocessor_artifact_type: str = "preprocessor"
    model_artifact_name: str = "model_artifact"
    model_file_name: str = "best_model.joblib"
    model_artifact_type: str = "model"
