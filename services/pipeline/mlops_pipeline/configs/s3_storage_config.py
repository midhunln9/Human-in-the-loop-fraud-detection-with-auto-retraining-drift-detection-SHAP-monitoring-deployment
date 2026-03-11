from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class S3StorageConfig:
    raw_data_key : str = "creditcard.csv"
    pipeline_id : str = field(default_factory=lambda: f"pipeline_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
    @property
    def training_data_key(self) -> str:
        return f"{self.pipeline_id}/data/training_data.csv"
    @property
    def validation_data_key(self) -> str:
        return f"{self.pipeline_id}/data/validation_data.csv"
    @property
    def test_data_key(self) -> str:
        return f"{self.pipeline_id}/data/test_data.csv"
    @property
    def preprocessed_train_data_key(self) -> str:
        return f"{self.pipeline_id}/data/preprocessed_train_data.csv"
    @property
    def preprocessed_test_data_key(self) -> str:
        return f"{self.pipeline_id}/data/preprocessed_test_data.csv"
    @property
    def preprocessed_validation_data_key(self) -> str:
        return f"{self.pipeline_id}/data/preprocessed_validation_data.csv"
    @property
    def model_key(self) -> str:
        return f"{self.pipeline_id}/models/model.joblib"