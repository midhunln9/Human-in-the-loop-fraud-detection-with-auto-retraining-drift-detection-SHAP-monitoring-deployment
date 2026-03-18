from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class S3StorageConfig:
    """Configuration for S3 storage paths and keys.

    This dataclass defines all S3 key paths used throughout the pipeline
    for storing raw data, preprocessed data, models, and evaluation reports.
    Keys are organized by pipeline ID with timestamp-based naming.

    Attributes:
        raw_data_key: S3 key for the raw input data file.
        pipeline_id: Unique identifier for the pipeline run with timestamp.
    """

    raw_data_key: str = "creditcard.csv"
    pipeline_id: str = field(default_factory=lambda: f"pipeline_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")

    @property
    def preprocessed_train_data_key_features(self) -> str:
        """S3 key for preprocessed training features."""
        return f"{self.pipeline_id}/data/preprocessed_train_data.csv"

    @property
    def preprocessed_test_data_key_features(self) -> str:
        """S3 key for preprocessed test features."""
        return f"{self.pipeline_id}/data/preprocessed_test_data.csv"

    @property
    def preprocessed_validation_data_key_features(self) -> str:
        """S3 key for preprocessed validation features."""
        return f"{self.pipeline_id}/data/preprocessed_validation_data.csv"

    @property
    def preprocessed_train_labels_key(self) -> str:
        """S3 key for preprocessed training labels."""
        return f"{self.pipeline_id}/data/preprocessed_train_labels.csv"

    @property
    def preprocessed_test_labels_key(self) -> str:
        """S3 key for preprocessed test labels."""
        return f"{self.pipeline_id}/data/preprocessed_test_labels.csv"

    @property
    def preprocessed_validation_labels_key(self) -> str:
        """S3 key for preprocessed validation labels."""
        return f"{self.pipeline_id}/data/preprocessed_validation_labels.csv"

    @property
    def model_key(self) -> str:
        """S3 key for the trained model artifact."""
        return f"{self.pipeline_id}/models/model.joblib"

    @property
    def eval_report_key(self) -> str:
        """S3 key for the evaluation HTML report."""
        return f"{self.pipeline_id}/reports/eval_report.html"