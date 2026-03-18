import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from mlops_pipeline.schemas.data import SplitDatasets, PreprocessedDatasets
from mlops_pipeline.configs.preprocessing_config import PreprocessingConfig
from mlops_pipeline.protocols.model_versioning_protocol import ModelVersioningProtocol
import logging
from mlops_pipeline.exceptions import PreprocessingError

logger = logging.getLogger(__name__)


class DataPreprocessing:
    """Handles data preprocessing including imputation, scaling, and artifact logging.

    This class builds a preprocessing pipeline with imputation and standard scaling,
    applies it to train, validation, and test datasets, and logs the preprocessor
    artifact to the model versioning system.

    Attributes:
        preprocessing_config: Configuration specifying numerical columns and settings.
        split_datasets: The split datasets (train/val/test) to preprocess.
        repository: Model versioning repository for logging preprocessor artifacts.
    """

    def __init__(self, preprocessing_config: PreprocessingConfig, split_datasets: SplitDatasets,
                 repository: ModelVersioningProtocol):
        """Initialize the DataPreprocessing with configuration and datasets.

        Args:
            preprocessing_config: Configuration for preprocessing operations.
            split_datasets: The split datasets containing train, validation, and test data.
            repository: Model versioning repository for artifact management.
        """
        self.preprocessing_config = preprocessing_config
        self.split_datasets = split_datasets
        self.repository = repository

    def _build_preprocessor(self) -> ColumnTransformer:
        """Build the preprocessing pipeline with imputation and scaling.

        Creates a ColumnTransformer that applies mean imputation and standard
        scaling to numerical columns.

        Returns:
            A fitted ColumnTransformer preprocessing pipeline.
        """
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
        return ColumnTransformer(
            transformers=[("num_pipeline", num_pipeline, self.preprocessing_config.numerical_columns)],
            remainder="drop",
        )

    def preprocess_data(self) -> PreprocessedDatasets:
        """Apply preprocessing to all datasets and log the preprocessor artifact.

        Fits the preprocessor on training data, transforms all datasets,
        logs the preprocessor as an artifact, and returns preprocessed datasets.

        Returns:
            A PreprocessedDatasets dataclass containing transformed features
            and original labels for train, validation, and test sets.

        Raises:
            PreprocessingError: If preprocessing or artifact logging fails.
        """
        preprocessor = self._build_preprocessor()
        preprocesed_train_data = pd.DataFrame(preprocessor.fit_transform(self.split_datasets.X_train), columns=self.preprocessing_config.numerical_columns)
        preprocesed_validation_data = pd.DataFrame(preprocessor.transform(self.split_datasets.X_val), columns=self.preprocessing_config.numerical_columns)
        preprocesed_test_data = pd.DataFrame(preprocessor.transform(self.split_datasets.X_test), columns=self.preprocessing_config.numerical_columns)
        try:
            self.repository.create_and_log_preprocessor_artifact_to_run(preprocessor, "staging")
        except Exception as e:
            logger.error(f"Error creating and logging preprocessor artifact: {e}")
            raise PreprocessingError(f"Error creating and logging preprocessor artifact") from e

        logger.info("successfully stored the preprocessed data inside the schema")

        return PreprocessedDatasets(X_train=preprocesed_train_data, X_val=preprocesed_validation_data, X_test=preprocesed_test_data,
                                      y_train=self.split_datasets.y_train, y_val=self.split_datasets.y_val, y_test=self.split_datasets.y_test)