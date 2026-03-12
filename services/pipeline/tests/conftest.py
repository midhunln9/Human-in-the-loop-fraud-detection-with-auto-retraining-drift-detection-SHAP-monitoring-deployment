import pytest
import numpy as np
from mlops_pipeline.protocols.storage_protocol import StorageProtocol
from mlops_pipeline.configs.transformation_config import TransformationConfig
from mlops_pipeline.src.xgboost_strategy import XGBoostStrategy
from mlops_pipeline.src.lightgbm_strategy import LightGBMStrategy
from tests.fakes.fake_storage import FakeStorageRepository
from mlops_pipeline.configs.s3_storage_config import S3StorageConfig
from mlops_pipeline.settings import Settings
from mlops_pipeline.configs.preprocessing_config import PreprocessingConfig
from mlops_pipeline.schemas.data import PreprocessedDatasets, SplitDatasets
import pandas as pd
from unittest.mock import MagicMock
from mlops_pipeline.protocols.model_versioning_protocol import ModelVersioningProtocol
from mlops_pipeline.src.master_tuner import MasterTuner
from mlops_pipeline.schemas.hyperparameter_tuning import HyperparameterTuningResult


def create_sample_features_dataframe():
    n_rows = 50
    v_columns = [f"V{i}" for i in range(29)]
    columns = v_columns + ["Amount"]
    data = np.random.randn(n_rows, len(columns))
    df = pd.DataFrame(data, columns=columns)
    return df

def create_sample_labels_series():
    n_rows = 50
    data = np.random.randint(0, 2, n_rows)
    series = pd.Series(data)
    return series

# storage Fixtures
@pytest.fixture
def fake_storage() -> StorageProtocol:
    return FakeStorageRepository()

# s3 config Fixtures
@pytest.fixture
def s3_config() -> S3StorageConfig:
    return S3StorageConfig()

# settings Fixtures
@pytest.fixture
def settings() -> Settings:
    return Settings()

# transformation config fixture
@pytest.fixture
def transformation_config():
    return TransformationConfig()

@pytest.fixture
def preprocessing_config():
    return PreprocessingConfig()

@pytest.fixture
def preprocessed_datasets():
    return PreprocessedDatasets(
        X_train=create_sample_features_dataframe(),
        X_val=create_sample_features_dataframe(),
        X_test=create_sample_features_dataframe(),
        y_train=create_sample_labels_series(),
        y_val=create_sample_labels_series(),
        y_test=create_sample_labels_series(),
    )

@pytest.fixture
def split_datasets():
    return SplitDatasets(
        X_train=create_sample_features_dataframe(),
        X_val=create_sample_features_dataframe(),
        X_test=create_sample_features_dataframe(),
        y_train=create_sample_labels_series(),
        y_val=create_sample_labels_series(),
        y_test=create_sample_labels_series(),
    )

@pytest.fixture
def model_versioning_repository_mock():
    return MagicMock(spec=ModelVersioningProtocol)

@pytest.fixture
def xgboost_strategy():
    return XGBoostStrategy(PreprocessedDatasets(
        X_train=create_sample_features_dataframe(),
        X_val=create_sample_features_dataframe(),
        X_test=create_sample_features_dataframe(),
        y_train=create_sample_labels_series(),
        y_val=create_sample_labels_series(),
        y_test=create_sample_labels_series(),
    ))

@pytest.fixture
def lightgbm_strategy():
    return LightGBMStrategy(PreprocessedDatasets(
        X_train=create_sample_features_dataframe(),
        X_val=create_sample_features_dataframe(),
        X_test=create_sample_features_dataframe(),
        y_train=create_sample_labels_series(),
        y_val=create_sample_labels_series(),
        y_test=create_sample_labels_series(),
    ))

@pytest.fixture
def master_tuner(xgboost_strategy, lightgbm_strategy, model_versioning_repository_mock):
    return MasterTuner(strategies=[xgboost_strategy, lightgbm_strategy], 
    model_versioning_repository=model_versioning_repository_mock)

@pytest.fixture
def hyperparameter_tuning_result():
    return HyperparameterTuningResult(
        name="xgboost",
        best_params={
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        best_pr_auc_score=0.9,
    )