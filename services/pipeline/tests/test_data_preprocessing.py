from mlops_pipeline.src.data_preprocessing import DataPreprocessing
import pandas as pd
from mlops_pipeline.schemas.data import PreprocessedDatasets
import pytest

def test_if_dataframe_after_preprocessing(preprocessing_config, split_datasets, model_versioning_repository_mock):
    data_preprocessing = DataPreprocessing(preprocessing_config, split_datasets, model_versioning_repository_mock)
    result = data_preprocessing.preprocess_data()
    assert isinstance(result.X_train, pd.DataFrame)
    assert isinstance(result.X_val, pd.DataFrame)
    assert isinstance(result.X_test, pd.DataFrame)
    assert isinstance(result.y_train, pd.DataFrame)
    assert isinstance(result.y_val, pd.DataFrame)
    assert isinstance(result.y_test, pd.DataFrame)

def test_if_stored_in_preprocessed_class(preprocessing_config, split_datasets, model_versioning_repository_mock):
    data_preprocessing = DataPreprocessing(preprocessing_config, split_datasets, model_versioning_repository_mock)
    result = data_preprocessing.preprocess_data()
    assert isinstance(result, PreprocessedDatasets)

def test_if_wandb_fails(preprocessing_config, split_datasets, model_versioning_repository_mock):
    model_versioning_repository_mock.create_and_log_preprocessor_artifact_to_run.side_effect = Exception("Wandb fails")
    data_preprocessing = DataPreprocessing(preprocessing_config, split_datasets, model_versioning_repository_mock)
    with pytest.raises(Exception, match="Wandb fails"):
        data_preprocessing.preprocess_data()








