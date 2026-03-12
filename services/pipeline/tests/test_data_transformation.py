from mlops_pipeline.src.data_transformation import DataTransformation
from mlops_pipeline.configs.transformation_config import TransformationConfig
from mlops_pipeline.schemas.data import SplitDatasets
import pandas as pd
import pytest

def test_data_transformation_splitting(transformation_config):
    data_transformation = DataTransformation(transformation_config)
    df = pd.DataFrame({
        "Feature1": [i for i in range(10)],
        "Feature2": [i for i in range(10)],
        "Class": [0 if i % 2 == 0 else 1 for i in range(10)]
    })
    result = data_transformation.transform_data(df)
    assert isinstance(result, SplitDatasets)
    assert len(result.X_train) == 8
    assert len(result.X_val) == 1
    assert len(result.X_test) == 1
    assert len(result.y_train) == 8
    assert len(result.y_val) == 1
    assert len(result.y_test) == 1

def test_wrong_target_columns(transformation_config):
    data_transformation = DataTransformation(transformation_config)
    df = pd.DataFrame({
        "Feature1": [i for i in range(10)],
        "Feature2": [i for i in range(10)],
        "target": [0 if i % 2 == 0 else 1 for i in range(10)]
    })
    with pytest.raises(ValueError, match = f"check the target column: {transformation_config.target_column}"):
        result = data_transformation.transform_data(df)

