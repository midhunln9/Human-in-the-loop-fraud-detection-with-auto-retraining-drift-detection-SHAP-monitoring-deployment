import pytest
from mlops_pipeline.schemas.hyperparameter_tuning import HyperparameterTuningResult

def test_lightgbm_strategy_returns_result(lightgbm_strategy):
    result = lightgbm_strategy.start_hyperparameter_tuning(trials=1)
    assert isinstance(result, HyperparameterTuningResult)

def test_light_strategy_result_metadata(lightgbm_strategy):
    result = lightgbm_strategy.start_hyperparameter_tuning(trials=1)
    assert result.name == "lightgbm"
    assert result.best_params is not None
    assert result.best_pr_auc_score is not None
