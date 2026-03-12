import pytest
from mlops_pipeline.schemas.hyperparameter_tuning import HyperparameterTuningResult

def test_xgboost_strategy_returns_result(xgboost_strategy):
    result = xgboost_strategy.start_hyperparameter_tuning(trials=1)
    assert isinstance(result, HyperparameterTuningResult)

def test_xgboost_strategy_result_metadata(xgboost_strategy):
    result = xgboost_strategy.start_hyperparameter_tuning(trials=1)
    assert result.name == "xgboost"
    assert result.best_params is not None
    assert result.best_pr_auc_score is not None
