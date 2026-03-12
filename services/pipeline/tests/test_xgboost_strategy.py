import pytest
from mlops_pipeline.schemas.hyperparameter_tuning import HyperparameterTuningResult

def test_xgboost_strategy_returns_result(xgboost_strategy, hyperparameter_tuning_result):
    result = xgboost_strategy.start_hyperparameter_tuning(trials=1)
    assert isinstance(result, HyperparameterTuningResult)

def test_xgboost_strategy_result_metadata(xgboost_strategy, hyperparameter_tuning_result):
    result = xgboost_strategy.start_hyperparameter_tuning(trials=1)
    assert result.name == "xgboost"
    assert result.best_pr_auc_score is not None
    assert isinstance(result.best_params, dict)
