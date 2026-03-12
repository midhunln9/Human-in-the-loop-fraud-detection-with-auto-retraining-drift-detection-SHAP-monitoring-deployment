from mlops_pipeline.schemas.hyperparameter_tuning import HyperparameterTuningResult

def test_if_hyperparameter_tuning_result_returned(master_tuner):
    result = master_tuner.start_hyperparameter_tuning()
    assert isinstance(result, HyperparameterTuningResult)

def test_metadata_of_result(master_tuner):
    result = master_tuner.start_hyperparameter_tuning()
    assert result.name in ["xgboost", "lightgbm"]
    assert result.best_params is not None
    assert result.best_pr_auc_score is not None