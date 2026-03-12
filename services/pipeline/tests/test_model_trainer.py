from mlops_pipeline.src.model_trainer import ModelTrainer
import pytest

def test_model_trainer(preprocessed_datasets, hyperparameter_tuning_result, model_versioning_repository_mock):
    model_trainer = ModelTrainer(preprocessed_datasets, hyperparameter_tuning_result, model_versioning_repository_mock)
    model, pr_auc_score = model_trainer.combine_data_and_train_model()
    assert model is not None
    assert pr_auc_score is not None
    assert model_versioning_repository_mock.create_and_log_model_artifact_to_run.call_count == 1
    