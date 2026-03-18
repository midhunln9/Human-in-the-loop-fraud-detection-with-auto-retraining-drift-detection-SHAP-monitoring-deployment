from mlops_pipeline.schemas.data import PreprocessedDatasets
from mlops_pipeline.schemas.hyperparameter_tuning import HyperparameterTuningResult
import pandas as pd
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import logging
from sklearn.metrics import average_precision_score
from mlops_pipeline.protocols.model_versioning_protocol import ModelVersioningProtocol
from mlops_pipeline.exceptions import TrainingError

logger = logging.getLogger(__name__)

MODELFACTORY: dict = {"xgboost": XGBClassifier, "lightgbm": LGBMClassifier}


class ModelTrainer:
    """Trains the best model on combined training and validation data.

    This class combines training and validation datasets, trains the best model
    from hyperparameter tuning, evaluates it, and logs the model artifact.

    Attributes:
        datasets: The preprocessed datasets containing train, validation, and test splits.
        best_result: The best hyperparameter tuning result with optimal parameters.
        model_versioning_repository: Repository for logging model artifacts.
    """

    def __init__(self, datasets: PreprocessedDatasets, best_result: HyperparameterTuningResult,
                 model_versioning_repository: ModelVersioningProtocol):
        """Initialize the ModelTrainer with datasets and best tuning result.

        Args:
            datasets: The preprocessed datasets for training.
            best_result: The best hyperparameter tuning result with model configuration.
            model_versioning_repository: Repository for model versioning and artifact management.
        """
        self.datasets = datasets
        self.best_result = best_result
        self.model_versioning_repository = model_versioning_repository

    def combine_data_and_train_model(self) -> BaseEstimator:
        """Combine train and validation data, train the best model, and log it.

        Concatenates training and validation datasets, trains the best model
        using optimal hyperparameters, evaluates PR-AUC on combined data,
        and logs the model artifact to the versioning system.

        Returns:
            The trained best model (BaseEstimator).

        Raises:
            TrainingError: If model training or artifact logging fails.
        """
        combined_df_features = pd.concat([self.datasets.X_train, self.datasets.X_val], axis=0)
        combined_df_labels = pd.concat([self.datasets.y_train, self.datasets.y_val], axis=0)
        try:
            best_model = MODELFACTORY[self.best_result.name](**self.best_result.best_params)
            best_model.fit(combined_df_features, combined_df_labels)
            pr_auc_score = average_precision_score(combined_df_labels, best_model.predict_proba(combined_df_features)[:, 1])
            self.model_versioning_repository.create_and_log_model_artifact_to_run(best_model, alias="staging", metric=pr_auc_score)
            logger.info(f"PR AUC score of staging model: {pr_auc_score}")
            logger.info(f"Model trained successfully with combined data")
            return best_model
        except Exception as e:
            logger.error(f"Error while instantiating best model: {e}")
            raise TrainingError(f"Error while instantiating best model") from e

