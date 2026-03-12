from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import List
import logging
from mlops_pipeline.protocols.model_versioning_protocol import ModelVersioningProtocol
from mlops_pipeline.strategies.base_strategy import BaseModelStrategy
from mlops_pipeline.schemas.hyperparameter_tuning import HyperparameterTuningResult
logger = logging.getLogger(__name__)

MODELFACTORY : dict = {"xgboost": XGBClassifier, "lightgbm": LGBMClassifier}

class MasterTuner:
    def __init__(self, strategies : List[BaseModelStrategy], model_versioning_repository: ModelVersioningProtocol):
        self.strategies = strategies
        self.results : List[HyperparameterTuningResult] = []
        self.model_versioning_repository = model_versioning_repository

    def start_hyperparameter_tuning(self):
        for strategy in self.strategies:
            result = strategy.start_hyperparameter_tuning()
            self.results.append(result)
        best_result = max(self.results, key=lambda x: x.best_pr_auc_score)
        try:
            best_model = MODELFACTORY[best_result.name](**best_result.best_params)
            self.model_versioning_repository.create_and_log_model_artifact_to_run(best_model, alias = "staging")
        except Exception as e:
            logger.error(f"Error while instantiating best model: {e}")
            raise
        return best_model
        