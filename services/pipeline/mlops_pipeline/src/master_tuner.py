from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import List, Tuple
import logging
from mlops_pipeline.protocols.model_versioning_protocol import ModelVersioningProtocol
from mlops_pipeline.strategies.base_strategy import BaseModelStrategy
from mlops_pipeline.schemas.hyperparameter_tuning import HyperparameterTuningResult
logger = logging.getLogger(__name__)


class MasterTuner:
    def __init__(self, strategies : List[BaseModelStrategy], model_versioning_repository: ModelVersioningProtocol):
        self.strategies = strategies
        self.results : List[HyperparameterTuningResult] = []
        self.model_versioning_repository = model_versioning_repository

    def start_hyperparameter_tuning(self) -> HyperparameterTuningResult:
        for strategy in self.strategies:
            result = strategy.start_hyperparameter_tuning()
            self.results.append(result)
        best_result = max(self.results, key=lambda x: x.best_pr_auc_score)
        return best_result
        