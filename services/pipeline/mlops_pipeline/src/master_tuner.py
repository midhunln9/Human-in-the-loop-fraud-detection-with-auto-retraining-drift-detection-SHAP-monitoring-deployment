from typing import List
import logging
from mlops_pipeline.protocols.model_versioning_protocol import ModelVersioningProtocol
from mlops_pipeline.strategies.base_strategy import BaseModelStrategy
from mlops_pipeline.schemas.hyperparameter_tuning import HyperparameterTuningResult
from mlops_pipeline.exceptions import TuningError

logger = logging.getLogger(__name__)


class MasterTuner:
    """Orchestrates hyperparameter tuning across multiple model strategies.

    This class manages the execution of hyperparameter tuning for multiple
    model strategies (e.g., XGBoost, LightGBM) and identifies the best performing
    model based on PR-AUC scores.

    Attributes:
        strategies: List of model strategy instances to tune.
        results: List of tuning results from all strategies.
        model_versioning_repository: Repository for model versioning operations.
    """

    def __init__(self, strategies: List[BaseModelStrategy], model_versioning_repository: ModelVersioningProtocol):
        """Initialize the MasterTuner with strategies and repository.

        Args:
            strategies: List of instantiated model strategies for hyperparameter tuning.
            model_versioning_repository: Repository for model versioning and artifact management.
        """
        self.strategies = strategies
        self.results: List[HyperparameterTuningResult] = []
        self.model_versioning_repository = model_versioning_repository

    def start_hyperparameter_tuning(self) -> HyperparameterTuningResult:
        """Execute hyperparameter tuning across all configured strategies.

        Runs hyperparameter tuning for each strategy and returns the best result
        based on PR-AUC score.

        Returns:
            The HyperparameterTuningResult with the highest PR-AUC score.

        Raises:
            TuningError: If the hyperparameter tuning process fails.
        """
        try:
            for strategy in self.strategies:
                result = strategy.start_hyperparameter_tuning()
                self.results.append(result)
            best_result = max(self.results, key=lambda x: x.best_pr_auc_score)
            return best_result
        except TuningError as e:
            raise
        except Exception as e:
            logger.error(f"Error starting hyperparameter tuning: {e}")
            raise TuningError(f"Error starting hyperparameter tuning") from e
        