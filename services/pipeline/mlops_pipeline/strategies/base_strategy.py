from abc import ABC, abstractmethod

from mlops_pipeline.schemas.hyperparameter_tuning import HyperparameterTuningResult


class BaseModelStrategy(ABC):
    """Abstract base class defining the interface for model training strategies.

    This class specifies the contract that all model training strategies must
    implement, including hyperparameter tuning capabilities.

    Implementations should provide model-specific logic for training and tuning
    using frameworks like XGBoost, LightGBM, etc.
    """

    @abstractmethod
    def objective(self, trial) -> float:
        """Define the objective function for hyperparameter optimization.

        This method is called by the Optuna framework during each trial to
        evaluate a set of hyperparameters.

        Args:
            trial: An Optuna trial object for suggesting hyperparameters.

        Returns:
            The evaluation metric score (typically PR-AUC) for the trial.
        """
        ...

    @abstractmethod
    def start_hyperparameter_tuning(self, trials: int) -> HyperparameterTuningResult:
        """Execute hyperparameter tuning using the strategy's implementation.

        Args:
            trials: The number of optimization trials to run.

        Returns:
            A HyperparameterTuningResult containing the best parameters and score.
        """
        ...