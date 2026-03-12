from abc import ABC, abstractmethod

from mlops_pipeline.schemas.hyperparameter_tuning import HyperparameterTuningResult

class BaseModelStrategy(ABC):
    @abstractmethod
    def objective(self) -> float:
        ...
    @abstractmethod
    def start_hyperparameter_tuning(self) -> HyperparameterTuningResult:
        ...