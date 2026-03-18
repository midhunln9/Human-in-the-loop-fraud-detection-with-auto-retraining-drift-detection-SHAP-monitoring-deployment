from dataclasses import dataclass


@dataclass
class HyperparameterTuningResult:
    """Result container for hyperparameter tuning operations.

    This dataclass stores the outcome of hyperparameter tuning including
    the model name, optimal hyperparameters, and the best achieved PR-AUC score.

    Attributes:
        name: The model identifier (e.g., "xgboost", "lightgbm").
        best_params: Dictionary containing the optimal hyperparameters.
        best_pr_auc_score: The best PR-AUC score achieved during tuning.
    """

    name: str
    best_params: dict
    best_pr_auc_score: float