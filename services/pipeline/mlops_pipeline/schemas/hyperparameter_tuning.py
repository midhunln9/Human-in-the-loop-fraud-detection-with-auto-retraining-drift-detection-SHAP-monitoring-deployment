from dataclasses import dataclass

@dataclass
class HyperparameterTuningResult:
    name: str
    best_params: dict
    best_pr_auc_score: float