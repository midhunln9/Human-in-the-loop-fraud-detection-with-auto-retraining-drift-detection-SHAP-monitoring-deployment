import logging

import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score

from mlops_pipeline.schemas.hyperparameter_tuning import HyperparameterTuningResult
from mlops_pipeline.strategies.base_strategy import BaseModelStrategy
from mlops_pipeline.schemas.data import PreprocessedDatasets


logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

class LightGBMStrategy(BaseModelStrategy):

    def __init__(self, datasets : PreprocessedDatasets) -> None:
        self.data = datasets
        self.scale_pos_weight = (datasets.y_train.squeeze().value_counts()[0] / datasets.y_train.squeeze().value_counts()[1])
        
    def objective(self, trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", -1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        }

        model = LGBMClassifier(
            **params,
            n_estimators=5000,
            objective="binary",
            n_jobs=-1,
            random_state=42,
            scale_pos_weight=self.scale_pos_weight,
            verbosity=-1,
        )

        model.fit(
            self.data.X_train, self.data.y_train,
            eval_set=[(self.data.X_val, self.data.y_val)],
            eval_metric="average_precision",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )

        trial.set_user_attr("best_iteration", model.best_iteration_)

        preds = model.predict_proba(self.data.X_val)[:, 1]
        return average_precision_score(self.data.y_val, preds)

    def start_hyperparameter_tuning(self) -> HyperparameterTuningResult:
        logger.info("LightGBM hyperparameter tuning started")

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=100)

        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        best_params.update({
            "n_estimators": best_trial.user_attrs["best_iteration"],
            "objective": "binary",
            "n_jobs": -1,
            "random_state": 42,
            "scale_pos_weight": self.scale_pos_weight,
            "verbosity": -1,
        })

        logger.info("LightGBM tuning complete — best PR-AUC: %.4f", study.best_value)

        return HyperparameterTuningResult(
            name="lightgbm",
            best_params=best_params,
            best_pr_auc_score=study.best_value,
        )
