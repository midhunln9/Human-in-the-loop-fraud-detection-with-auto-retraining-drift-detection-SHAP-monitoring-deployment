import optuna
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from mlops_pipeline.strategies.base_strategy import BaseModelStrategy
from mlops_pipeline.schemas.data import PreprocessedDatasets
import logging
from mlops_pipeline.schemas.hyperparameter_tuning import HyperparameterTuningResult

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

class XGBoostStrategy(BaseModelStrategy):
    def __init__(self, datasets : PreprocessedDatasets):
        self.data = datasets
        self.scale_pos_weight = (datasets.y_train.squeeze().value_counts()[0] / datasets.y_train.squeeze().value_counts()[1])
        logger.info(f"Scale pos weight: {self.scale_pos_weight}")

    def objective(self, trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20),
        }

        model = XGBClassifier(
            **params,
            n_jobs=-1,
            random_state=42,
            eval_metric="aucpr",
            tree_method="hist",
            scale_pos_weight=self.scale_pos_weight,
            early_stopping_rounds=50,
            n_estimators=5000,
        )

        model.fit(
            self.data.X_train, self.data.y_train,
            eval_set=[(self.data.X_val, self.data.y_val)],
            verbose=False,
        )

        trial.set_user_attr("best_iteration", model.best_iteration)

        preds = model.predict_proba(self.data.X_val)[:, 1]
        return average_precision_score(self.data.y_val, preds)

    def start_hyperparameter_tuning(self, trials : int = 100):
        logger.info("XGBoost hyperparameter tuning started")

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=trials)

        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        best_params.update({
            "n_estimators": best_trial.user_attrs["best_iteration"],
            "n_jobs": -1,
            "random_state": 42,
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "scale_pos_weight": self.scale_pos_weight,
        })

        logger.info("XGBoost tuning complete — best PR-AUC: %.4f", study.best_value)

        return HyperparameterTuningResult(
            name="xgboost",
            best_params=best_params,
            best_pr_auc_score=study.best_value,
        )
