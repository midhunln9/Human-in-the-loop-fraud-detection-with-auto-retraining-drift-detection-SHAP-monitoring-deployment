from mlops_pipeline.protocols.storage_protocol import StorageProtocol
from mlops_pipeline.configs.s3_storage_config import S3StorageConfig
from mlops_pipeline.schemas.data import PreprocessedDatasets
from sklearn.metrics import average_precision_score
from evidently import Report
from evidently.presets import ClassificationPreset
from evidently import DataDefinition
from evidently import BinaryClassification
from evidently import Dataset
import os
import tempfile
from sklearn.base import BaseEstimator
import pandas as pd
import logging
from mlops_pipeline.exceptions import EvaluationError

logger = logging.getLogger(__name__)

class ModelEvaluation:
    def __init__(self, best_model: BaseEstimator, storage: StorageProtocol, s3_config: S3StorageConfig,
                 preprocessed_datasets: PreprocessedDatasets, pos_label: int = 1) -> None:
        self.best_model = best_model
        self.pos_label = pos_label
        self.preprocessed_datasets = preprocessed_datasets
        self.storage = storage
        self.s3_config = s3_config

    def evaluate(self) -> float:
        try:
            proba = self.best_model.predict_proba(self.preprocessed_datasets.X_test)[:, 1]
            pred = self.best_model.predict(self.preprocessed_datasets.X_test)

            pr_auc = average_precision_score(self.preprocessed_datasets.y_test, proba)
            logger.info("Test PR-AUC: %.4f", pr_auc)

            eval_df = pd.DataFrame({
                "target": self.preprocessed_datasets.y_test,
                "prediction": pred,
                "prediction_proba": proba,
            })

            data_def = DataDefinition(
                classification=[
                    BinaryClassification(
                        target="target",
                        prediction_labels="prediction",
                        prediction_probas="prediction_proba",
                        pos_label=self.pos_label,
                    )
                ]
            )

            current = Dataset.from_pandas(eval_df, data_definition=data_def)
            report = Report(metrics=[ClassificationPreset()])
            evaluation = report.run(current, None)

            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "report.html")
                evaluation.save_html(path)
                self.storage.upload_html(path, self.s3_config.eval_report_key)

            logger.info("Evaluation report uploaded to S3")
            return pr_auc
        except Exception as exc:
            logger.error(f"Error evaluating model: {exc}")
            raise EvaluationError(f"Error evaluating model") from exc