from mlops_pipeline.protocols.model_versioning_protocol import ModelVersioningProtocol
from mlops_pipeline.configs.wandb_config import WandbConfig
import logging
from mlops_pipeline.exceptions import PromotionError

logger = logging.getLogger(__name__)


class ModelPromotion:
    """Handles model promotion from staging to production based on performance.

    This class compares the new staging model's performance against the
    current production model and promotes it if the new model performs better.

    Attributes:
        pr_auc_test: The PR-AUC score of the new model on test data.
        model_versioning_repository: Repository for model versioning operations.
        wandb_config: Configuration for W&B artifact names.
    """

    def __init__(self, pr_auc_test: float, model_versioning_repository: ModelVersioningProtocol,
                 wandb_config: WandbConfig) -> None:
        """Initialize the ModelPromotion with test metrics and configuration.

        Args:
            pr_auc_test: The PR-AUC score of the new model on the test dataset.
            model_versioning_repository: Repository for loading and promoting artifacts.
            wandb_config: W&B configuration containing artifact names.
        """
        self.pr_auc_test = pr_auc_test
        self.model_versioning_repository = model_versioning_repository
        self.wandb_config = wandb_config

    def promote_model(self) -> None:
        """Promote the staging model to production if it performs better.

        Compares the staging model's test PR-AUC against the current production
        model's PR-AUC. If no production model exists or the staging model is
        better, promotes both the model and preprocessor artifacts.

        Returns:
            None

        Raises:
            PromotionError: If the promotion operation fails.
        """
        try:
            production_model = None
            pr_auc_production = None
            try:
                production_model, pr_auc_production = self.model_versioning_repository.stream_load_from_alias(
                    artifact_name=self.wandb_config.model_artifact_name,
                    alias="production")
                logger.info(f"Production model loaded successfully and start testing it now")
            except Exception as e:
                logger.info(f"No model in production")

            if production_model is None or self.pr_auc_test > pr_auc_production:
                self.model_versioning_repository.promote_artifact(self.wandb_config.model_artifact_name,
                                                                    "staging", "production")
                self.model_versioning_repository.promote_artifact(self.wandb_config.preprocessor_artifact_name,
                                                                    "staging", "production")
                logger.info(f"Model promoted to production successfully")
            else:
                logger.info(f"Best staging model is not better than production model")
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            raise PromotionError(f"Error promoting model") from e